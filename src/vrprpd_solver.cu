// ============================================================================
// VRP-RPD Solver: ALNS → BRKGA Hybrid
//
// ALNS solves the full VRP-RPD problem (with cross-agent pickup, inventory
// constraints, etc.), then the solution's customer ordering is injected into
// BRKGA3 with a simplified greedy decoder for further refinement.
// ============================================================================

#include "vrprpd_solver.hpp"

// --- ALNS ---
#include "vrp_rpd_config.cuh"
#include <alns/gpu/multi_gpu.hpp>

// --- BRKGA3 ---
#include <brkga3/brkga.cuh>
#include <brkga3/config.hpp>
#include <brkga3/types.cuh>
#include "vrprpd_decoder.cuh"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <memory>
#include <sstream>
#include <algorithm>

// Convert ALNS VRP-RPD solution to BRKGA permutation chromosome.
// Extracts customer ordering from drop-off operations sorted by scheduled time.
static std::vector<brkga3::Gene> vrprpdSolutionToChromosome(
    const VRPRPDConfig::HostSolution& sol,
    int n_customers)
{
    // Collect all drop-off operations with their scheduled times
    std::vector<std::pair<float, int>> drops;
    for (const auto& route : sol.routes) {
        for (const auto& op : route.operations) {
            if (op.type == 0) {  // DROP_OFF
                drops.push_back({op.scheduled_time, op.customer});
            }
        }
    }

    // Sort by scheduled time → customer execution order
    std::sort(drops.begin(), drops.end());

    // Convert to random-key chromosome
    std::vector<brkga3::Gene> genes(n_customers, 0.5f);
    float inv_n = 1.0f / static_cast<float>(n_customers);
    for (std::size_t rank = 0; rank < drops.size() && rank < static_cast<std::size_t>(n_customers); ++rank) {
        int cust = drops[rank].second;
        if (cust >= 0 && cust < n_customers) {
            genes[cust] = (rank + 0.5f) * inv_n;
        }
    }
    return genes;
}

SolverResult solveVrpRpd(const std::string& instance_path, const SolverConfig& cfg) {
    SolverResult result;
    result.problem = "vrprpd";
    result.instance = instance_path;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ---- Load VRP-RPD instance ----
    if (cfg.verbose) std::printf("Loading VRP-RPD instance: %s\n", instance_path.c_str());

    VRPRPDConfig::ProblemData vrp_data;
    if (!cfg.processing_times_file.empty()) {
        vrp_data = VRPRPDConfig::load_problem(
            instance_path, cfg.processing_times_file,
            cfg.vrprpd_agents, cfg.vrprpd_resources);
    } else {
        vrp_data = VRPRPDConfig::load_problem(instance_path);
        vrp_data.num_agents = cfg.vrprpd_agents;
        vrp_data.resources_per_agent = cfg.vrprpd_resources;
    }

    int n_customers = vrp_data.num_customers;
    int n_agents = vrp_data.num_agents;
    int num_locations = vrp_data.num_locations;

    if (cfg.verbose) {
        std::printf("  Customers:  %d\n", n_customers);
        std::printf("  Agents:     %d\n", n_agents);
        std::printf("  Resources:  %d per agent\n", vrp_data.resources_per_agent);
        std::printf("  Locations:  %d (depot + customers)\n", num_locations);
    }

    // Keep host copies of travel times and processing times for BRKGA decoder
    std::vector<float> host_travel(vrp_data.travel_times,
        vrp_data.travel_times + num_locations * num_locations);
    std::vector<float> host_proc(vrp_data.processing_times,
        vrp_data.processing_times + n_customers);

    VRPRPDConfig::HostSolution alns_solution;
    bool have_alns_solution = false;

    // ================================================================
    // Phase 1: ALNS (full VRP-RPD)
    // ================================================================
    if (!cfg.cold_start) {
        if (cfg.verbose) {
            std::printf("\n--- Phase 1: ALNS (VRP-RPD with cross-agent pickup) ---\n");
        }

        alns::ALNSRuntimeConfig alns_cfg;
        alns_cfg.max_iterations = 25000;
        alns_cfg.max_time_seconds = cfg.alns_time;
        alns_cfg.num_gpus = cfg.alns_gpus;
        alns_cfg.solutions_per_gpu = cfg.alns_solutions_per_gpu;
        alns_cfg.cooling_rate = 0.99975f;
        alns_cfg.initial_temp_factor = 0.30f;
        alns_cfg.max_removal_fraction = 0.05f;
        alns_cfg.min_removal = 4;
        alns_cfg.stagnation_threshold = 2000;
        alns_cfg.reheat_factor = 0.5f;
        alns_cfg.verbose = cfg.verbose;

        if (cfg.verbose) {
            std::printf("  GPUs: %d | Solutions/GPU: %d | Time limit: %.0fs\n",
                        alns_cfg.num_gpus, alns_cfg.solutions_per_gpu, alns_cfg.max_time_seconds);
        }

        auto alns_start = std::chrono::high_resolution_clock::now();

        alns::MultiGPUSolver<VRPRPDConfig> solver(alns_cfg);
        alns_solution = solver.solve(vrp_data);
        have_alns_solution = true;

        auto alns_end = std::chrono::high_resolution_clock::now();
        result.alns_time_s = std::chrono::duration<double>(alns_end - alns_start).count();
        result.alns_objective = alns_solution.makespan;

        if (cfg.verbose) {
            std::printf("  ALNS completed: makespan = %.2f, time = %.1fs\n",
                        result.alns_objective, result.alns_time_s);
        }
    }

    // ================================================================
    // Phase 2: BRKGA (simplified VRP-RPD decoder)
    // ================================================================
    if (cfg.verbose) {
        std::printf("\n--- Phase 2: BRKGA (VRP-RPD greedy decoder) ---\n");
    }

    // Clear any leftover CUDA error state from ALNS
    {
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        for (int d = 0; d < dev_count; ++d) {
            cudaSetDevice(d);
            cudaDeviceSynchronize();
            cudaGetLastError();
        }
        cudaSetDevice(0);
    }

    int brkga_num_gpus = cfg.brkga_gpus;
    if (brkga_num_gpus < 0) {
        cudaGetDeviceCount(&brkga_num_gpus);
    }

    auto brkga_cfg = brkga3::BrkgaConfig::Builder()
        .chromosomeLength(n_customers)
        .decodeMode(brkga3::DecodeMode::PERMUTATION)
        .optimizationSense(brkga3::OptimizationSense::MINIMIZE)
        .populationSize(cfg.brkga_pop_size)
        .numElites(25)
        .numMutants(25)
        .numParents(2, brkga3::BiasType::LINEAR, 1)
        .numGpus(brkga_num_gpus)
        .numPopulations(cfg.brkga_pops_per_gpu)
        .migrationInterval(50)
        .numElitesToMigrate(2)
        .threadsPerBlock(256)
        .seed(cfg.brkga_seed)
        .build();

    brkga_cfg.bias_cdf = {0.75f, 1.0f};

    int resources_k = vrp_data.resources_per_agent;

    auto decoder_factory = [&](int gpu_id) -> std::unique_ptr<brkga3::GpuDecoder> {
        return std::make_unique<vrprpd::VrpRpdDecoder>(
            host_travel, host_proc, num_locations, n_agents, resources_k);
    };

    brkga3::Brkga brkga(brkga_cfg, decoder_factory);

    // Warm-start injection
    if (have_alns_solution) {
        auto genes = vrprpdSolutionToChromosome(alns_solution, n_customers);
        brkga.injectChromosome(genes);
        result.brkga_initial = brkga.getBestFitness();
        if (cfg.verbose) {
            std::printf("  Warm-start injected: %.2f\n", result.brkga_initial);
        }
    } else {
        result.brkga_initial = brkga.getBestFitness();
        if (cfg.verbose) {
            std::printf("  Cold start initial: %.2f\n", result.brkga_initial);
        }
    }

    if (cfg.verbose) {
        std::printf("  GPUs: %d | Populations: %u | Pop size: %u\n",
                    brkga_num_gpus, brkga_cfg.totalPopulations(), brkga_cfg.population_size);
    }

    auto brkga_start = std::chrono::high_resolution_clock::now();

    for (int g = 1; g <= cfg.brkga_gens; ++g) {
        brkga.evolve();

        if (cfg.verbose && (g % 100 == 0 || g == 1)) {
            auto best = brkga.getBestFitness();
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - brkga_start).count();
            std::printf("  Gen %5d | Best: %12.2f | Time: %.3fs\n", g, best, elapsed);
        }
    }

    auto brkga_end = std::chrono::high_resolution_clock::now();
    result.brkga_time_s = std::chrono::duration<double>(brkga_end - brkga_start).count();
    result.brkga_final = brkga.getBestFitness();
    result.brkga_generations = cfg.brkga_gens;

    if (cfg.verbose) {
        std::printf("  BRKGA completed: makespan = %.2f, time = %.1fs\n",
                    result.brkga_final, result.brkga_time_s);
    }

    // Final
    result.final_objective = result.brkga_final;
    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_s = std::chrono::duration<double>(total_end - total_start).count();

    // Reconstruct BRKGA solution routes from best permutation
    // (mirrors the resource-aware GPU kernel logic)
    auto best_perm = brkga.getBestPermutation();

    struct AgentState { float time = 0.0f; int pos = 0; int inv = 0; };
    std::vector<AgentState> agents(n_agents);
    for (int a = 0; a < n_agents; ++a) agents[a].inv = resources_k;

    struct PendingDrop { int node; float done_time; };
    std::vector<std::vector<PendingDrop>> pending(n_agents);

    struct StopInfo { int node; char op; float time; };
    std::vector<std::vector<StopInfo>> agent_stops(n_agents);

    for (int i = 0; i < n_customers; ++i) {
        int cust = best_perm[i];
        int node = cust + 1;

        // Find agent with min arrival time
        int best_a = 0;
        float best_t = agents[0].time
            + host_travel[agents[0].pos * num_locations + node];
        for (int a = 1; a < n_agents; ++a) {
            float t = agents[a].time
                + host_travel[agents[a].pos * num_locations + node];
            if (t < best_t) { best_t = t; best_a = a; }
        }

        // If out of resources, pick up earliest-completed drop
        if (agents[best_a].inv == 0 && !pending[best_a].empty()) {
            auto it = std::min_element(pending[best_a].begin(), pending[best_a].end(),
                [](const PendingDrop& a, const PendingDrop& b) { return a.done_time < b.done_time; });
            int pnode = it->node;
            float travel = host_travel[agents[best_a].pos * num_locations + pnode];
            float arrival = agents[best_a].time + travel;
            agents[best_a].time = std::max(arrival, it->done_time);
            agents[best_a].pos = pnode;
            agents[best_a].inv++;
            agent_stops[best_a].push_back({pnode, 'P', agents[best_a].time});
            pending[best_a].erase(it);

            // Recalculate arrival at target
            best_t = agents[best_a].time
                + host_travel[agents[best_a].pos * num_locations + node];
        }

        // Drop off
        agents[best_a].time = best_t;
        agents[best_a].pos = node;
        agents[best_a].inv--;
        agent_stops[best_a].push_back({node, 'D', best_t});
        pending[best_a].push_back({node, best_t + host_proc[cust]});
    }

    // Pick up all remaining pending drops
    for (int a = 0; a < n_agents; ++a) {
        while (!pending[a].empty()) {
            auto it = std::min_element(pending[a].begin(), pending[a].end(),
                [](const PendingDrop& x, const PendingDrop& y) { return x.done_time < y.done_time; });
            int pnode = it->node;
            float travel = host_travel[agents[a].pos * num_locations + pnode];
            float arrival = agents[a].time + travel;
            agents[a].time = std::max(arrival, it->done_time);
            agents[a].pos = pnode;
            agent_stops[a].push_back({pnode, 'P', agents[a].time});
            pending[a].erase(it);
        }
    }

    // Build solution JSON with full route details
    std::ostringstream ss;
    ss << std::fixed;
    ss << "{\n";
    ss << "    \"problem\": {\n";
    ss << "      \"n_customers\": " << n_customers << ",\n";
    ss << "      \"n_agents\": " << n_agents << ",\n";
    ss << "      \"resources_per_agent\": " << vrp_data.resources_per_agent << "\n";
    ss << "    },\n";
    ss << "    \"makespan\": " << result.brkga_final << ",\n";
    ss << "    \"routes\": [\n";

    for (int a = 0; a < n_agents; ++a) {
        float ft = agents[a].time
            + host_travel[agents[a].pos * num_locations + 0];
        ss << "      {\n";
        ss << "        \"agent\": " << a << ",\n";
        ss << "        \"finish_time\": " << ft << ",\n";
        ss << "        \"stops\": [\n";
        for (std::size_t j = 0; j < agent_stops[a].size(); ++j) {
            auto& s = agent_stops[a][j];
            ss << "          {\"node\": " << s.node
               << ", \"op\": \"" << s.op
               << "\", \"time\": " << s.time << "}";
            if (j + 1 < agent_stops[a].size()) ss << ",";
            ss << "\n";
        }
        ss << "        ]\n";
        ss << "      }";
        if (a + 1 < n_agents) ss << ",";
        ss << "\n";
    }

    ss << "    ]\n";
    ss << "  }";
    result.solution_json = ss.str();

    // Cleanup ALNS allocations
    delete[] vrp_data.travel_times;
    delete[] vrp_data.processing_times;

    return result;
}
