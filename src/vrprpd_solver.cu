// ============================================================================
// VRP-RPD Solver: ALNS → BRKGA Hybrid
//
// ALNS solves the full VRP-RPD problem, then its solution is encoded into
// BRKGA3 CHROMOSOME mode (4*N genes: drop priorities, pickup priorities,
// drop agent hints, pickup agent hints).
//
// The decoder controls the FULL schedule — both drops AND pickups — enabling
// exact warm-start from ALNS solutions with no information loss.
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
#include <numeric>

// ---- CPU-side decoder (mirrors GPU kernel exactly) ----
// Chromosome layout: [drop_prio | pickup_prio | drop_hint | pickup_hint]
//                     0..N-1      N..2N-1       2N..3N-1    3N..4N-1
static float cpuDecode(
    const std::vector<brkga3::Gene>& genes,
    const std::vector<float>& travel,
    const std::vector<float>& proc,
    int num_locations, int n_agents, int resources_k, int n_customers)
{
    int nc = n_customers;
    int n_ops = 2 * nc;

    // Build operation list and sort by priority
    std::vector<std::pair<float, int>> ops(n_ops);
    for (int i = 0; i < nc; ++i) {
        ops[i]      = {genes[i],      i};       // drop
        ops[nc + i] = {genes[nc + i], nc + i};   // pickup
    }
    std::sort(ops.begin(), ops.end());

    // Agent state
    struct AS { float time = 0; int pos = 0; int inv = 0; };
    std::vector<AS> ag(n_agents);
    for (int a = 0; a < n_agents; ++a) ag[a].inv = resources_k;

    std::vector<float> dropoff_time(nc, -1.0f);
    std::vector<bool> dropped(nc, false), picked_up(nc, false);
    std::vector<bool> op_done(n_ops, false);

    int completed = 0;
    for (int pass = 0; pass < n_ops && completed < n_ops; ++pass) {
        bool made_progress = false;

        for (int idx = 0; idx < n_ops; ++idx) {
            int op = ops[idx].second;
            if (op_done[op]) continue;

            bool is_drop = (op < nc);
            int cust = is_drop ? op : (op - nc);
            int node = cust + 1;

            int best_agent = -1;
            float best_time = FLT_MAX;

            if (is_drop) {
                if (dropped[cust]) continue;
                int hint_a = std::min(std::max((int)(genes[2 * nc + cust] * n_agents), 0),
                                      n_agents - 1);
                if (ag[hint_a].inv > 0) {
                    float t = ag[hint_a].time + travel[ag[hint_a].pos * num_locations + node];
                    best_time = t; best_agent = hint_a;
                } else {
                    for (int a = 0; a < n_agents; ++a) {
                        if (ag[a].inv > 0) {
                            float t = ag[a].time + travel[ag[a].pos * num_locations + node];
                            if (t < best_time) { best_time = t; best_agent = a; }
                        }
                    }
                }
                if (best_agent >= 0) {
                    ag[best_agent].time += travel[ag[best_agent].pos * num_locations + node];
                    ag[best_agent].pos = node;
                    ag[best_agent].inv--;
                    dropoff_time[cust] = ag[best_agent].time;
                    dropped[cust] = true;
                    op_done[op] = true;
                    completed++;
                    made_progress = true;
                }
            } else {
                if (!dropped[cust] || picked_up[cust]) continue;
                float ready = dropoff_time[cust] + proc[cust];
                int hint_a = std::min(std::max((int)(genes[3 * nc + cust] * n_agents), 0),
                                      n_agents - 1);
                if (ag[hint_a].inv < resources_k) {
                    float t = ag[hint_a].time + travel[ag[hint_a].pos * num_locations + node];
                    best_time = std::max(t, ready); best_agent = hint_a;
                } else {
                    for (int a = 0; a < n_agents; ++a) {
                        if (ag[a].inv < resources_k) {
                            float t = ag[a].time + travel[ag[a].pos * num_locations + node];
                            float arrival = std::max(t, ready);
                            if (arrival < best_time) { best_time = arrival; best_agent = a; }
                        }
                    }
                }
                if (best_agent >= 0) {
                    float t = ag[best_agent].time + travel[ag[best_agent].pos * num_locations + node];
                    ag[best_agent].time = std::max(t, ready);
                    ag[best_agent].pos = node;
                    ag[best_agent].inv++;
                    picked_up[cust] = true;
                    op_done[op] = true;
                    completed++;
                    made_progress = true;
                }
            }
        }
        if (!made_progress) break;
    }

    float ms = 0;
    for (int a = 0; a < n_agents; ++a) {
        float t = ag[a].time + travel[ag[a].pos * num_locations + 0];
        if (t > ms) ms = t;
    }
    int unserviced = 0;
    for (int c = 0; c < nc; ++c) {
        if (!dropped[c])   unserviced++;
        if (!picked_up[c]) unserviced++;
    }
    return ms + unserviced * 100000.0f;
}

// ---- Encode FULL ALNS solution as 4*N chromosome ----
// genes[0..N-1]:     drop priorities (time-sorted)
// genes[N..2N-1]:    pickup priorities (time-sorted)
// genes[2N..3N-1]:   drop agent hints
// genes[3N..4N-1]:   pickup agent hints
static std::vector<brkga3::Gene> alnsToChromosome(
    const VRPRPDConfig::HostSolution& sol,
    int n_customers, int n_agents)
{
    int nc = n_customers;
    std::vector<brkga3::Gene> genes(4 * nc, 0.5f);

    // Collect ALL operations with their times and agents
    struct OpInfo { float time; int customer; int agent; bool is_drop; };
    std::vector<OpInfo> drops, pickups;

    for (const auto& route : sol.routes) {
        for (const auto& op : route.operations) {
            if (op.customer < 0 || op.customer >= nc) continue;
            if (op.type == 0)  // DROP_OFF
                drops.push_back({op.scheduled_time, op.customer, route.agent_id, true});
            else  // PICK_UP
                pickups.push_back({op.scheduled_time, op.customer, route.agent_id, false});
        }
    }

    // Sort drops and pickups by scheduled time
    std::sort(drops.begin(), drops.end(),
        [](const OpInfo& a, const OpInfo& b) { return a.time < b.time; });
    std::sort(pickups.begin(), pickups.end(),
        [](const OpInfo& a, const OpInfo& b) { return a.time < b.time; });

    // Interleave drop and pickup priorities so that the ALNS execution order
    // is preserved. We assign priorities in [0, 1) such that the merged sort
    // reproduces the ALNS interleaved sequence.

    // Collect all ops in ALNS execution order (time-sorted across both types)
    std::vector<OpInfo> all_ops;
    all_ops.insert(all_ops.end(), drops.begin(), drops.end());
    all_ops.insert(all_ops.end(), pickups.begin(), pickups.end());
    std::sort(all_ops.begin(), all_ops.end(),
        [](const OpInfo& a, const OpInfo& b) { return a.time < b.time; });

    // Assign priorities: operation at rank r gets priority (r + 0.5) / total_ops
    int total_ops = (int)all_ops.size();
    float inv_total = 1.0f / std::max(total_ops, 1);

    for (int rank = 0; rank < total_ops; ++rank) {
        const auto& op = all_ops[rank];
        float priority = (rank + 0.5f) * inv_total;

        if (op.is_drop) {
            genes[op.customer] = priority;                          // drop priority
            genes[2 * nc + op.customer] = (op.agent + 0.5f) / n_agents;  // drop agent hint
        } else {
            genes[nc + op.customer] = priority;                     // pickup priority
            genes[3 * nc + op.customer] = (op.agent + 0.5f) / n_agents;  // pickup agent hint
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
    int resources_k = vrp_data.resources_per_agent;

    if (cfg.verbose) {
        std::printf("  Customers:  %d\n", n_customers);
        std::printf("  Agents:     %d\n", n_agents);
        std::printf("  Resources:  %d per agent\n", resources_k);
        std::printf("  Locations:  %d (depot + customers)\n", num_locations);
    }

    std::vector<float> host_travel(vrp_data.travel_times,
        vrp_data.travel_times + num_locations * num_locations);
    std::vector<float> host_proc(vrp_data.processing_times,
        vrp_data.processing_times + n_customers);

    VRPRPDConfig::HostSolution alns_solution;
    bool have_alns_solution = false;

    // ================================================================
    // Phase 1: ALNS
    // ================================================================
    if (!cfg.cold_start) {
        if (cfg.verbose)
            std::printf("\n--- Phase 1: ALNS ---\n");

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
    // Phase 2: BRKGA (4*N genes: full operation control)
    // ================================================================
    if (cfg.verbose)
        std::printf("\n--- Phase 2: BRKGA (4*N chromosome, full op control) ---\n");

    // Clear CUDA state
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
    if (brkga_num_gpus < 0)
        cudaGetDeviceCount(&brkga_num_gpus);

    auto brkga_cfg = brkga3::BrkgaConfig::Builder()
        .chromosomeLength(4 * n_customers)
        .decodeMode(brkga3::DecodeMode::CHROMOSOME)
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

    auto decoder_factory = [&](int gpu_id) -> std::unique_ptr<brkga3::GpuDecoder> {
        return std::make_unique<vrprpd::VrpRpdDecoder>(
            host_travel, host_proc, num_locations, n_agents, resources_k);
    };

    brkga3::Brkga brkga(brkga_cfg, decoder_factory);

    // Warm-start: encode full ALNS solution
    if (have_alns_solution) {
        auto genes = alnsToChromosome(alns_solution, n_customers, n_agents);

        float ws_val = cpuDecode(genes, host_travel, host_proc,
                                 num_locations, n_agents, resources_k, n_customers);

        if (cfg.verbose) {
            std::printf("  Warm-start CPU decode: %.2f (ALNS = %.2f)\n",
                        ws_val, alns_solution.makespan);
        }

        brkga.injectChromosome(genes);
        result.brkga_initial = brkga.getBestFitness();
        if (cfg.verbose)
            std::printf("  Warm-start injected (GPU): %.2f\n", result.brkga_initial);
    } else {
        result.brkga_initial = brkga.getBestFitness();
        if (cfg.verbose)
            std::printf("  Cold start initial: %.2f\n", result.brkga_initial);
    }

    if (cfg.verbose) {
        std::printf("  GPUs: %d | Populations: %u | Pop size: %u | Genes: %d\n",
                    brkga_num_gpus, brkga_cfg.totalPopulations(),
                    brkga_cfg.population_size, 4 * n_customers);
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

    result.final_objective = result.brkga_final;
    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_s = std::chrono::duration<double>(total_end - total_start).count();

    // ================================================================
    // Route reconstruction from best chromosome
    // ================================================================
    auto best_genes = brkga.getBestChromosome();

    int nc = n_customers;
    int n_ops = 2 * nc;

    // Sort operations by priority (same as decoder)
    std::vector<std::pair<float, int>> ops(n_ops);
    for (int i = 0; i < nc; ++i) {
        ops[i]      = {best_genes[i],      i};
        ops[nc + i] = {best_genes[nc + i], nc + i};
    }
    std::sort(ops.begin(), ops.end());

    // Reconstruct routes
    struct AgentState { float time = 0; int pos = 0; int inv = 0; };
    std::vector<AgentState> agents(n_agents);
    for (int a = 0; a < n_agents; ++a) agents[a].inv = resources_k;

    std::vector<float> dropoff_time(nc, -1.0f);
    std::vector<bool> dropped(nc, false), picked_up(nc, false);
    std::vector<bool> op_done(n_ops, false);

    struct StopInfo { int node; char op; float time; };
    std::vector<std::vector<StopInfo>> agent_stops(n_agents);

    int completed = 0;
    for (int pass = 0; pass < n_ops && completed < n_ops; ++pass) {
        bool made_progress = false;
        for (int idx = 0; idx < n_ops; ++idx) {
            int op = ops[idx].second;
            if (op_done[op]) continue;

            bool is_drop = (op < nc);
            int cust = is_drop ? op : (op - nc);
            int node = cust + 1;

            int best_agent = -1;
            float best_time = FLT_MAX;

            if (is_drop) {
                if (dropped[cust]) continue;
                int hint_a = std::min(std::max((int)(best_genes[2*nc + cust] * n_agents), 0),
                                      n_agents - 1);
                if (agents[hint_a].inv > 0) {
                    best_time = agents[hint_a].time + host_travel[agents[hint_a].pos * num_locations + node];
                    best_agent = hint_a;
                } else {
                    for (int a = 0; a < n_agents; ++a) {
                        if (agents[a].inv > 0) {
                            float t = agents[a].time + host_travel[agents[a].pos * num_locations + node];
                            if (t < best_time) { best_time = t; best_agent = a; }
                        }
                    }
                }
                if (best_agent >= 0) {
                    agents[best_agent].time += host_travel[agents[best_agent].pos * num_locations + node];
                    agents[best_agent].pos = node;
                    agents[best_agent].inv--;
                    dropoff_time[cust] = agents[best_agent].time;
                    dropped[cust] = true;
                    op_done[op] = true;
                    agent_stops[best_agent].push_back({node, 'D', agents[best_agent].time});
                    completed++;
                    made_progress = true;
                }
            } else {
                if (!dropped[cust] || picked_up[cust]) continue;
                float ready = dropoff_time[cust] + host_proc[cust];
                int hint_a = std::min(std::max((int)(best_genes[3*nc + cust] * n_agents), 0),
                                      n_agents - 1);
                if (agents[hint_a].inv < resources_k) {
                    float t = agents[hint_a].time + host_travel[agents[hint_a].pos * num_locations + node];
                    best_time = std::max(t, ready); best_agent = hint_a;
                } else {
                    for (int a = 0; a < n_agents; ++a) {
                        if (agents[a].inv < resources_k) {
                            float t = agents[a].time + host_travel[agents[a].pos * num_locations + node];
                            float arrival = std::max(t, ready);
                            if (arrival < best_time) { best_time = arrival; best_agent = a; }
                        }
                    }
                }
                if (best_agent >= 0) {
                    float t = agents[best_agent].time + host_travel[agents[best_agent].pos * num_locations + node];
                    agents[best_agent].time = std::max(t, ready);
                    agents[best_agent].pos = node;
                    agents[best_agent].inv++;
                    picked_up[cust] = true;
                    op_done[op] = true;
                    agent_stops[best_agent].push_back({node, 'P', agents[best_agent].time});
                    completed++;
                    made_progress = true;
                }
            }
        }
        if (!made_progress) break;
    }

    // Build solution JSON
    std::ostringstream ss;
    ss << std::fixed;
    ss << "{\n";
    ss << "    \"problem\": {\n";
    ss << "      \"n_customers\": " << n_customers << ",\n";
    ss << "      \"n_agents\": " << n_agents << ",\n";
    ss << "      \"resources_per_agent\": " << resources_k << "\n";
    ss << "    },\n";
    ss << "    \"makespan\": " << result.brkga_final << ",\n";
    ss << "    \"routes\": [\n";

    for (int a = 0; a < n_agents; ++a) {
        float ft = agents[a].time + host_travel[agents[a].pos * num_locations + 0];
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

    delete[] vrp_data.travel_times;
    delete[] vrp_data.processing_times;

    return result;
}
