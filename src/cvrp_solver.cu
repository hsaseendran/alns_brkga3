// ============================================================================
// CVRP Solver: ALNS (TSP relaxation) → BRKGA (CVRP decoder) Hybrid
//
// Key insight: CVRP = TSP + capacity constraints. ALNS solves the TSP
// relaxation (ignoring capacity) to find a good client ordering. BRKGA3's
// greedy-split decoder then handles capacity by splitting the permutation
// into feasible routes.
// ============================================================================

#include "cvrp_solver.hpp"

// --- ALNS (reuse TSP config for the relaxation) ---
#include "tsp_config.cuh"
#include <alns/gpu/multi_gpu.hpp>

// --- BRKGA3 ---
#include <brkga3/brkga.cuh>
#include <brkga3/config.hpp>
#include <brkga3/types.cuh>
#include "cvrp_decoder.cuh"
#include "cvrp_instance.hpp"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <memory>
#include <sstream>
#include <algorithm>

// Convert ALNS TSP tour (over all nodes including depot) to BRKGA CVRP chromosome.
// The CVRP decoder uses PERMUTATION mode with n_clients genes.
// We extract client ordering from the TSP tour, skipping the depot.
static std::vector<brkga3::Gene> tspTourToCvrpChromosome(
    const std::vector<int>& tour,
    std::uint32_t n_clients,
    std::uint32_t depot)
{
    std::vector<brkga3::Gene> genes(n_clients, 0.5f);

    // Extract clients in tour order (skip depot)
    std::vector<int> client_order;
    client_order.reserve(n_clients);
    for (int node : tour) {
        if (static_cast<std::uint32_t>(node) == depot) continue;
        // Map node index to 0-based client index
        int client_idx = (static_cast<std::uint32_t>(node) > depot)
                       ? node - 1 : node;
        if (client_idx >= 0 && static_cast<std::uint32_t>(client_idx) < n_clients)
            client_order.push_back(client_idx);
    }

    float inv_n = 1.0f / static_cast<float>(n_clients);
    for (std::uint32_t rank = 0; rank < client_order.size() && rank < n_clients; ++rank) {
        genes[client_order[rank]] = (rank + 0.5f) * inv_n;
    }
    return genes;
}

SolverResult solveCvrp(const std::string& instance_path, const SolverConfig& cfg) {
    SolverResult result;
    result.problem = "cvrp";
    result.instance = instance_path;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ---- Load CVRP instance ----
    if (cfg.verbose) std::printf("Loading CVRP instance: %s\n", instance_path.c_str());
    auto instance = cvrp::loadCVRPLIB(instance_path);
    if (cfg.verbose) {
        std::printf("  Name:      %s\n", instance.name.c_str());
        std::printf("  Dimension: %u (depot + %u clients)\n",
                    instance.dimension, instance.n_clients);
        std::printf("  Capacity:  %u\n", instance.capacity);
        std::printf("  Depot:     node %u\n", instance.depot);
        if (instance.optimal > 0)
            std::printf("  Known optimal: %.2f\n", instance.optimal);
    }

    std::vector<int> alns_tour;

    // ================================================================
    // Phase 1: ALNS (TSP relaxation on CVRP distance matrix)
    // ================================================================
    if (!cfg.cold_start) {
        if (cfg.verbose) {
            std::printf("\n--- Phase 1: ALNS (TSP relaxation on CVRP distances) ---\n");
        }

        // Create ALNS TSP ProblemData from CVRP distance matrix
        TSPConfig::ProblemData alns_data;
        alns_data.num_cities = instance.dimension;  // all nodes including depot
        alns_data.pitch = instance.dimension;
        alns_data.distances = new float[instance.dimension * instance.dimension];
        std::memcpy(alns_data.distances, instance.distances.data(),
                    instance.dimension * instance.dimension * sizeof(float));

        // Configure ALNS
        alns::ALNSRuntimeConfig alns_cfg;
        alns_cfg.max_iterations = 500000;
        alns_cfg.max_time_seconds = cfg.alns_time;
        alns_cfg.num_gpus = cfg.alns_gpus;
        alns_cfg.solutions_per_gpu = cfg.alns_solutions_per_gpu;
        alns_cfg.cooling_rate = 0.9998f;
        alns_cfg.initial_temp_factor = 0.05f;
        alns_cfg.max_removal_fraction = 0.3f;
        alns_cfg.min_removal = 3;
        alns_cfg.verbose = cfg.verbose;

        if (cfg.verbose) {
            std::printf("  GPUs: %d | Solutions/GPU: %d | Time limit: %.0fs\n",
                        alns_cfg.num_gpus, alns_cfg.solutions_per_gpu, alns_cfg.max_time_seconds);
            std::printf("  Note: solving TSP relaxation (ignoring capacity)\n");
        }

        auto alns_start = std::chrono::high_resolution_clock::now();

        alns::MultiGPUSolver<TSPConfig> solver(alns_cfg);
        TSPConfig::HostSolution alns_result = solver.solve(alns_data);

        auto alns_end = std::chrono::high_resolution_clock::now();
        result.alns_time_s = std::chrono::duration<double>(alns_end - alns_start).count();
        result.alns_objective = alns_result.total_distance;
        alns_tour = alns_result.tour;

        if (cfg.verbose) {
            std::printf("  ALNS completed: TSP objective = %.2f, time = %.1fs\n",
                        result.alns_objective, result.alns_time_s);
        }

        delete[] alns_data.distances;
    }

    // ================================================================
    // Phase 2: BRKGA (CVRP decoder with greedy split)
    // ================================================================
    if (cfg.verbose) {
        std::printf("\n--- Phase 2: BRKGA (CVRP greedy-split decoder) ---\n");
    }

    int brkga_num_gpus = cfg.brkga_gpus;
    if (brkga_num_gpus < 0) {
        cudaGetDeviceCount(&brkga_num_gpus);
    }

    const int num_elites = std::max(2, cfg.brkga_pop_size / 10);
    const int num_mutants = std::max(2, cfg.brkga_pop_size / 10);

    auto brkga_cfg = brkga3::BrkgaConfig::Builder()
        .chromosomeLength(instance.n_clients)
        .decodeMode(brkga3::DecodeMode::PERMUTATION)
        .optimizationSense(brkga3::OptimizationSense::MINIMIZE)
        .populationSize(cfg.brkga_pop_size)
        .numElites(num_elites)
        .numMutants(num_mutants)
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
        return std::make_unique<cvrp::CvrpDecoder>(
            instance.distances, instance.demands,
            instance.dimension, instance.depot, instance.capacity);
    };

    brkga3::Brkga brkga(brkga_cfg, decoder_factory);

    // Warm-start injection
    if (!alns_tour.empty()) {
        auto genes = tspTourToCvrpChromosome(alns_tour, instance.n_clients, instance.depot);
        brkga.injectChromosome(genes);
        result.brkga_initial = brkga.getBestFitness();
        if (cfg.verbose) {
            std::printf("  Warm-start injected: %.2f (from TSP tour)\n", result.brkga_initial);
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
            if (instance.optimal > 0) {
                float gap = (best - instance.optimal) / instance.optimal * 100.0f;
                std::printf("  Gen %5d | Best: %12.2f | Gap: %5.2f%% | Time: %.3fs\n",
                            g, best, gap, elapsed);
            } else {
                std::printf("  Gen %5d | Best: %12.2f | Time: %.3fs\n", g, best, elapsed);
            }
        }
    }

    auto brkga_end = std::chrono::high_resolution_clock::now();
    result.brkga_time_s = std::chrono::duration<double>(brkga_end - brkga_start).count();
    result.brkga_final = brkga.getBestFitness();
    result.brkga_generations = cfg.brkga_gens;

    if (cfg.verbose) {
        std::printf("  BRKGA completed: objective = %.2f, time = %.1fs\n",
                    result.brkga_final, result.brkga_time_s);
    }

    // Final
    result.final_objective = result.brkga_final;
    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_s = std::chrono::duration<double>(total_end - total_start).count();

    // Build solution JSON
    std::ostringstream ss;
    ss << "{\n    \"total_distance\": " << result.brkga_final << ",\n";
    ss << "    \"n_clients\": " << instance.n_clients << ",\n";
    ss << "    \"capacity\": " << instance.capacity;
    if (instance.optimal > 0) {
        float gap = (result.brkga_final - instance.optimal) / instance.optimal * 100.0f;
        ss << ",\n    \"known_optimal\": " << instance.optimal;
        ss << ",\n    \"gap_pct\": " << gap;
    }
    ss << "\n  }";
    result.solution_json = ss.str();

    return result;
}
