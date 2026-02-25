// ============================================================================
// TSP Solver: ALNS → BRKGA Hybrid
// ============================================================================

#include "tsp_solver.hpp"

// --- ALNS ---
#include "tsp_config.cuh"
#include <alns/gpu/multi_gpu.hpp>

// --- BRKGA3 ---
#include <brkga3/brkga.cuh>
#include <brkga3/config.hpp>
#include <brkga3/types.cuh>
#include "tsp_decoder.cuh"
#include "tsp_instance.hpp"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <memory>
#include <sstream>

// Convert ALNS tour (ordered city indices) to BRKGA random-key chromosome.
// gene[tour[rank]] = (rank + 0.5) / n ensures sorting genes reproduces the tour.
static std::vector<brkga3::Gene> tourToChromosome(const std::vector<int>& tour,
                                                   std::uint32_t n) {
    std::vector<brkga3::Gene> genes(n);
    float inv_n = 1.0f / static_cast<float>(n);
    for (std::uint32_t rank = 0; rank < n; ++rank) {
        genes[tour[rank]] = (rank + 0.5f) * inv_n;
    }
    return genes;
}

SolverResult solveTsp(const std::string& instance_path, const SolverConfig& cfg) {
    SolverResult result;
    result.problem = "tsp";
    result.instance = instance_path;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ---- Load instance using BRKGA3's loader (shared distance matrix) ----
    if (cfg.verbose) std::printf("Loading TSP instance: %s\n", instance_path.c_str());
    auto instance = tsp::loadTSPLIB(instance_path);
    if (cfg.verbose) {
        std::printf("  Name:   %s\n", instance.name.c_str());
        std::printf("  Cities: %u\n", instance.n);
        if (instance.optimal > 0)
            std::printf("  Known optimal: %.2f\n", instance.optimal);
    }

    std::vector<int> alns_tour;

    // ================================================================
    // Phase 1: ALNS
    // ================================================================
    if (!cfg.cold_start) {
        if (cfg.verbose) {
            std::printf("\n--- Phase 1: ALNS (Adaptive Large Neighborhood Search) ---\n");
        }

        // Create ALNS ProblemData from BRKGA3's TspInstance
        TSPConfig::ProblemData alns_data;
        alns_data.num_cities = instance.n;
        alns_data.pitch = instance.n;
        alns_data.distances = new float[instance.n * instance.n];
        std::memcpy(alns_data.distances, instance.distances.data(),
                    instance.n * instance.n * sizeof(float));

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
        }

        auto alns_start = std::chrono::high_resolution_clock::now();

        alns::MultiGPUSolver<TSPConfig> solver(alns_cfg);
        TSPConfig::HostSolution alns_result = solver.solve(alns_data);

        auto alns_end = std::chrono::high_resolution_clock::now();
        result.alns_time_s = std::chrono::duration<double>(alns_end - alns_start).count();
        result.alns_objective = alns_result.total_distance;
        alns_tour = alns_result.tour;

        if (cfg.verbose) {
            std::printf("  ALNS completed: objective = %.2f, time = %.1fs\n",
                        result.alns_objective, result.alns_time_s);
        }

        delete[] alns_data.distances;
    }

    // ================================================================
    // Phase 2: BRKGA
    // ================================================================
    if (cfg.verbose) {
        std::printf("\n--- Phase 2: BRKGA (Biased Random-Key Genetic Algorithm) ---\n");
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

    // Determine number of GPUs for BRKGA
    int brkga_num_gpus = cfg.brkga_gpus;
    if (brkga_num_gpus < 0) {
        cudaGetDeviceCount(&brkga_num_gpus);
    }

    auto brkga_cfg = brkga3::BrkgaConfig::Builder()
        .chromosomeLength(instance.n)
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

    auto decoder_factory = [&](int gpu_id) -> std::unique_ptr<brkga3::GpuDecoder> {
        return std::make_unique<tsp::TspDecoder>(instance.distances, instance.n);
    };

    brkga3::Brkga brkga(brkga_cfg, decoder_factory);

    // Warm-start injection
    if (!alns_tour.empty()) {
        auto genes = tourToChromosome(alns_tour, instance.n);
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
        std::printf("  BRKGA completed: objective = %.2f, time = %.1fs\n",
                    result.brkga_final, result.brkga_time_s);
    }

    // Final results
    result.final_objective = result.brkga_final;
    auto total_end = std::chrono::high_resolution_clock::now();
    result.total_time_s = std::chrono::duration<double>(total_end - total_start).count();

    // Build solution JSON
    auto best_perm = brkga.getBestPermutation();
    std::ostringstream ss;
    ss << "{\n    \"total_distance\": " << result.brkga_final << ",\n";
    ss << "    \"num_cities\": " << instance.n << ",\n";
    ss << "    \"tour\": [";
    for (std::uint32_t i = 0; i < instance.n; ++i) {
        if (i > 0) ss << ", ";
        ss << best_perm[i];
    }
    ss << "]\n  }";
    result.solution_json = ss.str();

    return result;
}
