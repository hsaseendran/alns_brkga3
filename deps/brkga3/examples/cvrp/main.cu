#include <brkga3/brkga.cuh>
#include <brkga3/config.hpp>
#include <brkga3/types.cuh>

#include "cvrp_decoder.cuh"
#include "cvrp_instance.hpp"
#include "../common/json_reader.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <memory>
#include <array>
#include <vector>

// Convert CVRP route solution to BRKGA random-key chromosome.
// Input: routes as nested array of node indices (1-based, depot excluded).
// Output: random-key chromosome where sorting produces the client visit order.
static std::vector<brkga3::Gene> routesToChromosome(
    const std::vector<std::vector<int>>& routes,
    std::uint32_t n_clients,
    std::uint32_t depot)
{
    // Flatten routes into a single client permutation
    std::vector<int> client_order;
    client_order.reserve(n_clients);
    for (const auto& route : routes) {
        for (int node : route) {
            if (static_cast<std::uint32_t>(node) == depot) continue; // skip depot
            // Map node index to 0-based client index
            // In CVRP decoder: client_node >= depot → client_node + 1 = node
            // So: node > depot → client_index = node - 1
            //     node < depot → client_index = node (but depot is typically 0)
            int client_idx = (static_cast<std::uint32_t>(node) > depot)
                           ? node - 1 : node;
            client_order.push_back(client_idx);
        }
    }

    if (client_order.size() != n_clients) {
        std::fprintf(stderr, "Warning: warm-start has %zu clients, expected %u\n",
                     client_order.size(), n_clients);
    }

    std::vector<brkga3::Gene> genes(n_clients);
    float inv_n = 1.0f / static_cast<float>(n_clients);
    for (std::uint32_t rank = 0; rank < client_order.size() && rank < n_clients; ++rank) {
        genes[client_order[rank]] = (rank + 0.5f) * inv_n;
    }
    return genes;
}

// Parse CLI args: positional + optional --warm-start <file>
struct Args {
    std::string vrp_file;
    int num_gpus    = 8;
    int num_pops    = 3;
    int pop_size    = 16384;
    int max_gens    = 500000;
    int seed        = 42;
    std::string warm_start_file;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <vrp_file> [num_gpus] [num_pops] [pop_size] [max_gens] [seed] "
            "[--warm-start <json>]\n", argv[0]);
        std::exit(1);
    }

    a.vrp_file = argv[1];

    std::vector<std::string> positional;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warm-start" && i + 1 < argc) {
            a.warm_start_file = argv[++i];
        } else if (arg[0] != '-') {
            positional.push_back(arg);
        }
    }

    if (positional.size() > 0) a.num_gpus  = std::atoi(positional[0].c_str());
    if (positional.size() > 1) a.num_pops  = std::atoi(positional[1].c_str());
    if (positional.size() > 2) a.pop_size  = std::atoi(positional[2].c_str());
    if (positional.size() > 3) a.max_gens  = std::atoi(positional[3].c_str());
    if (positional.size() > 4) a.seed      = std::atoi(positional[4].c_str());

    return a;
}

int main(int argc, char** argv) {
    auto args = parseArgs(argc, argv);

    // --- Load CVRP instance ---
    std::printf("Loading CVRP instance: %s\n", args.vrp_file.c_str());
    auto instance = cvrp::loadCVRPLIB(args.vrp_file);
    std::printf("  Name:      %s\n", instance.name.c_str());
    std::printf("  Dimension: %u (depot + %u clients)\n",
                instance.dimension, instance.n_clients);
    std::printf("  Capacity:  %u\n", instance.capacity);
    std::printf("  Depot:     node %u\n", instance.depot);
    if (instance.optimal > 0)
        std::printf("  Optimal:   %.2f\n", instance.optimal);

    // --- Configure BRKGA ---
    // Scale elites/mutants to ~10% of population each
    const int num_elites  = std::max(2, args.pop_size / 10);
    const int num_mutants = std::max(2, args.pop_size / 10);

    auto cfg = brkga3::BrkgaConfig::Builder()
        .chromosomeLength(instance.n_clients)
        .decodeMode(brkga3::DecodeMode::PERMUTATION)
        .optimizationSense(brkga3::OptimizationSense::MINIMIZE)
        .populationSize(args.pop_size)
        .numElites(num_elites)
        .numMutants(num_mutants)
        .numParents(2, brkga3::BiasType::LINEAR, 1)
        .numGpus(args.num_gpus)
        .numPopulations(args.num_pops)
        .migrationInterval(50)
        .numElitesToMigrate(5)
        .threadsPerBlock(256)
        .seed(args.seed)
        .build();

    cfg.bias_cdf = {0.75f, 1.0f};

    std::printf("\nBRKGA Configuration:\n");
    std::printf("  GPUs:        %u\n", cfg.num_gpus);
    std::printf("  Pops/GPU:    %u\n", cfg.num_populations);
    std::printf("  Total pops:  %u\n", cfg.totalPopulations());
    std::printf("  Pop size:    %u (per population)\n", cfg.population_size);
    std::printf("  Chrom len:   %u (= n_clients)\n", cfg.chromosome_length);
    std::printf("  Elites:      %u\n", cfg.num_elites);
    std::printf("  Mutants:     %u\n", cfg.num_mutants);
    std::printf("  Crossover:   %u\n", cfg.num_crossover);
    std::printf("  Parents:     %u\n", cfg.num_parents);
    std::printf("  Max gens:    %d\n", args.max_gens);
    if (!args.warm_start_file.empty())
        std::printf("  Warm-start:  %s\n", args.warm_start_file.c_str());

    // --- Time-to-target thresholds ---
    constexpr int N_THRESHOLDS = 4;
    const std::array<float, N_THRESHOLDS> gap_thresholds = {10.0f, 5.0f, 2.0f, 1.0f};
    std::array<double, N_THRESHOLDS> ttt_seconds;
    std::array<int, N_THRESHOLDS>    ttt_gen;
    std::array<bool, N_THRESHOLDS>   ttt_reached;
    for (int i = 0; i < N_THRESHOLDS; ++i) {
        ttt_seconds[i] = -1.0;
        ttt_gen[i] = -1;
        ttt_reached[i] = false;
    }
    const bool has_optimal = (instance.optimal > 0);
    const float target_gap = 1.0f;

    // --- Create decoder factory ---
    auto decoder_factory = [&](int gpu_id) -> std::unique_ptr<brkga3::GpuDecoder> {
        return std::make_unique<cvrp::CvrpDecoder>(
            instance.distances, instance.demands,
            instance.dimension, instance.depot, instance.capacity);
    };

    // --- Run BRKGA ---
    std::printf("\nStarting evolution...\n");
    auto t_start = std::chrono::high_resolution_clock::now();

    brkga3::Brkga brkga(cfg, decoder_factory);

    // --- Warm-start injection ---
    if (!args.warm_start_file.empty()) {
        std::printf("Loading warm-start solution from: %s\n", args.warm_start_file.c_str());
        auto json = json_reader::readFile(args.warm_start_file);
        auto routes = json_reader::readNestedIntArray(json, "routes");

        float alns_obj = json_reader::readFloat(json, "total_distance");
        std::printf("  ALNS objective: %.2f\n", alns_obj);
        std::printf("  Routes: %zu\n", routes.size());

        auto genes = routesToChromosome(routes, instance.n_clients, instance.depot);
        brkga.injectChromosome(genes);

        auto init_best = brkga.getBestFitness();
        std::printf("  BRKGA initial best after injection: %.2f\n", init_best);
    }

    // Track last improvement
    float prev_best = std::numeric_limits<float>::max();
    int   last_improve_gen = 0;
    double last_improve_time = 0.0;
    float last_improve_gap = -1.0f;

    int final_gen = args.max_gens;
    for (int g = 1; g <= args.max_gens; ++g) {
        brkga.evolve();

        auto best = brkga.getBestFitness();
        auto t_now = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();

        if (best < prev_best) {
            prev_best = best;
            last_improve_gen = g;
            last_improve_time = elapsed_s;
            if (has_optimal)
                last_improve_gap = (best - instance.optimal) / instance.optimal * 100.0f;
        }

        float current_gap = -1.0f;
        if (has_optimal) {
            current_gap = (best - instance.optimal) / instance.optimal * 100.0f;

            for (int t = 0; t < N_THRESHOLDS; ++t) {
                if (!ttt_reached[t] && current_gap <= gap_thresholds[t]) {
                    ttt_reached[t] = true;
                    ttt_seconds[t] = elapsed_s;
                    ttt_gen[t] = g;
                    std::printf("  >>> TTT %5.1f%% reached at gen %5d | Best: %12.2f | Gap: %6.2f%% | Time: %.3fs\n",
                                gap_thresholds[t], g, best, current_gap, elapsed_s);
                }
            }
        }

        if (g % 500 == 0 || g == 1) {
            if (has_optimal)
                std::printf("  Gen %5d | Best: %12.2f | Gap: %6.2f%% | Time: %.3fs\n",
                            g, best, current_gap, elapsed_s);
            else
                std::printf("  Gen %5d | Best: %12.2f | Time: %.3fs\n",
                            g, best, elapsed_s);
        }

        if (has_optimal && current_gap <= target_gap) {
            std::printf("  Target gap %.1f%% reached at gen %d. Stopping.\n", target_gap, g);
            final_gen = g;
            break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();

    // --- Results ---
    auto best_fitness = brkga.getBestFitness();

    std::printf("\n--- Results ---\n");
    std::printf("Best fitness: %.2f\n", best_fitness);
    std::printf("Generations:  %d\n", final_gen);
    std::printf("Total time:   %.3f s\n", total_s);
    std::printf("Time/gen:     %.3f ms\n", total_s / final_gen * 1000.0);
    std::printf("Last improve: gen %d @ %.3fs (best=%.2f",
                last_improve_gen, last_improve_time, prev_best);
    if (has_optimal)
        std::printf(", gap=%.2f%%", last_improve_gap);
    std::printf(")\n");

    if (has_optimal) {
        float gap = (best_fitness - instance.optimal) / instance.optimal * 100.0f;
        std::printf("Known optimal: %.2f\n", instance.optimal);
        std::printf("Final gap:     %.2f%%\n", gap);

        std::printf("\n--- Time to Target ---\n");
        std::printf("  %-10s %-10s %-12s %-10s\n", "Gap", "Gen", "Time (s)", "Reached");
        for (int t = 0; t < N_THRESHOLDS; ++t) {
            if (ttt_reached[t]) {
                std::printf("  %-10.1f%% %-10d %-12.3f YES\n",
                            gap_thresholds[t], ttt_gen[t], ttt_seconds[t]);
            } else {
                std::printf("  %-10.1f%% %-10s %-12s NO\n",
                            gap_thresholds[t], "-", "-");
            }
        }
    }

    // Machine-readable summary line for scripting
    std::printf("\nCSV: %s,%.2f,%.2f,%.2f%%,%d,%.3f,%d,%.3f",
                instance.name.c_str(), instance.optimal, best_fitness,
                has_optimal ? (best_fitness - instance.optimal) / instance.optimal * 100.0f : -1.0f,
                final_gen, total_s, last_improve_gen, last_improve_time);
    for (int t = 0; t < N_THRESHOLDS; ++t) {
        std::printf(",%.3f", ttt_seconds[t]);
    }
    std::printf("\n");

    return 0;
}
