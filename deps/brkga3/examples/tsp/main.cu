#include <brkga3/brkga.cuh>
#include <brkga3/config.hpp>
#include <brkga3/types.cuh>

#include "tsp_decoder.cuh"
#include "tsp_instance.hpp"
#include "../common/json_reader.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <memory>
#include <vector>

// Convert an ALNS tour (ordered city indices) to BRKGA random-key chromosome.
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

// Parse CLI args: positional + optional --warm-start <file>
struct Args {
    std::string tsp_file;
    int num_gpus    = 1;
    int num_pops    = 1;
    int pop_size    = 512;
    int generations = 1000;
    int seed        = 42;
    std::string warm_start_file;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <tsp_file> [num_gpus] [num_pops] [pop_size] [gens] [seed] "
            "[--warm-start <json>]\n", argv[0]);
        std::exit(1);
    }

    a.tsp_file = argv[1];

    // Collect positional args (skip --flags)
    std::vector<std::string> positional;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warm-start" && i + 1 < argc) {
            a.warm_start_file = argv[++i];
        } else if (arg[0] != '-') {
            positional.push_back(arg);
        }
    }

    if (positional.size() > 0) a.num_gpus    = std::atoi(positional[0].c_str());
    if (positional.size() > 1) a.num_pops    = std::atoi(positional[1].c_str());
    if (positional.size() > 2) a.pop_size    = std::atoi(positional[2].c_str());
    if (positional.size() > 3) a.generations = std::atoi(positional[3].c_str());
    if (positional.size() > 4) a.seed        = std::atoi(positional[4].c_str());

    return a;
}

int main(int argc, char** argv) {
    auto args = parseArgs(argc, argv);

    // --- Load TSP instance ---
    std::printf("Loading TSP instance: %s\n", args.tsp_file.c_str());
    auto instance = tsp::loadTSPLIB(args.tsp_file);
    std::printf("  Name: %s\n", instance.name.c_str());
    std::printf("  Cities: %u\n", instance.n);

    // --- Configure BRKGA ---
    auto cfg = brkga3::BrkgaConfig::Builder()
        .chromosomeLength(instance.n)
        .decodeMode(brkga3::DecodeMode::PERMUTATION)
        .optimizationSense(brkga3::OptimizationSense::MINIMIZE)
        .populationSize(args.pop_size)
        .numElites(25)
        .numMutants(25)
        .numParents(2, brkga3::BiasType::LINEAR, 1)
        .numGpus(args.num_gpus)
        .numPopulations(args.num_pops)
        .migrationInterval(50)
        .numElitesToMigrate(2)
        .threadsPerBlock(256)
        .seed(args.seed)
        .build();

    // Override bias CDF to match paper's rho=0.75 (classic 2-parent BRKGA)
    cfg.bias_cdf = {0.75f, 1.0f};

    std::printf("\nBRKGA Configuration:\n");
    std::printf("  GPUs:        %u\n", cfg.num_gpus);
    std::printf("  Pops/GPU:    %u\n", cfg.num_populations);
    std::printf("  Total pops:  %u\n", cfg.totalPopulations());
    std::printf("  Pop size:    %u (per population)\n", cfg.population_size);
    std::printf("  Elites:      %u\n", cfg.num_elites);
    std::printf("  Mutants:     %u\n", cfg.num_mutants);
    std::printf("  Crossover:   %u\n", cfg.num_crossover);
    std::printf("  Parents:     %u\n", cfg.num_parents);
    std::printf("  Generations: %d\n", args.generations);
    if (!args.warm_start_file.empty())
        std::printf("  Warm-start:  %s\n", args.warm_start_file.c_str());

    // --- Create decoder factory ---
    auto decoder_factory = [&](int gpu_id) -> std::unique_ptr<brkga3::GpuDecoder> {
        return std::make_unique<tsp::TspDecoder>(instance.distances, instance.n);
    };

    // --- Initialize BRKGA ---
    std::printf("\nInitializing BRKGA...\n");
    brkga3::Brkga brkga(cfg, decoder_factory);

    // --- Warm-start injection ---
    if (!args.warm_start_file.empty()) {
        std::printf("Loading warm-start solution from: %s\n", args.warm_start_file.c_str());
        auto json = json_reader::readFile(args.warm_start_file);
        auto tour = json_reader::readIntArray(json, "tour");

        if (tour.size() != instance.n) {
            std::fprintf(stderr, "Error: warm-start tour has %zu cities, expected %u\n",
                         tour.size(), instance.n);
            return 1;
        }

        float alns_obj = json_reader::readFloat(json, "total_distance");
        std::printf("  ALNS objective: %.2f\n", alns_obj);

        auto genes = tourToChromosome(tour, instance.n);
        brkga.injectChromosome(genes);

        auto init_best = brkga.getBestFitness();
        std::printf("  BRKGA initial best after injection: %.2f\n", init_best);
    }

    // --- Run evolution ---
    std::printf("\nStarting evolution...\n");
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int g = 1; g <= args.generations; ++g) {
        brkga.evolve();

        if (g % 100 == 0 || g == 1) {
            auto best = brkga.getBestFitness();
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(t_now - t_start).count();
            std::printf("  Gen %5d | Best: %12.2f | Time: %.3fs\n",
                        g, best, elapsed_s);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double evo_s = std::chrono::duration<double>(t_end - t_start).count();

    // --- Results ---
    auto best_fitness = brkga.getBestFitness();
    auto best_perm = brkga.getBestPermutation();

    std::printf("\n--- Results ---\n");
    std::printf("Best fitness: %.2f\n", best_fitness);
    std::printf("Total time:   %.3f s\n", evo_s);
    std::printf("Time/gen:     %.3f ms\n", evo_s / args.generations * 1000.0);
    std::printf("Tour: ");
    for (std::uint32_t i = 0; i < std::min(instance.n, 20u); ++i) {
        std::printf("%u ", best_perm[i]);
    }
    if (instance.n > 20) std::printf("...");
    std::printf("\n");

    if (instance.optimal > 0) {
        float gap = (best_fitness - instance.optimal) / instance.optimal * 100.0f;
        std::printf("Known optimal: %.2f (gap: %.2f%%)\n",
                    instance.optimal, gap);
    }

    return 0;
}
