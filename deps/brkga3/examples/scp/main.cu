#include <brkga3/brkga.cuh>
#include <brkga3/config.hpp>
#include <brkga3/types.cuh>

#include "scp_decoder.cuh"
#include "scp_instance.hpp"
#include "../common/json_reader.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>
#include <memory>
#include <vector>

// Convert SCP solution (list of selected set indices) to BRKGA chromosome.
// Selected sets get gene < 0.5, unselected get gene > 0.5.
static std::vector<brkga3::Gene> setsToChromosome(const std::vector<int>& selected_sets,
                                                    std::uint32_t n_sets) {
    std::vector<brkga3::Gene> genes(n_sets, 0.75f);  // default: unselected
    for (int s : selected_sets) {
        if (s >= 0 && static_cast<std::uint32_t>(s) < n_sets) {
            genes[s] = 0.25f;  // selected
        }
    }
    return genes;
}

// Parse CLI args: positional + optional --warm-start <file>
struct Args {
    std::string scp_file;
    int num_gpus    = 1;
    int num_pops    = 1;
    int pop_size    = 256;
    int generations = 1000;
    int seed        = 42;
    std::string warm_start_file;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <scp_file> [num_gpus] [num_pops] [pop_size] [gens] [seed] "
            "[--warm-start <json>]\n", argv[0]);
        std::exit(1);
    }

    a.scp_file = argv[1];

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

    // --- Load SCP instance ---
    std::printf("Loading SCP instance: %s\n", args.scp_file.c_str());
    auto instance = scp::loadORLibrary(args.scp_file);
    std::printf("  Name:     %s\n", instance.name.c_str());
    std::printf("  Elements: %u\n", instance.n_elements);
    std::printf("  Sets:     %u\n", instance.n_sets);

    // --- Configure BRKGA (paper parameters) ---
    auto cfg = brkga3::BrkgaConfig::Builder()
        .chromosomeLength(instance.n_sets)
        .decodeMode(brkga3::DecodeMode::CHROMOSOME)
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
    std::printf("  Chrom len:   %u (= n_sets)\n", cfg.chromosome_length);
    std::printf("  Elites:      %u\n", cfg.num_elites);
    std::printf("  Mutants:     %u\n", cfg.num_mutants);
    std::printf("  Crossover:   %u\n", cfg.num_crossover);
    std::printf("  Parents:     %u\n", cfg.num_parents);
    std::printf("  Generations: %d\n", args.generations);
    if (!args.warm_start_file.empty())
        std::printf("  Warm-start:  %s\n", args.warm_start_file.c_str());

    // --- Create decoder factory ---
    auto decoder_factory = [&](int gpu_id) -> std::unique_ptr<brkga3::GpuDecoder> {
        return std::make_unique<scp::ScpDecoder>(
            instance.costs, instance.set_covers, instance.set_offsets,
            instance.n_elements);
    };

    // --- Run BRKGA ---
    std::printf("\nStarting evolution...\n");
    auto t_start = std::chrono::high_resolution_clock::now();

    brkga3::Brkga brkga(cfg, decoder_factory);

    // --- Warm-start injection ---
    if (!args.warm_start_file.empty()) {
        std::printf("Loading warm-start solution from: %s\n", args.warm_start_file.c_str());
        auto json = json_reader::readFile(args.warm_start_file);
        auto selected = json_reader::readIntArray(json, "selected_sets");

        float alns_obj = json_reader::readFloat(json, "objective");
        std::printf("  ALNS objective: %.2f\n", alns_obj);
        std::printf("  Selected sets: %zu\n", selected.size());

        auto genes = setsToChromosome(selected, instance.n_sets);
        brkga.injectChromosome(genes);

        auto init_best = brkga.getBestFitness();
        std::printf("  BRKGA initial best after injection: %.2f\n", init_best);
    }

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
    double total_s = std::chrono::duration<double>(t_end - t_start).count();

    // --- Results ---
    auto best_fitness = brkga.getBestFitness();
    auto best_chromosome = brkga.getBestChromosome();

    // Count selected sets
    std::uint32_t selected_count = 0;
    for (std::uint32_t s = 0; s < instance.n_sets; ++s) {
        if (best_chromosome[s] < 0.5f) selected_count++;
    }

    std::printf("\n--- Results ---\n");
    std::printf("Best fitness:    %.2f\n", best_fitness);
    std::printf("Selected sets:   %u / %u\n", selected_count, instance.n_sets);
    std::printf("Total time:      %.3f s\n", total_s);
    std::printf("Time/gen:        %.3f ms\n", total_s / args.generations * 1000.0);

    if (instance.optimal > 0) {
        float gap = (best_fitness - instance.optimal) / instance.optimal * 100.0f;
        std::printf("Known optimal:   %.2f (gap: %.2f%%)\n",
                    instance.optimal, gap);
    }

    return 0;
}
