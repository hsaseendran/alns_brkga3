#pragma once

#include "config.hpp"
#include "decoder.cuh"
#include "island_manager.cuh"
#include <memory>
#include <vector>
#include <functional>

namespace brkga3 {

// ============================================================================
// Brkga — public API for the BRKGA 3.0 framework
//
// Usage:
//   auto cfg = BrkgaConfig::Builder()
//       .chromosomeLength(n)
//       .decodeMode(DecodeMode::PERMUTATION)
//       .populationSize(512)
//       .numElites(51)
//       .numMutants(51)
//       .numParents(3, BiasType::LINEAR, 1)
//       .numGpus(8)
//       .seed(42)
//       .build();
//
//   auto factory = [&](int gpu_id) -> std::unique_ptr<GpuDecoder> {
//       return std::make_unique<MyDecoder>(problem_data);
//   };
//
//   Brkga brkga(cfg, factory);
//   brkga.evolve(1000);
//   printf("Best: %.2f\n", brkga.getBestFitness());
// ============================================================================

class Brkga {
public:
    using DecoderFactory = std::function<std::unique_ptr<GpuDecoder>(int gpu_id)>;

    // Construct and initialize: allocates GPU memory, generates initial
    // population, runs initial decode + sort on all GPUs.
    Brkga(const BrkgaConfig& cfg, DecoderFactory decoder_factory);
    ~Brkga();

    // Non-copyable
    Brkga(const Brkga&) = delete;
    Brkga& operator=(const Brkga&) = delete;

    // --- Evolution ---

    // Evolve all populations by one generation.
    // Automatically handles migration at the configured interval.
    void evolve();

    // Evolve for num_generations generations.
    void evolve(std::uint32_t num_generations);

    // --- Warm-start ---

    // Inject a chromosome into slot 0 of every population, then re-decode
    // and re-sort so the injected solution gets properly ranked.
    // Must be called after construction, before evolve().
    // genes.size() must equal chromosome_length.
    void injectChromosome(const std::vector<Gene>& genes);

    // --- Queries (these sync GPUs and do D→H transfer) ---

    // Best fitness across all GPUs.
    Fitness getBestFitness();

    // Best chromosome (raw genes).
    std::vector<Gene> getBestChromosome();

    // Best permutation (only valid for PERMUTATION decode mode).
    std::vector<GeneIndex> getBestPermutation();

    // --- State ---

    std::uint32_t generation() const { return gen_; }
    const BrkgaConfig& config() const { return cfg_; }

private:
    BrkgaConfig cfg_;
    std::unique_ptr<IslandManager> island_mgr_;
    std::uint32_t gen_ = 0;

    // Migration state
    bool     migration_pending_ = false;
    std::uint32_t migration_apply_gen_ = 0;
};

} // namespace brkga3
