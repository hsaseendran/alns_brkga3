#pragma once

#include "config.hpp"
#include "population.cuh"
#include "decoder.cuh"
#include <vector>
#include <memory>
#include <functional>

namespace brkga3 {

// ============================================================================
// IslandManager — orchestrates multi-population, multi-GPU evolution pipeline
//
// Supports N populations per GPU, each on its own CUDA stream.
// Total populations = num_gpus * num_populations.
//
// The manager:
//   1. Launches the per-generation pipeline on all populations concurrently
//   2. Handles all-pairs elite migration (same-GPU via DtoD, cross-GPU via P2P)
//   3. Queries results (best fitness/chromosome across all populations)
//
// All synchronization uses cudaEvent + cudaStreamWaitEvent.
// No cudaDeviceSynchronize is ever called.
// ============================================================================

class IslandManager {
public:
    // Factory function: called once per GPU to create a decoder instance.
    // Each GPU gets its own decoder with its own copy of problem data.
    using DecoderFactory = std::function<std::unique_ptr<GpuDecoder>(int gpu_id)>;

    IslandManager(const BrkgaConfig& cfg, DecoderFactory factory);
    ~IslandManager();

    // Non-copyable, non-movable
    IslandManager(const IslandManager&) = delete;
    IslandManager& operator=(const IslandManager&) = delete;

    // ---- Per-generation pipeline ----

    // Launch the full pipeline for one generation on all populations.
    void launchGeneration();

    // Block host until all populations finish the current generation.
    void syncAllPops();

    // ---- Migration ----

    // Send top elites from each population to all others (all-pairs).
    void migrateSend();

    // Apply previously received immigrants (overwrites worst chromosomes).
    void migrateApply();

    // Re-decode and re-sort all populations (e.g., after migration apply).
    void redecodeAllPops();

    // ---- Warm-start ----

    // Inject a chromosome into slot 0 of every population, re-decode, re-sort.
    void injectChromosome(const std::vector<Gene>& genes);

    // ---- Queries (involves D→H transfer) ----

    Fitness getBestFitness();
    std::vector<Gene> getBestChromosome();
    std::vector<GeneIndex> getBestPermutation();

    // ---- Access ----
    std::uint32_t totalPops() const { return cfg_.totalPopulations(); }
    const Population& population(int pop_idx) const { return populations_[pop_idx]; }
    Population& population(int pop_idx) { return populations_[pop_idx]; }

private:
    BrkgaConfig cfg_;
    std::vector<Population> populations_;           // [totalPopulations]
    std::vector<std::unique_ptr<GpuDecoder>> decoders_;  // [num_gpus] — one per GPU

    // Mapping helpers
    int gpuOf(int pop_idx) const { return pop_idx / static_cast<int>(cfg_.num_populations); }

    // Internal pipeline steps (operate on a single population)
    void evolveOnPop(int pop_idx);
    void prepareDecodeOnPop(int pop_idx);
    void decodeOnPop(int pop_idx);
    void sortFitnessOnPop(int pop_idx);

    // Find which population has the best fitness (syncs first)
    int findBestPop();
};

} // namespace brkga3
