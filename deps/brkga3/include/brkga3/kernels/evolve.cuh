#pragma once

#include "../types.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

namespace brkga3 {

// ============================================================================
// RNG state initialization
//   One curandState per crossover chromosome, used for parent selection.
// ============================================================================

void initRngStates(
    cudaStream_t   stream,
    curandState_t* d_rng_states,    // [num_crossover]
    std::uint64_t  seed,
    std::uint32_t  num_crossover,
    std::uint32_t  gpu_offset = 0); // offset for multi-GPU determinism

// ============================================================================
// Parent selection
//   One thread per crossover chromosome.
//   Samples num_elite_parents from [0, num_elites) and
//   (num_parents - num_elite_parents) from [num_elites, pop_size).
// ============================================================================

void selectParents(
    cudaStream_t   stream,
    GeneIndex*     d_parents,       // [num_crossover * num_parents]
    curandState_t* d_rng_states,    // [num_crossover]
    std::uint32_t  num_crossover,
    std::uint32_t  num_parents,
    std::uint32_t  num_elite_parents,
    std::uint32_t  num_elites,
    std::uint32_t  pop_size,
    std::uint32_t  threads_per_block);

// ============================================================================
// Elite copy
//   Copies elite chromosomes from old generation to new generation.
//   Uses fitness index to find elite chromosomes by rank.
//   One thread per gene of all elite chromosomes.
// ============================================================================

void evolveCopyElite(
    cudaStream_t    stream,
    Gene*           d_new_pop,      // output: new generation
    const Gene*     d_old_pop,      // input: current generation
    const GeneIndex* d_fitness_idx, // rank -> chromosome index
    std::uint32_t   num_elites,
    std::uint32_t   chrom_len,
    std::uint32_t   threads_per_block);

// ============================================================================
// Mate (crossover)
//   Generates crossover chromosomes using biased parent selection.
//   d_new_pop already contains random toss values (from curandGenerateUniform).
//   The random toss is read, used for parent selection via bias CDF, and
//   overwritten with the selected parent's gene value.
//
//   One thread per gene of all crossover chromosomes.
//   Bias CDF loaded into shared memory.
// ============================================================================

void evolveMate(
    cudaStream_t     stream,
    Gene*            d_new_pop,      // in/out: random toss -> offspring genes
    const Gene*      d_old_pop,      // input: current generation
    const GeneIndex* d_fitness_idx,  // rank -> chromosome index
    const GeneIndex* d_parents,      // [num_crossover * num_parents]
    const float*     d_bias_cdf,     // [num_parents] cumulative distribution
    std::uint32_t    num_elites,
    std::uint32_t    num_mutants,
    std::uint32_t    num_parents,
    std::uint32_t    pop_size,
    std::uint32_t    chrom_len,
    std::uint32_t    threads_per_block);

// ============================================================================
// Pack elites into migration buffer
//   Copies the top M elite chromosomes into a contiguous buffer for migration.
// ============================================================================

void packElites(
    cudaStream_t     stream,
    Gene*            d_send_buf,     // output: [num_migrate * chrom_len]
    const Gene*      d_chromosomes,  // source population
    const GeneIndex* d_fitness_idx,  // rank -> chromosome index
    std::uint32_t    num_migrate,
    std::uint32_t    chrom_len,
    std::uint32_t    threads_per_block);

// ============================================================================
// Unpack immigrants into worst chromosome slots
//   Overwrites the worst M chromosomes with received immigrants.
// ============================================================================

void unpackImmigrants(
    cudaStream_t     stream,
    Gene*            d_chromosomes,  // population to modify
    const Gene*      d_recv_buf,     // [num_migrate * chrom_len]
    const GeneIndex* d_fitness_idx,  // rank -> chromosome index
    std::uint32_t    num_migrate,
    std::uint32_t    pop_size,
    std::uint32_t    chrom_len,
    std::uint32_t    threads_per_block);

} // namespace brkga3
