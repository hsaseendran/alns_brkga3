#pragma once

#include "types.cuh"
#include "config.hpp"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstddef>

namespace brkga3 {

// ============================================================================
// Population — all device memory for one island on one GPU
//
// Fully GPU-resident: all data stays on-device during evolution.
// The only host-device transfers are at initialization and result query.
// ============================================================================

struct Population {
    int gpu_id = -1;
    int pop_global_idx = -1;  // unique index across all GPUs [0, totalPopulations)

    // --- Evolution storage (AoS layout) ---
    // d_chromosomes[chrom_i * chrom_len + gene_j]
    Gene* d_chromosomes     = nullptr;  // [pop_size * chrom_len] current gen
    Gene* d_chromosomes_tmp = nullptr;  // [pop_size * chrom_len] next gen workspace

    // --- Decoder storage (SoA layout, for CHROMOSOME mode) ---
    // d_chromosomes_soa[gene_j * pop_size + chrom_i]
    Gene* d_chromosomes_soa = nullptr;  // [chrom_len * pop_size]

    // --- Permutation decoder storage ---
    GeneIndex* d_permutations    = nullptr;  // [pop_size * chrom_len] sorted indices
    Gene*      d_sort_keys_tmp   = nullptr;  // [pop_size * chrom_len] scratch for CUB seg sort

    // --- Fitness ---
    Fitness*   d_fitness         = nullptr;  // [pop_size] raw fitness from decoder
    Fitness*   d_fitness_sorted  = nullptr;  // [pop_size] sorted fitness (output of radix sort)
    GeneIndex* d_fitness_idx     = nullptr;  // [pop_size] rank -> chromosome index

    // --- Parent selection ---
    GeneIndex*    d_parents     = nullptr;  // [num_crossover * num_parents]
    curandState_t* d_rng_states = nullptr;  // [num_crossover] for parent selection RNG

    // --- Bias CDF (uploaded once at init) ---
    float* d_bias_cdf = nullptr;            // [num_parents]

    // --- Sort input arrays (constant, initialized once) ---
    GeneIndex* d_iota_pop   = nullptr;      // [pop_size] = {0, 1, ..., pop_size-1}
    GeneIndex* d_iota_genes = nullptr;      // [pop_size * chrom_len] = {0,1,...,L-1, 0,1,...}

    // --- CUB temporary storage (pre-allocated) ---
    void*       d_cub_sort_fitness_tmp = nullptr;
    std::size_t cub_sort_fitness_tmp_bytes = 0;
    void*       d_cub_seg_sort_tmp = nullptr;
    std::size_t cub_seg_sort_tmp_bytes = 0;

    // --- Bulk RNG (cuRAND host API for generating random floats) ---
    curandGenerator_t rng_generator = nullptr;

    // --- Streams ---
    cudaStream_t compute_stream = nullptr;  // evolve / transpose / decode / sort
    cudaStream_t comm_stream    = nullptr;  // migration transfers

    // --- Events ---
    cudaEvent_t generation_done = nullptr;  // signaled after sort completes
    cudaEvent_t migration_sent  = nullptr;  // signaled on THIS GPU after outgoing transfer completes

    // --- Migration staging buffers ---
    Gene* d_migration_send_buf = nullptr;   // [num_migrate * chrom_len]
    Gene* d_migration_recv_buf = nullptr;   // [num_migrate * chrom_len]

    // --- Lifecycle ---

    // Allocate all GPU memory and initialize.
    // Sets the CUDA device to gpu_id internally.
    static Population create(int gpu_id, int pop_global_idx, const BrkgaConfig& cfg);

    // Free all GPU memory and destroy streams/events.
    static void destroy(Population& pop);

    // Initialize population with random chromosomes and run initial decode+sort.
    // Must be called after decoder is initialized.
    void initialize(const BrkgaConfig& cfg);
};

} // namespace brkga3
