#pragma once

#include "types.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

namespace brkga3 {

// ============================================================================
// GpuDecoder — abstract interface for GPU-only decoders
//
// The user implements a concrete subclass that:
//   1. Uploads problem data to GPU in initialize()
//   2. Launches custom CUDA kernels in decode()
//   3. Frees GPU resources in finalize()
//
// Key contract:
//   - decode() must launch ALL work on the provided stream
//   - decode() must NOT call cudaDeviceSynchronize or cudaStreamSynchronize
//   - The framework guarantees d_genes_soa / d_permutations are valid
//     on the current GPU when decode() is called
//
// Memory layouts:
//   CHROMOSOME mode:
//     d_genes_soa[gene_j * pop_size + chrom_i]
//     → Adjacent threads read adjacent chromosomes at the same gene (coalesced)
//
//   PERMUTATION mode:
//     d_permutations[chrom_i * chrom_len + rank_k]
//     → The permutation for chromosome i: visit order of gene indices
// ============================================================================

class GpuDecoder {
public:
    virtual ~GpuDecoder() = default;

    // Called once per GPU after all GPU memory is allocated.
    // Upload problem-specific data (e.g., distance matrix) here.
    virtual void initialize(int gpu_id,
                            std::uint32_t chromosome_length,
                            std::uint32_t population_size) = 0;

    // Decode all chromosomes in the population and write fitness values.
    //
    // Parameters:
    //   stream          — launch all kernels on this stream
    //   population_size — number of chromosomes to decode
    //   chromosome_length — number of genes per chromosome
    //   d_genes_soa     — [chrom_len x pop_size] SoA layout (CHROMOSOME mode)
    //                     nullptr if using PERMUTATION mode
    //   d_permutations  — [pop_size x chrom_len] permutation indices
    //                     nullptr if using CHROMOSOME mode
    //   d_fitness       — [pop_size] output: write fitness[i] for chromosome i
    //
    virtual void decode(cudaStream_t stream,
                        std::uint32_t population_size,
                        std::uint32_t chromosome_length,
                        const Gene*      d_genes_soa,
                        const GeneIndex* d_permutations,
                        Fitness*         d_fitness) = 0;

    // Called once at shutdown before GPU memory is freed.
    virtual void finalize() = 0;
};

} // namespace brkga3
