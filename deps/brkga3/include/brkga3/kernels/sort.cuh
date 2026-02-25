#pragma once

#include "../types.cuh"
#include <cuda_runtime.h>
#include <cstddef>

namespace brkga3 {

// ============================================================================
// Sort A: Fitness ranking
//   Sorts fitness values ascending (or descending) to produce ranking.
//   Uses CUB DeviceRadixSort::SortPairs — stream-compatible, no device sync.
// ============================================================================

void sortFitness(
    cudaStream_t    stream,
    Fitness*        d_keys_out,      // [pop_size] sorted fitness
    GeneIndex*      d_values_out,    // [pop_size] rank -> chromosome index
    const Fitness*  d_keys_in,       // [pop_size] unsorted fitness
    const GeneIndex* d_values_in,    // [pop_size] iota {0,1,...,pop_size-1}
    void*           d_tmp,           // pre-allocated temp storage
    std::size_t     tmp_bytes,
    std::uint32_t   pop_size);

// ============================================================================
// Sort B: Gene permutation (segmented sort)
//   Sorts genes within each chromosome to produce permutation indices.
//   Uses CUB DeviceSegmentedSort::SortPairs — stream-compatible.
//   Segment offsets computed via transform iterator (no device array needed).
// ============================================================================

void sortGenesSegmented(
    cudaStream_t     stream,
    Gene*            d_keys_out,     // [pop_size * chrom_len] sorted genes
    GeneIndex*       d_values_out,   // [pop_size * chrom_len] permutation
    const Gene*      d_keys_in,      // [pop_size * chrom_len] input genes
    const GeneIndex* d_values_in,    // [pop_size * chrom_len] iota mod chrom_len
    void*            d_tmp,
    std::size_t      tmp_bytes,
    std::uint32_t    pop_size,
    std::uint32_t    chrom_len);

// ============================================================================
// Query CUB temp storage sizes (called once during Population::create)
// ============================================================================

std::size_t querySortFitnessTmpBytes(std::uint32_t pop_size);

std::size_t querySortGenesSegmentedTmpBytes(
    std::uint32_t pop_size,
    std::uint32_t chrom_len);

// ============================================================================
// Fill iota arrays on device (used to initialize sort value inputs)
// ============================================================================

// Fill d_out with {0, 1, 2, ..., n-1}
void fillIota(cudaStream_t stream, GeneIndex* d_out, std::uint32_t n);

// Fill d_out with {0, 1, ..., mod-1, 0, 1, ..., mod-1, ...} for n elements
void fillIotaMod(cudaStream_t stream, GeneIndex* d_out, std::uint32_t n,
                 std::uint32_t mod);

} // namespace brkga3
