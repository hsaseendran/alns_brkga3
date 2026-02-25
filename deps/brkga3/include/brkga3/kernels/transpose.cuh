#pragma once

#include "../types.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace brkga3 {

// ============================================================================
// Shared-memory tiled matrix transpose
//
// AoS -> SoA: [rows x cols] -> [cols x rows]
//   Input  (AoS): d_aos[chrom_i * chrom_len + gene_j]    (row = chromosome)
//   Output (SoA): d_soa[gene_j * pop_size   + chrom_i]   (row = gene index)
//
// Uses TILE_DIM=32 x TILE_DIM tiles with +1 padding to avoid shared memory
// bank conflicts. BLOCK_ROWS=8 means each thread processes 4 elements.
// ============================================================================

void transposeAoStoSoA(
    cudaStream_t stream,
    Gene*        d_soa,       // output: [cols x rows]
    const Gene*  d_aos,       // input:  [rows x cols]
    std::uint32_t rows,       // pop_size
    std::uint32_t cols);      // chrom_len

} // namespace brkga3
