#include <brkga3/kernels/transpose.cuh>
#include <brkga3/utils/cuda_check.cuh>

namespace brkga3 {

// ============================================================================
// Shared-memory tiled matrix transpose kernel
//
// Standard optimized transpose:
//   - TILE_DIM x TILE_DIM tiles
//   - BLOCK_ROWS threads per tile row (each thread handles TILE_DIM/BLOCK_ROWS elements)
//   - +1 column padding in shared memory to avoid bank conflicts
//   - Coalesced reads from input, coalesced writes to output
//
// Reference: NVIDIA CUDA SDK "Matrix Transpose" sample
// ============================================================================

constexpr unsigned TILE_DIM   = 32;
constexpr unsigned BLOCK_ROWS = 8;

__global__ void transposeKernel(
    Gene*       __restrict__ d_out,   // [cols x rows] output (SoA)
    const Gene* __restrict__ d_in,    // [rows x cols] input  (AoS)
    std::uint32_t rows,
    std::uint32_t cols)
{
    __shared__ Gene tile[TILE_DIM][TILE_DIM + 1];  // +1 avoids bank conflicts

    // Input coordinates
    unsigned x = blockIdx.x * TILE_DIM + threadIdx.x;  // column
    unsigned y = blockIdx.y * TILE_DIM + threadIdx.y;   // row

    // Load tile from input (coalesced along columns)
    #pragma unroll
    for (unsigned j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < rows && x < cols) {
            tile[threadIdx.y + j][threadIdx.x] = d_in[(y + j) * cols + x];
        }
    }

    __syncthreads();

    // Output coordinates: swap block indices for transposed write
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write tile to output (coalesced along the transposed dimension)
    #pragma unroll
    for (unsigned j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < cols && x < rows) {
            d_out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void transposeAoStoSoA(
    cudaStream_t  stream,
    Gene*         d_soa,
    const Gene*   d_aos,
    std::uint32_t rows,
    std::uint32_t cols)
{
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM,
              (rows + TILE_DIM - 1) / TILE_DIM);

    transposeKernel<<<grid, block, 0, stream>>>(d_soa, d_aos, rows, cols);
    BRKGA_CUDA_CHECK_LAST();
}

} // namespace brkga3
