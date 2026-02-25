#pragma once

#include <brkga3/decoder.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <vector>
#include <cstdint>

namespace tsp {

// ============================================================================
// TSP GPU Decoder — warp-per-chromosome
//
// Uses PERMUTATION decode mode: the framework sorts genes to produce a visit
// order (permutation). This decoder computes the total tour cost.
//
// Each warp (32 threads) decodes one chromosome:
//   - Each lane sums a strided portion of the tour edges
//   - Warp shuffle reduction aggregates partial sums (~5 cycles, no smem)
//
// Distance matrix stored in global memory (for small n, could use shared mem).
// ============================================================================

__global__ void tspDecodeKernel(
    const brkga3::GeneIndex* __restrict__ d_perms,   // [pop_size x n]
    const float*             __restrict__ d_dist,     // [n x n]
    brkga3::Fitness*         __restrict__ d_fitness,  // [pop_size]
    std::uint32_t pop_size,
    std::uint32_t n)
{
    const unsigned global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned warp_id    = global_tid / 32;
    const unsigned lane       = global_tid % 32;

    if (warp_id >= pop_size) return;

    const brkga3::GeneIndex* perm = d_perms + warp_id * n;

    // Each lane computes a partial sum over strided edges
    float local_cost = 0.0f;
    for (unsigned i = lane; i < n; i += 32) {
        unsigned from = perm[i];
        unsigned to   = perm[(i + 1) % n];  // wrap around for last edge
        local_cost += d_dist[from * n + to];
    }

    // Warp-level sum reduction via shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_cost += __shfl_down_sync(0xFFFFFFFF, local_cost, offset);
    }

    // Lane 0 writes the total fitness
    if (lane == 0) {
        d_fitness[warp_id] = local_cost;
    }
}

class TspDecoder : public brkga3::GpuDecoder {
public:
    // Construct with a host-side distance matrix [n x n].
    // The matrix is uploaded to each GPU during initialize().
    explicit TspDecoder(const std::vector<float>& distances, std::uint32_t n)
        : host_dist_(distances), n_(n) {}

    void initialize(int gpu_id, std::uint32_t chromosome_length,
                    std::uint32_t population_size) override {
        gpu_id_ = gpu_id;
        pop_size_ = population_size;
        brkga3::setDevice(gpu_id);

        BRKGA_CUDA_CHECK(cudaMalloc(&d_dist_, n_ * n_ * sizeof(float)));
        BRKGA_CUDA_CHECK(cudaMemcpy(d_dist_, host_dist_.data(),
                                     n_ * n_ * sizeof(float),
                                     cudaMemcpyHostToDevice));
    }

    void decode(cudaStream_t stream,
                std::uint32_t population_size,
                std::uint32_t chromosome_length,
                const brkga3::Gene*      /*d_genes_soa*/,
                const brkga3::GeneIndex* d_permutations,
                brkga3::Fitness*         d_fitness) override {
        // Warp-per-chromosome: 32 threads per chromosome
        constexpr unsigned WARPS_PER_BLOCK = 4;
        constexpr unsigned THREADS = WARPS_PER_BLOCK * 32;  // 128 threads

        unsigned blocks = (population_size + WARPS_PER_BLOCK - 1)
                          / WARPS_PER_BLOCK;

        tspDecodeKernel<<<blocks, THREADS, 0, stream>>>(
            d_permutations, d_dist_, d_fitness,
            population_size, chromosome_length);
        BRKGA_CUDA_CHECK_LAST();
    }

    void finalize() override {
        if (d_dist_) {
            brkga3::setDevice(gpu_id_);
            BRKGA_CUDA_CHECK(cudaFree(d_dist_));
            d_dist_ = nullptr;
        }
    }

private:
    const std::vector<float>& host_dist_;
    std::uint32_t n_ = 0;
    int gpu_id_ = -1;
    std::uint32_t pop_size_ = 0;
    float* d_dist_ = nullptr;
};

} // namespace tsp
