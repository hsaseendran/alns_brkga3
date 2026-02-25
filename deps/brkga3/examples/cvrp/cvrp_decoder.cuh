#pragma once

#include <brkga3/decoder.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <vector>
#include <cstdint>

namespace cvrp {

// ============================================================================
// CVRP GPU Decoder — thread-per-chromosome, PERMUTATION mode, greedy split
//
// Each thread gets a permutation of client indices [0, n_clients-1].
// The greedy split processes clients in permutation order:
//   - If adding the next client exceeds vehicle capacity, close current route
//     (return to depot) and start a new one.
//   - Fitness = total distance of all routes.
//
// Permutation layout: d_permutations[chrom_i * chrom_len + rank_k]
//   where chrom_len = n_clients, and values are in [0, n_clients-1]
//   representing client indices (0-based, NOT including depot).
//
// Distance matrix: d_dist[node_i * dimension + node_j]
//   where dimension = n_clients + 1 (includes depot at index 0).
//   Client k in permutation maps to distance matrix node (k + 1) if depot=0,
//   or we remap so depot is always index 0.
// ============================================================================

__global__ void cvrpGreedyDecodeKernel(
    const brkga3::GeneIndex* __restrict__ d_permutations,  // [pop_size x n_clients]
    const float*             __restrict__ d_dist,           // [dimension x dimension]
    const std::uint32_t*     __restrict__ d_demands,        // [dimension]
    brkga3::Fitness*         __restrict__ d_fitness,        // [pop_size]
    std::uint32_t pop_size,
    std::uint32_t n_clients,
    std::uint32_t dimension,
    std::uint32_t depot,
    std::uint32_t capacity)
{
    const unsigned chrom_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chrom_id >= pop_size) return;

    const brkga3::GeneIndex* perm = d_permutations + chrom_id * n_clients;

    float total_cost = 0.0f;
    std::uint32_t current_load = 0;
    std::uint32_t prev_node = depot;

    for (std::uint32_t k = 0; k < n_clients; ++k) {
        // Map permutation index to actual node in distance matrix
        // Permutation values are [0, n_clients-1]
        // Node indices: depot=0, clients=1..n_clients (if depot is index 0)
        // We need to skip the depot index
        std::uint32_t client_node = perm[k];
        if (client_node >= depot) client_node += 1;  // skip depot index

        std::uint32_t demand = d_demands[client_node];

        if (current_load + demand > capacity) {
            // Close current route: return to depot
            total_cost += d_dist[prev_node * dimension + depot];
            // Start new route from depot
            prev_node = depot;
            current_load = 0;
        }

        // Visit this client
        total_cost += d_dist[prev_node * dimension + client_node];
        current_load += demand;
        prev_node = client_node;
    }

    // Close the last route
    total_cost += d_dist[prev_node * dimension + depot];

    d_fitness[chrom_id] = total_cost;
}

class CvrpDecoder : public brkga3::GpuDecoder {
public:
    CvrpDecoder(const std::vector<float>& distances,
                const std::vector<std::uint32_t>& demands,
                std::uint32_t dimension,
                std::uint32_t depot,
                std::uint32_t capacity)
        : host_distances_(distances)
        , host_demands_(demands)
        , dimension_(dimension)
        , depot_(depot)
        , capacity_(capacity)
    {}

    void initialize(int gpu_id, std::uint32_t chromosome_length,
                    std::uint32_t population_size) override {
        gpu_id_ = gpu_id;
        brkga3::setDevice(gpu_id);

        // Upload distance matrix
        BRKGA_CUDA_CHECK(cudaMalloc(&d_dist_,
                                     dimension_ * dimension_ * sizeof(float)));
        BRKGA_CUDA_CHECK(cudaMemcpy(d_dist_, host_distances_.data(),
                                     dimension_ * dimension_ * sizeof(float),
                                     cudaMemcpyHostToDevice));

        // Upload demands
        BRKGA_CUDA_CHECK(cudaMalloc(&d_demands_,
                                     dimension_ * sizeof(std::uint32_t)));
        BRKGA_CUDA_CHECK(cudaMemcpy(d_demands_, host_demands_.data(),
                                     dimension_ * sizeof(std::uint32_t),
                                     cudaMemcpyHostToDevice));
    }

    void decode(cudaStream_t stream,
                std::uint32_t population_size,
                std::uint32_t chromosome_length,
                const brkga3::Gene*      /*d_genes_soa*/,
                const brkga3::GeneIndex* d_permutations,
                brkga3::Fitness*         d_fitness) override {
        constexpr unsigned BLOCK = 256;
        unsigned grid = (population_size + BLOCK - 1) / BLOCK;

        cvrpGreedyDecodeKernel<<<grid, BLOCK, 0, stream>>>(
            d_permutations, d_dist_, d_demands_, d_fitness,
            population_size, chromosome_length, dimension_,
            depot_, capacity_);
        BRKGA_CUDA_CHECK_LAST();
    }

    void finalize() override {
        brkga3::setDevice(gpu_id_);
        if (d_dist_)    { cudaFree(d_dist_);    d_dist_ = nullptr; }
        if (d_demands_) { cudaFree(d_demands_); d_demands_ = nullptr; }
    }

private:
    const std::vector<float>& host_distances_;
    const std::vector<std::uint32_t>& host_demands_;
    std::uint32_t dimension_ = 0;
    std::uint32_t depot_ = 0;
    std::uint32_t capacity_ = 0;
    int gpu_id_ = -1;

    float*         d_dist_    = nullptr;
    std::uint32_t* d_demands_ = nullptr;
};

} // namespace cvrp
