#pragma once

// ============================================================================
// VRP-RPD GPU Decoder — PERMUTATION mode, greedy agent assignment
//
// Simplified Vehicle Routing with Release and Pickup-Delivery:
//   - N customers, M agents, all agents start at depot (node 0)
//   - Permutation encodes customer visit priority
//   - Greedy decoder assigns each customer to the agent that can reach it
//     earliest, then the agent performs: travel → drop → processing → pickup
//   - Objective: minimize makespan (max completion time across all agents)
//
// Simplifications vs full VRP-RPD:
//   - Same agent handles both drop and pick for each customer (no cross-agent)
//   - Resource constraints not enforced (simplified)
//   - Processing time included as service time per customer
//
// Permutation layout: d_permutations[chrom_i * chrom_len + rank_k]
//   where chrom_len = n_customers, values in [0, n_customers-1]
//
// Distance matrix: d_travel_times[loc_i * num_locations + loc_j]
//   where num_locations = n_customers + 1 (depot = index 0, customers = 1..N)
// ============================================================================

#include <brkga3/decoder.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <vector>
#include <cstdint>

namespace vrprpd {

constexpr int MAX_AGENTS = 32;

__global__ void vrprpdGreedyDecodeKernel(
    const brkga3::GeneIndex* __restrict__ d_permutations,
    const float*             __restrict__ d_travel_times,
    const float*             __restrict__ d_proc_times,
    brkga3::Fitness*         __restrict__ d_fitness,
    std::uint32_t pop_size,
    std::uint32_t n_customers,
    std::uint32_t num_locations,
    std::uint32_t n_agents)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;

    const brkga3::GeneIndex* perm = d_permutations + tid * n_customers;

    // Agent state: current time and position
    float agent_time[MAX_AGENTS];
    int   agent_pos[MAX_AGENTS];
    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
        agent_time[a] = 0.0f;
        agent_pos[a] = 0;  // all start at depot (location 0)
    }

    // Process customers in permutation order
    for (unsigned i = 0; i < n_customers; ++i) {
        int cust = perm[i];                    // 0-based customer index
        int node = cust + 1;                   // location index (0 = depot)

        // Find agent with minimum arrival time at this customer
        int best_a = 0;
        float best_arrival = agent_time[0]
            + d_travel_times[agent_pos[0] * num_locations + node];

        for (unsigned a = 1; a < n_agents && a < MAX_AGENTS; ++a) {
            float arrival = agent_time[a]
                + d_travel_times[agent_pos[a] * num_locations + node];
            if (arrival < best_arrival) {
                best_arrival = arrival;
                best_a = a;
            }
        }

        // Agent travels to customer, drops, waits processing time, picks up
        agent_time[best_a] = best_arrival + d_proc_times[cust];
        agent_pos[best_a] = node;
    }

    // All agents return to depot
    float makespan = 0.0f;
    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
        float t = agent_time[a]
            + d_travel_times[agent_pos[a] * num_locations + 0];
        if (t > makespan) makespan = t;
    }

    d_fitness[tid] = makespan;
}

class VrpRpdDecoder : public brkga3::GpuDecoder {
public:
    VrpRpdDecoder(const std::vector<float>& travel_times,
                  const std::vector<float>& proc_times,
                  std::uint32_t num_locations,
                  std::uint32_t n_agents)
        : host_travel_(travel_times)
        , host_proc_(proc_times)
        , num_locations_(num_locations)
        , n_agents_(n_agents)
    {}

    void initialize(int gpu_id, std::uint32_t chromosome_length,
                    std::uint32_t population_size) override {
        gpu_id_ = gpu_id;
        brkga3::setDevice(gpu_id);

        std::size_t matrix_bytes = num_locations_ * num_locations_ * sizeof(float);
        BRKGA_CUDA_CHECK(cudaMalloc(&d_travel_, matrix_bytes));
        BRKGA_CUDA_CHECK(cudaMemcpy(d_travel_, host_travel_.data(),
                                     matrix_bytes, cudaMemcpyHostToDevice));

        std::size_t proc_bytes = host_proc_.size() * sizeof(float);
        BRKGA_CUDA_CHECK(cudaMalloc(&d_proc_, proc_bytes));
        BRKGA_CUDA_CHECK(cudaMemcpy(d_proc_, host_proc_.data(),
                                     proc_bytes, cudaMemcpyHostToDevice));
    }

    void decode(cudaStream_t stream,
                std::uint32_t population_size,
                std::uint32_t chromosome_length,
                const brkga3::Gene*      /*d_genes_soa*/,
                const brkga3::GeneIndex* d_permutations,
                brkga3::Fitness*         d_fitness) override {
        constexpr unsigned BLOCK = 256;
        unsigned grid = (population_size + BLOCK - 1) / BLOCK;

        vrprpdGreedyDecodeKernel<<<grid, BLOCK, 0, stream>>>(
            d_permutations, d_travel_, d_proc_, d_fitness,
            population_size, chromosome_length, num_locations_, n_agents_);
        BRKGA_CUDA_CHECK_LAST();
    }

    void finalize() override {
        brkga3::setDevice(gpu_id_);
        if (d_travel_) { cudaFree(d_travel_); d_travel_ = nullptr; }
        if (d_proc_)   { cudaFree(d_proc_);   d_proc_ = nullptr; }
    }

private:
    const std::vector<float>& host_travel_;
    const std::vector<float>& host_proc_;
    std::uint32_t num_locations_ = 0;
    std::uint32_t n_agents_ = 0;
    int gpu_id_ = -1;

    float* d_travel_ = nullptr;
    float* d_proc_   = nullptr;
};

} // namespace vrprpd
