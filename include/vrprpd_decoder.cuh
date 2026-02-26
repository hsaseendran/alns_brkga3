#pragma once

// ============================================================================
// VRP-RPD GPU Decoder — PERMUTATION mode, resource-aware greedy assignment
//
// Vehicle Routing with Release & Pickup-Delivery:
//   - N customers, M agents, all agents start at depot (node 0)
//   - Each agent carries k resources (resources_per_agent)
//   - Permutation encodes customer visit priority
//   - Greedy decoder assigns each customer to the agent that can reach it
//     earliest, performing a dropoff (uses 1 resource)
//   - When an agent runs out of resources, it picks up the earliest-completed
//     customer to reclaim a resource before continuing
//   - After all customers are assigned, agents pick up remaining drops
//   - Objective: minimize makespan (max completion time + return to depot)
//
// Key improvement over simplified decoder: agents can have up to k drops
// in-flight simultaneously, with processing overlapping travel time.
//
// Permutation layout: d_permutations[chrom_i * chrom_len + rank_k]
// Distance matrix: d_travel_times[loc_i * num_locations + loc_j]
// ============================================================================

#include <brkga3/decoder.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <vector>
#include <cstdint>

namespace vrprpd {

constexpr int MAX_AGENTS = 32;
constexpr int MAX_RESOURCES = 16;

__global__ void vrprpdGreedyDecodeKernel(
    const brkga3::GeneIndex* __restrict__ d_permutations,
    const float*             __restrict__ d_travel_times,
    const float*             __restrict__ d_proc_times,
    brkga3::Fitness*         __restrict__ d_fitness,
    std::uint32_t pop_size,
    std::uint32_t n_customers,
    std::uint32_t num_locations,
    std::uint32_t n_agents,
    std::uint32_t resources_per_agent)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;

    const brkga3::GeneIndex* perm = d_permutations + tid * n_customers;
    const unsigned k = min(resources_per_agent, (unsigned)MAX_RESOURCES);

    // Per-agent state
    float agent_time[MAX_AGENTS];
    int   agent_pos[MAX_AGENTS];
    int   agent_inv[MAX_AGENTS];        // resources in hand

    // Pending drops per agent: node and completion time
    int   pending_node[MAX_AGENTS][MAX_RESOURCES];
    float pending_done[MAX_AGENTS][MAX_RESOURCES];
    int   pending_cnt[MAX_AGENTS];

    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
        agent_time[a] = 0.0f;
        agent_pos[a] = 0;
        agent_inv[a] = k;
        pending_cnt[a] = 0;
    }

    // Assign customers in permutation order
    for (unsigned i = 0; i < n_customers; ++i) {
        int cust = perm[i];
        int node = cust + 1;

        // Find agent with minimum arrival time at this customer
        int best_a = 0;
        float best_arrival = agent_time[0]
            + d_travel_times[agent_pos[0] * num_locations + node];

        for (unsigned a = 1; a < n_agents && a < MAX_AGENTS; ++a) {
            float arr = agent_time[a]
                + d_travel_times[agent_pos[a] * num_locations + node];
            if (arr < best_arrival) {
                best_arrival = arr;
                best_a = a;
            }
        }

        // If agent has no resources, pick up earliest-completed drop first
        if (agent_inv[best_a] == 0 && pending_cnt[best_a] > 0) {
            // Find drop with earliest completion time
            int ei = 0;
            float et = pending_done[best_a][0];
            for (int d = 1; d < pending_cnt[best_a]; ++d) {
                if (pending_done[best_a][d] < et) {
                    et = pending_done[best_a][d];
                    ei = d;
                }
            }

            int pnode = pending_node[best_a][ei];
            float travel = d_travel_times[agent_pos[best_a] * num_locations + pnode];
            float arrival = agent_time[best_a] + travel;
            agent_time[best_a] = fmaxf(arrival, et);  // wait if processing not done
            agent_pos[best_a] = pnode;
            agent_inv[best_a]++;

            // Remove from pending (swap with last)
            pending_cnt[best_a]--;
            if (ei < pending_cnt[best_a]) {
                pending_node[best_a][ei] = pending_node[best_a][pending_cnt[best_a]];
                pending_done[best_a][ei] = pending_done[best_a][pending_cnt[best_a]];
            }

            // Recalculate arrival at target customer from new position
            best_arrival = agent_time[best_a]
                + d_travel_times[agent_pos[best_a] * num_locations + node];
        }

        // Travel to customer and drop off
        agent_time[best_a] = best_arrival;
        agent_pos[best_a] = node;

        // Record pending drop
        int dc = pending_cnt[best_a];
        if (dc < (int)k) {
            pending_node[best_a][dc] = node;
            pending_done[best_a][dc] = best_arrival + d_proc_times[cust];
            pending_cnt[best_a]++;
        }
        agent_inv[best_a]--;
    }

    // Pick up all remaining pending drops
    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
        while (pending_cnt[a] > 0) {
            // Find earliest completed
            int ei = 0;
            float et = pending_done[a][0];
            for (int d = 1; d < pending_cnt[a]; ++d) {
                if (pending_done[a][d] < et) {
                    et = pending_done[a][d];
                    ei = d;
                }
            }

            int pnode = pending_node[a][ei];
            float travel = d_travel_times[agent_pos[a] * num_locations + pnode];
            float arrival = agent_time[a] + travel;
            agent_time[a] = fmaxf(arrival, et);
            agent_pos[a] = pnode;

            // Remove
            pending_cnt[a]--;
            if (ei < pending_cnt[a]) {
                pending_node[a][ei] = pending_node[a][pending_cnt[a]];
                pending_done[a][ei] = pending_done[a][pending_cnt[a]];
            }
        }
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
                  std::uint32_t n_agents,
                  std::uint32_t resources_per_agent)
        : host_travel_(travel_times)
        , host_proc_(proc_times)
        , num_locations_(num_locations)
        , n_agents_(n_agents)
        , resources_per_agent_(resources_per_agent)
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
        constexpr unsigned BLOCK = 128;
        unsigned grid = (population_size + BLOCK - 1) / BLOCK;

        vrprpdGreedyDecodeKernel<<<grid, BLOCK, 0, stream>>>(
            d_permutations, d_travel_, d_proc_, d_fitness,
            population_size, chromosome_length, num_locations_,
            n_agents_, resources_per_agent_);
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
    std::uint32_t resources_per_agent_ = 1;
    int gpu_id_ = -1;

    float* d_travel_ = nullptr;
    float* d_proc_   = nullptr;
};

} // namespace vrprpd
