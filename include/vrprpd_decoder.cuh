#pragma once

// ============================================================================
// VRP-RPD GPU Decoder — PERMUTATION mode, interleaved drops & pickups
//
// Models the ALNS VRP-RPD problem exactly:
//   - N customers, M agents, each with k resources, all start at depot
//   - Permutation encodes customer DROP priority
//   - Between consecutive drops, agents can do standalone PICKUPS of ready
//     items from the global pool (cross-agent pickup)
//   - At each step the decoder chooses: do the next DROP, or do a PICKUP
//     — whichever produces the lower makespan
//   - Agents carry inventory [0, k]; DROP costs 1, PICKUP restores 1
//   - Processing: after drop at time t, pickup available at t + proc_time
//   - Objective: minimize makespan (max agent completion + return to depot)
//
// This interleaving matches the ALNS model where agent routes contain
// arbitrary sequences of DROP and PICKUP operations.
// ============================================================================

#include <brkga3/decoder.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <vector>
#include <cstdint>
#include <cfloat>

namespace vrprpd {

constexpr int MAX_AGENTS = 32;
constexpr int MAX_POOL = 128;

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
    const int k = min((int)resources_per_agent, MAX_POOL / (int)n_agents);

    // Per-agent state
    float agent_time[MAX_AGENTS];
    int   agent_pos[MAX_AGENTS];
    int   agent_inv[MAX_AGENTS];

    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
        agent_time[a] = 0.0f;
        agent_pos[a] = 0;
        agent_inv[a] = k;
    }

    // Global pool of pending drops (cross-agent pickup enabled)
    int   pool_node[MAX_POOL];
    float pool_done[MAX_POOL];
    int   pool_cnt = 0;

    // ----------------------------------------------------------------
    // Main loop: interleave drops and pickups
    // At each step, choose between:
    //   (A) DROP the next customer in the permutation
    //   (B) PICKUP a ready item from the global pool (standalone)
    // Each iteration does exactly one operation.
    // Total iterations <= 2 * n_customers (N drops + at most N pickups).
    // ----------------------------------------------------------------
    unsigned drop_idx = 0;

    while (drop_idx < n_customers || pool_cnt > 0) {

        // Current bottleneck
        float max_all = 0.0f;
        for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
            if (agent_time[a] > max_all) max_all = agent_time[a];
        }

        // --- Evaluate best DROP option ---
        int   drop_a = -1;
        float drop_ms = FLT_MAX;
        float drop_arr = FLT_MAX;
        int   drop_pickup = -1;  // optional pickup-before-drop

        if (drop_idx < n_customers) {
            int cust = perm[drop_idx];
            int node = cust + 1;

            for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
                // Direct drop (agent must have inventory)
                if (agent_inv[a] > 0) {
                    float new_time = agent_time[a]
                        + d_travel_times[agent_pos[a] * num_locations + node];
                    float est = fmaxf(new_time, max_all);
                    if (est < drop_ms
                        || (est == drop_ms && new_time < drop_arr)) {
                        drop_ms = est; drop_arr = new_time;
                        drop_a = a;     drop_pickup = -1;
                    }
                }

                // Pickup-then-drop (agent needs inv < k to hold pickup)
                if (pool_cnt > 0 && agent_inv[a] < k) {
                    for (int p = 0; p < pool_cnt; ++p) {
                        float to_p = d_travel_times[agent_pos[a] * num_locations + pool_node[p]];
                        float at_p = fmaxf(agent_time[a] + to_p, pool_done[p]);
                        float to_n = d_travel_times[pool_node[p] * num_locations + node];
                        float new_time = at_p + to_n;
                        float est = fmaxf(new_time, max_all);
                        if (est < drop_ms
                            || (est == drop_ms && new_time < drop_arr)) {
                            drop_ms = est; drop_arr = new_time;
                            drop_a = a;     drop_pickup = p;
                        }
                    }
                }
            }
        }

        // --- Evaluate best standalone PICKUP option ---
        int   pu_a = -1;
        int   pu_p = -1;
        float pu_ms = FLT_MAX;
        float pu_cost = FLT_MAX;

        if (pool_cnt > 0) {
            for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
                if (agent_inv[a] >= k) continue;  // can't hold more
                for (int p = 0; p < pool_cnt; ++p) {
                    float to_p = d_travel_times[agent_pos[a] * num_locations + pool_node[p]];
                    float cost = fmaxf(agent_time[a] + to_p, pool_done[p]);
                    float est = fmaxf(cost, max_all);
                    if (est < pu_ms
                        || (est == pu_ms && cost < pu_cost)) {
                        pu_ms = est; pu_cost = cost;
                        pu_a = a;    pu_p = p;
                    }
                }
            }
        }

        // --- Decision: DROP or standalone PICKUP? ---
        // Do a standalone pickup if:
        //   1. No more drops to do (drop_idx >= n_customers), OR
        //   2. The pickup is "free" (doesn't increase bottleneck) AND
        //      the pickup agent's completion stays below the drop arrival
        //      (i.e., the pickup uses idle time that would be wasted anyway)
        bool do_pickup = false;

        if (drop_idx >= n_customers) {
            // Only pickups remain
            do_pickup = true;
        } else if (pu_a >= 0 && pu_cost <= max_all) {
            // Pickup is free: the agent finishes it without exceeding
            // the current bottleneck, so no delay to any future drop
            do_pickup = true;
        }

        if (do_pickup && pu_a >= 0) {
            // Execute standalone pickup
            float to_p = d_travel_times[agent_pos[pu_a] * num_locations + pool_node[pu_p]];
            agent_time[pu_a] = fmaxf(agent_time[pu_a] + to_p, pool_done[pu_p]);
            agent_pos[pu_a] = pool_node[pu_p];
            agent_inv[pu_a]++;

            pool_cnt--;
            if (pu_p < pool_cnt) {
                pool_node[pu_p] = pool_node[pool_cnt];
                pool_done[pu_p] = pool_done[pool_cnt];
            }
        } else {
            // Execute drop (possibly with a pickup-before-drop)
            if (drop_a < 0) drop_a = 0;
            int cust = perm[drop_idx];
            int node = cust + 1;

            if (drop_pickup >= 0) {
                int pnode = pool_node[drop_pickup];
                float to_p = d_travel_times[agent_pos[drop_a] * num_locations + pnode];
                agent_time[drop_a] = fmaxf(agent_time[drop_a] + to_p,
                                            pool_done[drop_pickup]);
                agent_pos[drop_a] = pnode;
                agent_inv[drop_a]++;

                pool_cnt--;
                if (drop_pickup < pool_cnt) {
                    pool_node[drop_pickup] = pool_node[pool_cnt];
                    pool_done[drop_pickup] = pool_done[pool_cnt];
                }
            }

            float travel = d_travel_times[agent_pos[drop_a] * num_locations + node];
            agent_time[drop_a] += travel;
            agent_pos[drop_a] = node;
            agent_inv[drop_a]--;

            if (pool_cnt < MAX_POOL) {
                pool_node[pool_cnt] = node;
                pool_done[pool_cnt] = agent_time[drop_a] + d_proc_times[cust];
                pool_cnt++;
            }
            drop_idx++;
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
