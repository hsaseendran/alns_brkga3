#pragma once

// ============================================================================
// VRP-RPD GPU Decoder — CHROMOSOME mode, 4*N genes
//
// Chromosome structure (4 * num_customers genes):
//   Component 0: Drop priorities   [0 .. N-1]
//   Component 1: Pickup priorities  [N .. 2N-1]
//   Component 2: Drop agent hints   [2N .. 3N-1]
//   Component 3: Pickup agent hints [3N .. 4N-1]
//
// The decoder:
//   1. Merges drop & pickup priorities into a unified 2N operation list
//   2. Sorts by priority to get execution order
//   3. Processes operations in order with dependency checking:
//      - Drop: agent must have inventory (hint-guided agent selection)
//      - Pickup: must be dropped + processing done, agent needs capacity
//   4. Multiple passes until all operations complete
//   5. Cross-agent pickups supported: any agent can pick up any dropped item
//
// This matches the RCMADP config approach — the GA controls the FULL schedule
// (both drops AND pickups), enabling exact warm-start from ALNS solutions.
// ============================================================================

#include <brkga3/decoder.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <vector>
#include <cstdint>
#include <cfloat>

namespace vrprpd {

constexpr int MAX_AGENTS    = 16;
constexpr int MAX_CUSTOMERS = 64;
constexpr int MAX_OPS       = 2 * MAX_CUSTOMERS;

__global__ void vrprpdDecodeKernel(
    const brkga3::Gene*  __restrict__ d_genes_soa,
    const float*         __restrict__ d_travel_times,
    const float*         __restrict__ d_proc_times,
    brkga3::Fitness*     __restrict__ d_fitness,
    std::uint32_t pop_size,
    std::uint32_t n_customers,
    std::uint32_t num_locations,
    std::uint32_t n_agents,
    std::uint32_t resources_per_agent)
{
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pop_size) return;

    const int nc     = min((int)n_customers, MAX_CUSTOMERS);
    const int k      = (int)resources_per_agent;
    const int n_ops  = 2 * nc;

    // ----------------------------------------------------------------
    // Step 1: Read all 2N operation priorities and sort
    //   op 0..N-1   = drop customer i   (priority from genes[i])
    //   op N..2N-1  = pickup customer i  (priority from genes[N+i])
    // ----------------------------------------------------------------
    float op_priority[MAX_OPS];
    int   op_order[MAX_OPS];

    for (int i = 0; i < nc; ++i) {
        op_priority[i]      = d_genes_soa[i * pop_size + tid];           // drop priorities
        op_priority[nc + i] = d_genes_soa[(nc + i) * pop_size + tid];    // pickup priorities
        op_order[i]      = i;
        op_order[nc + i] = nc + i;
    }

    // Insertion sort by priority (fine for N ≤ ~500)
    for (int i = 1; i < n_ops; ++i) {
        float key = op_priority[i];
        int   val = op_order[i];
        int j = i - 1;
        while (j >= 0 && op_priority[j] > key) {
            op_priority[j + 1] = op_priority[j];
            op_order[j + 1]    = op_order[j];
            j--;
        }
        op_priority[j + 1] = key;
        op_order[j + 1]    = val;
    }

    // ----------------------------------------------------------------
    // Step 2: Read agent hints
    //   genes[2N .. 3N-1] = drop agent hints
    //   genes[3N .. 4N-1] = pickup agent hints
    // ----------------------------------------------------------------
    float drop_hint[MAX_CUSTOMERS];
    float pickup_hint[MAX_CUSTOMERS];
    for (int i = 0; i < nc; ++i) {
        drop_hint[i]   = d_genes_soa[(2 * nc + i) * pop_size + tid];
        pickup_hint[i] = d_genes_soa[(3 * nc + i) * pop_size + tid];
    }

    // ----------------------------------------------------------------
    // Step 3: Process operations in priority order (multi-pass)
    // ----------------------------------------------------------------
    float agent_time[MAX_AGENTS];
    int   agent_pos[MAX_AGENTS];
    int   agent_inv[MAX_AGENTS];

    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
        agent_time[a] = 0.0f;
        agent_pos[a]  = 0;
        agent_inv[a]  = k;
    }

    float dropoff_time[MAX_CUSTOMERS];
    bool  dropped[MAX_CUSTOMERS];
    bool  picked_up[MAX_CUSTOMERS];
    bool  op_done[MAX_OPS];

    for (int i = 0; i < nc; ++i) {
        dropoff_time[i] = -1.0f;
        dropped[i]  = false;
        picked_up[i] = false;
    }
    for (int i = 0; i < n_ops; ++i)
        op_done[i] = false;

    int completed = 0;
    // Multiple passes to resolve dependencies (pickup must wait for drop + processing)
    for (int pass = 0; pass < n_ops && completed < n_ops; ++pass) {
        bool made_progress = false;

        for (int idx = 0; idx < n_ops; ++idx) {
            int op = op_order[idx];
            if (op_done[op]) continue;

            bool is_drop = (op < nc);
            int cust = is_drop ? op : (op - nc);  // 0-indexed customer
            int node = cust + 1;                   // 1-indexed location

            int best_agent = -1;
            float best_time = FLT_MAX;

            if (is_drop) {
                // Drop: need agent with inventory > 0
                if (dropped[cust]) continue;  // already dropped

                int hint_a = min(max((int)(drop_hint[cust] * (float)n_agents), 0),
                                 (int)n_agents - 1);

                // Try hinted agent first
                if (agent_inv[hint_a] > 0) {
                    float travel = d_travel_times[agent_pos[hint_a] * num_locations + node];
                    best_time = agent_time[hint_a] + travel;
                    best_agent = hint_a;
                } else {
                    // Fallback: earliest arrival among feasible agents
                    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
                        if (agent_inv[a] > 0) {
                            float travel = d_travel_times[agent_pos[a] * num_locations + node];
                            float arrival = agent_time[a] + travel;
                            if (arrival < best_time) {
                                best_time = arrival;
                                best_agent = a;
                            }
                        }
                    }
                }

                if (best_agent >= 0) {
                    float travel = d_travel_times[agent_pos[best_agent] * num_locations + node];
                    agent_time[best_agent] += travel;
                    agent_pos[best_agent] = node;
                    agent_inv[best_agent]--;
                    dropoff_time[cust] = agent_time[best_agent];
                    dropped[cust] = true;
                    op_done[op] = true;
                    completed++;
                    made_progress = true;
                }
            } else {
                // Pickup: need dropped + processing done + agent with capacity
                if (!dropped[cust] || picked_up[cust]) continue;

                float ready_time = dropoff_time[cust] + d_proc_times[cust];

                int hint_a = min(max((int)(pickup_hint[cust] * (float)n_agents), 0),
                                 (int)n_agents - 1);

                // Try hinted agent first
                if (agent_inv[hint_a] < k) {
                    float travel = d_travel_times[agent_pos[hint_a] * num_locations + node];
                    float arrival = fmaxf(agent_time[hint_a] + travel, ready_time);
                    best_time = arrival;
                    best_agent = hint_a;
                } else {
                    // Fallback: earliest arrival among feasible agents
                    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
                        if (agent_inv[a] < k) {
                            float travel = d_travel_times[agent_pos[a] * num_locations + node];
                            float arrival = fmaxf(agent_time[a] + travel, ready_time);
                            if (arrival < best_time) {
                                best_time = arrival;
                                best_agent = a;
                            }
                        }
                    }
                }

                if (best_agent >= 0) {
                    float travel = d_travel_times[agent_pos[best_agent] * num_locations + node];
                    float arrival = fmaxf(agent_time[best_agent] + travel, ready_time);
                    agent_time[best_agent] = arrival;
                    agent_pos[best_agent] = node;
                    agent_inv[best_agent]++;
                    picked_up[cust] = true;
                    op_done[op] = true;
                    completed++;
                    made_progress = true;
                }
            }
        }

        if (!made_progress) break;
    }

    // All agents return to depot
    float makespan = 0.0f;
    for (unsigned a = 0; a < n_agents && a < MAX_AGENTS; ++a) {
        float t = agent_time[a]
                + d_travel_times[agent_pos[a] * num_locations + 0];
        if (t > makespan) makespan = t;
    }

    // Penalty for unserviced operations
    int unserviced = 0;
    for (int c = 0; c < nc; ++c) {
        if (!dropped[c])   unserviced++;
        if (!picked_up[c]) unserviced++;
    }

    d_fitness[tid] = makespan + unserviced * 100000.0f;
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
                const brkga3::Gene*      d_genes_soa,
                const brkga3::GeneIndex* /*d_permutations*/,
                brkga3::Fitness*         d_fitness) override {
        constexpr unsigned BLOCK = 128;
        unsigned grid = (population_size + BLOCK - 1) / BLOCK;

        // chromosome_length = 4 * n_customers
        std::uint32_t n_cust = chromosome_length / 4;

        vrprpdDecodeKernel<<<grid, BLOCK, 0, stream>>>(
            d_genes_soa, d_travel_, d_proc_, d_fitness,
            population_size, n_cust, num_locations_,
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
