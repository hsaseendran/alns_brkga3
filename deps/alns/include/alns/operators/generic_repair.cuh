#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <alns/gpu/rng.cuh>

namespace alns {

// ============================================================================
// GENERIC REPAIR OPERATORS
// ============================================================================
//
// All generic repair operators use Config hook functions:
//   Config::get_num_components(sol, data)
//   Config::num_insertion_positions(sol, element_id, component, data)
//   Config::insertion_cost(sol, element_id, component, position, data)
//   Config::insert_element(sol, element_id, component, position, data)
//
// Serial versions run on thread 0 only.
// Parallel versions use all threads for cost evaluation with tree reduction.

// ============================================================================
// 1. GREEDY REPAIR (serial)
// ============================================================================
// Insert element with globally lowest insertion cost, one at a time.

template<typename Config>
__device__ void generic_greedy_repair(
    typename Config::Solution& sol,
    const int* removed, int num_removed,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    int pending[Config::MAX_ELEMENTS];
    int num_pending = num_removed;
    for (int i = 0; i < num_removed; i++) pending[i] = removed[i];

    while (num_pending > 0) {
        float best_cost = FLT_MAX;
        int best_pending = -1;
        int best_comp = -1;
        int best_pos = -1;

        int num_components = Config::get_num_components(sol, data);

        for (int p = 0; p < num_pending; p++) {
            int elem = pending[p];

            for (int c = 0; c < num_components; c++) {
                int num_pos = Config::num_insertion_positions(sol, elem, c, data);
                for (int pos = 0; pos < num_pos; pos++) {
                    float cost = Config::insertion_cost(sol, elem, c, pos, data);
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_pending = p;
                        best_comp = c;
                        best_pos = pos;
                    }
                }
            }
        }

        if (best_pending < 0) break;

        Config::insert_element(sol, pending[best_pending], best_comp, best_pos, data);

        // Remove from pending (swap with last)
        pending[best_pending] = pending[num_pending - 1];
        num_pending--;
    }
}

// ============================================================================
// 2. GREEDY REPAIR (parallel)
// ============================================================================
// All threads evaluate insertion costs, tree reduction to find best.

template<typename Config>
__device__ void generic_greedy_repair_parallel(
    typename Config::Solution& sol,
    const int* removed, int num_removed,
    const typename Config::ProblemData& data,
    XorShift128& rng,
    int tid, int num_threads,
    float* shared_mem)
{
    // Shared memory layout:
    // [0 .. num_threads-1]:             best_cost per thread
    // [num_threads .. 2*num_threads-1]: best_elem (as float-encoded int)
    // [2*num_threads .. 3*num_threads-1]: best_comp
    // [3*num_threads .. 4*num_threads-1]: best_pos
    float* s_cost = shared_mem;
    int* s_elem = reinterpret_cast<int*>(shared_mem + num_threads);
    int* s_comp = reinterpret_cast<int*>(shared_mem + 2 * num_threads);
    int* s_pos  = reinterpret_cast<int*>(shared_mem + 3 * num_threads);

    __shared__ int pending[Config::MAX_ELEMENTS];
    __shared__ int num_pending;

    if (tid == 0) {
        num_pending = num_removed;
        for (int i = 0; i < num_removed; i++) pending[i] = removed[i];
    }
    __syncthreads();

    while (num_pending > 0) {
        float my_best_cost = FLT_MAX;
        int my_elem = -1, my_comp = -1, my_pos = -1;

        int num_components = Config::get_num_components(sol, data);

        // Distribute work: linearize (pending_idx, component) across threads
        int total_work = num_pending * num_components;

        for (int work_idx = tid; work_idx < total_work; work_idx += num_threads) {
            int pi = work_idx / num_components;
            int ci = work_idx % num_components;
            int elem = pending[pi];

            int num_pos = Config::num_insertion_positions(sol, elem, ci, data);
            for (int pos = 0; pos < num_pos; pos++) {
                float cost = Config::insertion_cost(sol, elem, ci, pos, data);
                if (cost < my_best_cost) {
                    my_best_cost = cost;
                    my_elem = elem;
                    my_comp = ci;
                    my_pos = pos;
                }
            }
        }

        // Write to shared memory
        s_cost[tid] = my_best_cost;
        s_elem[tid] = my_elem;
        s_comp[tid] = my_comp;
        s_pos[tid]  = my_pos;
        __syncthreads();

        // Tree reduction
        for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride && s_cost[tid + stride] < s_cost[tid]) {
                s_cost[tid] = s_cost[tid + stride];
                s_elem[tid] = s_elem[tid + stride];
                s_comp[tid] = s_comp[tid + stride];
                s_pos[tid]  = s_pos[tid + stride];
            }
            __syncthreads();
        }

        // Thread 0 applies the insertion
        if (tid == 0) {
            if (s_elem[0] >= 0) {
                Config::insert_element(sol, s_elem[0], s_comp[0], s_pos[0], data);

                // Remove from pending
                for (int i = 0; i < num_pending; i++) {
                    if (pending[i] == s_elem[0]) {
                        pending[i] = pending[num_pending - 1];
                        num_pending--;
                        break;
                    }
                }
            } else {
                num_pending = 0;  // No feasible insertion
            }
        }
        __syncthreads();
    }
}

// ============================================================================
// 3. REGRET-K REPAIR (serial)
// ============================================================================
// For each element, find K-best insertions. Insert element with highest regret.
// regret = sum(cost[i] - cost[0]) for i=1..K-1

template<typename Config>
__device__ void generic_regret_repair(
    typename Config::Solution& sol,
    const int* removed, int num_removed,
    int k,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    int pending[Config::MAX_ELEMENTS];
    int num_pending = num_removed;
    for (int i = 0; i < num_removed; i++) pending[i] = removed[i];

    while (num_pending > 0) {
        float best_regret = -FLT_MAX;
        int best_pending = -1;
        int best_comp = -1;
        int best_pos = -1;
        float best_cost = FLT_MAX;

        int num_components = Config::get_num_components(sol, data);

        for (int p = 0; p < num_pending; p++) {
            int elem = pending[p];

            // Find best insertion per component
            float comp_costs[Config::MAX_COMPONENTS];
            int comp_positions[Config::MAX_COMPONENTS];
            int num_valid = 0;

            for (int c = 0; c < num_components; c++) {
                float c_best = FLT_MAX;
                int c_pos = -1;

                int num_pos = Config::num_insertion_positions(sol, elem, c, data);
                for (int pos = 0; pos < num_pos; pos++) {
                    float cost = Config::insertion_cost(sol, elem, c, pos, data);
                    if (cost < c_best) {
                        c_best = cost;
                        c_pos = pos;
                    }
                }

                if (c_pos >= 0) {
                    comp_costs[num_valid] = c_best;
                    comp_positions[num_valid] = c;
                    num_valid++;
                }
            }

            if (num_valid == 0) continue;

            // Sort by cost (ascending)
            for (int a = 0; a < num_valid - 1; a++) {
                for (int b = a + 1; b < num_valid; b++) {
                    if (comp_costs[b] < comp_costs[a]) {
                        float tc = comp_costs[a]; comp_costs[a] = comp_costs[b]; comp_costs[b] = tc;
                        int tp = comp_positions[a]; comp_positions[a] = comp_positions[b]; comp_positions[b] = tp;
                    }
                }
            }

            // Compute regret
            float regret = 0.0f;
            int eff_k = min(k, num_valid);
            for (int j = 1; j < eff_k; j++) {
                regret += comp_costs[j] - comp_costs[0];
            }

            // Find the actual best position for the best component
            int b_comp = comp_positions[0];
            float b_cost = comp_costs[0];
            int b_pos = -1;

            int num_pos = Config::num_insertion_positions(sol, elem, b_comp, data);
            for (int pos = 0; pos < num_pos; pos++) {
                float cost = Config::insertion_cost(sol, elem, b_comp, pos, data);
                if (cost <= b_cost + 0.001f) {
                    b_pos = pos;
                    break;
                }
            }
            if (b_pos < 0) b_pos = 0;

            if (regret > best_regret || (regret == best_regret && b_cost < best_cost)) {
                best_regret = regret;
                best_pending = p;
                best_comp = b_comp;
                best_pos = b_pos;
                best_cost = b_cost;
            }
        }

        if (best_pending < 0) break;

        Config::insert_element(sol, pending[best_pending], best_comp, best_pos, data);

        pending[best_pending] = pending[num_pending - 1];
        num_pending--;
    }
}

// ============================================================================
// 4. RANDOM REPAIR (serial)
// ============================================================================
// Insert each element at a random valid position.

template<typename Config>
__device__ void generic_random_repair(
    typename Config::Solution& sol,
    const int* removed, int num_removed,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    // Shuffle insertion order
    int order[Config::MAX_ELEMENTS];
    for (int i = 0; i < num_removed; i++) order[i] = removed[i];

    for (int i = num_removed - 1; i > 0; i--) {
        int j = rng.nextInt(i + 1);
        int temp = order[i];
        order[i] = order[j];
        order[j] = temp;
    }

    int num_components = Config::get_num_components(sol, data);

    for (int i = 0; i < num_removed; i++) {
        int elem = order[i];

        // Collect all valid (component, position) pairs
        int valid_comps[Config::MAX_COMPONENTS * 64];
        int valid_pos[Config::MAX_COMPONENTS * 64];
        int num_valid = 0;
        constexpr int MAX_VALID = Config::MAX_COMPONENTS * 64;

        for (int c = 0; c < num_components && num_valid < MAX_VALID; c++) {
            int num_pos = Config::num_insertion_positions(sol, elem, c, data);
            for (int pos = 0; pos < num_pos && num_valid < MAX_VALID; pos++) {
                float cost = Config::insertion_cost(sol, elem, c, pos, data);
                if (cost < FLT_MAX * 0.5f) {
                    valid_comps[num_valid] = c;
                    valid_pos[num_valid] = pos;
                    num_valid++;
                }
            }
        }

        if (num_valid > 0) {
            int choice = rng.nextInt(num_valid);
            Config::insert_element(sol, elem, valid_comps[choice], valid_pos[choice], data);
        }
    }
}

} // namespace alns
