#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <type_traits>

#include <alns/core/alns_state.cuh>
#include <alns/core/runtime_config.hpp>
#include <alns/gpu/rng.cuh>
#include <alns/gpu/gpu_utils.cuh>
#include <alns/gpu/memory.cuh>

namespace alns {

// ============================================================================
// SFINAE: detect optional functions in Config
// ============================================================================

// Detect Config::repair_parallel
template<typename C, typename = void>
struct has_repair_parallel : std::false_type {};

template<typename C>
struct has_repair_parallel<C, std::void_t<decltype(
    C::repair_parallel(
        0, std::declval<typename C::Solution&>(),
        (const int*)nullptr, 0,
        std::declval<const typename C::ProblemData&>(),
        std::declval<XorShift128&>(),
        0, 0, (float*)nullptr)
)>> : std::true_type {};

// Detect Config::evaluate_parallel
template<typename C, typename = void>
struct has_evaluate_parallel : std::false_type {};

template<typename C>
struct has_evaluate_parallel<C, std::void_t<decltype(
    C::evaluate_parallel(
        std::declval<const typename C::Solution&>(),
        std::declval<const typename C::ProblemData&>(),
        0, 0, (float*)nullptr)
)>> : std::true_type {};

// Detect Config::local_search
template<typename C, typename = void>
struct has_local_search : std::false_type {};

template<typename C>
struct has_local_search<C, std::void_t<decltype(
    C::local_search(
        std::declval<typename C::Solution&>(),
        std::declval<const typename C::ProblemData&>(),
        std::declval<XorShift128&>(),
        0, 0, (float*)nullptr)
)>> : std::true_type {};

// ============================================================================
// GLOBAL BEST MANAGEMENT (generic)
// ============================================================================

template<typename Config>
__device__ void tryUpdateGlobalBest(
    SharedBest<typename Config::Solution>* global_best,
    const typename Config::Solution* candidate,
    float candidate_obj,
    int gpu_id, int block_id, uint64_t iteration)
{
    // Quick check without lock
    bool dominated = Config::MINIMIZE ?
        (candidate_obj >= global_best->objective) :
        (candidate_obj <= global_best->objective);
    if (dominated) return;

    // Try to acquire lock
    if (atomicCAS(&global_best->lock, 0, 1) == 0) {
        bool improved = Config::MINIMIZE ?
            (candidate_obj < global_best->objective) :
            (candidate_obj > global_best->objective);

        if (improved) {
            copySolutionVectorized(&global_best->solution, candidate, 0, 1);
            global_best->objective = candidate_obj;
            global_best->source_gpu = gpu_id;
            global_best->source_block = block_id;
            global_best->found_at_iteration = iteration;
        }

        __threadfence();
        atomicExch(&global_best->lock, 0);
    }
}

template<typename Config>
__device__ void checkAndImportGlobalBest(
    SharedBest<typename Config::Solution>* global_best,
    typename Config::Solution* local_best,
    float* local_best_obj)
{
    bool dominated = Config::MINIMIZE ?
        (global_best->objective >= *local_best_obj) :
        (global_best->objective <= *local_best_obj);
    if (dominated) return;

    if (atomicCAS(&global_best->lock, 0, 1) == 0) {
        bool improved = Config::MINIMIZE ?
            (global_best->objective < *local_best_obj) :
            (global_best->objective > *local_best_obj);

        if (improved) {
            copySolutionVectorized(local_best, &global_best->solution, 0, 1);
            *local_best_obj = global_best->objective;
        }
        __threadfence();
        atomicExch(&global_best->lock, 0);
    }
}

// ============================================================================
// ALNS MAIN LOOP KERNEL (generic, templated on Config)
// ============================================================================

template<typename Config>
__global__ void alnsMainKernel(
    typename Config::Solution* __restrict__ current_solutions,
    typename Config::Solution* __restrict__ best_solutions,
    typename Config::Solution* __restrict__ working_solutions,
    ALNSState<Config::NUM_DESTROY_OPS, Config::NUM_REPAIR_OPS>* __restrict__ states,
    const typename Config::ProblemData* __restrict__ problem_data,
    SharedBest<typename Config::Solution>* __restrict__ global_best,
    DeviceRuntimeParams params,
    int iterations_per_launch,
    int gpu_id)
{
    using Solution = typename Config::Solution;
    using State = ALNSState<Config::NUM_DESTROY_OPS, Config::NUM_REPAIR_OPS>;

    const int block_id = blockIdx.x;
    const int tid = threadIdx.x;

    Solution* current = &current_solutions[block_id];
    Solution* best = &best_solutions[block_id];
    Solution* working = &working_solutions[block_id];
    State* state = &states[block_id];
    const auto& data = *problem_data;

    // Shared memory
    extern __shared__ char shared_raw[];
    float* shared_mem = reinterpret_cast<float*>(shared_raw);

    // Removed elements buffer (shared memory)
    __shared__ int removed[Config::MAX_ELEMENTS];
    __shared__ int num_removed;
    __shared__ int destroy_op, repair_op, removal_size;
    __shared__ float new_objective;
    __shared__ uint32_t new_feasibility;
    __shared__ int accept;
    __shared__ float current_objective;

    // Per-thread RNG
    XorShift128 rng;
    rng.init(12345ULL + gpu_id * 1000 + block_id * 100, tid);

    if (tid == 0) {
        current_objective = state->current_objective;
    }
    __syncthreads();

    // Main ALNS iteration loop
    for (int iter = 0; iter < iterations_per_launch; iter++) {

        // ============================================================
        // STEP 1: Select operators (thread 0)
        // ============================================================
        if (tid == 0) {
            destroy_op = rouletteWheelSelect(
                state->destroy_weights, Config::NUM_DESTROY_OPS, rng);
            repair_op = rouletteWheelSelect(
                state->repair_weights, Config::NUM_REPAIR_OPS, rng);

            int num_elements = Config::get_num_elements(data);
            int max_remove = max(params.min_removal,
                min(num_elements / 2,
                    static_cast<int>(params.max_removal_fraction * num_elements)));
            int min_remove = max(params.min_removal, max_remove / 2);
            removal_size = rng.nextIntRange(min_remove, max_remove);
        }
        __syncthreads();

        // ============================================================
        // STEP 2: Copy current → working (all threads)
        // ============================================================
        copySolutionVectorized(working, current, tid, blockDim.x);
        __syncthreads();

        // ============================================================
        // STEP 3: Destroy (thread 0)
        // ============================================================
        if (tid == 0) {
            Config::destroy(destroy_op, *working, removed, num_removed,
                           removal_size, data, rng);
        }
        __syncthreads();

        // ============================================================
        // STEP 4: Repair
        // ============================================================
        if constexpr (has_repair_parallel<Config>::value) {
            Config::repair_parallel(repair_op, *working, removed, num_removed,
                                     data, rng, tid, blockDim.x, shared_mem);
        } else {
            if (tid == 0) {
                Config::repair(repair_op, *working, removed, num_removed, data, rng);
            }
        }
        __syncthreads();

        // ============================================================
        // STEP 5: Evaluate
        // ============================================================
        if constexpr (has_evaluate_parallel<Config>::value) {
            float obj = Config::evaluate_parallel(*working, data, tid, blockDim.x, shared_mem);
            if (tid == 0) {
                new_objective = obj;
                new_feasibility = Config::check_feasibility(*working, data);
                working->objective = new_objective;
                working->feasibility_flags = new_feasibility;
            }
        } else {
            if (tid == 0) {
                new_objective = Config::evaluate(*working, data);
                new_feasibility = Config::check_feasibility(*working, data);
                working->objective = new_objective;
                working->feasibility_flags = new_feasibility;
            }
        }
        __syncthreads();

        // ============================================================
        // STEP 6: Acceptance (Simulated Annealing)
        // ============================================================
        if (tid == 0) {
            accept = 0;
            bool feasible = (new_feasibility == 0);

            if (feasible) {
                bool improved = Config::MINIMIZE ?
                    (new_objective < current_objective) :
                    (new_objective > current_objective);

                float delta = Config::MINIMIZE ?
                    (new_objective - current_objective) :
                    (current_objective - new_objective);

                if (improved) {
                    accept = 1;
                    state->destroy_scores[destroy_op] += params.score_improved;
                    state->repair_scores[repair_op] += params.score_improved;

                    bool new_best = Config::MINIMIZE ?
                        (new_objective < state->best_objective) :
                        (new_objective > state->best_objective);

                    if (new_best) {
                        state->destroy_scores[destroy_op] +=
                            (params.score_global_best - params.score_improved);
                        state->repair_scores[repair_op] +=
                            (params.score_global_best - params.score_improved);
                        state->improvements++;
                        state->last_improvement_iter = state->iteration;
                    }
                } else {
                    float prob = expf(-delta / state->temperature);
                    if (rng.nextFloat() < prob) {
                        accept = 1;
                        state->destroy_scores[destroy_op] += params.score_accepted;
                        state->repair_scores[repair_op] += params.score_accepted;
                    }
                }
            }

            state->destroy_attempts[destroy_op]++;
            state->repair_attempts[repair_op]++;
            if (accept) state->acceptances++;
        }
        __syncthreads();

        // ============================================================
        // STEP 7: Update solutions
        // ============================================================
        if (accept) {
            copySolutionVectorized(current, working, tid, blockDim.x);

            if (tid == 0) {
                current_objective = new_objective;

                bool new_best = Config::MINIMIZE ?
                    (new_objective < state->best_objective) :
                    (new_objective > state->best_objective);

                if (new_best) {
                    copySolutionVectorized(best, working, 0, 1);
                    state->best_objective = new_objective;

                    tryUpdateGlobalBest<Config>(
                        global_best, working, new_objective,
                        gpu_id, block_id, state->iteration);
                }
            }
        }
        __syncthreads();

        // ============================================================
        // STEP 8: SA cooling and periodic updates
        // ============================================================
        if (tid == 0) {
            state->temperature *= params.cooling_rate;
            state->iteration++;
            state->current_objective = current_objective;

            // Reheating
            uint64_t stagnation = state->iteration - state->last_improvement_iter;
            if (stagnation > static_cast<uint64_t>(params.stagnation_threshold) &&
                state->temperature < state->initial_temperature * 0.01f) {
                state->temperature = state->initial_temperature * params.reheat_factor;
            }

            // Periodic weight update
            if (state->iteration % params.weight_update_interval == 0) {
                updateOperatorWeights(state, params.reaction_factor);
            }

            // Periodic global best check
            if (state->iteration % params.best_check_interval == 0) {
                checkAndImportGlobalBest<Config>(
                    global_best, best, &state->best_objective);
            }
        }
        __syncthreads();

        // ============================================================
        // STEP 9: Optional local search
        // ============================================================
        if constexpr (has_local_search<Config>::value) {
            // Run every 200 iterations
            if (state->iteration % 200 == 0 && state->iteration > 0) {
                Config::local_search(*current, data, rng, tid, blockDim.x, shared_mem);
                __syncthreads();

                if (tid == 0) {
                    float ls_obj = Config::evaluate(*current, data);
                    current_objective = ls_obj;

                    bool new_best = Config::MINIMIZE ?
                        (ls_obj < state->best_objective) :
                        (ls_obj > state->best_objective);

                    if (new_best) {
                        copySolutionVectorized(best, current, 0, 1);
                        state->best_objective = ls_obj;
                        tryUpdateGlobalBest<Config>(
                            global_best, current, ls_obj,
                            gpu_id, block_id, state->iteration);
                    }
                }
                __syncthreads();
            }
        }
    }
}

// ============================================================================
// INITIALIZATION KERNEL
// ============================================================================

template<typename Config>
__global__ void initStateKernel(
    ALNSState<Config::NUM_DESTROY_OPS, Config::NUM_REPAIR_OPS>* __restrict__ states,
    typename Config::Solution* __restrict__ current_solutions,
    typename Config::Solution* __restrict__ best_solutions,
    const typename Config::ProblemData* __restrict__ problem_data,
    float initial_temperature,
    int num_instances)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_instances) return;

    using State = ALNSState<Config::NUM_DESTROY_OPS, Config::NUM_REPAIR_OPS>;
    State* state = &states[idx];
    const auto& data = *problem_data;

    for (int i = 0; i < Config::NUM_DESTROY_OPS; i++) {
        state->destroy_weights[i] = 1.0f;
        state->destroy_scores[i] = 0.0f;
        state->destroy_attempts[i] = 0;
    }

    for (int i = 0; i < Config::NUM_REPAIR_OPS; i++) {
        state->repair_weights[i] = 1.0f;
        state->repair_scores[i] = 0.0f;
        state->repair_attempts[i] = 0;
    }

    state->temperature = initial_temperature;
    state->initial_temperature = initial_temperature;
    state->iteration = 0;
    state->improvements = 0;
    state->acceptances = 0;
    state->last_improvement_iter = 0;

    // Evaluate the initial solution
    float obj = Config::evaluate(current_solutions[idx], data);
    state->best_objective = obj;
    state->current_objective = obj;
}

} // namespace alns
