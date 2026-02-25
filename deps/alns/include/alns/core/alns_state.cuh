#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace alns {

// ============================================================================
// ALNS STATE (templatized on operator counts)
// ============================================================================

template<int NUM_DESTROY_OPS, int NUM_REPAIR_OPS>
struct ALNSState {
    // Operator weights (adaptive selection)
    float destroy_weights[NUM_DESTROY_OPS];
    float repair_weights[NUM_REPAIR_OPS];

    // Score accumulators (reset after weight update)
    float destroy_scores[NUM_DESTROY_OPS];
    float repair_scores[NUM_REPAIR_OPS];
    int destroy_attempts[NUM_DESTROY_OPS];
    int repair_attempts[NUM_REPAIR_OPS];

    // Simulated annealing
    float temperature;
    float initial_temperature;

    // Progress tracking
    uint64_t iteration;
    uint64_t improvements;
    uint64_t acceptances;
    uint64_t last_improvement_iter;

    // Best solution tracking
    float best_objective;
    float current_objective;
};

// ============================================================================
// WEIGHT UPDATE (generic, works for any operator count)
// ============================================================================

template<int NUM_DESTROY_OPS, int NUM_REPAIR_OPS>
__device__ void updateOperatorWeights(
    ALNSState<NUM_DESTROY_OPS, NUM_REPAIR_OPS>* state,
    float reaction_factor)
{
    for (int i = 0; i < NUM_DESTROY_OPS; i++) {
        if (state->destroy_attempts[i] > 0) {
            float avg_score = state->destroy_scores[i] / state->destroy_attempts[i];
            state->destroy_weights[i] =
                state->destroy_weights[i] * (1.0f - reaction_factor) +
                reaction_factor * avg_score;
            state->destroy_weights[i] = fmaxf(state->destroy_weights[i], 0.1f);
        }
        state->destroy_scores[i] = 0.0f;
        state->destroy_attempts[i] = 0;
    }

    for (int i = 0; i < NUM_REPAIR_OPS; i++) {
        if (state->repair_attempts[i] > 0) {
            float avg_score = state->repair_scores[i] / state->repair_attempts[i];
            state->repair_weights[i] =
                state->repair_weights[i] * (1.0f - reaction_factor) +
                reaction_factor * avg_score;
            state->repair_weights[i] = fmaxf(state->repair_weights[i], 0.1f);
        }
        state->repair_scores[i] = 0.0f;
        state->repair_attempts[i] = 0;
    }
}

// ============================================================================
// SHARED BEST SOLUTION (across GPUs, in unified memory)
// ============================================================================

template<typename Solution>
struct alignas(256) SharedBest {
    Solution solution;
    float objective;
    int source_gpu;
    int source_block;
    uint64_t found_at_iteration;
    int lock;
    int _padding[3];
};

} // namespace alns
