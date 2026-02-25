#pragma once

#include <cstdint>
#include <string>

namespace alns {

// ============================================================================
// RUNTIME CONFIGURATION (host-side, passed to engine)
// ============================================================================

struct ALNSRuntimeConfig {
    // Termination
    int64_t max_iterations = 25000;
    double max_time_seconds = 300.0;
    int64_t max_no_improvement = 5000;
    double target_objective = -1.0;  // -1 = disabled

    // Multi-GPU
    int num_gpus = -1;  // -1 = auto-detect
    int solutions_per_gpu = 32;

    // ALNS parameters
    float initial_temp_factor = 0.05f;
    float cooling_rate = 0.9998f;
    float reaction_factor = 0.1f;
    float max_removal_fraction = 0.4f;
    int min_removal = 4;

    // Scoring
    float score_global_best = 33.0f;
    float score_improved = 9.0f;
    float score_accepted = 13.0f;

    // Reheating
    int stagnation_threshold = 2000;
    float reheat_factor = 0.5f;

    // Periodic intervals
    int weight_update_interval = 100;
    int best_check_interval = 1000;
    int iterations_per_launch = 100;

    // Output
    bool verbose = true;
    int log_interval = 1000;
};

// ============================================================================
// DEVICE-SIDE RUNTIME PARAMS (compact, passed to kernel)
// ============================================================================

struct DeviceRuntimeParams {
    float cooling_rate;
    float reaction_factor;
    float max_removal_fraction;
    int min_removal;
    float score_global_best;
    float score_improved;
    float score_accepted;
    int stagnation_threshold;
    float reheat_factor;
    int weight_update_interval;
    int best_check_interval;
    int block_size;

    static DeviceRuntimeParams fromConfig(const ALNSRuntimeConfig& cfg, int block_size_val) {
        DeviceRuntimeParams p;
        p.cooling_rate = cfg.cooling_rate;
        p.reaction_factor = cfg.reaction_factor;
        p.max_removal_fraction = cfg.max_removal_fraction;
        p.min_removal = cfg.min_removal;
        p.score_global_best = cfg.score_global_best;
        p.score_improved = cfg.score_improved;
        p.score_accepted = cfg.score_accepted;
        p.stagnation_threshold = cfg.stagnation_threshold;
        p.reheat_factor = cfg.reheat_factor;
        p.weight_update_interval = cfg.weight_update_interval;
        p.best_check_interval = cfg.best_check_interval;
        p.block_size = block_size_val;
        return p;
    }
};

// ============================================================================
// SOLVER STATISTICS
// ============================================================================

struct SolverStatistics {
    uint64_t total_iterations = 0;
    double elapsed_seconds = 0.0;
    uint64_t improvements = 0;
    float initial_objective = 0.0f;
    float final_objective = 0.0f;
    std::vector<float> convergence_history;
};

} // namespace alns
