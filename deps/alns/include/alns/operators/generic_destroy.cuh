#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <alns/gpu/rng.cuh>

namespace alns {

// ============================================================================
// GENERIC DESTROY OPERATORS
// ============================================================================
//
// All generic destroy operators are templates parameterized on Config.
// They use only the hook functions defined by the configurator:
//   Config::get_num_elements(data)
//   Config::is_assigned(sol, element_id)
//   Config::remove_element(sol, element_id, data)
//   Config::removal_cost(sol, element_id, data)
//   Config::relatedness(sol, elem_a, elem_b, data)
//   Config::element_distance(elem_a, elem_b, data)
//   Config::get_num_components(sol, data)
//   Config::get_component_size(sol, component_id)
//   Config::get_element_component(sol, element_id)
//
// All operators run on thread 0 (sequential removal).
// They populate removed[] and set num_removed.

// ============================================================================
// 1. RANDOM REMOVAL
// ============================================================================
// Fisher-Yates shuffle, remove random subset.
// No cost function needed — simplest operator.

template<typename Config>
__device__ void generic_random_destroy(
    typename Config::Solution& sol,
    int* removed, int& num_removed,
    int removal_size,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    int assigned[Config::MAX_ELEMENTS];
    int num_assigned = 0;

    int num_elements = Config::get_num_elements(data);
    for (int i = 0; i < num_elements; i++) {
        if (Config::is_assigned(sol, i)) {
            assigned[num_assigned++] = i;
        }
    }

    int to_remove = min(removal_size, num_assigned);
    num_removed = 0;

    // Fisher-Yates partial shuffle
    for (int i = 0; i < to_remove; i++) {
        int j = i + rng.nextInt(num_assigned - i);
        int temp = assigned[i];
        assigned[i] = assigned[j];
        assigned[j] = temp;

        removed[num_removed++] = assigned[i];
        Config::remove_element(sol, assigned[i], data);
    }
}

// ============================================================================
// 2. WORST REMOVAL
// ============================================================================
// Remove elements with highest removal_cost (randomized power-law selection).
// Requires: Config::removal_cost()

template<typename Config>
__device__ void generic_worst_destroy(
    typename Config::Solution& sol,
    int* removed, int& num_removed,
    int removal_size,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    float costs[Config::MAX_ELEMENTS];
    int ids[Config::MAX_ELEMENTS];
    int num_assigned = 0;

    int num_elements = Config::get_num_elements(data);
    for (int i = 0; i < num_elements; i++) {
        if (Config::is_assigned(sol, i)) {
            costs[num_assigned] = Config::removal_cost(sol, i, data);
            ids[num_assigned] = i;
            num_assigned++;
        }
    }

    const float p = 6.0f;  // Determinism parameter
    int to_remove = min(removal_size, num_assigned);
    num_removed = 0;

    for (int iter = 0; iter < to_remove; iter++) {
        int remaining = num_assigned - iter;

        // Randomized rank selection via power-law
        float y = powf(rng.nextFloat(), p);
        int rank = static_cast<int>(y * remaining);
        rank = min(rank, remaining - 1);

        // Find element at this rank (rank-th highest cost)
        int best_idx = iter;
        for (int i = iter; i < num_assigned; i++) {
            int count_higher = 0;
            for (int j = iter; j < num_assigned; j++) {
                if (j != i && costs[j] > costs[i]) count_higher++;
            }
            if (count_higher == rank) {
                best_idx = i;
                break;
            }
        }

        // Swap to front
        float temp_cost = costs[iter];
        costs[iter] = costs[best_idx];
        costs[best_idx] = temp_cost;

        int temp_id = ids[iter];
        ids[iter] = ids[best_idx];
        ids[best_idx] = temp_id;

        removed[num_removed++] = ids[iter];
        Config::remove_element(sol, ids[iter], data);
    }
}

// ============================================================================
// 3. SHAW (RELATEDNESS) REMOVAL
// ============================================================================
// Seed-based: pick random element, iteratively remove most-related elements.
// Requires: Config::relatedness()

template<typename Config>
__device__ void generic_shaw_destroy(
    typename Config::Solution& sol,
    int* removed, int& num_removed,
    int removal_size,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    int num_elements = Config::get_num_elements(data);
    bool is_removed[Config::MAX_ELEMENTS];
    int assigned[Config::MAX_ELEMENTS];
    int num_assigned = 0;

    for (int i = 0; i < num_elements; i++) {
        is_removed[i] = false;
        if (Config::is_assigned(sol, i)) {
            assigned[num_assigned++] = i;
        }
    }

    if (num_assigned == 0) { num_removed = 0; return; }

    // Pick seed
    int seed = assigned[rng.nextInt(num_assigned)];
    removed[0] = seed;
    num_removed = 1;
    is_removed[seed] = true;
    Config::remove_element(sol, seed, data);

    const float p = 6.0f;
    int to_remove = min(removal_size, num_assigned);

    while (num_removed < to_remove) {
        float rel_scores[Config::MAX_ELEMENTS];
        int candidates[Config::MAX_ELEMENTS];
        int num_candidates = 0;

        for (int i = 0; i < num_elements; i++) {
            if (!Config::is_assigned(sol, i) || is_removed[i]) continue;

            // Minimum relatedness to any removed element
            float min_rel = FLT_MAX;
            for (int r = 0; r < num_removed; r++) {
                float rel = Config::relatedness(sol, i, removed[r], data);
                min_rel = fminf(min_rel, rel);
            }

            rel_scores[num_candidates] = min_rel;
            candidates[num_candidates] = i;
            num_candidates++;
        }

        if (num_candidates == 0) break;

        // Sort by relatedness (ascending = most related first)
        for (int i = 1; i < num_candidates; i++) {
            float key_rel = rel_scores[i];
            int key_cust = candidates[i];
            int j = i - 1;

            while (j >= 0 && rel_scores[j] > key_rel) {
                rel_scores[j + 1] = rel_scores[j];
                candidates[j + 1] = candidates[j];
                j--;
            }
            rel_scores[j + 1] = key_rel;
            candidates[j + 1] = key_cust;
        }

        // Select using power-law
        float y = powf(rng.nextFloat(), p);
        int pos = min(static_cast<int>(y * num_candidates), num_candidates - 1);

        int selected = candidates[pos];
        removed[num_removed++] = selected;
        is_removed[selected] = true;
        Config::remove_element(sol, selected, data);
    }
}

// ============================================================================
// 4. CLUSTER REMOVAL
// ============================================================================
// Pick random center, remove nearest elements by element_distance().
// Requires: Config::element_distance()

template<typename Config>
__device__ void generic_cluster_destroy(
    typename Config::Solution& sol,
    int* removed, int& num_removed,
    int removal_size,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    int assigned[Config::MAX_ELEMENTS];
    int num_assigned = 0;

    int num_elements = Config::get_num_elements(data);
    for (int i = 0; i < num_elements; i++) {
        if (Config::is_assigned(sol, i)) {
            assigned[num_assigned++] = i;
        }
    }

    if (num_assigned == 0) { num_removed = 0; return; }

    // Pick random center
    int center = assigned[rng.nextInt(num_assigned)];

    // Compute distances and sort
    float distances[Config::MAX_ELEMENTS];
    int sorted[Config::MAX_ELEMENTS];

    for (int i = 0; i < num_assigned; i++) {
        distances[i] = Config::element_distance(center, assigned[i], data);
        sorted[i] = assigned[i];
    }

    // Insertion sort by distance (ascending)
    for (int i = 1; i < num_assigned; i++) {
        float key_d = distances[i];
        int key_e = sorted[i];
        int j = i - 1;

        while (j >= 0 && distances[j] > key_d) {
            distances[j + 1] = distances[j];
            sorted[j + 1] = sorted[j];
            j--;
        }
        distances[j + 1] = key_d;
        sorted[j + 1] = key_e;
    }

    // Remove nearest
    int to_remove = min(removal_size, num_assigned);
    num_removed = 0;

    for (int i = 0; i < to_remove; i++) {
        removed[num_removed++] = sorted[i];
        Config::remove_element(sol, sorted[i], data);
    }
}

// ============================================================================
// 5. COMPONENT REMOVAL
// ============================================================================
// Remove all elements from a randomly selected component.
// Generalizes "route removal" in VRP.
// Requires: Config::get_num_components(), Config::get_component_size(),
//           Config::get_element_component()

template<typename Config>
__device__ void generic_component_destroy(
    typename Config::Solution& sol,
    int* removed, int& num_removed,
    int removal_size,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    int num_components = Config::get_num_components(sol, data);
    if (num_components == 0) { num_removed = 0; return; }

    // Find non-empty components
    int non_empty[Config::MAX_COMPONENTS];
    int num_non_empty = 0;

    for (int c = 0; c < num_components; c++) {
        if (Config::get_component_size(sol, c) > 0) {
            non_empty[num_non_empty++] = c;
        }
    }

    if (num_non_empty == 0) { num_removed = 0; return; }

    int target = non_empty[rng.nextInt(num_non_empty)];
    num_removed = 0;

    // Collect elements in this component
    int num_elements = Config::get_num_elements(data);
    for (int i = 0; i < num_elements && num_removed < removal_size; i++) {
        if (Config::get_element_component(sol, i) == target) {
            removed[num_removed++] = i;
        }
    }

    // Remove them (in reverse to avoid position invalidation issues)
    for (int i = num_removed - 1; i >= 0; i--) {
        Config::remove_element(sol, removed[i], data);
    }
}

// ============================================================================
// 6. CRITICAL COMPONENT REMOVAL
// ============================================================================
// Remove elements from the component contributing most to the objective.
// For minimize-makespan: remove from component with highest completion.
// For minimize-cost: remove from component with highest cost contribution.
// Uses removal_cost to identify which component is "worst".
// Requires: Config::removal_cost(), Config::get_element_component()

template<typename Config>
__device__ void generic_critical_destroy(
    typename Config::Solution& sol,
    int* removed, int& num_removed,
    int removal_size,
    const typename Config::ProblemData& data,
    XorShift128& rng)
{
    int num_elements = Config::get_num_elements(data);
    int num_components = Config::get_num_components(sol, data);

    // Find the component with the highest sum of removal costs
    float component_cost[Config::MAX_COMPONENTS];
    for (int c = 0; c < num_components; c++) component_cost[c] = 0.0f;

    for (int i = 0; i < num_elements; i++) {
        if (!Config::is_assigned(sol, i)) continue;
        int comp = Config::get_element_component(sol, i);
        if (comp >= 0 && comp < num_components) {
            component_cost[comp] += Config::removal_cost(sol, i, data);
        }
    }

    // Find critical component (highest total removal cost = most savings if emptied)
    int critical = 0;
    float max_cost = component_cost[0];
    for (int c = 1; c < num_components; c++) {
        if (component_cost[c] > max_cost) {
            max_cost = component_cost[c];
            critical = c;
        }
    }

    // Collect elements from critical component
    int candidates[Config::MAX_ELEMENTS];
    int num_candidates = 0;

    for (int i = 0; i < num_elements; i++) {
        if (Config::get_element_component(sol, i) == critical) {
            candidates[num_candidates++] = i;
        }
    }

    // Shuffle and remove up to removal_size
    for (int i = num_candidates - 1; i > 0; i--) {
        int j = rng.nextInt(i + 1);
        int temp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = temp;
    }

    int to_remove = min(removal_size, num_candidates);
    num_removed = 0;

    for (int i = 0; i < to_remove; i++) {
        removed[num_removed++] = candidates[i];
        Config::remove_element(sol, candidates[i], data);
    }

    // If we need more, take from other components
    if (num_removed < removal_size) {
        for (int i = 0; i < num_elements && num_removed < removal_size; i++) {
            if (Config::is_assigned(sol, i)) {
                removed[num_removed++] = i;
                Config::remove_element(sol, i, data);
            }
        }
    }
}

} // namespace alns
