#pragma once

// ============================================================================
// VRP-RPD (Vehicle Routing Problem with Release and Pickup-Delivery)
// ALNS Framework Configurator
// ============================================================================
//
// Features:
//   - Multiple agents (vehicles) with capacity constraints
//   - Each customer requires a DROP-OFF followed by a PICK-UP
//   - Cross-agent pickup: drop on agent A, pick on agent B
//   - Objective: minimize makespan (max completion time across all agents)
//   - Inventory constraints: agents carry limited resources
//   - Precedence: pickup must wait for drop-off + processing time
//
// Destroy operators:
//   0: Random removal (generic)
//   1: Worst removal (generic)
//   2: Shaw/relatedness removal (generic)
//   3: Cluster removal (generic)
//   4: Route removal (custom)
//   5: Critical path removal (custom)
//
// Repair operators (ALL custom due to paired drop/pick insertion):
//   0: Greedy repair (makespan-aware)
//   1: Regret-2 repair
//   2: Regret-3 repair
//   3: Regret-M repair (regret across all agents)

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <cmath>
#include <iostream>

#include <alns/gpu/rng.cuh>
#include <alns/gpu/gpu_utils.cuh>
#include <alns/operators/generic_destroy.cuh>

// ============================================================================
// OPERATION ID ENCODING (16-bit)
// ============================================================================
//   Bits [0-9]:   customer_id (0-1023)
//   Bit  [10]:    operation_type (0=DROP_OFF, 1=PICK_UP)
//   Bits [11-15]: agent_id (0-31)

typedef uint16_t OperationID;

static constexpr int OP_DROP_OFF = 0;
static constexpr int OP_PICK_UP = 1;

__host__ __device__ __forceinline__
OperationID vrp_makeOperationID(int customer, int op_type, int agent) {
    return static_cast<OperationID>(
        (customer & 0x3FF) | ((op_type & 0x1) << 10) | ((agent & 0x1F) << 11));
}

__host__ __device__ __forceinline__
int vrp_getCustomer(OperationID op_id) { return op_id & 0x3FF; }

__host__ __device__ __forceinline__
int vrp_getOpType(OperationID op_id) { return (op_id >> 10) & 0x1; }

__host__ __device__ __forceinline__
int vrp_getAgent(OperationID op_id) { return (op_id >> 11) & 0x1F; }

// ============================================================================
// VRP-RPD CONFIGURATOR
// ============================================================================

struct VRPRPDConfig {

    // ========================================================================
    // CONSTANTS
    // ========================================================================

    static constexpr int MAX_CUSTOMERS = 1024;
    static constexpr int MAX_AGENTS = 32;
    static constexpr int MAX_ROUTE_LENGTH = 512;

    static constexpr int MAX_ELEMENTS = MAX_CUSTOMERS;
    static constexpr int MAX_COMPONENTS = MAX_AGENTS;
    static constexpr int MAX_COMPONENT_SIZE = MAX_ROUTE_LENGTH;
    static constexpr int NUM_DESTROY_OPS = 6;
    static constexpr int NUM_REPAIR_OPS = 4;
    static constexpr bool MINIMIZE = true;
    static constexpr int SHARED_MEM_BYTES = 4096;

    // ========================================================================
    // TYPES
    // ========================================================================

    struct alignas(256) Solution {
        // Per-agent route data
        OperationID route_ops[MAX_AGENTS][MAX_ROUTE_LENGTH];
        int16_t route_lengths[MAX_AGENTS];
        float completion_times[MAX_AGENTS];

        // Customer-indexed fast lookup
        int8_t dropoff_agent[MAX_CUSTOMERS];
        int8_t pickup_agent[MAX_CUSTOMERS];
        int16_t dropoff_position[MAX_CUSTOMERS];
        int16_t pickup_position[MAX_CUSTOMERS];

        // Timing
        float dropoff_time[MAX_CUSTOMERS];
        float pickup_ready_time[MAX_CUSTOMERS];
        float pickup_actual_time[MAX_CUSTOMERS];

        // Solution quality (objective = makespan)
        float objective;
        float total_travel_time;
        uint32_t hash;
        uint32_t feasibility_flags;
        uint32_t _padding[2];
    };

    struct ProblemData {
        float* travel_times;       // (num_locations x num_locations) matrix
        float* processing_times;   // [num_customers] array
        int num_customers;
        int num_agents;
        int resources_per_agent;
        int num_locations;         // num_customers + 1
        int travel_pitch;          // row stride for travel_times
        float max_travel_time;
    };

    // Host-side solution
    struct Operation {
        int customer;
        int type;  // 0=DROP, 1=PICK
        float scheduled_time;
    };

    struct Route {
        int agent_id;
        std::vector<Operation> operations;
        float completion_time;
    };

    struct HostSolution {
        std::vector<Route> routes;
        float makespan;
        float total_travel_time;
        bool is_feasible;
    };

    // ========================================================================
    // INTERNAL HELPER: INSERTION RESULT
    // ========================================================================

    struct InsertionResult {
        float cost;
        int drop_agent;
        int drop_pos;
        int pick_agent;
        int pick_pos;
        bool feasible;
    };

    // ========================================================================
    // DEVICE FUNCTIONS: EVALUATION
    // ========================================================================

    __device__ static float evaluate(const Solution& sol, const ProblemData& data) {
        // Two-pass makespan computation for cross-agent timing
        float pickup_ready[MAX_CUSTOMERS];
        for (int c = 0; c < data.num_customers; c++) pickup_ready[c] = 0.0f;

        // Pass 1: compute all DROP times
        for (int agent = 0; agent < data.num_agents; agent++) {
            int route_len = sol.route_lengths[agent];
            float time = 0.0f;
            int location = 0;
            for (int i = 0; i < route_len; i++) {
                OperationID op = sol.route_ops[agent][i];
                int c = vrp_getCustomer(op);
                int type = vrp_getOpType(op);
                time += data.travel_times[location * data.travel_pitch + (c + 1)];
                if (type == OP_DROP_OFF) {
                    pickup_ready[c] = time + data.processing_times[c];
                } else {
                    time = fmaxf(time, pickup_ready[c]);
                }
                location = c + 1;
            }
        }

        // Pass 2: recompute with correct pickup_ready for all PICKs
        float makespan = 0.0f;
        for (int agent = 0; agent < data.num_agents; agent++) {
            int route_len = sol.route_lengths[agent];
            float time = 0.0f;
            int location = 0;
            for (int i = 0; i < route_len; i++) {
                OperationID op = sol.route_ops[agent][i];
                int c = vrp_getCustomer(op);
                int type = vrp_getOpType(op);
                time += data.travel_times[location * data.travel_pitch + (c + 1)];
                if (type == OP_DROP_OFF) {
                    pickup_ready[c] = time + data.processing_times[c];
                } else {
                    time = fmaxf(time, pickup_ready[c]);
                }
                location = c + 1;
            }
            time += data.travel_times[location * data.travel_pitch + 0];
            makespan = fmaxf(makespan, time);
        }
        return makespan;
    }

    __device__ static uint32_t check_feasibility(const Solution& sol,
                                                  const ProblemData& data) {
        uint32_t flags = 0;
        for (int agent = 0; agent < data.num_agents; agent++) {
            int inventory = data.resources_per_agent;
            for (int i = 0; i < sol.route_lengths[agent]; i++) {
                int type = vrp_getOpType(sol.route_ops[agent][i]);
                if (type == OP_DROP_OFF) {
                    if (inventory <= 0) flags |= 1;
                    inventory--;
                } else {
                    if (inventory >= data.resources_per_agent) flags |= 1;
                    inventory++;
                }
            }
        }
        // Check all customers assigned
        for (int c = 0; c < data.num_customers; c++) {
            if (sol.dropoff_agent[c] < 0 || sol.pickup_agent[c] < 0) flags |= 2;
        }
        return flags;
    }

    // ========================================================================
    // DEVICE FUNCTIONS: ELEMENT ACCESS
    // ========================================================================

    __device__ static int get_num_elements(const ProblemData& data) {
        return data.num_customers;
    }

    __device__ static int get_num_components(const Solution& sol, const ProblemData& data) {
        return data.num_agents;
    }

    __device__ static int get_element_component(const Solution& sol, int element_id) {
        return sol.dropoff_agent[element_id];
    }

    __device__ static int get_element_position(const Solution& sol, int element_id) {
        return sol.dropoff_position[element_id];
    }

    __device__ static int get_component_size(const Solution& sol, int component_id) {
        return sol.route_lengths[component_id];
    }

    __device__ static bool is_assigned(const Solution& sol, int element_id) {
        return sol.dropoff_agent[element_id] >= 0;
    }

    // ========================================================================
    // DEVICE FUNCTIONS: ELEMENT MUTATION
    // ========================================================================

    __device__ static void remove_element(Solution& sol, int customer,
                                           const ProblemData& data) {
        // Remove drop-off
        int drop_agent = sol.dropoff_agent[customer];
        if (drop_agent >= 0) {
            int drop_pos = sol.dropoff_position[customer];
            int route_len = sol.route_lengths[drop_agent];
            for (int i = drop_pos; i < route_len - 1; i++)
                sol.route_ops[drop_agent][i] = sol.route_ops[drop_agent][i + 1];
            sol.route_lengths[drop_agent]--;
            for (int c = 0; c < data.num_customers; c++) {
                if (sol.dropoff_agent[c] == drop_agent && sol.dropoff_position[c] > drop_pos)
                    sol.dropoff_position[c]--;
                if (sol.pickup_agent[c] == drop_agent && sol.pickup_position[c] > drop_pos)
                    sol.pickup_position[c]--;
            }
            sol.dropoff_agent[customer] = -1;
            sol.dropoff_position[customer] = -1;
        }

        // Remove pick-up
        int pick_agent = sol.pickup_agent[customer];
        if (pick_agent >= 0) {
            int pick_pos = sol.pickup_position[customer];
            int route_len = sol.route_lengths[pick_agent];
            for (int i = pick_pos; i < route_len - 1; i++)
                sol.route_ops[pick_agent][i] = sol.route_ops[pick_agent][i + 1];
            sol.route_lengths[pick_agent]--;
            for (int c = 0; c < data.num_customers; c++) {
                if (sol.dropoff_agent[c] == pick_agent && sol.dropoff_position[c] > pick_pos)
                    sol.dropoff_position[c]--;
                if (sol.pickup_agent[c] == pick_agent && sol.pickup_position[c] > pick_pos)
                    sol.pickup_position[c]--;
            }
            sol.pickup_agent[customer] = -1;
            sol.pickup_position[customer] = -1;
        }
    }

    // Same-agent insertion (component = agent, position = drop position)
    // Pickup placed right after drop for simplicity. Used by generic operators.
    __device__ static bool insert_element(Solution& sol, int customer,
                                           int component, int position,
                                           const ProblemData& data) {
        int agent = component;
        int drop_pos = position;
        int pick_pos = drop_pos + 1;
        insert_customer(sol, customer, agent, drop_pos, agent, pick_pos, data);
        return true;
    }

    // Full insertion with separate drop/pick agents and positions
    __device__ static void insert_customer(Solution& sol, int customer,
                                            int drop_agent, int drop_pos,
                                            int pick_agent, int pick_pos,
                                            const ProblemData& data) {
        // Insert drop-off
        int drop_route_len = sol.route_lengths[drop_agent];
        for (int i = drop_route_len; i > drop_pos; i--)
            sol.route_ops[drop_agent][i] = sol.route_ops[drop_agent][i - 1];
        sol.route_ops[drop_agent][drop_pos] = vrp_makeOperationID(customer, OP_DROP_OFF, drop_agent);
        sol.route_lengths[drop_agent]++;
        for (int c = 0; c < data.num_customers; c++) {
            if (sol.dropoff_agent[c] == drop_agent && sol.dropoff_position[c] >= drop_pos)
                sol.dropoff_position[c]++;
            if (sol.pickup_agent[c] == drop_agent && sol.pickup_position[c] >= drop_pos)
                sol.pickup_position[c]++;
        }
        sol.dropoff_agent[customer] = drop_agent;
        sol.dropoff_position[customer] = drop_pos;

        // Insert pick-up
        int pick_route_len = sol.route_lengths[pick_agent];
        for (int i = pick_route_len; i > pick_pos; i--)
            sol.route_ops[pick_agent][i] = sol.route_ops[pick_agent][i - 1];
        sol.route_ops[pick_agent][pick_pos] = vrp_makeOperationID(customer, OP_PICK_UP, pick_agent);
        sol.route_lengths[pick_agent]++;
        for (int c = 0; c < data.num_customers; c++) {
            if (sol.dropoff_agent[c] == pick_agent && sol.dropoff_position[c] >= pick_pos)
                sol.dropoff_position[c]++;
            if (sol.pickup_agent[c] == pick_agent && sol.pickup_position[c] >= pick_pos)
                sol.pickup_position[c]++;
        }
        sol.pickup_agent[customer] = pick_agent;
        sol.pickup_position[customer] = pick_pos;
    }

    // ========================================================================
    // DEVICE FUNCTIONS: COST FUNCTIONS
    // ========================================================================

    __device__ static float removal_cost(const Solution& sol, int customer,
                                          const ProblemData& data) {
        float total_delta = 0.0f;
        int drop_agent = sol.dropoff_agent[customer];
        int pick_agent = sol.pickup_agent[customer];

        if (drop_agent >= 0) {
            int drop_pos = sol.dropoff_position[customer];
            int route_len = sol.route_lengths[drop_agent];
            int pred_loc = (drop_pos > 0) ?
                vrp_getCustomer(sol.route_ops[drop_agent][drop_pos - 1]) + 1 : 0;
            int curr_loc = customer + 1;
            int succ_loc = (drop_pos < route_len - 1) ?
                vrp_getCustomer(sol.route_ops[drop_agent][drop_pos + 1]) + 1 : 0;
            float old_cost = data.travel_times[pred_loc * data.travel_pitch + curr_loc] +
                             data.travel_times[curr_loc * data.travel_pitch + succ_loc];
            float new_cost = data.travel_times[pred_loc * data.travel_pitch + succ_loc];
            total_delta += new_cost - old_cost;
        }

        if (pick_agent >= 0 && pick_agent != drop_agent) {
            int pick_pos = sol.pickup_position[customer];
            int route_len = sol.route_lengths[pick_agent];
            int pred_loc = (pick_pos > 0) ?
                vrp_getCustomer(sol.route_ops[pick_agent][pick_pos - 1]) + 1 : 0;
            int curr_loc = customer + 1;
            int succ_loc = (pick_pos < route_len - 1) ?
                vrp_getCustomer(sol.route_ops[pick_agent][pick_pos + 1]) + 1 : 0;
            float old_cost = data.travel_times[pred_loc * data.travel_pitch + curr_loc] +
                             data.travel_times[curr_loc * data.travel_pitch + succ_loc];
            float new_cost = data.travel_times[pred_loc * data.travel_pitch + succ_loc];
            total_delta += new_cost - old_cost;
        }

        return -total_delta;  // Positive = good to remove
    }

    __device__ static float relatedness(const Solution& sol, int i, int j,
                                         const ProblemData& data) {
        const float phi = 9.0f, chi = 3.0f, omega = 5.0f;
        float dist = data.travel_times[(i + 1) * data.travel_pitch + (j + 1)];
        float time_diff = fabsf(sol.dropoff_time[i] - sol.dropoff_time[j]);
        float agent_term = 0.0f;
        if (sol.dropoff_agent[i] != sol.dropoff_agent[j]) agent_term += omega * 0.5f;
        if (sol.pickup_agent[i] != sol.pickup_agent[j]) agent_term += omega * 0.5f;
        return phi * dist + chi * time_diff + agent_term;
    }

    __device__ static float element_distance(int a, int b, const ProblemData& data) {
        return data.travel_times[(a + 1) * data.travel_pitch + (b + 1)];
    }

    // These are required by the interface but VRP-RPD uses custom repair
    __device__ static int num_insertion_positions(const Solution& sol, int element_id,
                                                   int component, const ProblemData& data) {
        return sol.route_lengths[component] + 1;
    }

    __device__ static float insertion_cost(const Solution& sol, int element_id,
                                            int component, int position,
                                            const ProblemData& data) {
        int agent = component;
        int drop_pos = position;
        int route_len = sol.route_lengths[agent];

        // Quick travel cost estimate for drop insertion
        int pred_loc = (drop_pos > 0) ?
            vrp_getCustomer(sol.route_ops[agent][drop_pos - 1]) + 1 : 0;
        int succ_loc = (drop_pos < route_len) ?
            vrp_getCustomer(sol.route_ops[agent][drop_pos]) + 1 : 0;
        int curr_loc = element_id + 1;

        float old_cost = data.travel_times[pred_loc * data.travel_pitch + succ_loc];
        float new_cost = data.travel_times[pred_loc * data.travel_pitch + curr_loc] +
                         data.travel_times[curr_loc * data.travel_pitch + succ_loc];
        return new_cost - old_cost;
    }

    // ========================================================================
    // DEVICE FUNCTIONS: CUSTOM DESTROY OPERATORS
    // ========================================================================

    __device__ static void destroy_route(Solution& sol, int* removed,
                                          int& num_removed, int removal_size,
                                          const ProblemData& data,
                                          alns::XorShift128& rng) {
        int non_empty[MAX_AGENTS];
        int num_non_empty = 0;
        for (int a = 0; a < data.num_agents; a++) {
            if (sol.route_lengths[a] > 0) non_empty[num_non_empty++] = a;
        }
        if (num_non_empty == 0) { num_removed = 0; return; }

        int agent = non_empty[rng.nextInt(num_non_empty)];
        int route_len = sol.route_lengths[agent];
        bool seen[MAX_CUSTOMERS];
        for (int c = 0; c < data.num_customers; c++) seen[c] = false;

        int customers[MAX_ROUTE_LENGTH];
        int count = 0;
        for (int i = 0; i < route_len && count < removal_size; i++) {
            int c = vrp_getCustomer(sol.route_ops[agent][i]);
            if (!seen[c]) { seen[c] = true; customers[count++] = c; }
        }

        num_removed = 0;
        for (int i = 0; i < count; i++) {
            removed[num_removed++] = customers[i];
            remove_element(sol, customers[i], data);
        }
    }

    __device__ static void destroy_critical_path(Solution& sol, int* removed,
                                                  int& num_removed, int removal_size,
                                                  const ProblemData& data,
                                                  alns::XorShift128& rng) {
        int critical_agent = 0;
        float max_completion = 0.0f;
        for (int a = 0; a < data.num_agents; a++) {
            if (sol.completion_times[a] > max_completion) {
                max_completion = sol.completion_times[a];
                critical_agent = a;
            }
        }

        int route_len = sol.route_lengths[critical_agent];
        if (route_len == 0) {
            alns::generic_random_destroy<VRPRPDConfig>(sol, removed, num_removed,
                                                       removal_size, data, rng);
            return;
        }

        bool seen[MAX_CUSTOMERS];
        for (int c = 0; c < data.num_customers; c++) seen[c] = false;

        int customers[MAX_ROUTE_LENGTH];
        int count = 0;
        for (int i = 0; i < route_len && count < removal_size; i++) {
            int c = vrp_getCustomer(sol.route_ops[critical_agent][i]);
            if (!seen[c]) { seen[c] = true; customers[count++] = c; }
        }

        // Shuffle
        for (int i = count - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int temp = customers[i]; customers[i] = customers[j]; customers[j] = temp;
        }

        num_removed = 0;
        int to_remove = min(removal_size, count);
        for (int i = 0; i < to_remove; i++) {
            removed[num_removed++] = customers[i];
            remove_element(sol, customers[i], data);
        }

        // If we need more, take from other routes
        if (num_removed < removal_size) {
            for (int a = 0; a < data.num_agents && num_removed < removal_size; a++) {
                if (a == critical_agent) continue;
                route_len = sol.route_lengths[a];
                for (int i = 0; i < route_len && num_removed < removal_size; i++) {
                    int c = vrp_getCustomer(sol.route_ops[a][i]);
                    if (sol.dropoff_agent[c] >= 0) {
                        removed[num_removed++] = c;
                        remove_element(sol, c, data);
                    }
                }
            }
        }
    }

    // ========================================================================
    // DEVICE FUNCTIONS: CUSTOM REPAIR HELPERS
    // ========================================================================

    // Compute agent completion time with hypothetical insertions
    __device__ static float compute_agent_completion_with_insertion(
        const Solution& sol, const ProblemData& data,
        int agent, int customer, int drop_pos, int pick_pos,
        float pickup_ready, float* out_drop_time) {

        int route_len = sol.route_lengths[agent];
        float current_time = 0.0f;
        int location = 0;
        int inventory = data.resources_per_agent;
        float drop_time = 0.0f;
        int new_idx = 0;

        for (int old_idx = 0; old_idx <= route_len; old_idx++) {
            if (drop_pos >= 0 && new_idx == drop_pos) {
                float travel = data.travel_times[location * data.travel_pitch + (customer + 1)];
                current_time += travel;
                if (inventory <= 0) return FLT_MAX;
                inventory--;
                drop_time = current_time;
                location = customer + 1;
                new_idx++;
            }

            if (pick_pos >= 0 && new_idx == pick_pos) {
                float travel = data.travel_times[location * data.travel_pitch + (customer + 1)];
                current_time += travel;
                float ready = (drop_pos >= 0) ?
                    (drop_time + data.processing_times[customer]) : pickup_ready;
                current_time = fmaxf(current_time, ready);
                if (inventory >= data.resources_per_agent) return FLT_MAX;
                inventory++;
                location = customer + 1;
                new_idx++;
            }

            if (old_idx < route_len) {
                OperationID op = sol.route_ops[agent][old_idx];
                int c = vrp_getCustomer(op);
                int type = vrp_getOpType(op);
                float travel = data.travel_times[location * data.travel_pitch + (c + 1)];
                current_time += travel;
                if (type == OP_DROP_OFF) {
                    if (inventory <= 0) return FLT_MAX;
                    inventory--;
                } else {
                    if (inventory >= data.resources_per_agent) return FLT_MAX;
                    inventory++;
                }
                location = c + 1;
                new_idx++;
            }
        }

        current_time += data.travel_times[location * data.travel_pitch + 0];
        if (out_drop_time) *out_drop_time = drop_time;
        return current_time;
    }

    // Quick drop insertion feasibility (travel delta + inventory check)
    __device__ static float compute_drop_insertion_cost(
        const Solution& sol, const ProblemData& data,
        int agent, int pos, int customer) {

        int route_len = sol.route_lengths[agent];
        int inventory = data.resources_per_agent;
        for (int i = 0; i < pos; i++) {
            int type = vrp_getOpType(sol.route_ops[agent][i]);
            inventory += (type == OP_PICK_UP) ? 1 : -1;
        }
        if (inventory <= 0) return FLT_MAX;

        int pred_loc = (pos > 0) ? vrp_getCustomer(sol.route_ops[agent][pos - 1]) + 1 : 0;
        int succ_loc = (pos < route_len) ? vrp_getCustomer(sol.route_ops[agent][pos]) + 1 : 0;
        int curr_loc = customer + 1;

        float old_cost = data.travel_times[pred_loc * data.travel_pitch + succ_loc];
        float new_cost = data.travel_times[pred_loc * data.travel_pitch + curr_loc] +
                         data.travel_times[curr_loc * data.travel_pitch + succ_loc];
        return new_cost - old_cost;
    }

    // Estimate drop time at position
    __device__ static float estimate_drop_time(
        const Solution& sol, const ProblemData& data,
        int agent, int pos, int customer) {

        float time = 0.0f;
        int location = 0;
        for (int i = 0; i < pos; i++) {
            int c = vrp_getCustomer(sol.route_ops[agent][i]);
            time += data.travel_times[location * data.travel_pitch + (c + 1)];
            location = c + 1;
        }
        time += data.travel_times[location * data.travel_pitch + (customer + 1)];
        return time;
    }

    // Find best insertion for a customer (cross-agent capable, makespan-aware)
    __device__ static InsertionResult find_best_insertion(
        const Solution& sol, const ProblemData& data, int customer) {

        InsertionResult best_same, best_cross;
        best_same.cost = FLT_MAX; best_same.feasible = false;
        best_cross.cost = FLT_MAX; best_cross.feasible = false;

        float current_makespan = sol.objective;

        for (int drop_agent = 0; drop_agent < data.num_agents; drop_agent++) {
            int drop_route_len = sol.route_lengths[drop_agent];

            for (int drop_pos = 0; drop_pos <= drop_route_len; drop_pos++) {
                float drop_feas = compute_drop_insertion_cost(sol, data, drop_agent,
                                                              drop_pos, customer);
                if (drop_feas >= FLT_MAX * 0.5f) continue;

                float drop_time_out = 0.0f;
                float new_drop_completion = compute_agent_completion_with_insertion(
                    sol, data, drop_agent, customer, drop_pos, -1, 0.0f, &drop_time_out);
                if (new_drop_completion >= FLT_MAX * 0.5f) continue;

                float pickup_ready = drop_time_out + data.processing_times[customer];

                for (int pick_agent = 0; pick_agent < data.num_agents; pick_agent++) {
                    int pick_route_len = sol.route_lengths[pick_agent];
                    int extra = (pick_agent == drop_agent) ? 1 : 0;
                    int start = (pick_agent == drop_agent) ? drop_pos + 1 : 0;

                    for (int pick_pos = start; pick_pos <= pick_route_len + extra; pick_pos++) {
                        float new_pick_completion;
                        if (pick_agent == drop_agent) {
                            new_pick_completion = compute_agent_completion_with_insertion(
                                sol, data, drop_agent, customer, drop_pos, pick_pos,
                                pickup_ready, nullptr);
                        } else {
                            new_pick_completion = compute_agent_completion_with_insertion(
                                sol, data, pick_agent, customer, -1, pick_pos,
                                pickup_ready, nullptr);
                        }
                        if (new_pick_completion >= FLT_MAX * 0.5f) continue;

                        float new_makespan = 0.0f;
                        for (int a = 0; a < data.num_agents; a++) {
                            float comp;
                            if (a == drop_agent && a == pick_agent)
                                comp = new_pick_completion;
                            else if (a == drop_agent)
                                comp = new_drop_completion;
                            else if (a == pick_agent)
                                comp = new_pick_completion;
                            else
                                comp = sol.completion_times[a];
                            new_makespan = fmaxf(new_makespan, comp);
                        }

                        float cost = new_makespan - current_makespan;
                        // Balance penalty
                        float max_new = fmaxf(new_drop_completion,
                            (pick_agent != drop_agent) ? new_pick_completion : 0.0f);
                        cost += 0.001f * max_new;

                        InsertionResult& target =
                            (pick_agent != drop_agent) ? best_cross : best_same;

                        if (cost < target.cost) {
                            target.cost = cost;
                            target.drop_agent = drop_agent;
                            target.drop_pos = drop_pos;
                            target.pick_agent = pick_agent;
                            target.pick_pos = pick_pos;
                            target.feasible = true;
                        }
                    }
                }
            }
        }

        if (best_same.feasible && best_cross.feasible)
            return (best_same.cost <= best_cross.cost) ? best_same : best_cross;
        if (best_same.feasible) return best_same;
        return best_cross;
    }

    // Find best insertion with drop on a specific agent
    __device__ static InsertionResult find_best_insertion_for_agent(
        const Solution& sol, const ProblemData& data,
        int customer, int drop_agent) {

        InsertionResult best;
        best.cost = FLT_MAX;
        best.feasible = false;
        int drop_route_len = sol.route_lengths[drop_agent];

        for (int drop_pos = 0; drop_pos <= drop_route_len; drop_pos++) {
            float drop_cost = compute_drop_insertion_cost(sol, data, drop_agent,
                                                          drop_pos, customer);
            if (drop_cost >= FLT_MAX * 0.5f) continue;

            for (int pick_agent = 0; pick_agent < data.num_agents; pick_agent++) {
                int pick_route_len = sol.route_lengths[pick_agent];
                int extra = (pick_agent == drop_agent) ? 1 : 0;
                int start = (pick_agent == drop_agent) ? drop_pos + 1 : 0;

                for (int pick_pos = start; pick_pos <= pick_route_len + extra; pick_pos++) {
                    // Quick travel cost estimate for pick
                    int pred_loc, succ_loc;
                    if (pick_agent == drop_agent) {
                        if (pick_pos == drop_pos + 1)
                            pred_loc = customer + 1;
                        else if (pick_pos > 0)
                            pred_loc = vrp_getCustomer(
                                sol.route_ops[pick_agent]
                                [(pick_pos > drop_pos + 1) ? pick_pos - 2 : pick_pos - 1]) + 1;
                        else
                            pred_loc = 0;
                        succ_loc = ((pick_pos - 1) < pick_route_len) ?
                            vrp_getCustomer(sol.route_ops[pick_agent][pick_pos - 1]) + 1 : 0;
                    } else {
                        pred_loc = (pick_pos > 0) ?
                            vrp_getCustomer(sol.route_ops[pick_agent][pick_pos - 1]) + 1 : 0;
                        succ_loc = (pick_pos < pick_route_len) ?
                            vrp_getCustomer(sol.route_ops[pick_agent][pick_pos]) + 1 : 0;
                    }

                    float pick_cost = data.travel_times[pred_loc * data.travel_pitch + (customer + 1)] +
                                      data.travel_times[(customer + 1) * data.travel_pitch + succ_loc] -
                                      data.travel_times[pred_loc * data.travel_pitch + succ_loc];

                    float total = drop_cost + pick_cost;
                    if (total < best.cost) {
                        best.cost = total;
                        best.drop_agent = drop_agent;
                        best.drop_pos = drop_pos;
                        best.pick_agent = pick_agent;
                        best.pick_pos = pick_pos;
                        best.feasible = true;
                    }
                }
            }
        }
        return best;
    }

    // Update completion times after insertion (two-pass for cross-agent)
    __device__ static void update_completion_times(Solution& sol, const ProblemData& data) {
        float pickup_ready[MAX_CUSTOMERS];
        for (int c = 0; c < data.num_customers; c++) pickup_ready[c] = 0.0f;

        // Pass 1
        for (int agent = 0; agent < data.num_agents; agent++) {
            float time = 0.0f;
            int location = 0;
            for (int i = 0; i < sol.route_lengths[agent]; i++) {
                OperationID op = sol.route_ops[agent][i];
                int c = vrp_getCustomer(op);
                int type = vrp_getOpType(op);
                time += data.travel_times[location * data.travel_pitch + (c + 1)];
                if (type == OP_DROP_OFF) {
                    pickup_ready[c] = time + data.processing_times[c];
                    sol.dropoff_time[c] = time;
                    sol.pickup_ready_time[c] = pickup_ready[c];
                } else {
                    time = fmaxf(time, pickup_ready[c]);
                    sol.pickup_actual_time[c] = time;
                }
                location = c + 1;
            }
            time += data.travel_times[location * data.travel_pitch + 0];
            sol.completion_times[agent] = time;
        }

        // Pass 2 (handles cross-agent timing)
        float makespan = 0.0f;
        for (int agent = 0; agent < data.num_agents; agent++) {
            float time = 0.0f;
            int location = 0;
            for (int i = 0; i < sol.route_lengths[agent]; i++) {
                OperationID op = sol.route_ops[agent][i];
                int c = vrp_getCustomer(op);
                int type = vrp_getOpType(op);
                time += data.travel_times[location * data.travel_pitch + (c + 1)];
                if (type == OP_DROP_OFF) {
                    pickup_ready[c] = time + data.processing_times[c];
                    sol.dropoff_time[c] = time;
                    sol.pickup_ready_time[c] = pickup_ready[c];
                } else {
                    time = fmaxf(time, pickup_ready[c]);
                    sol.pickup_actual_time[c] = time;
                }
                location = c + 1;
            }
            time += data.travel_times[location * data.travel_pitch + 0];
            sol.completion_times[agent] = time;
            makespan = fmaxf(makespan, time);
        }
        sol.objective = makespan;
    }

    // ========================================================================
    // DEVICE FUNCTIONS: CUSTOM REPAIR OPERATORS
    // ========================================================================

    __device__ static void repair_greedy(Solution& sol, const int* removed,
                                          int num_removed, const ProblemData& data) {
        int pending[MAX_CUSTOMERS];
        int num_pending = num_removed;
        for (int i = 0; i < num_removed; i++) pending[i] = removed[i];

        while (num_pending > 0) {
            int best_idx = -1;
            InsertionResult best_result;
            best_result.cost = FLT_MAX;

            for (int i = 0; i < num_pending; i++) {
                InsertionResult result = find_best_insertion(sol, data, pending[i]);
                if (result.feasible && result.cost < best_result.cost) {
                    best_result = result;
                    best_idx = i;
                }
            }

            if (best_idx < 0) break;

            insert_customer(sol, pending[best_idx],
                           best_result.drop_agent, best_result.drop_pos,
                           best_result.pick_agent, best_result.pick_pos, data);
            update_completion_times(sol, data);

            pending[best_idx] = pending[num_pending - 1];
            num_pending--;
        }
    }

    __device__ static void repair_regret_k(Solution& sol, const int* removed,
                                            int num_removed, int k,
                                            const ProblemData& data) {
        int pending[MAX_CUSTOMERS];
        int num_pending = num_removed;
        for (int i = 0; i < num_removed; i++) pending[i] = removed[i];

        while (num_pending > 0) {
            int best_idx = -1;
            float best_regret = -FLT_MAX;
            InsertionResult best_result;

            for (int i = 0; i < num_pending; i++) {
                int customer = pending[i];
                float costs[MAX_AGENTS];
                InsertionResult results[MAX_AGENTS];
                int num_results = 0;

                for (int da = 0; da < data.num_agents; da++) {
                    InsertionResult r = find_best_insertion_for_agent(sol, data, customer, da);
                    if (r.feasible) {
                        costs[num_results] = r.cost;
                        results[num_results] = r;
                        num_results++;
                    }
                }

                if (num_results == 0) continue;

                // Sort by cost
                for (int a = 0; a < num_results - 1; a++) {
                    for (int b = a + 1; b < num_results; b++) {
                        if (costs[b] < costs[a]) {
                            float tc = costs[a]; costs[a] = costs[b]; costs[b] = tc;
                            InsertionResult tr = results[a]; results[a] = results[b]; results[b] = tr;
                        }
                    }
                }

                float regret = 0.0f;
                int eff_k = min(k, num_results);
                for (int j = 1; j < eff_k; j++)
                    regret += costs[j] - costs[0];

                if (regret > best_regret ||
                    (regret == best_regret && costs[0] < best_result.cost)) {
                    best_regret = regret;
                    best_result = results[0];
                    best_idx = i;
                }
            }

            if (best_idx < 0) break;

            insert_customer(sol, pending[best_idx],
                           best_result.drop_agent, best_result.drop_pos,
                           best_result.pick_agent, best_result.pick_pos, data);
            update_completion_times(sol, data);

            pending[best_idx] = pending[num_pending - 1];
            num_pending--;
        }
    }

    // ========================================================================
    // DEVICE FUNCTIONS: OPERATOR DISPATCH
    // ========================================================================

    __device__ static void destroy(int op_id, Solution& sol,
                                    int* removed, int& num_removed,
                                    int removal_size,
                                    const ProblemData& data, alns::XorShift128& rng) {
        switch (op_id) {
            case 0: alns::generic_random_destroy<VRPRPDConfig>(
                        sol, removed, num_removed, removal_size, data, rng); break;
            case 1: alns::generic_worst_destroy<VRPRPDConfig>(
                        sol, removed, num_removed, removal_size, data, rng); break;
            case 2: alns::generic_shaw_destroy<VRPRPDConfig>(
                        sol, removed, num_removed, removal_size, data, rng); break;
            case 3: alns::generic_cluster_destroy<VRPRPDConfig>(
                        sol, removed, num_removed, removal_size, data, rng); break;
            case 4: destroy_route(sol, removed, num_removed, removal_size, data, rng); break;
            case 5: destroy_critical_path(sol, removed, num_removed, removal_size, data, rng); break;
            default: alns::generic_random_destroy<VRPRPDConfig>(
                         sol, removed, num_removed, removal_size, data, rng); break;
        }
    }

    __device__ static void repair(int op_id, Solution& sol,
                                   const int* removed, int num_removed,
                                   const ProblemData& data, alns::XorShift128& rng) {
        switch (op_id) {
            case 0: repair_greedy(sol, removed, num_removed, data); break;
            case 1: repair_regret_k(sol, removed, num_removed, 2, data); break;
            case 2: repair_regret_k(sol, removed, num_removed, 3, data); break;
            case 3: repair_regret_k(sol, removed, num_removed, data.num_agents, data); break;
        }
    }

    // ========================================================================
    // HOST FUNCTIONS
    // ========================================================================

    static ProblemData load_problem(const std::string& distance_path,
                                    const std::string& processing_path,
                                    int num_agents, int resources_per_agent) {
        ProblemData data;

        // Load distance matrix
        std::ifstream dfile(distance_path);
        if (!dfile.is_open())
            throw std::runtime_error("Cannot open: " + distance_path);

        std::vector<std::vector<float>> rows;
        std::string line;
        while (std::getline(dfile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<float> row;
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ',')) {
                size_t s = val.find_first_not_of(" \t\r\n");
                size_t e = val.find_last_not_of(" \t\r\n");
                if (s != std::string::npos) row.push_back(std::stof(val.substr(s, e - s + 1)));
            }
            if (!row.empty()) rows.push_back(row);
        }
        dfile.close();

        int matrix_size = rows.size();
        data.num_customers = matrix_size - 1;
        data.num_agents = num_agents;
        data.resources_per_agent = resources_per_agent;
        data.num_locations = matrix_size;
        data.travel_pitch = matrix_size;

        data.travel_times = new float[matrix_size * matrix_size];
        data.max_travel_time = 0;
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                float v = rows[i][j];
                data.travel_times[i * matrix_size + j] = v;
                data.max_travel_time = std::max(data.max_travel_time, v);
            }
        }

        // Load processing times
        data.processing_times = new float[data.num_customers];
        for (int c = 0; c < data.num_customers; c++) data.processing_times[c] = 0.0f;

        std::ifstream pfile(processing_path);
        if (pfile.is_open()) {
            bool header_skipped = false;
            while (std::getline(pfile, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::stringstream ss(line);
                std::string id_str, pt_str;
                if (!std::getline(ss, id_str, ',')) continue;
                if (!std::getline(ss, pt_str, ',')) continue;

                if (!header_skipped) {
                    try { std::stoi(id_str); } catch (...) { header_skipped = true; continue; }
                    header_skipped = true;
                }

                int cid = std::stoi(id_str);
                float pt = std::stof(pt_str);
                if (cid >= 1 && cid <= data.num_customers)
                    data.processing_times[cid - 1] = pt;
            }
            pfile.close();
        }

        std::cout << "VRP-RPD Problem: " << data.num_customers << " customers, "
                  << data.num_agents << " agents, "
                  << data.resources_per_agent << " resources/agent" << std::endl;

        return data;
    }

    // Single-argument load_problem (distance matrix only, no processing times)
    static ProblemData load_problem(const std::string& path) {
        // Default: 5 agents, 2 resources, no processing times
        ProblemData data;

        std::ifstream file(path);
        if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);

        std::vector<std::vector<float>> rows;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<float> row;
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ',')) {
                size_t s = val.find_first_not_of(" \t\r\n");
                size_t e = val.find_last_not_of(" \t\r\n");
                if (s != std::string::npos) row.push_back(std::stof(val.substr(s, e - s + 1)));
            }
            if (!row.empty()) rows.push_back(row);
        }
        file.close();

        int matrix_size = rows.size();
        data.num_customers = matrix_size - 1;
        data.num_agents = 5;
        data.resources_per_agent = 2;
        data.num_locations = matrix_size;
        data.travel_pitch = matrix_size;

        data.travel_times = new float[matrix_size * matrix_size];
        data.max_travel_time = 0;
        for (int i = 0; i < matrix_size; i++) {
            for (int j = 0; j < matrix_size; j++) {
                float v = rows[i][j];
                data.travel_times[i * matrix_size + j] = v;
                data.max_travel_time = std::max(data.max_travel_time, v);
            }
        }

        data.processing_times = new float[data.num_customers];
        for (int c = 0; c < data.num_customers; c++) data.processing_times[c] = 0.0f;

        return data;
    }

    static void upload_problem_data(const ProblemData& host_data, ProblemData** d_data) {
        ProblemData h_copy = host_data;
        size_t matrix_size = host_data.num_locations * host_data.num_locations * sizeof(float);
        size_t proc_size = host_data.num_customers * sizeof(float);

        ALNS_CUDA_CHECK(cudaMalloc(&h_copy.travel_times, matrix_size));
        ALNS_CUDA_CHECK(cudaMemcpy(h_copy.travel_times, host_data.travel_times,
                                    matrix_size, cudaMemcpyHostToDevice));

        ALNS_CUDA_CHECK(cudaMalloc(&h_copy.processing_times, proc_size));
        ALNS_CUDA_CHECK(cudaMemcpy(h_copy.processing_times, host_data.processing_times,
                                    proc_size, cudaMemcpyHostToDevice));

        ALNS_CUDA_CHECK(cudaMalloc(d_data, sizeof(ProblemData)));
        ALNS_CUDA_CHECK(cudaMemcpy(*d_data, &h_copy, sizeof(ProblemData), cudaMemcpyHostToDevice));
    }

    static HostSolution create_initial_solution(const ProblemData& data) {
        const int N = data.num_customers;
        const int M = data.num_agents;
        const int K = data.resources_per_agent;

        HostSolution sol;
        sol.routes.resize(M);
        for (int a = 0; a < M; a++) sol.routes[a].agent_id = a;

        // Geographic sectoring via pseudo-angles
        struct CInfo { int id; float angle; };
        std::vector<CInfo> sorted(N);

        // Find reference customers
        int ref1 = 0, ref2 = 0;
        float md1 = 0, md2 = 0;
        for (int c = 0; c < N; c++) {
            float d = data.travel_times[0 * data.travel_pitch + (c + 1)];
            if (d > md1) { md2 = md1; ref2 = ref1; md1 = d; ref1 = c; }
            else if (d > md2) { md2 = d; ref2 = c; }
        }

        for (int c = 0; c < N; c++) {
            float d_depot = data.travel_times[0 * data.travel_pitch + (c + 1)];
            float d_ref1 = data.travel_times[(ref1 + 1) * data.travel_pitch + (c + 1)];
            float d_ref2 = data.travel_times[(ref2 + 1) * data.travel_pitch + (c + 1)];
            float x = (d_ref1 - d_ref2) / (md1 + 1.0f);
            float y = (d_depot - (d_ref1 + d_ref2) / 2.0f) / (md1 + 1.0f);
            sorted[c].id = c;
            sorted[c].angle = std::atan2(y, x);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const CInfo& a, const CInfo& b) { return a.angle < b.angle; });

        // Assign to agents
        std::vector<std::vector<int>> agent_custs(M);
        int per_agent = (N + M - 1) / M;
        for (int i = 0; i < N; i++)
            agent_custs[std::min(i / per_agent, M - 1)].push_back(sorted[i].id);

        // Build routes with nearest-neighbor + interleaved D/P
        float makespan = 0;
        float total_travel = 0;
        for (int a = 0; a < M; a++) {
            auto& custs = agent_custs[a];
            if (custs.empty()) { sol.routes[a].completion_time = 0; continue; }

            std::set<int> unvisited(custs.begin(), custs.end());
            std::vector<std::pair<float, int>> pending_picks;
            int location = 0;
            float time = 0;
            int inventory = K;

            while (!unvisited.empty() || !pending_picks.empty()) {
                int best_c = -1;
                int best_op = -1;
                float best_cost = FLT_MAX;
                float best_arrival = 0;

                if (inventory > 0 && !unvisited.empty()) {
                    for (int c : unvisited) {
                        float t = data.travel_times[location * data.travel_pitch + (c + 1)];
                        if (t < best_cost) {
                            best_cost = t; best_c = c; best_op = 0;
                            best_arrival = time + t;
                        }
                    }
                }

                if (inventory < K && !pending_picks.empty()) {
                    for (auto& p : pending_picks) {
                        float t = data.travel_times[location * data.travel_pitch + (p.second + 1)];
                        float arr = time + t;
                        float wait = std::max(0.0f, p.first - arr);
                        if (t + wait < best_cost) {
                            best_cost = t + wait; best_c = p.second; best_op = 1;
                            best_arrival = std::max(arr, p.first);
                        }
                    }
                }

                if (inventory == 0 && !pending_picks.empty() && best_op != 1) {
                    float min_ready = FLT_MAX;
                    int bp = -1;
                    for (size_t i = 0; i < pending_picks.size(); i++) {
                        if (pending_picks[i].first < min_ready) {
                            min_ready = pending_picks[i].first; bp = i;
                        }
                    }
                    if (bp >= 0) {
                        best_c = pending_picks[bp].second; best_op = 1;
                        float t = data.travel_times[location * data.travel_pitch + (best_c + 1)];
                        best_arrival = std::max(time + t, pending_picks[bp].first);
                    }
                }

                if (best_c < 0) break;

                Operation op;
                op.customer = best_c;
                op.type = best_op;
                op.scheduled_time = best_arrival;
                total_travel += data.travel_times[location * data.travel_pitch + (best_c + 1)];
                time = best_arrival;
                location = best_c + 1;

                if (best_op == 0) {
                    unvisited.erase(best_c);
                    inventory--;
                    pending_picks.push_back({time + data.processing_times[best_c], best_c});
                } else {
                    for (auto it = pending_picks.begin(); it != pending_picks.end(); ++it) {
                        if (it->second == best_c) { pending_picks.erase(it); break; }
                    }
                    inventory++;
                }
                sol.routes[a].operations.push_back(op);
            }

            if (location != 0) {
                float t = data.travel_times[location * data.travel_pitch + 0];
                time += t;
                total_travel += t;
            }
            sol.routes[a].completion_time = time;
            makespan = std::max(makespan, time);
        }

        sol.makespan = makespan;
        sol.total_travel_time = total_travel;
        sol.is_feasible = true;
        return sol;
    }

    static float host_evaluate(const HostSolution& sol, const ProblemData& data) {
        return sol.makespan;
    }

    static Solution to_device_solution(const HostSolution& host, const ProblemData& data) {
        Solution sol;
        std::memset(&sol, 0, sizeof(Solution));

        for (int c = 0; c < data.num_customers; c++) {
            sol.dropoff_agent[c] = -1;
            sol.pickup_agent[c] = -1;
            sol.dropoff_position[c] = -1;
            sol.pickup_position[c] = -1;
        }

        for (const auto& route : host.routes) {
            int agent = route.agent_id;
            int pos = 0;
            for (const auto& op : route.operations) {
                sol.route_ops[agent][pos] = vrp_makeOperationID(op.customer, op.type, agent);
                if (op.type == 0) {
                    sol.dropoff_agent[op.customer] = agent;
                    sol.dropoff_position[op.customer] = pos;
                    sol.dropoff_time[op.customer] = op.scheduled_time;
                    sol.pickup_ready_time[op.customer] =
                        op.scheduled_time + data.processing_times[op.customer];
                } else {
                    sol.pickup_agent[op.customer] = agent;
                    sol.pickup_position[op.customer] = pos;
                    sol.pickup_actual_time[op.customer] = op.scheduled_time;
                }
                pos++;
            }
            sol.route_lengths[agent] = pos;
            sol.completion_times[agent] = route.completion_time;
        }

        sol.objective = host.makespan;
        sol.total_travel_time = host.total_travel_time;
        sol.feasibility_flags = host.is_feasible ? 0 : 1;
        return sol;
    }

    static HostSolution from_device_solution(const Solution& sol, const ProblemData& data) {
        HostSolution host;
        host.routes.resize(data.num_agents);
        host.makespan = sol.objective;
        host.total_travel_time = sol.total_travel_time;
        host.is_feasible = (sol.feasibility_flags == 0);

        for (int agent = 0; agent < data.num_agents; agent++) {
            host.routes[agent].agent_id = agent;
            host.routes[agent].completion_time = sol.completion_times[agent];
            for (int i = 0; i < sol.route_lengths[agent]; i++) {
                OperationID op_id = sol.route_ops[agent][i];
                Operation op;
                op.customer = vrp_getCustomer(op_id);
                op.type = vrp_getOpType(op_id);
                op.scheduled_time = (op.type == 0) ?
                    sol.dropoff_time[op.customer] : sol.pickup_actual_time[op.customer];
                host.routes[agent].operations.push_back(op);
            }
        }
        return host;
    }

    static void output_solution(const HostSolution& sol, const ProblemData& data,
                                const std::string& path) {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"problem\": {\n";
        file << "    \"num_customers\": " << data.num_customers << ",\n";
        file << "    \"num_agents\": " << data.num_agents << ",\n";
        file << "    \"resources_per_agent\": " << data.resources_per_agent << "\n";
        file << "  },\n";
        file << "  \"makespan\": " << sol.makespan << ",\n";
        file << "  \"total_travel_time\": " << sol.total_travel_time << ",\n";
        file << "  \"routes\": [\n";
        for (size_t a = 0; a < sol.routes.size(); a++) {
            const auto& route = sol.routes[a];
            file << "    {\n";
            file << "      \"agent\": " << route.agent_id << ",\n";
            file << "      \"completion_time\": " << route.completion_time << ",\n";
            file << "      \"operations\": [";
            for (size_t i = 0; i < route.operations.size(); i++) {
                if (i > 0) file << ", ";
                file << "{\"customer\": " << (route.operations[i].customer + 1)
                     << ", \"type\": \"" << (route.operations[i].type == 0 ? "D" : "P")
                     << "\", \"time\": " << route.operations[i].scheduled_time << "}";
            }
            file << "]\n";
            file << "    }" << (a < sol.routes.size() - 1 ? "," : "") << "\n";
        }
        file << "  ]\n";
        file << "}\n";
        file.close();
    }
};
