# ALNS Framework Configurator Specification v1.0

## What This Is

This document specifies how to write a **configurator** for the GPU-accelerated ALNS (Adaptive Large Neighborhood Search) framework. A configurator is a single C++/CUDA header file (`.cuh`) that adapts the generic ALNS engine to solve a specific optimization problem.

The ALNS engine handles: simulated annealing, adaptive operator weights, multi-GPU orchestration, and convergence tracking. You provide: the solution representation, objective function, and how elements get removed/inserted.

## Quick Start Template

Your configurator is a single struct in a `.cuh` file:

```cpp
#pragma once
#include <alns/gpu/rng.cuh>
#include <alns/operators/generic_destroy.cuh>
#include <alns/operators/generic_repair.cuh>

struct MyConfig {
    // 1. Types
    struct alignas(256) Solution { ... };
    struct ProblemData { ... };
    struct HostSolution { ... };

    // 2. Constants
    static constexpr int MAX_ELEMENTS = ...;
    static constexpr int MAX_COMPONENTS = ...;
    static constexpr int MAX_COMPONENT_SIZE = ...;
    static constexpr int NUM_DESTROY_OPS = ...;
    static constexpr int NUM_REPAIR_OPS = ...;
    static constexpr bool MINIMIZE = true;

    // 3. Device functions (see below)
    // 4. Host functions (see below)
};
```

## Core Concept: Elements and Components

ALNS works by repeatedly **removing** elements from a solution and **reinserting** them. Every problem must define:

- **Elements**: The atomic units that get removed/reinserted
  - TSP: cities in the tour
  - VRP: customers to visit
  - Job Shop: operations to schedule
  - Bin Packing: items to pack

- **Components**: The grouping structures that contain elements
  - TSP: the single tour (1 component)
  - VRP: the routes (M components, one per vehicle)
  - Job Shop: the machines (M components)
  - Bin Packing: the bins (K components)

## Required Constants

```cpp
static constexpr int MAX_ELEMENTS = 1024;       // Max removable elements
static constexpr int MAX_COMPONENTS = 1;         // Max groups (routes, machines, bins)
static constexpr int MAX_COMPONENT_SIZE = 1024;  // Max elements per group
static constexpr int NUM_DESTROY_OPS = 4;        // Total destroy operators
static constexpr int NUM_REPAIR_OPS = 3;         // Total repair operators
static constexpr bool MINIMIZE = true;           // true=minimize, false=maximize
static constexpr int SHARED_MEM_BYTES = 4096;    // Shared memory for parallel ops
```

## Required Types

### Solution (GPU, device-side)
```cpp
struct alignas(256) Solution {
    // Your solution representation here.
    // MUST be:
    //   - POD type (no pointers to heap, no std::vector)
    //   - Fixed size (all arrays have compile-time sizes)
    //   - Aligned to at least 128 bytes (256 recommended)
    //   - sizeof(Solution) must be a multiple of 16
    float objective;
    uint32_t feasibility_flags;  // 0 = feasible
};
```

### ProblemData (device-side)
```cpp
struct ProblemData {
    // Problem input data accessible on GPU.
    // Can contain device pointers to global memory arrays.
    float* distances;    // Example: device pointer to distance matrix
    int num_cities;
};
```

### HostSolution (CPU-side)
```cpp
struct HostSolution {
    // CPU-side solution for I/O. Can use std::vector, std::string, etc.
    std::vector<int> tour;
    float total_distance;
};
```

## Required Device Functions (~15 functions)

All device functions are `__device__ static` members of your config struct.

### Evaluation (2 functions)

```cpp
// Compute objective value from scratch
__device__ static float evaluate(const Solution& sol, const ProblemData& data);

// Check feasibility. Return 0 if feasible, non-zero bitflags otherwise.
__device__ static uint32_t check_feasibility(const Solution& sol, const ProblemData& data);
```

### Element Access (6 functions)

These are called by generic operators to interact with your solution.

```cpp
// Total number of elements in the problem
__device__ static int get_num_elements(const ProblemData& data);

// Number of components (routes, machines, bins, etc.)
__device__ static int get_num_components(const Solution& sol, const ProblemData& data);

// Which component is element_id in? Return -1 if unassigned.
__device__ static int get_element_component(const Solution& sol, int element_id);

// Position of element within its component. Undefined if unassigned.
__device__ static int get_element_position(const Solution& sol, int element_id);

// Number of elements in a component
__device__ static int get_component_size(const Solution& sol, int component_id);

// Is element currently assigned?
__device__ static bool is_assigned(const Solution& sol, int element_id);
```

### Element Mutation (2 functions)

```cpp
// Remove element from its current component.
// Must update ALL relevant data structures (positions, sizes, inverse lookups).
__device__ static void remove_element(Solution& sol, int element_id, const ProblemData& data);

// Insert element into component at position. Return true if successful.
__device__ static bool insert_element(Solution& sol, int element_id,
                                       int component, int position,
                                       const ProblemData& data);
```

### Cost Functions (5 functions)

These are used by generic operators to make intelligent choices.

```cpp
// Cost saved by removing element (positive = removing is beneficial).
// Used by: generic_worst_destroy
__device__ static float removal_cost(const Solution& sol, int element_id,
                                      const ProblemData& data);

// Cost of inserting element at (component, position).
// Used by: generic_greedy_repair, generic_regret_repair
__device__ static float insertion_cost(const Solution& sol, int element_id,
                                        int component, int position,
                                        const ProblemData& data);

// Relatedness between two elements (lower = more related).
// Used by: generic_shaw_destroy
__device__ static float relatedness(const Solution& sol, int elem_a, int elem_b,
                                     const ProblemData& data);

// Distance/dissimilarity between two elements.
// Used by: generic_cluster_destroy
__device__ static float element_distance(int elem_a, int elem_b,
                                          const ProblemData& data);

// Number of valid insertion positions for element in component.
// Used by: generic repair operators to iterate positions.
__device__ static int num_insertion_positions(const Solution& sol, int element_id,
                                               int component, const ProblemData& data);
```

### Operator Dispatch (2-3 functions)

```cpp
// Dispatch destroy operator by ID. Populate removed[] and set num_removed.
__device__ static void destroy(int op_id, Solution& sol,
                                int* removed, int& num_removed,
                                int removal_size,
                                const ProblemData& data, alns::XorShift128& rng);

// Dispatch repair operator by ID. Insert all removed elements back.
__device__ static void repair(int op_id, Solution& sol,
                               const int* removed, int num_removed,
                               const ProblemData& data, alns::XorShift128& rng);

// OPTIONAL: Parallel repair using all block threads.
// If defined, the engine calls this instead of repair().
__device__ static void repair_parallel(int op_id, Solution& sol,
                                        const int* removed, int num_removed,
                                        const ProblemData& data, alns::XorShift128& rng,
                                        int tid, int num_threads, float* shared_mem);
```

## Required Host Functions (6 functions)

```cpp
// Load problem from file. Returns ProblemData with host-side distances pointer.
static ProblemData load_problem(const std::string& path);

// Upload ProblemData to GPU. Allocates device memory, copies data.
static void upload_problem_data(const ProblemData& host_data, ProblemData** d_data);

// Construct initial feasible solution on CPU.
static HostSolution create_initial_solution(const ProblemData& data);

// Evaluate host solution (for initial objective value).
static float host_evaluate(const HostSolution& sol, const ProblemData& data);

// Convert between host and device formats.
static Solution to_device_solution(const HostSolution& host, const ProblemData& data);
static HostSolution from_device_solution(const Solution& sol, const ProblemData& data);

// Write solution to file.
static void output_solution(const HostSolution& sol, const ProblemData& data,
                            const std::string& path);
```

## Available Generic Operators

### Destroy Operators

| ID | Name | What it does | Config functions used |
|----|------|-------------|----------------------|
| - | `generic_random_destroy<Config>` | Remove random elements (Fisher-Yates) | `is_assigned()`, `remove_element()` |
| - | `generic_worst_destroy<Config>` | Remove elements with highest removal cost | `removal_cost()`, `remove_element()` |
| - | `generic_shaw_destroy<Config>` | Remove related elements (seed + expand) | `relatedness()`, `remove_element()` |
| - | `generic_cluster_destroy<Config>` | Remove geographically close elements | `element_distance()`, `remove_element()` |
| - | `generic_component_destroy<Config>` | Empty a random component | `get_component_size()`, `remove_element()` |
| - | `generic_critical_destroy<Config>` | Empty the worst component | `removal_cost()`, `remove_element()` |

### Repair Operators

| ID | Name | What it does | Parallel? |
|----|------|-------------|-----------|
| - | `generic_greedy_repair<Config>` | Insert cheapest element at cheapest position | Serial |
| - | `generic_greedy_repair_parallel<Config>` | Same, but all threads evaluate costs | Parallel |
| - | `generic_regret_repair<Config>(sol, removed, n, k, data, rng)` | Insert highest-regret element | Serial |
| - | `generic_random_repair<Config>` | Insert at random valid positions | Serial |

### How to Use Generic Operators

In your `destroy()` / `repair()` dispatch, call them like this:

```cpp
__device__ static void destroy(int op_id, Solution& sol,
                                int* removed, int& num_removed,
                                int removal_size,
                                const ProblemData& data, alns::XorShift128& rng) {
    switch (op_id) {
        case 0: alns::generic_random_destroy<MyConfig>(sol, removed, num_removed, removal_size, data, rng); break;
        case 1: alns::generic_worst_destroy<MyConfig>(sol, removed, num_removed, removal_size, data, rng); break;
        case 2: my_custom_destroy(sol, removed, num_removed, removal_size, data, rng); break;
    }
}
```

## Writing Custom Operators

If the generic operators are insufficient, write custom ones:

```cpp
// Custom destroy: remove elements that violate a specific constraint
__device__ static void my_custom_destroy(Solution& sol, int* removed, int& num_removed,
                                          int removal_size, const ProblemData& data,
                                          alns::XorShift128& rng) {
    num_removed = 0;
    // Your logic: identify elements to remove, call remove_element() for each
    // Populate removed[] with the element IDs you removed
}
```

## Choosing Operators for Your Problem

**For routing problems (TSP, VRP, CVRP):**
- Destroy: random, worst, Shaw (relatedness = distance), cluster
- Repair: greedy, regret-2 or regret-3

**For scheduling problems (Job Shop, Flow Shop):**
- Destroy: random, worst (removal cost = tardiness contribution), critical (bottleneck machine)
- Repair: greedy (earliest feasible position), regret

**For packing problems (Bin Packing, Knapsack):**
- Destroy: random, worst (removal cost = wasted space), component (empty a bin)
- Repair: greedy (best-fit), random

**For assignment problems:**
- Destroy: random, worst, Shaw (relatedness = assignment similarity)
- Repair: greedy, regret

## GPU Programming Notes

1. **No heap allocation**: All arrays must be stack-allocated with compile-time sizes. Use `int arr[MAX_ELEMENTS]` not `new int[n]`.

2. **No std:: on device**: STL containers cannot be used in `__device__` functions. Use C arrays.

3. **Solution struct must be POD**: No virtual functions, no constructors, no pointers to dynamically allocated memory.

4. **Alignment matters**: `alignas(256)` on Solution ensures cache-friendly access.

5. **sizeof(Solution) % 16 == 0**: Required for vectorized copy between solutions.

6. **Thread safety**: `destroy()` and serial `repair()` run on thread 0 only. `repair_parallel()` runs on all threads — use `__syncthreads()` for synchronization, shared memory for communication.

## Building

1. Create your configurator: `my_problem/my_config.cuh`
2. Create your main: `my_problem/main.cu`
3. Add to CMakeLists.txt:
```cmake
add_executable(alns_my_problem my_problem/main.cu)
target_link_libraries(alns_my_problem alns_framework)
set_target_properties(alns_my_problem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```
4. Build: `mkdir build && cd build && cmake .. && make`

## Main File Template

```cpp
#include "my_config.cuh"
#include <alns/gpu/multi_gpu.hpp>

int main(int argc, char* argv[]) {
    auto data = MyConfig::load_problem(argv[1]);

    alns::ALNSRuntimeConfig config;
    config.max_iterations = 50000;
    config.max_time_seconds = 120.0;

    alns::MultiGPUSolver<MyConfig> solver(config);
    auto result = solver.solve(data);

    MyConfig::output_solution(result, data, "solution.json");
    return 0;
}
```

## Complete Example: TSP

See `framework/examples/tsp/tsp_config.cuh` for a fully working TSP configurator that uses only generic operators (4 destroy + 3 repair).
