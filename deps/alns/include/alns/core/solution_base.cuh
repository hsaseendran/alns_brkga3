#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace alns {

// ============================================================================
// GENERIC SOLUTION TEMPLATE
// ============================================================================
//
// A default solution representation that covers ~80% of combinatorial
// optimization problems. Elements are assigned to components at positions.
//
// Examples:
//   TSP:  1 component (tour), N elements (cities), ordered
//   VRP:  M components (routes), N elements (customers), ordered
//   Job Shop: M components (machines), N elements (operations), ordered
//   Bin Packing: K components (bins), N elements (items), unordered
//
// For problems that need a custom solution (e.g., paired operations like
// VRP-RPD), define your own Solution struct instead of using this template.
//
// Requirements for any Solution struct:
//   - Must be a POD type (trivially copyable)
//   - Must be aligned to at least 128 bytes
//   - sizeof(Solution) must be a multiple of 16 (for vectorized copy)

template<int MAX_ELEMENTS, int MAX_COMPONENTS, int MAX_COMPONENT_SIZE>
struct alignas(256) GenericSolution {
    // What element is at each position in each component
    int16_t component_elements[MAX_COMPONENTS][MAX_COMPONENT_SIZE];

    // Number of elements in each component
    int16_t component_sizes[MAX_COMPONENTS];

    // Inverse lookup: which component is each element in? (-1 = unassigned)
    int16_t element_component[MAX_ELEMENTS];

    // Inverse lookup: position within its component (-1 = unassigned)
    int16_t element_position[MAX_ELEMENTS];

    // Solution quality
    float objective;
    uint32_t feasibility_flags;

    // Padding to ensure size is multiple of 16
    uint32_t _padding[2];

    // Feasibility constants
    static constexpr uint32_t FEASIBLE = 0;
};

} // namespace alns
