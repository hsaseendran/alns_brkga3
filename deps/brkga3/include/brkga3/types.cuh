#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace brkga3 {

// ============================================================================
// Core value types
// ============================================================================

using Gene      = float;           // Random key in (0, 1]
using Fitness   = float;           // Objective value (minimize by default)
using GeneIndex = std::uint32_t;   // Permutation index / chromosome index

// ============================================================================
// Bias function for multi-parent crossover
// ============================================================================

enum class BiasType : std::uint8_t {
    CONSTANT,      // All parents equally likely
    LINEAR,        // Probability decreases linearly with rank
    QUADRATIC,     // Probability decreases quadratically
    CUBIC,         // Probability decreases cubically
    EXPONENTIAL,   // Probability decreases exponentially
    LOGARITHMIC    // Probability decreases logarithmically
};

// ============================================================================
// Decode mode: determines what the decoder receives
// ============================================================================

enum class DecodeMode : std::uint8_t {
    CHROMOSOME,    // Decoder receives raw Gene values in SoA layout
                   //   d_genes_soa[gene_j * pop_size + chrom_i]
    PERMUTATION    // Framework sorts genes; decoder receives GeneIndex permutation
                   //   d_permutations[chrom_i * chrom_len + rank_k]
};

// ============================================================================
// Migration topology between GPUs
// ============================================================================

enum class MigrationTopology : std::uint8_t {
    RING,          // GPU i sends to GPU (i+1) % N
    ALL_TO_ALL     // Every GPU sends to every other GPU
};

// ============================================================================
// Optimization sense
// ============================================================================

enum class OptimizationSense : std::uint8_t {
    MINIMIZE,      // Lower fitness is better (default)
    MAXIMIZE       // Higher fitness is better
};

} // namespace brkga3
