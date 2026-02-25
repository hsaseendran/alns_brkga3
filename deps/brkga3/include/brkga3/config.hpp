#pragma once

#include "types.cuh"
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace brkga3 {

// ============================================================================
// BrkgaConfig — immutable after construction via Builder::build()
// ============================================================================

struct BrkgaConfig {
    // --- Problem ---
    std::uint32_t chromosome_length = 0;
    DecodeMode    decode_mode       = DecodeMode::PERMUTATION;
    OptimizationSense sense         = OptimizationSense::MINIMIZE;

    // --- Population (per GPU) ---
    std::uint32_t population_size   = 256;
    std::uint32_t num_elites        = 25;
    std::uint32_t num_mutants       = 25;

    // --- Crossover ---
    std::uint32_t num_parents       = 2;
    std::uint32_t num_elite_parents = 1;
    BiasType      bias_type         = BiasType::LINEAR;

    // --- Multi-GPU island model ---
    std::uint32_t     num_gpus              = 1;
    std::uint32_t     num_populations       = 1;   // populations per GPU
    std::uint32_t     migration_interval    = 50;
    std::uint32_t     num_elites_to_migrate = 2;
    std::uint32_t     migration_delay       = 2;
    MigrationTopology migration_topology    = MigrationTopology::RING;

    // --- Execution ---
    std::uint32_t threads_per_block = 256;
    std::uint64_t seed              = 42;

    // --- Adaptive parameters ---
    bool  adaptive_enabled  = false;
    float diversity_low     = 0.1f;
    float diversity_high    = 0.5f;

    // --- Derived fields (computed by Builder::build()) ---
    std::uint32_t       num_crossover = 0;  // pop_size - num_elites - num_mutants
    std::vector<float>  bias_cdf;           // cumulative bias distribution [num_parents]

    // Convenience
    std::uint32_t totalGenes() const { return population_size * chromosome_length; }
    std::uint32_t totalPopulations() const { return num_gpus * num_populations; }

    // --- Builder ---
    class Builder;
};

// ============================================================================
// Builder — validates constraints and computes derived fields
// ============================================================================

class BrkgaConfig::Builder {
public:
    Builder& chromosomeLength(std::uint32_t n) {
        cfg_.chromosome_length = n;
        return *this;
    }

    Builder& decodeMode(DecodeMode m) {
        cfg_.decode_mode = m;
        return *this;
    }

    Builder& optimizationSense(OptimizationSense s) {
        cfg_.sense = s;
        return *this;
    }

    Builder& populationSize(std::uint32_t n) {
        cfg_.population_size = n;
        return *this;
    }

    Builder& numElites(std::uint32_t n) {
        cfg_.num_elites = n;
        return *this;
    }

    Builder& elitePercentage(float p) {
        elite_pct_ = p;
        use_elite_pct_ = true;
        return *this;
    }

    Builder& numMutants(std::uint32_t n) {
        cfg_.num_mutants = n;
        return *this;
    }

    Builder& mutantPercentage(float p) {
        mutant_pct_ = p;
        use_mutant_pct_ = true;
        return *this;
    }

    Builder& numParents(std::uint32_t n, BiasType bt = BiasType::LINEAR,
                        std::uint32_t numEliteParents = 1) {
        cfg_.num_parents = n;
        cfg_.bias_type = bt;
        cfg_.num_elite_parents = numEliteParents;
        return *this;
    }

    Builder& numGpus(std::uint32_t n) {
        cfg_.num_gpus = n;
        return *this;
    }

    Builder& numPopulations(std::uint32_t n) {
        cfg_.num_populations = n;
        return *this;
    }

    Builder& migrationInterval(std::uint32_t k) {
        cfg_.migration_interval = k;
        return *this;
    }

    Builder& numElitesToMigrate(std::uint32_t k) {
        cfg_.num_elites_to_migrate = k;
        return *this;
    }

    Builder& migrationDelay(std::uint32_t d) {
        cfg_.migration_delay = d;
        return *this;
    }

    Builder& migrationTopology(MigrationTopology t) {
        cfg_.migration_topology = t;
        return *this;
    }

    Builder& threadsPerBlock(std::uint32_t t) {
        cfg_.threads_per_block = t;
        return *this;
    }

    Builder& seed(std::uint64_t s) {
        cfg_.seed = s;
        return *this;
    }

    Builder& adaptive(bool enabled, float divLow = 0.1f, float divHigh = 0.5f) {
        cfg_.adaptive_enabled = enabled;
        cfg_.diversity_low = divLow;
        cfg_.diversity_high = divHigh;
        return *this;
    }

    // Validates all parameters and computes derived fields.
    BrkgaConfig build() const;

private:
    BrkgaConfig cfg_;
    float elite_pct_  = 0.0f;
    float mutant_pct_ = 0.0f;
    bool  use_elite_pct_  = false;
    bool  use_mutant_pct_ = false;
};

} // namespace brkga3
