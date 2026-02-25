#include <brkga3/config.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>

namespace brkga3 {

namespace {

// Compute cumulative bias distribution for parent selection.
// bias[i] = unnormalized probability of selecting parent i (rank 0 = best).
// Returns CDF: bias_cdf[i] = sum(bias[0..i]) / sum(bias[0..n-1]).
std::vector<float> computeBiasCdf(BiasType type, std::uint32_t num_parents) {
    std::vector<float> weights(num_parents);

    for (std::uint32_t i = 0; i < num_parents; ++i) {
        float rank = static_cast<float>(i + 1);  // 1-indexed rank
        switch (type) {
            case BiasType::CONSTANT:
                weights[i] = 1.0f;
                break;
            case BiasType::LINEAR:
                weights[i] = 1.0f / rank;
                break;
            case BiasType::QUADRATIC:
                weights[i] = 1.0f / (rank * rank);
                break;
            case BiasType::CUBIC:
                weights[i] = 1.0f / (rank * rank * rank);
                break;
            case BiasType::EXPONENTIAL:
                weights[i] = std::exp(-rank);
                break;
            case BiasType::LOGARITHMIC:
                weights[i] = 1.0f / std::log(rank + 1.0f);
                break;
        }
    }

    // Compute cumulative sum
    std::vector<float> cdf(num_parents);
    float total = 0.0f;
    for (std::uint32_t i = 0; i < num_parents; ++i) {
        total += weights[i];
        cdf[i] = total;
    }

    // Normalize to [0, 1]
    for (std::uint32_t i = 0; i < num_parents; ++i) {
        cdf[i] /= total;
    }
    // Ensure last element is exactly 1.0
    cdf[num_parents - 1] = 1.0f;

    return cdf;
}

} // anonymous namespace

BrkgaConfig BrkgaConfig::Builder::build() const {
    BrkgaConfig cfg = cfg_;

    // --- Resolve percentages ---
    if (use_elite_pct_) {
        cfg.num_elites = static_cast<std::uint32_t>(
            std::round(elite_pct_ * cfg.population_size));
    }
    if (use_mutant_pct_) {
        cfg.num_mutants = static_cast<std::uint32_t>(
            std::round(mutant_pct_ * cfg.population_size));
    }

    // --- Validate ---
    if (cfg.chromosome_length == 0)
        throw std::invalid_argument("chromosome_length must be > 0");

    if (cfg.population_size < 3)
        throw std::invalid_argument("population_size must be >= 3");

    if (cfg.num_elites == 0)
        throw std::invalid_argument("num_elites must be > 0");

    if (cfg.num_elites >= cfg.population_size)
        throw std::invalid_argument("num_elites must be < population_size");

    if (cfg.num_elites + cfg.num_mutants >= cfg.population_size)
        throw std::invalid_argument(
            "num_elites + num_mutants must be < population_size");

    if (cfg.num_parents < 2)
        throw std::invalid_argument("num_parents must be >= 2");

    if (cfg.num_elite_parents < 1)
        throw std::invalid_argument("num_elite_parents must be >= 1");

    if (cfg.num_elite_parents >= cfg.num_parents)
        throw std::invalid_argument(
            "num_elite_parents must be < num_parents");

    if (cfg.num_gpus == 0)
        throw std::invalid_argument("num_gpus must be >= 1");

    if (cfg.num_populations == 0)
        throw std::invalid_argument("num_populations must be >= 1");

    if (cfg.totalPopulations() > 1) {
        if (cfg.migration_interval == 0)
            throw std::invalid_argument(
                "migration_interval must be > 0 for multiple populations");

        if (cfg.num_elites_to_migrate == 0)
            throw std::invalid_argument(
                "num_elites_to_migrate must be > 0 for multiple populations");

        if (cfg.num_elites_to_migrate > cfg.num_elites)
            throw std::invalid_argument(
                "num_elites_to_migrate must be <= num_elites");

        if (cfg.migration_delay < 1)
            throw std::invalid_argument("migration_delay must be >= 1");
    }

    if (cfg.threads_per_block == 0 || (cfg.threads_per_block & (cfg.threads_per_block - 1)) != 0)
        throw std::invalid_argument("threads_per_block must be a power of 2");

    if (cfg.threads_per_block > 1024)
        throw std::invalid_argument("threads_per_block must be <= 1024");

    // --- Compute derived fields ---
    cfg.num_crossover = cfg.population_size - cfg.num_elites - cfg.num_mutants;
    cfg.bias_cdf = computeBiasCdf(cfg.bias_type, cfg.num_parents);

    return cfg;
}

} // namespace brkga3
