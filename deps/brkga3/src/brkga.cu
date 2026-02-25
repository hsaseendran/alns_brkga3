#include <brkga3/brkga.cuh>
#include <brkga3/utils/log.hpp>

#include <stdexcept>

namespace brkga3 {

Brkga::Brkga(const BrkgaConfig& cfg, DecoderFactory decoder_factory)
    : cfg_(cfg)
{
    log::info("Creating BRKGA 3.0 instance: %u GPU(s), %u pop(s)/GPU, pop_size=%u, chrom_len=%u",
              cfg_.num_gpus, cfg_.num_populations, cfg_.population_size, cfg_.chromosome_length);

    island_mgr_ = std::make_unique<IslandManager>(cfg_, std::move(decoder_factory));

    log::info("BRKGA 3.0 ready (generation 0, initial decode+sort complete)");
}

Brkga::~Brkga() = default;

void Brkga::injectChromosome(const std::vector<Gene>& genes) {
    if (genes.size() != cfg_.chromosome_length) {
        throw std::invalid_argument(
            "injectChromosome: genes.size()=" + std::to_string(genes.size()) +
            " != chromosome_length=" + std::to_string(cfg_.chromosome_length));
    }
    island_mgr_->injectChromosome(genes);
    log::info("Warm-start chromosome injected into all %u population(s)",
              cfg_.totalPopulations());
}

void Brkga::evolve() {
    ++gen_;

    // Apply previously-sent migration if it's time
    if (migration_pending_ && gen_ >= migration_apply_gen_) {
        island_mgr_->migrateApply();
        // Re-decode and re-sort so immigrants get proper fitness rankings
        island_mgr_->redecodeAllPops();
        migration_pending_ = false;
    }

    // Run the full pipeline on all GPUs
    island_mgr_->launchGeneration();

    // Check if we should send migration this generation
    if (cfg_.totalPopulations() > 1 && gen_ % cfg_.migration_interval == 0) {
        // Must sync all populations so fitness sort is complete before packing elites
        island_mgr_->syncAllPops();
        island_mgr_->migrateSend();

        migration_pending_ = true;
        migration_apply_gen_ = gen_ + cfg_.migration_delay;

        log::debug("Gen %u: migration sent, will apply at gen %u",
                   gen_, migration_apply_gen_);
    }
}

void Brkga::evolve(std::uint32_t num_generations) {
    for (std::uint32_t g = 0; g < num_generations; ++g) {
        evolve();
    }
}

Fitness Brkga::getBestFitness() {
    return island_mgr_->getBestFitness();
}

std::vector<Gene> Brkga::getBestChromosome() {
    return island_mgr_->getBestChromosome();
}

std::vector<GeneIndex> Brkga::getBestPermutation() {
    return island_mgr_->getBestPermutation();
}

} // namespace brkga3
