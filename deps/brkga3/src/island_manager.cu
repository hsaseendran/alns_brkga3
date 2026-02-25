#include <brkga3/island_manager.cuh>
#include <brkga3/kernels/evolve.cuh>
#include <brkga3/kernels/transpose.cuh>
#include <brkga3/kernels/sort.cuh>
#include <brkga3/utils/cuda_check.cuh>
#include <brkga3/utils/log.hpp>

#include <algorithm>
#include <limits>
#include <utility>

namespace brkga3 {

// ============================================================================
// Construction / Destruction
// ============================================================================

IslandManager::IslandManager(const BrkgaConfig& cfg, DecoderFactory factory)
    : cfg_(cfg)
{
    const int num_gpus = static_cast<int>(cfg_.num_gpus);
    const int num_pops_per_gpu = static_cast<int>(cfg_.num_populations);
    const int total_pops = static_cast<int>(cfg_.totalPopulations());
    const int available = getDeviceCount();

    if (num_gpus > available) {
        log::error("Requested %d GPUs but only %d available", num_gpus, available);
        throw std::runtime_error("Insufficient GPUs");
    }

    log::info("Initializing %d population(s) across %d GPU(s) (%d per GPU)",
              total_pops, num_gpus, num_pops_per_gpu);

    // Enable P2P access between all GPU pairs (for cross-GPU migration)
    if (num_gpus > 1) {
        for (int i = 0; i < num_gpus; ++i) {
            setDevice(i);
            for (int j = 0; j < num_gpus; ++j) {
                if (i == j) continue;
                int can_access = 0;
                BRKGA_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        cudaGetLastError();  // clear the error
                    } else {
                        BRKGA_CUDA_CHECK(err);
                    }
                    log::debug("GPU %d -> GPU %d: P2P enabled", i, j);
                } else {
                    log::warn("GPU %d -> GPU %d: P2P not available, "
                              "migration will use staged host transfer", i, j);
                }
            }
        }
    }

    // Create all populations and one decoder per GPU
    populations_.reserve(total_pops);
    decoders_.reserve(num_gpus);

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        // Create decoder for this GPU (shared across its populations)
        decoders_.push_back(factory(gpu));

        // Create populations on this GPU
        for (int local = 0; local < num_pops_per_gpu; ++local) {
            int global_idx = gpu * num_pops_per_gpu + local;
            populations_.push_back(Population::create(gpu, global_idx, cfg_));
        }
    }

    // Initialize all populations and decoders
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        setDevice(gpu);

        // Initialize decoder once per GPU
        decoders_[gpu]->initialize(gpu, cfg_.chromosome_length, cfg_.population_size);

        // Initialize each population on this GPU
        for (int local = 0; local < num_pops_per_gpu; ++local) {
            int p = gpu * num_pops_per_gpu + local;
            populations_[p].initialize(cfg_);
        }
    }

    // Run initial decode + sort on all populations
    for (int p = 0; p < total_pops; ++p) {
        prepareDecodeOnPop(p);
        decodeOnPop(p);
        sortFitnessOnPop(p);
    }

    // Sync all populations to ensure initialization is complete
    syncAllPops();

    log::info("All %d population(s) initialized across %d GPU(s)", total_pops, num_gpus);
}

IslandManager::~IslandManager() {
    const int num_gpus = static_cast<int>(cfg_.num_gpus);
    const int total_pops = static_cast<int>(cfg_.totalPopulations());

    // Finalize decoders (one per GPU)
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        setDevice(gpu);
        if (decoders_[gpu]) {
            decoders_[gpu]->finalize();
        }
    }

    // Destroy all populations
    for (int p = 0; p < total_pops; ++p) {
        Population::destroy(populations_[p]);
    }
}

// ============================================================================
// Per-generation pipeline
// ============================================================================

void IslandManager::launchGeneration() {
    const int total_pops = static_cast<int>(cfg_.totalPopulations());

    // Launch the full pipeline on each population (non-blocking to host)
    for (int p = 0; p < total_pops; ++p) {
        evolveOnPop(p);
        prepareDecodeOnPop(p);
        decodeOnPop(p);
        sortFitnessOnPop(p);

        // Record that this generation is complete on this population
        int gpu = gpuOf(p);
        setDevice(gpu);
        BRKGA_CUDA_CHECK(cudaEventRecord(populations_[p].generation_done,
                                          populations_[p].compute_stream));
    }
}

void IslandManager::syncAllPops() {
    const int total_pops = static_cast<int>(cfg_.totalPopulations());
    for (int p = 0; p < total_pops; ++p) {
        setDevice(gpuOf(p));
        BRKGA_CUDA_CHECK(cudaStreamSynchronize(populations_[p].compute_stream));
    }
}

void IslandManager::redecodeAllPops() {
    const int total_pops = static_cast<int>(cfg_.totalPopulations());
    for (int p = 0; p < total_pops; ++p) {
        prepareDecodeOnPop(p);
        decodeOnPop(p);
        sortFitnessOnPop(p);
    }
}

// ============================================================================
// Evolution step (per population)
// ============================================================================

void IslandManager::evolveOnPop(int pop_idx) {
    int gpu = gpuOf(pop_idx);
    setDevice(gpu);
    Population& pop = populations_[pop_idx];
    cudaStream_t stream = pop.compute_stream;

    // Step 1: Fill d_chromosomes_tmp with random floats
    BRKGA_CURAND_CHECK(curandGenerateUniform(
        pop.rng_generator, pop.d_chromosomes_tmp, cfg_.totalGenes()));

    // Step 2: Select parents for crossover chromosomes
    selectParents(
        stream,
        pop.d_parents,
        pop.d_rng_states,
        cfg_.num_crossover,
        cfg_.num_parents,
        cfg_.num_elite_parents,
        cfg_.num_elites,
        cfg_.population_size,
        cfg_.threads_per_block);

    // Step 3: Copy elite chromosomes from old generation to new
    evolveCopyElite(
        stream,
        pop.d_chromosomes_tmp,
        pop.d_chromosomes,
        pop.d_fitness_idx,
        cfg_.num_elites,
        cfg_.chromosome_length,
        cfg_.threads_per_block);

    // Step 4: Crossover to generate offspring
    evolveMate(
        stream,
        pop.d_chromosomes_tmp,
        pop.d_chromosomes,
        pop.d_fitness_idx,
        pop.d_parents,
        pop.d_bias_cdf,
        cfg_.num_elites,
        cfg_.num_mutants,
        cfg_.num_parents,
        cfg_.population_size,
        cfg_.chromosome_length,
        cfg_.threads_per_block);

    // Step 5: Swap generations
    std::swap(pop.d_chromosomes, pop.d_chromosomes_tmp);
}

// ============================================================================
// Prepare decode (transpose or segmented sort)
// ============================================================================

void IslandManager::prepareDecodeOnPop(int pop_idx) {
    int gpu = gpuOf(pop_idx);
    setDevice(gpu);
    Population& pop = populations_[pop_idx];
    cudaStream_t stream = pop.compute_stream;

    if (cfg_.decode_mode == DecodeMode::CHROMOSOME) {
        transposeAoStoSoA(
            stream,
            pop.d_chromosomes_soa,
            pop.d_chromosomes,
            cfg_.population_size,
            cfg_.chromosome_length);
    } else {
        BRKGA_CUDA_CHECK(cudaMemcpyAsync(
            pop.d_sort_keys_tmp, pop.d_chromosomes,
            cfg_.totalGenes() * sizeof(Gene),
            cudaMemcpyDeviceToDevice, stream));

        fillIotaMod(stream, pop.d_iota_genes,
                    cfg_.totalGenes(), cfg_.chromosome_length);

        sortGenesSegmented(
            stream,
            pop.d_sort_keys_tmp,
            pop.d_permutations,
            pop.d_sort_keys_tmp,
            pop.d_iota_genes,
            pop.d_cub_seg_sort_tmp,
            pop.cub_seg_sort_tmp_bytes,
            cfg_.population_size,
            cfg_.chromosome_length);
    }
}

// ============================================================================
// Decode (call user's decoder)
// ============================================================================

void IslandManager::decodeOnPop(int pop_idx) {
    int gpu = gpuOf(pop_idx);
    setDevice(gpu);
    Population& pop = populations_[pop_idx];

    // Use the decoder for this GPU (shared across populations on the same GPU)
    decoders_[gpu]->decode(
        pop.compute_stream,
        cfg_.population_size,
        cfg_.chromosome_length,
        cfg_.decode_mode == DecodeMode::CHROMOSOME ? pop.d_chromosomes_soa : nullptr,
        cfg_.decode_mode == DecodeMode::PERMUTATION ? pop.d_permutations : nullptr,
        pop.d_fitness);
}

// ============================================================================
// Sort fitness to produce ranking
// ============================================================================

void IslandManager::sortFitnessOnPop(int pop_idx) {
    int gpu = gpuOf(pop_idx);
    setDevice(gpu);
    Population& pop = populations_[pop_idx];
    cudaStream_t stream = pop.compute_stream;

    fillIota(stream, pop.d_iota_pop, cfg_.population_size);

    sortFitness(
        stream,
        pop.d_fitness_sorted,
        pop.d_fitness_idx,
        pop.d_fitness,
        pop.d_iota_pop,
        pop.d_cub_sort_fitness_tmp,
        pop.cub_sort_fitness_tmp_bytes,
        cfg_.population_size);
}

// ============================================================================
// Migration: all-pairs exchange with delayed application
// Handles both same-GPU (cudaMemcpyAsync DtoD) and cross-GPU (cudaMemcpyPeerAsync)
// ============================================================================

void IslandManager::migrateSend() {
    const int N = static_cast<int>(cfg_.totalPopulations());
    if (N <= 1) return;

    const std::size_t elites_bytes = cfg_.num_elites_to_migrate
                                     * cfg_.chromosome_length * sizeof(Gene);

    for (int i = 0; i < N; ++i) {
        int gpu_i = gpuOf(i);
        setDevice(gpu_i);
        Population& src = populations_[i];

        // Pack elites into send buffer (on compute_stream, after sort)
        packElites(
            src.compute_stream,
            src.d_migration_send_buf,
            src.d_chromosomes,
            src.d_fitness_idx,
            cfg_.num_elites_to_migrate,
            cfg_.chromosome_length,
            cfg_.threads_per_block);

        // Wait for pack to complete before comm_stream reads the buffer
        BRKGA_CUDA_CHECK(cudaEventRecord(src.generation_done,
                                          src.compute_stream));
        BRKGA_CUDA_CHECK(cudaStreamWaitEvent(src.comm_stream,
                                              src.generation_done, 0));

        // Send elites to ALL other populations
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            Population& dst = populations_[j];

            // Unique slot for sender i in destination j's recv_buf
            int slot = (i < j) ? i : i - 1;

            Gene* dst_offset = dst.d_migration_recv_buf
                               + slot * cfg_.num_elites_to_migrate
                                      * cfg_.chromosome_length;

            int gpu_j = gpuOf(j);
            if (gpu_i == gpu_j) {
                // Same GPU: device-to-device copy on same device
                BRKGA_CUDA_CHECK(cudaMemcpyAsync(
                    dst_offset,
                    src.d_migration_send_buf,
                    elites_bytes,
                    cudaMemcpyDeviceToDevice,
                    src.comm_stream));
            } else {
                // Cross-GPU: peer-to-peer transfer
                BRKGA_CUDA_CHECK(cudaMemcpyPeerAsync(
                    dst_offset, gpu_j,
                    src.d_migration_send_buf, gpu_i,
                    elites_bytes, src.comm_stream));
            }
        }

        // Record that all sends from this population are complete
        BRKGA_CUDA_CHECK(cudaEventRecord(src.migration_sent,
                                          src.comm_stream));
    }

    log::debug("Migration sent (all-pairs, %u populations across %u GPUs)",
               cfg_.totalPopulations(), cfg_.num_gpus);
}

void IslandManager::migrateApply() {
    const int N = static_cast<int>(cfg_.totalPopulations());
    if (N <= 1) return;

    const std::uint32_t total_immigrants =
        (N - 1) * cfg_.num_elites_to_migrate;

    for (int i = 0; i < N; ++i) {
        setDevice(gpuOf(i));
        Population& pop = populations_[i];

        // Wait for ALL source populations to finish sending
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            BRKGA_CUDA_CHECK(cudaStreamWaitEvent(pop.compute_stream,
                                                  populations_[j].migration_sent, 0));
        }

        // Unpack all immigrants into worst chromosome slots
        unpackImmigrants(
            pop.compute_stream,
            pop.d_chromosomes,
            pop.d_migration_recv_buf,
            pop.d_fitness_idx,
            total_immigrants,
            cfg_.population_size,
            cfg_.chromosome_length,
            cfg_.threads_per_block);
    }

    log::debug("Migration applied (%u immigrants per population from %d senders)",
               total_immigrants, N - 1);
}

// ============================================================================
// Warm-start injection
// ============================================================================

void IslandManager::injectChromosome(const std::vector<Gene>& genes) {
    const int total_pops = static_cast<int>(cfg_.totalPopulations());
    const std::size_t bytes = cfg_.chromosome_length * sizeof(Gene);

    // Copy the chromosome into slot 0 of every population
    for (int p = 0; p < total_pops; ++p) {
        setDevice(gpuOf(p));
        Population& pop = populations_[p];

        BRKGA_CUDA_CHECK(cudaMemcpyAsync(
            pop.d_chromosomes,  // slot 0
            genes.data(),
            bytes,
            cudaMemcpyHostToDevice,
            pop.compute_stream));
    }

    // Re-decode and re-sort so the injected chromosome gets ranked
    for (int p = 0; p < total_pops; ++p) {
        prepareDecodeOnPop(p);
        decodeOnPop(p);
        sortFitnessOnPop(p);
    }

    syncAllPops();
}

// ============================================================================
// Queries
// ============================================================================

int IslandManager::findBestPop() {
    syncAllPops();

    int best_pop = 0;
    Fitness best_fitness = std::numeric_limits<Fitness>::max();

    const int total_pops = static_cast<int>(cfg_.totalPopulations());
    for (int p = 0; p < total_pops; ++p) {
        setDevice(gpuOf(p));
        Fitness f;
        BRKGA_CUDA_CHECK(cudaMemcpy(
            &f, populations_[p].d_fitness_sorted,
            sizeof(Fitness), cudaMemcpyDeviceToHost));

        if (cfg_.sense == OptimizationSense::MINIMIZE) {
            if (f < best_fitness) { best_fitness = f; best_pop = p; }
        } else {
            if (f > best_fitness) { best_fitness = f; best_pop = p; }
        }
    }

    return best_pop;
}

Fitness IslandManager::getBestFitness() {
    int best_pop = findBestPop();
    setDevice(gpuOf(best_pop));

    Fitness f;
    BRKGA_CUDA_CHECK(cudaMemcpy(
        &f, populations_[best_pop].d_fitness_sorted,
        sizeof(Fitness), cudaMemcpyDeviceToHost));
    return f;
}

std::vector<Gene> IslandManager::getBestChromosome() {
    int best_pop = findBestPop();
    setDevice(gpuOf(best_pop));
    Population& pop = populations_[best_pop];

    GeneIndex best_idx;
    BRKGA_CUDA_CHECK(cudaMemcpy(
        &best_idx, pop.d_fitness_idx,
        sizeof(GeneIndex), cudaMemcpyDeviceToHost));

    std::vector<Gene> chromosome(cfg_.chromosome_length);
    BRKGA_CUDA_CHECK(cudaMemcpy(
        chromosome.data(),
        pop.d_chromosomes + best_idx * cfg_.chromosome_length,
        cfg_.chromosome_length * sizeof(Gene),
        cudaMemcpyDeviceToHost));

    return chromosome;
}

std::vector<GeneIndex> IslandManager::getBestPermutation() {
    int best_pop = findBestPop();
    setDevice(gpuOf(best_pop));
    Population& pop = populations_[best_pop];

    GeneIndex best_idx;
    BRKGA_CUDA_CHECK(cudaMemcpy(
        &best_idx, pop.d_fitness_idx,
        sizeof(GeneIndex), cudaMemcpyDeviceToHost));

    std::vector<GeneIndex> perm(cfg_.chromosome_length);
    BRKGA_CUDA_CHECK(cudaMemcpy(
        perm.data(),
        pop.d_permutations + best_idx * cfg_.chromosome_length,
        cfg_.chromosome_length * sizeof(GeneIndex),
        cudaMemcpyDeviceToHost));

    return perm;
}

} // namespace brkga3
