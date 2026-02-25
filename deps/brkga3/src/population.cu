#include <brkga3/population.cuh>
#include <brkga3/kernels/sort.cuh>
#include <brkga3/kernels/evolve.cuh>
#include <brkga3/utils/cuda_check.cuh>
#include <brkga3/utils/log.hpp>

namespace brkga3 {

namespace {

template <typename T>
T* deviceAlloc(std::size_t count) {
    T* ptr = nullptr;
    BRKGA_CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template <typename T>
void deviceFree(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

} // anonymous namespace

Population Population::create(int gpu_id, int pop_global_idx, const BrkgaConfig& cfg) {
    Population pop;
    pop.gpu_id = gpu_id;
    pop.pop_global_idx = pop_global_idx;
    setDevice(gpu_id);

    const std::size_t total_genes = cfg.totalGenes();
    const std::uint32_t pop_size  = cfg.population_size;
    const std::uint32_t chrom_len = cfg.chromosome_length;

    log::info("GPU %d, Pop %d: Allocating population (pop=%u, chrom_len=%u, %.1f MB)",
              gpu_id, pop_global_idx, pop_size, chrom_len,
              static_cast<float>(total_genes * sizeof(Gene) * 2) / (1024.0f * 1024.0f));

    // --- Evolution storage (AoS) ---
    pop.d_chromosomes     = deviceAlloc<Gene>(total_genes);
    pop.d_chromosomes_tmp = deviceAlloc<Gene>(total_genes);

    // --- Decoder storage ---
    if (cfg.decode_mode == DecodeMode::CHROMOSOME) {
        pop.d_chromosomes_soa = deviceAlloc<Gene>(total_genes);
    } else {
        pop.d_permutations  = deviceAlloc<GeneIndex>(total_genes);
        pop.d_sort_keys_tmp = deviceAlloc<Gene>(total_genes);
    }

    // --- Fitness ---
    pop.d_fitness        = deviceAlloc<Fitness>(pop_size);
    pop.d_fitness_sorted = deviceAlloc<Fitness>(pop_size);
    pop.d_fitness_idx    = deviceAlloc<GeneIndex>(pop_size);

    // --- Parent selection ---
    pop.d_parents    = deviceAlloc<GeneIndex>(cfg.num_crossover * cfg.num_parents);
    pop.d_rng_states = deviceAlloc<curandState_t>(cfg.num_crossover);

    // --- Bias CDF ---
    pop.d_bias_cdf = deviceAlloc<float>(cfg.num_parents);

    // --- Sort input arrays ---
    pop.d_iota_pop = deviceAlloc<GeneIndex>(pop_size);
    if (cfg.decode_mode == DecodeMode::PERMUTATION) {
        pop.d_iota_genes = deviceAlloc<GeneIndex>(total_genes);
    }

    // --- CUB temp storage ---
    pop.cub_sort_fitness_tmp_bytes = querySortFitnessTmpBytes(pop_size);
    BRKGA_CUDA_CHECK(cudaMalloc(&pop.d_cub_sort_fitness_tmp,
                                pop.cub_sort_fitness_tmp_bytes));

    if (cfg.decode_mode == DecodeMode::PERMUTATION) {
        pop.cub_seg_sort_tmp_bytes = querySortGenesSegmentedTmpBytes(pop_size, chrom_len);
        BRKGA_CUDA_CHECK(cudaMalloc(&pop.d_cub_seg_sort_tmp,
                                    pop.cub_seg_sort_tmp_bytes));
    }

    // --- Streams ---
    BRKGA_CUDA_CHECK(cudaStreamCreateWithFlags(&pop.compute_stream,
                                                cudaStreamNonBlocking));
    BRKGA_CUDA_CHECK(cudaStreamCreateWithFlags(&pop.comm_stream,
                                                cudaStreamNonBlocking));

    // --- Events ---
    BRKGA_CUDA_CHECK(cudaEventCreateWithFlags(&pop.generation_done,
                                               cudaEventDisableTiming));
    BRKGA_CUDA_CHECK(cudaEventCreateWithFlags(&pop.migration_sent,
                                               cudaEventDisableTiming));

    // --- Migration staging ---
    if (cfg.totalPopulations() > 1) {
        std::size_t send_size = cfg.num_elites_to_migrate * chrom_len;
        // Recv buffer holds immigrants from ALL other populations (all-pairs exchange)
        std::size_t recv_size = (cfg.totalPopulations() - 1) * cfg.num_elites_to_migrate * chrom_len;
        pop.d_migration_send_buf = deviceAlloc<Gene>(send_size);
        pop.d_migration_recv_buf = deviceAlloc<Gene>(recv_size);
    }

    // --- Bulk RNG generator ---
    BRKGA_CURAND_CHECK(curandCreateGenerator(&pop.rng_generator,
                                              CURAND_RNG_PSEUDO_DEFAULT));
    BRKGA_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(
        pop.rng_generator, cfg.seed + pop_global_idx));
    BRKGA_CURAND_CHECK(curandSetStream(pop.rng_generator, pop.compute_stream));

    return pop;
}

void Population::initialize(const BrkgaConfig& cfg) {
    setDevice(gpu_id);

    // Upload bias CDF (one-time host->device transfer)
    BRKGA_CUDA_CHECK(cudaMemcpyAsync(
        d_bias_cdf, cfg.bias_cdf.data(),
        cfg.num_parents * sizeof(float),
        cudaMemcpyHostToDevice, compute_stream));

    // Initialize iota arrays
    fillIota(compute_stream, d_iota_pop, cfg.population_size);
    if (cfg.decode_mode == DecodeMode::PERMUTATION) {
        fillIotaMod(compute_stream, d_iota_genes,
                    cfg.totalGenes(), cfg.chromosome_length);
    }

    // Initialize RNG states for parent selection
    // pop_global_idx * num_crossover ensures non-overlapping subsequences across populations
    initRngStates(compute_stream, d_rng_states, cfg.seed,
                  cfg.num_crossover, pop_global_idx * cfg.num_crossover);

    // Generate initial random population
    BRKGA_CURAND_CHECK(curandGenerateUniform(
        rng_generator, d_chromosomes, cfg.totalGenes()));

    log::info("GPU %d, Pop %d: Population initialized with random chromosomes",
              gpu_id, pop_global_idx);
}

void Population::destroy(Population& pop) {
    if (pop.gpu_id < 0) return;  // not initialized
    setDevice(pop.gpu_id);

    // Destroy cuRAND generator
    if (pop.rng_generator) {
        curandDestroyGenerator(pop.rng_generator);
        pop.rng_generator = nullptr;
    }

    // Destroy events
    if (pop.generation_done) { cudaEventDestroy(pop.generation_done); pop.generation_done = nullptr; }
    if (pop.migration_sent)  { cudaEventDestroy(pop.migration_sent);  pop.migration_sent = nullptr; }

    // Destroy streams
    if (pop.compute_stream) { cudaStreamDestroy(pop.compute_stream); pop.compute_stream = nullptr; }
    if (pop.comm_stream)    { cudaStreamDestroy(pop.comm_stream);    pop.comm_stream = nullptr; }

    // Free device memory
    deviceFree(pop.d_chromosomes);
    deviceFree(pop.d_chromosomes_tmp);
    deviceFree(pop.d_chromosomes_soa);
    deviceFree(pop.d_permutations);
    deviceFree(pop.d_sort_keys_tmp);
    deviceFree(pop.d_fitness);
    deviceFree(pop.d_fitness_sorted);
    deviceFree(pop.d_fitness_idx);
    deviceFree(pop.d_parents);
    deviceFree(pop.d_rng_states);
    deviceFree(pop.d_bias_cdf);
    deviceFree(pop.d_iota_pop);
    deviceFree(pop.d_iota_genes);
    deviceFree(pop.d_migration_send_buf);
    deviceFree(pop.d_migration_recv_buf);

    if (pop.d_cub_sort_fitness_tmp) {
        cudaFree(pop.d_cub_sort_fitness_tmp);
        pop.d_cub_sort_fitness_tmp = nullptr;
    }
    if (pop.d_cub_seg_sort_tmp) {
        cudaFree(pop.d_cub_seg_sort_tmp);
        pop.d_cub_seg_sort_tmp = nullptr;
    }

    pop.gpu_id = -1;
    log::debug("Population destroyed");
}

} // namespace brkga3
