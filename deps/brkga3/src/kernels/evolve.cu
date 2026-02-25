#include <brkga3/kernels/evolve.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <curand_kernel.h>

namespace brkga3 {

// ============================================================================
// RNG state initialization
// ============================================================================

__global__ void initRngStatesKernel(
    curandState_t* d_states,
    std::uint64_t  seed,
    std::uint32_t  n,
    std::uint32_t  offset)
{
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Each state gets a unique subsequence for reproducibility.
        // offset ensures different GPUs get non-overlapping subsequences.
        curand_init(seed, static_cast<unsigned long long>(offset + idx),
                    0, &d_states[idx]);
    }
}

void initRngStates(
    cudaStream_t   stream,
    curandState_t* d_rng_states,
    std::uint64_t  seed,
    std::uint32_t  num_crossover,
    std::uint32_t  gpu_offset)
{
    constexpr std::uint32_t BLOCK = 256;
    std::uint32_t grid = (num_crossover + BLOCK - 1) / BLOCK;
    initRngStatesKernel<<<grid, BLOCK, 0, stream>>>(
        d_rng_states, seed, num_crossover, gpu_offset);
    BRKGA_CUDA_CHECK_LAST();
}

// ============================================================================
// Parent selection kernel
//
// One thread per crossover chromosome.
// Samples num_elite_parents from [0, num_elites) and
// (num_parents - num_elite_parents) from [num_elites, pop_size).
// Parents are stored as fitness-rank indices (not chromosome indices).
// ============================================================================

__global__ void selectParentsKernel(
    GeneIndex*     d_parents,
    curandState_t* d_rng_states,
    std::uint32_t  num_crossover,
    std::uint32_t  num_parents,
    std::uint32_t  num_elite_parents,
    std::uint32_t  num_elites,
    std::uint32_t  pop_size)
{
    std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_crossover) return;

    curandState_t local_state = d_rng_states[tid];
    GeneIndex* my_parents = d_parents + tid * num_parents;

    // Sample elite parents from [0, num_elites)
    for (std::uint32_t p = 0; p < num_elite_parents; ++p) {
        std::uint32_t r;
        bool unique;
        do {
            r = curand(&local_state) % num_elites;
            unique = true;
            for (std::uint32_t q = 0; q < p; ++q) {
                if (my_parents[q] == r) { unique = false; break; }
            }
        } while (!unique);
        my_parents[p] = r;
    }

    // Sample non-elite parents from [num_elites, pop_size)
    std::uint32_t non_elite_range = pop_size - num_elites;
    for (std::uint32_t p = num_elite_parents; p < num_parents; ++p) {
        std::uint32_t r;
        bool unique;
        do {
            r = num_elites + (curand(&local_state) % non_elite_range);
            unique = true;
            for (std::uint32_t q = num_elite_parents; q < p; ++q) {
                if (my_parents[q] == r) { unique = false; break; }
            }
        } while (!unique);
        my_parents[p] = r;
    }

    // Sort all parents by rank (ascending) so better-ranked parents
    // get higher bias weight — matches BRKGA 2.0 behavior.
    for (std::uint32_t i = 1; i < num_parents; ++i) {
        GeneIndex key = my_parents[i];
        std::uint32_t j = i;
        while (j > 0 && my_parents[j - 1] > key) {
            my_parents[j] = my_parents[j - 1];
            --j;
        }
        my_parents[j] = key;
    }

    d_rng_states[tid] = local_state;
}

void selectParents(
    cudaStream_t   stream,
    GeneIndex*     d_parents,
    curandState_t* d_rng_states,
    std::uint32_t  num_crossover,
    std::uint32_t  num_parents,
    std::uint32_t  num_elite_parents,
    std::uint32_t  num_elites,
    std::uint32_t  pop_size,
    std::uint32_t  threads_per_block)
{
    std::uint32_t grid = (num_crossover + threads_per_block - 1) / threads_per_block;
    selectParentsKernel<<<grid, threads_per_block, 0, stream>>>(
        d_parents, d_rng_states, num_crossover, num_parents,
        num_elite_parents, num_elites, pop_size);
    BRKGA_CUDA_CHECK_LAST();
}

// ============================================================================
// Elite copy kernel
//
// Copies the top num_elites chromosomes from old generation to new.
// One thread per gene across all elite chromosomes.
// Uses d_fitness_idx to map from rank to actual chromosome position.
// ============================================================================

__global__ void evolveCopyEliteKernel(
    Gene*            d_new_pop,
    const Gene*      d_old_pop,
    const GeneIndex* d_fitness_idx,
    std::uint32_t    num_elites,
    std::uint32_t    chrom_len)
{
    std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t total = num_elites * chrom_len;
    if (tid >= total) return;

    std::uint32_t elite_rank = tid / chrom_len;
    std::uint32_t gene_pos   = tid % chrom_len;

    // Source: the chromosome at fitness rank elite_rank
    std::uint32_t src_chrom = d_fitness_idx[elite_rank];

    d_new_pop[elite_rank * chrom_len + gene_pos] =
        d_old_pop[src_chrom * chrom_len + gene_pos];
}

void evolveCopyElite(
    cudaStream_t     stream,
    Gene*            d_new_pop,
    const Gene*      d_old_pop,
    const GeneIndex* d_fitness_idx,
    std::uint32_t    num_elites,
    std::uint32_t    chrom_len,
    std::uint32_t    threads_per_block)
{
    std::uint32_t total = num_elites * chrom_len;
    std::uint32_t grid  = (total + threads_per_block - 1) / threads_per_block;
    evolveCopyEliteKernel<<<grid, threads_per_block, 0, stream>>>(
        d_new_pop, d_old_pop, d_fitness_idx, num_elites, chrom_len);
    BRKGA_CUDA_CHECK_LAST();
}

// ============================================================================
// Mate (crossover) kernel
//
// Generates crossover offspring using biased multi-parent selection.
//
// Layout: d_new_pop is the new generation array:
//   [0, num_elites)                      : already filled by evolveCopyElite
//   [num_elites, num_elites+num_crossover): filled here by crossover
//   [num_elites+num_crossover, pop_size) : mutants (random, from curandGenerateUniform)
//
// One thread per gene of all crossover chromosomes.
// The random toss value is read from d_new_pop (pre-filled by curandGenerateUniform).
// Bias CDF is loaded into shared memory for fast lookup.
// ============================================================================

__global__ void evolveMateKernel(
    Gene*            d_new_pop,
    const Gene*      d_old_pop,
    const GeneIndex* d_fitness_idx,
    const GeneIndex* d_parents,
    const float*     d_bias_cdf,
    std::uint32_t    num_elites,
    std::uint32_t    num_crossover,
    std::uint32_t    num_parents,
    std::uint32_t    chrom_len)
{
    // Load bias CDF into shared memory (one entry per parent)
    extern __shared__ float s_bias_cdf[];
    if (threadIdx.x < num_parents) {
        s_bias_cdf[threadIdx.x] = d_bias_cdf[threadIdx.x];
    }
    __syncthreads();

    std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t total = num_crossover * chrom_len;
    if (tid >= total) return;

    std::uint32_t offspring_idx = tid / chrom_len;  // which crossover chromosome
    std::uint32_t gene_pos      = tid % chrom_len;

    // Absolute position in the new population array
    std::uint32_t new_chrom_idx = num_elites + offspring_idx;

    // Read the random toss value (pre-filled by curandGenerateUniform)
    float toss = d_new_pop[new_chrom_idx * chrom_len + gene_pos];

    // Select parent via bias CDF (roulette wheel)
    std::uint32_t selected_parent = 0;
    for (std::uint32_t p = 0; p < num_parents; ++p) {
        if (toss <= s_bias_cdf[p]) {
            selected_parent = p;
            break;
        }
    }

    // Look up the parent's fitness rank, then the actual chromosome index
    GeneIndex parent_rank = d_parents[offspring_idx * num_parents + selected_parent];
    GeneIndex parent_chrom = d_fitness_idx[parent_rank];

    // Write the selected parent's gene
    d_new_pop[new_chrom_idx * chrom_len + gene_pos] =
        d_old_pop[parent_chrom * chrom_len + gene_pos];
}

void evolveMate(
    cudaStream_t     stream,
    Gene*            d_new_pop,
    const Gene*      d_old_pop,
    const GeneIndex* d_fitness_idx,
    const GeneIndex* d_parents,
    const float*     d_bias_cdf,
    std::uint32_t    num_elites,
    std::uint32_t    num_mutants,
    std::uint32_t    num_parents,
    std::uint32_t    pop_size,
    std::uint32_t    chrom_len,
    std::uint32_t    threads_per_block)
{
    std::uint32_t num_crossover = pop_size - num_elites - num_mutants;
    std::uint32_t total = num_crossover * chrom_len;
    std::uint32_t grid  = (total + threads_per_block - 1) / threads_per_block;

    // Shared memory for bias CDF
    std::size_t smem = num_parents * sizeof(float);

    evolveMateKernel<<<grid, threads_per_block, smem, stream>>>(
        d_new_pop, d_old_pop, d_fitness_idx, d_parents, d_bias_cdf,
        num_elites, num_crossover, num_parents, chrom_len);
    BRKGA_CUDA_CHECK_LAST();
}

// ============================================================================
// Pack elites for migration
// ============================================================================

__global__ void packElitesKernel(
    Gene*            d_send_buf,
    const Gene*      d_chromosomes,
    const GeneIndex* d_fitness_idx,
    std::uint32_t    num_migrate,
    std::uint32_t    chrom_len)
{
    std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t total = num_migrate * chrom_len;
    if (tid >= total) return;

    std::uint32_t elite_rank = tid / chrom_len;
    std::uint32_t gene_pos   = tid % chrom_len;

    GeneIndex src_chrom = d_fitness_idx[elite_rank];
    d_send_buf[elite_rank * chrom_len + gene_pos] =
        d_chromosomes[src_chrom * chrom_len + gene_pos];
}

void packElites(
    cudaStream_t     stream,
    Gene*            d_send_buf,
    const Gene*      d_chromosomes,
    const GeneIndex* d_fitness_idx,
    std::uint32_t    num_migrate,
    std::uint32_t    chrom_len,
    std::uint32_t    threads_per_block)
{
    std::uint32_t total = num_migrate * chrom_len;
    std::uint32_t grid  = (total + threads_per_block - 1) / threads_per_block;
    packElitesKernel<<<grid, threads_per_block, 0, stream>>>(
        d_send_buf, d_chromosomes, d_fitness_idx, num_migrate, chrom_len);
    BRKGA_CUDA_CHECK_LAST();
}

// ============================================================================
// Unpack immigrants into worst chromosome slots
// ============================================================================

__global__ void unpackImmigrantsKernel(
    Gene*            d_chromosomes,
    const Gene*      d_recv_buf,
    const GeneIndex* d_fitness_idx,
    std::uint32_t    num_migrate,
    std::uint32_t    pop_size,
    std::uint32_t    chrom_len)
{
    std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t total = num_migrate * chrom_len;
    if (tid >= total) return;

    std::uint32_t migrant_idx = tid / chrom_len;
    std::uint32_t gene_pos    = tid % chrom_len;

    // Overwrite the worst chromosomes (at the end of the sorted ranking)
    GeneIndex dest_chrom = d_fitness_idx[pop_size - 1 - migrant_idx];
    d_chromosomes[dest_chrom * chrom_len + gene_pos] =
        d_recv_buf[migrant_idx * chrom_len + gene_pos];
}

void unpackImmigrants(
    cudaStream_t     stream,
    Gene*            d_chromosomes,
    const Gene*      d_recv_buf,
    const GeneIndex* d_fitness_idx,
    std::uint32_t    num_migrate,
    std::uint32_t    pop_size,
    std::uint32_t    chrom_len,
    std::uint32_t    threads_per_block)
{
    std::uint32_t total = num_migrate * chrom_len;
    std::uint32_t grid  = (total + threads_per_block - 1) / threads_per_block;
    unpackImmigrantsKernel<<<grid, threads_per_block, 0, stream>>>(
        d_chromosomes, d_recv_buf, d_fitness_idx, num_migrate, pop_size, chrom_len);
    BRKGA_CUDA_CHECK_LAST();
}

} // namespace brkga3
