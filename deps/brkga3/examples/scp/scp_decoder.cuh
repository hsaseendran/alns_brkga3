#pragma once

#include <brkga3/decoder.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <vector>
#include <cstdint>

namespace scp {

// ============================================================================
// SCP GPU Decoder — thread-per-chromosome, CHROMOSOME mode
//
// Uses SoA layout: d_genes_soa[set_j * pop_size + chrom_i]
// Adjacent threads (adjacent chrom_i) access adjacent memory — coalesced!
//
// Decoding:
//   Gene < 0.5 => set is selected; add its cost
//   If any element is uncovered, add large penalty
//
// Coverage tracking uses a per-thread bitmap in local memory instead of
// a global scratch buffer.  This makes the decoder safe for multi-pop-per-GPU
// where multiple populations decode concurrently on different streams sharing
// the same decoder instance.
//
// Set membership stored in CSR (Compressed Sparse Row) format:
//   d_set_covers[d_set_offsets[s] .. d_set_offsets[s+1]) = elements covered by set s
// ============================================================================

// Max elements supported by the bitmap approach (32 * 32 = 1024)
constexpr unsigned MAX_BITMAP_WORDS = 32;

__global__ void scpDecodeKernel(
    const brkga3::Gene* __restrict__ d_genes_soa,     // [n_sets x pop_size] SoA
    const float*        __restrict__ d_costs,          // [n_sets]
    const std::uint32_t* __restrict__ d_set_covers,    // CSR values
    const std::uint32_t* __restrict__ d_set_offsets,   // CSR offsets [n_sets+1]
    brkga3::Fitness*     __restrict__ d_fitness,       // [pop_size]
    std::uint32_t pop_size,
    std::uint32_t n_sets,
    std::uint32_t n_elements)
{
    const unsigned chrom_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chrom_id >= pop_size) return;

    // Per-thread coverage bitmap in local memory (no shared scratch needed)
    const unsigned bitmap_words = (n_elements + 31) / 32;
    std::uint32_t covered_bm[MAX_BITMAP_WORDS];
    for (unsigned w = 0; w < bitmap_words; ++w) {
        covered_bm[w] = 0;
    }

    float cost = 0.0f;

    // Evaluate each set
    for (std::uint32_t s = 0; s < n_sets; ++s) {
        // SoA access: coalesced across threads!
        float gene = d_genes_soa[s * pop_size + chrom_id];

        if (gene < 0.5f) {
            cost += d_costs[s];
            // Mark covered elements in bitmap
            for (std::uint32_t j = d_set_offsets[s]; j < d_set_offsets[s + 1]; ++j) {
                std::uint32_t e = d_set_covers[j];
                covered_bm[e >> 5] |= (1u << (e & 31));
            }
        }
    }

    // Count uncovered elements using popcount
    std::uint32_t covered_count = 0;
    for (unsigned w = 0; w < bitmap_words; ++w) {
        covered_count += __popc(covered_bm[w]);
    }
    cost += (n_elements - covered_count) * 1e6f;

    d_fitness[chrom_id] = cost;
}

class ScpDecoder : public brkga3::GpuDecoder {
public:
    ScpDecoder(const std::vector<float>& costs,
               const std::vector<std::uint32_t>& set_covers,
               const std::vector<std::uint32_t>& set_offsets,
               std::uint32_t n_elements)
        : host_costs_(costs)
        , host_set_covers_(set_covers)
        , host_set_offsets_(set_offsets)
        , n_elements_(n_elements)
        , n_sets_(static_cast<std::uint32_t>(costs.size()))
    {}

    void initialize(int gpu_id, std::uint32_t chromosome_length,
                    std::uint32_t population_size) override {
        gpu_id_ = gpu_id;
        brkga3::setDevice(gpu_id);

        // Upload costs
        BRKGA_CUDA_CHECK(cudaMalloc(&d_costs_, n_sets_ * sizeof(float)));
        BRKGA_CUDA_CHECK(cudaMemcpy(d_costs_, host_costs_.data(),
                                     n_sets_ * sizeof(float),
                                     cudaMemcpyHostToDevice));

        // Upload CSR structure
        BRKGA_CUDA_CHECK(cudaMalloc(&d_set_covers_,
                                     host_set_covers_.size() * sizeof(std::uint32_t)));
        BRKGA_CUDA_CHECK(cudaMemcpy(d_set_covers_, host_set_covers_.data(),
                                     host_set_covers_.size() * sizeof(std::uint32_t),
                                     cudaMemcpyHostToDevice));

        BRKGA_CUDA_CHECK(cudaMalloc(&d_set_offsets_,
                                     host_set_offsets_.size() * sizeof(std::uint32_t)));
        BRKGA_CUDA_CHECK(cudaMemcpy(d_set_offsets_, host_set_offsets_.data(),
                                     host_set_offsets_.size() * sizeof(std::uint32_t),
                                     cudaMemcpyHostToDevice));
    }

    void decode(cudaStream_t stream,
                std::uint32_t population_size,
                std::uint32_t chromosome_length,
                const brkga3::Gene*      d_genes_soa,
                const brkga3::GeneIndex* /*d_permutations*/,
                brkga3::Fitness*         d_fitness) override {
        constexpr unsigned BLOCK = 256;
        unsigned grid = (population_size + BLOCK - 1) / BLOCK;

        scpDecodeKernel<<<grid, BLOCK, 0, stream>>>(
            d_genes_soa, d_costs_, d_set_covers_, d_set_offsets_,
            d_fitness,
            population_size, n_sets_, n_elements_);
        BRKGA_CUDA_CHECK_LAST();
    }

    void finalize() override {
        brkga3::setDevice(gpu_id_);
        if (d_costs_)       { cudaFree(d_costs_);       d_costs_ = nullptr; }
        if (d_set_covers_)  { cudaFree(d_set_covers_);  d_set_covers_ = nullptr; }
        if (d_set_offsets_) { cudaFree(d_set_offsets_); d_set_offsets_ = nullptr; }
    }

private:
    const std::vector<float>& host_costs_;
    const std::vector<std::uint32_t>& host_set_covers_;
    const std::vector<std::uint32_t>& host_set_offsets_;
    std::uint32_t n_elements_ = 0;
    std::uint32_t n_sets_ = 0;
    int gpu_id_ = -1;

    float*         d_costs_       = nullptr;
    std::uint32_t* d_set_covers_  = nullptr;
    std::uint32_t* d_set_offsets_ = nullptr;
};

} // namespace scp
