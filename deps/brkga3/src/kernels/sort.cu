#include <brkga3/kernels/sort.cuh>
#include <brkga3/utils/cuda_check.cuh>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace brkga3 {

// ============================================================================
// Functor to compute segment offsets on-the-fly: offset(i) = i * segment_size
// Avoids storing a device array of offsets.
// ============================================================================

struct MultiplyByN {
    GeneIndex n;
    __host__ __device__ __forceinline__
    GeneIndex operator()(GeneIndex i) const { return i * n; }
};

// ============================================================================
// Sort A: Fitness ranking via CUB DeviceRadixSort
// ============================================================================

void sortFitness(
    cudaStream_t    stream,
    Fitness*        d_keys_out,
    GeneIndex*      d_values_out,
    const Fitness*  d_keys_in,
    const GeneIndex* d_values_in,
    void*           d_tmp,
    std::size_t     tmp_bytes,
    std::uint32_t   pop_size)
{
    BRKGA_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            d_tmp, tmp_bytes,
            d_keys_in, d_keys_out,
            d_values_in, d_values_out,
            static_cast<int>(pop_size),
            0,                              // begin_bit
            sizeof(Fitness) * 8,            // end_bit (all bits of float)
            stream));
}

std::size_t querySortFitnessTmpBytes(std::uint32_t pop_size) {
    std::size_t tmp_bytes = 0;
    BRKGA_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            nullptr, tmp_bytes,
            static_cast<const Fitness*>(nullptr),
            static_cast<Fitness*>(nullptr),
            static_cast<const GeneIndex*>(nullptr),
            static_cast<GeneIndex*>(nullptr),
            static_cast<int>(pop_size),
            0, sizeof(Fitness) * 8,
            static_cast<cudaStream_t>(nullptr)));
    return tmp_bytes;
}

// ============================================================================
// Sort B: Gene permutation via CUB DeviceSegmentedSort
// ============================================================================

void sortGenesSegmented(
    cudaStream_t     stream,
    Gene*            d_keys_out,
    GeneIndex*       d_values_out,
    const Gene*      d_keys_in,
    const GeneIndex* d_values_in,
    void*            d_tmp,
    std::size_t      tmp_bytes,
    std::uint32_t    pop_size,
    std::uint32_t    chrom_len)
{
    // Compute segment offsets on-the-fly using a transform iterator.
    // offset[i] = i * chrom_len, no device memory needed.
    cub::CountingInputIterator<GeneIndex> counting(0);
    MultiplyByN mul{chrom_len};
    auto offsets_begin = cub::TransformInputIterator<GeneIndex, MultiplyByN,
                                                     cub::CountingInputIterator<GeneIndex>>(
        counting, mul);

    int total_items  = static_cast<int>(pop_size) * static_cast<int>(chrom_len);
    int num_segments = static_cast<int>(pop_size);

    BRKGA_CUDA_CHECK(
        cub::DeviceSegmentedSort::SortPairs(
            d_tmp, tmp_bytes,
            d_keys_in, d_keys_out,
            d_values_in, d_values_out,
            total_items,
            num_segments,
            offsets_begin,           // d_begin_offsets
            offsets_begin + 1,       // d_end_offsets
            stream));
}

std::size_t querySortGenesSegmentedTmpBytes(
    std::uint32_t pop_size,
    std::uint32_t chrom_len)
{
    std::size_t tmp_bytes = 0;

    cub::CountingInputIterator<GeneIndex> counting(0);
    MultiplyByN mul{chrom_len};
    auto offsets_begin = cub::TransformInputIterator<GeneIndex, MultiplyByN,
                                                     cub::CountingInputIterator<GeneIndex>>(
        counting, mul);

    int total_items  = static_cast<int>(pop_size) * static_cast<int>(chrom_len);
    int num_segments = static_cast<int>(pop_size);

    BRKGA_CUDA_CHECK(
        cub::DeviceSegmentedSort::SortPairs(
            nullptr, tmp_bytes,
            static_cast<const Gene*>(nullptr),
            static_cast<Gene*>(nullptr),
            static_cast<const GeneIndex*>(nullptr),
            static_cast<GeneIndex*>(nullptr),
            total_items,
            num_segments,
            offsets_begin,
            offsets_begin + 1,
            static_cast<cudaStream_t>(nullptr)));

    return tmp_bytes;
}

// ============================================================================
// Iota kernels
// ============================================================================

__global__ void iotaKernel(GeneIndex* d_out, std::uint32_t n) {
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = idx;
    }
}

__global__ void iotaModKernel(GeneIndex* d_out, std::uint32_t n,
                              std::uint32_t mod) {
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = idx % mod;
    }
}

void fillIota(cudaStream_t stream, GeneIndex* d_out, std::uint32_t n) {
    constexpr std::uint32_t BLOCK = 256;
    std::uint32_t grid = (n + BLOCK - 1) / BLOCK;
    iotaKernel<<<grid, BLOCK, 0, stream>>>(d_out, n);
    BRKGA_CUDA_CHECK_LAST();
}

void fillIotaMod(cudaStream_t stream, GeneIndex* d_out, std::uint32_t n,
                 std::uint32_t mod) {
    constexpr std::uint32_t BLOCK = 256;
    std::uint32_t grid = (n + BLOCK - 1) / BLOCK;
    iotaModKernel<<<grid, BLOCK, 0, stream>>>(d_out, n, mod);
    BRKGA_CUDA_CHECK_LAST();
}

} // namespace brkga3
