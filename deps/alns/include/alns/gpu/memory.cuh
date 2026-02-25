#pragma once

#include <cuda_runtime.h>

namespace alns {

// ============================================================================
// VECTORIZED SOLUTION COPY
// ============================================================================

// Copy a solution using vectorized loads (float4 = 16 bytes at a time).
// The solution type must be a multiple of 16 bytes in size and aligned >= 16.
// All threads in the block participate.
template<typename SolutionType>
__device__ __forceinline__
void copySolutionVectorized(SolutionType* __restrict__ dst,
                            const SolutionType* __restrict__ src,
                            int tid, int stride) {
    static_assert(sizeof(SolutionType) % sizeof(float4) == 0,
                  "Solution size must be a multiple of sizeof(float4) for vectorized copy");

    float4* dst_vec = reinterpret_cast<float4*>(dst);
    const float4* src_vec = reinterpret_cast<const float4*>(src);
    constexpr int num_vec = sizeof(SolutionType) / sizeof(float4);

    for (int i = tid; i < num_vec; i += stride) {
        dst_vec[i] = src_vec[i];
    }
}

// Copy raw bytes (for solutions with non-standard size)
template<int NUM_BYTES>
__device__ __forceinline__
void copyBytesParallel(void* __restrict__ dst,
                       const void* __restrict__ src,
                       int tid, int stride) {
    constexpr int num_int4 = NUM_BYTES / sizeof(int4);
    constexpr int remainder = NUM_BYTES % sizeof(int4);

    int4* dst_vec = reinterpret_cast<int4*>(dst);
    const int4* src_vec = reinterpret_cast<const int4*>(src);

    for (int i = tid; i < num_int4; i += stride) {
        dst_vec[i] = src_vec[i];
    }

    // Handle remainder bytes with thread 0
    if (remainder > 0 && tid == 0) {
        char* dst_c = reinterpret_cast<char*>(dst) + num_int4 * sizeof(int4);
        const char* src_c = reinterpret_cast<const char*>(src) + num_int4 * sizeof(int4);
        for (int i = 0; i < remainder; i++) {
            dst_c[i] = src_c[i];
        }
    }
}

} // namespace alns
