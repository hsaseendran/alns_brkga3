#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace alns {

constexpr int WARP_SIZE = 32;

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define ALNS_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// WARP-LEVEL PRIMITIVES
// ============================================================================

__device__ __forceinline__
float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__
float warpReduceMin(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fminf(val, other);
    }
    return val;
}

__device__ __forceinline__
float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__
int warpReduceMinIndex(float val, int idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }
    return idx;
}

__device__ __forceinline__
int warpReduceMaxIndex(float val, int idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    return idx;
}

__device__ __forceinline__
float warpPrefixSum(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % WARP_SIZE >= offset) {
            val += n;
        }
    }
    return val;
}

__device__ __forceinline__
float warpPrefixMax(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % WARP_SIZE >= offset) {
            val = fmaxf(val, n);
        }
    }
    return val;
}

// ============================================================================
// BLOCK-LEVEL PRIMITIVES
// ============================================================================

__device__ __forceinline__
float blockReduceMax(float val, float* shared_data) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);

    if (lane == 0) {
        shared_data[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared_data[lane] : -FLT_MAX;
        val = warpReduceMax(val);
    }

    return val;
}

__device__ __forceinline__
float blockReduceMin(float val, float* shared_data) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warpReduceMin(val);

    if (lane == 0) {
        shared_data[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared_data[lane] : FLT_MAX;
        val = warpReduceMin(val);
    }

    return val;
}

// ============================================================================
// ATOMIC OPERATIONS
// ============================================================================

__device__ __forceinline__
float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ __forceinline__
float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);

    return __int_as_float(old);
}

// ============================================================================
// SPIN LOCK
// ============================================================================

__device__ __forceinline__
void spinLock(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
        // Spin
    }
    __threadfence();
}

__device__ __forceinline__
void spinUnlock(int* lock) {
    __threadfence();
    atomicExch(lock, 0);
}

} // namespace alns
