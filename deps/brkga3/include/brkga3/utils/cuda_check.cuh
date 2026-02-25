#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// CUDA error checking macros
// ============================================================================

#define BRKGA_CUDA_CHECK(call)                                              \
    do {                                                                    \
        cudaError_t brkga_err_ = (call);                                    \
        if (brkga_err_ != cudaSuccess) {                                    \
            std::fprintf(stderr,                                            \
                "CUDA error at %s:%d in %s: %s (code %d)\n",               \
                __FILE__, __LINE__, __func__,                               \
                cudaGetErrorString(brkga_err_),                             \
                static_cast<int>(brkga_err_));                              \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

#define BRKGA_CUDA_CHECK_LAST()                                             \
    BRKGA_CUDA_CHECK(cudaPeekAtLastError())

#define BRKGA_CURAND_CHECK(call)                                            \
    do {                                                                    \
        curandStatus_t status = (call);                                     \
        if (status != CURAND_STATUS_SUCCESS) {                              \
            std::fprintf(stderr,                                            \
                "cuRAND error at %s:%d in %s: code %d\n",                   \
                __FILE__, __LINE__, __func__, static_cast<int>(status));    \
            std::abort();                                                   \
        }                                                                   \
    } while (0)

namespace brkga3 {

// Convenience: query and return a value, checking errors
inline int getDeviceCount() {
    int count = 0;
    BRKGA_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

inline void setDevice(int gpu_id) {
    BRKGA_CUDA_CHECK(cudaSetDevice(gpu_id));
}

inline int getDevice() {
    int dev = 0;
    BRKGA_CUDA_CHECK(cudaGetDevice(&dev));
    return dev;
}

} // namespace brkga3
