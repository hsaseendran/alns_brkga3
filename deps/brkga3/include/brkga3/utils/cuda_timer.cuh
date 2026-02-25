#pragma once

#include "cuda_check.cuh"
#include <cuda_runtime.h>

namespace brkga3 {

// Event-based GPU timer for profiling kernel execution.
// Usage:
//   CudaTimer timer;
//   timer.start(stream);
//   myKernel<<<...>>>();
//   timer.stop(stream);
//   float ms = timer.elapsed_ms();  // blocks until stop event completes

class CudaTimer {
public:
    CudaTimer() {
        BRKGA_CUDA_CHECK(cudaEventCreate(&start_));
        BRKGA_CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    CudaTimer(CudaTimer&& other) noexcept
        : start_(other.start_), stop_(other.stop_) {
        other.start_ = nullptr;
        other.stop_ = nullptr;
    }

    CudaTimer& operator=(CudaTimer&& other) noexcept {
        if (this != &other) {
            if (start_) cudaEventDestroy(start_);
            if (stop_) cudaEventDestroy(stop_);
            start_ = other.start_;
            stop_ = other.stop_;
            other.start_ = nullptr;
            other.stop_ = nullptr;
        }
        return *this;
    }

    void start(cudaStream_t stream = nullptr) {
        BRKGA_CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = nullptr) {
        BRKGA_CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    // Blocks host until stop event completes, returns elapsed time in ms.
    float elapsed_ms() {
        BRKGA_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        BRKGA_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_  = nullptr;
};

} // namespace brkga3
