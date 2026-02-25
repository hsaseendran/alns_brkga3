#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace alns {

// ============================================================================
// FAST XORSHIFT RNG (GPU-friendly)
// ============================================================================

struct XorShift128 {
    uint32_t state[4];

    __device__ __forceinline__
    void init(uint64_t seed, int thread_id) {
        uint64_t s = seed + thread_id * 0x9E3779B97F4A7C15ULL;

        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
        state[0] = static_cast<uint32_t>(s);
        state[1] = static_cast<uint32_t>(s >> 32);

        s = (s ^ (s >> 27)) * 0x94D049BB133111EBULL;
        state[2] = static_cast<uint32_t>(s);
        state[3] = static_cast<uint32_t>(s >> 32);
    }

    __device__ __forceinline__
    uint32_t next() {
        uint32_t t = state[3];
        uint32_t s = state[0];
        state[3] = state[2];
        state[2] = state[1];
        state[1] = s;
        t ^= t << 11;
        t ^= t >> 8;
        state[0] = t ^ s ^ (s >> 19);
        return state[0];
    }

    __device__ __forceinline__
    float nextFloat() {
        return (next() >> 8) * (1.0f / 16777216.0f);
    }

    __device__ __forceinline__
    int nextInt(int max) {
        return static_cast<int>(nextFloat() * max);
    }

    __device__ __forceinline__
    int nextIntRange(int min_val, int max_val) {
        return min_val + nextInt(max_val - min_val + 1);
    }
};

// ============================================================================
// ROULETTE WHEEL SELECTION
// ============================================================================

__device__ __forceinline__
int rouletteWheelSelect(const float* weights, int n, XorShift128& rng) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += weights[i];
    }

    float r = rng.nextFloat() * sum;
    float cumulative = 0.0f;

    for (int i = 0; i < n - 1; i++) {
        cumulative += weights[i];
        if (r < cumulative) {
            return i;
        }
    }
    return n - 1;
}

} // namespace alns
