#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <cstring>
#include <type_traits>

#include <alns/core/engine.cuh>
#include <alns/core/runtime_config.hpp>

namespace alns {

// SFINAE helper: detect Config::SHARED_MEM_BYTES
template<typename C, typename = void>
struct has_shared_mem_bytes : std::false_type {};

template<typename C>
struct has_shared_mem_bytes<C, std::void_t<decltype(C::SHARED_MEM_BYTES)>> : std::true_type {};

template<typename C>
int computeSharedMemSize(int base) {
    if constexpr (has_shared_mem_bytes<C>::value) {
        return base > static_cast<int>(C::SHARED_MEM_BYTES) ? base : static_cast<int>(C::SHARED_MEM_BYTES);
    } else {
        return base;
    }
}

// ============================================================================
// MULTI-GPU CONTROLLER (templated on Config)
// ============================================================================

template<typename Config>
class MultiGPUSolver {
public:
    using Solution = typename Config::Solution;
    using HostSolution = typename Config::HostSolution;
    using ProblemData = typename Config::ProblemData;
    using State = ALNSState<Config::NUM_DESTROY_OPS, Config::NUM_REPAIR_OPS>;

    MultiGPUSolver(const ALNSRuntimeConfig& config) : config_(config) {}
    ~MultiGPUSolver() { cleanup(); }

    HostSolution solve(const ProblemData& data) {
        // Build initial solution on CPU
        HostSolution initial_host = Config::create_initial_solution(data);
        float initial_obj = Config::host_evaluate(initial_host, data);

        if (config_.verbose) {
            std::cout << "Initial objective: " << initial_obj << std::endl;
        }

        // Initialize GPUs
        initialize(data, initial_host, initial_obj);

        // Run optimization
        HostSolution result = run(data);

        return result;
    }

    SolverStatistics getStatistics() const { return stats_; }

private:
    ALNSRuntimeConfig config_;
    SolverStatistics stats_;

    int num_gpus_ = 0;
    std::vector<cudaStream_t> streams_;

    struct GPUData {
        Solution* d_current;
        Solution* d_best;
        Solution* d_working;
        State* d_states;
    };
    std::vector<GPUData> gpu_data_;

    std::vector<ProblemData*> d_problem_data_;  // one per GPU (replicated)
    SharedBest<Solution>* d_global_best_ = nullptr;

    void cleanup() {
        for (int gpu = 0; gpu < num_gpus_; gpu++) {
            cudaSetDevice(gpu);
            if (gpu < (int)gpu_data_.size()) {
                cudaFree(gpu_data_[gpu].d_current);
                cudaFree(gpu_data_[gpu].d_best);
                cudaFree(gpu_data_[gpu].d_working);
                cudaFree(gpu_data_[gpu].d_states);
            }
            if (gpu < (int)d_problem_data_.size() && d_problem_data_[gpu]) {
                cudaFree(d_problem_data_[gpu]);
                d_problem_data_[gpu] = nullptr;
            }
            if (gpu < (int)streams_.size()) {
                cudaStreamDestroy(streams_[gpu]);
            }
        }
        d_problem_data_.clear();
        if (d_global_best_) { cudaFree(d_global_best_); d_global_best_ = nullptr; }
        gpu_data_.clear();
        streams_.clear();
    }

    void initialize(const ProblemData& data, const HostSolution& initial_host, float initial_obj) {
        if (config_.num_gpus < 0) {
            cudaGetDeviceCount(&num_gpus_);
        } else {
            num_gpus_ = config_.num_gpus;
        }

        if (num_gpus_ == 0) {
            throw std::runtime_error("No CUDA-capable devices found");
        }

        if (config_.verbose) {
            std::cout << "Initializing ALNS with " << num_gpus_ << " GPU(s), "
                      << config_.solutions_per_gpu << " solutions/GPU" << std::endl;
        }

        int num_blocks = config_.solutions_per_gpu;

        // Convert initial solution to device format
        Solution h_initial = Config::to_device_solution(initial_host, data);

        // Allocate global best in managed memory
        ALNS_CUDA_CHECK(cudaMallocManaged(&d_global_best_, sizeof(SharedBest<Solution>)));
        std::memset(d_global_best_, 0, sizeof(SharedBest<Solution>));
        std::memcpy(&d_global_best_->solution, &h_initial, sizeof(Solution));
        d_global_best_->objective = initial_obj;
        d_global_best_->source_gpu = -1;
        d_global_best_->source_block = -1;
        d_global_best_->found_at_iteration = 0;
        d_global_best_->lock = 0;

        gpu_data_.resize(num_gpus_);
        streams_.resize(num_gpus_);

        for (int gpu = 0; gpu < num_gpus_; gpu++) {
            cudaSetDevice(gpu);
            ALNS_CUDA_CHECK(cudaStreamCreate(&streams_[gpu]));

            GPUData& gd = gpu_data_[gpu];

            ALNS_CUDA_CHECK(cudaMalloc(&gd.d_current, num_blocks * sizeof(Solution)));
            ALNS_CUDA_CHECK(cudaMalloc(&gd.d_best, num_blocks * sizeof(Solution)));
            ALNS_CUDA_CHECK(cudaMalloc(&gd.d_working, num_blocks * sizeof(Solution)));
            ALNS_CUDA_CHECK(cudaMalloc(&gd.d_states, num_blocks * sizeof(State)));

            // Initialize all solution slots with the initial solution
            std::vector<Solution> h_solutions(num_blocks, h_initial);

            ALNS_CUDA_CHECK(cudaMemcpy(gd.d_current, h_solutions.data(),
                                        num_blocks * sizeof(Solution), cudaMemcpyHostToDevice));
            ALNS_CUDA_CHECK(cudaMemcpy(gd.d_best, h_solutions.data(),
                                        num_blocks * sizeof(Solution), cudaMemcpyHostToDevice));
        }

        // Upload problem data to each GPU (replicated)
        d_problem_data_.resize(num_gpus_, nullptr);
        for (int gpu = 0; gpu < num_gpus_; gpu++) {
            cudaSetDevice(gpu);
            Config::upload_problem_data(data, &d_problem_data_[gpu]);
        }

        // Initialize ALNS states
        float initial_temp = config_.initial_temp_factor * initial_obj;

        for (int gpu = 0; gpu < num_gpus_; gpu++) {
            cudaSetDevice(gpu);

            int threads = 256;
            int blocks = (num_blocks + threads - 1) / threads;

            initStateKernel<Config><<<blocks, threads, 0, streams_[gpu]>>>(
                gpu_data_[gpu].d_states,
                gpu_data_[gpu].d_current,
                gpu_data_[gpu].d_best,
                d_problem_data_[gpu],
                initial_temp,
                num_blocks);
        }

        // Sync all
        for (int gpu = 0; gpu < num_gpus_; gpu++) {
            cudaSetDevice(gpu);
            ALNS_CUDA_CHECK(cudaStreamSynchronize(streams_[gpu]));
        }

        stats_.initial_objective = initial_obj;
    }

    HostSolution run(const ProblemData& data) {
        auto start_time = std::chrono::high_resolution_clock::now();

        int64_t total_iterations = 0;
        int ipl = config_.iterations_per_launch;
        int num_blocks = config_.solutions_per_gpu;
        float best_obj = stats_.initial_objective;

        // Compute shared memory size:
        // Engine needs 4 * block_size * sizeof(int) for parallel repair reduction
        // Plus Config::SHARED_MEM_BYTES if defined
        constexpr int BLOCK_SIZE = 256;
        int shared_mem_size = 4 * BLOCK_SIZE * sizeof(float);
        shared_mem_size = computeSharedMemSize<Config>(shared_mem_size);
        // Add buffer for removed elements in shared mem
        shared_mem_size += Config::MAX_ELEMENTS * sizeof(int);

        DeviceRuntimeParams dev_params = DeviceRuntimeParams::fromConfig(config_, BLOCK_SIZE);

        bool should_stop = false;

        while (!should_stop) {
            // Launch ALNS on all GPUs
            for (int gpu = 0; gpu < num_gpus_; gpu++) {
                cudaSetDevice(gpu);

                alnsMainKernel<Config><<<num_blocks, BLOCK_SIZE, shared_mem_size, streams_[gpu]>>>(
                    gpu_data_[gpu].d_current,
                    gpu_data_[gpu].d_best,
                    gpu_data_[gpu].d_working,
                    gpu_data_[gpu].d_states,
                    d_problem_data_[gpu],
                    d_global_best_,
                    dev_params,
                    ipl,
                    gpu);
            }

            // Synchronize
            for (int gpu = 0; gpu < num_gpus_; gpu++) {
                cudaSetDevice(gpu);
                ALNS_CUDA_CHECK(cudaStreamSynchronize(streams_[gpu]));
            }

            total_iterations += static_cast<int64_t>(ipl) * num_blocks * num_gpus_;

            // Check global best
            bool improved = Config::MINIMIZE ?
                (d_global_best_->objective < best_obj) :
                (d_global_best_->objective > best_obj);

            if (improved) {
                best_obj = d_global_best_->objective;
                stats_.convergence_history.push_back(best_obj);
                stats_.improvements++;

                if (config_.verbose) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double>(now - start_time).count();
                    std::cout << "[" << elapsed << "s] Iter " << total_iterations
                              << ": Best = " << best_obj << std::endl;
                }
            }

            // Termination criteria
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();

            if (config_.max_iterations > 0 && total_iterations >= config_.max_iterations) {
                if (config_.verbose) std::cout << "Stopping: max iterations" << std::endl;
                should_stop = true;
            }

            if (config_.max_time_seconds > 0 && elapsed >= config_.max_time_seconds) {
                if (config_.verbose) std::cout << "Stopping: time limit" << std::endl;
                should_stop = true;
            }

            if (config_.target_objective > 0) {
                bool target_met = Config::MINIMIZE ?
                    (best_obj <= config_.target_objective) :
                    (best_obj >= config_.target_objective);
                if (target_met) {
                    if (config_.verbose) std::cout << "Stopping: target achieved" << std::endl;
                    should_stop = true;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        stats_.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
        stats_.total_iterations = total_iterations;
        stats_.final_objective = best_obj;

        // Convert best solution back to host format
        HostSolution result = Config::from_device_solution(d_global_best_->solution, data);

        if (config_.verbose) {
            std::cout << "\nALNS completed:" << std::endl;
            std::cout << "  Iterations: " << stats_.total_iterations << std::endl;
            std::cout << "  Time: " << stats_.elapsed_seconds << "s" << std::endl;
            std::cout << "  Initial: " << stats_.initial_objective << std::endl;
            std::cout << "  Final: " << stats_.final_objective << std::endl;
            double improvement = (1.0 - stats_.final_objective / stats_.initial_objective) * 100.0;
            std::cout << "  Improvement: " << improvement << "%" << std::endl;
        }

        return result;
    }
};

} // namespace alns
