#pragma once

// ============================================================================
// TSP CONFIGURATOR
// ============================================================================
// A complete example of an ALNS configurator for the Traveling Salesman Problem.
// This is the simplest non-trivial configurator:
//   - 1 component (the tour)
//   - N elements (cities)
//   - Objective: minimize total tour distance
//   - All generic operators work out of the box

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>

#include <alns/gpu/rng.cuh>
#include <alns/gpu/gpu_utils.cuh>
#include <alns/operators/generic_destroy.cuh>
#include <alns/operators/generic_repair.cuh>

struct TSPConfig {

    // ================================================================
    // TYPES
    // ================================================================

    static constexpr int MAX_CITIES = 1024;

    struct alignas(256) Solution {
        int16_t tour[MAX_CITIES];       // tour[i] = city at position i
        int16_t position[MAX_CITIES];   // position[city] = index in tour (-1 = removed)
        int16_t tour_size;              // Current number of cities in tour
        int16_t _pad1;
        float objective;
        uint32_t feasibility_flags;
        float _pad2[2];
    };

    struct ProblemData {
        float* distances;    // N x N distance matrix in device global memory
        int num_cities;
        int pitch;           // Row stride (= num_cities for simplicity)
    };

    struct HostSolution {
        std::vector<int> tour;
        float total_distance;
    };

    // ================================================================
    // CONSTANTS
    // ================================================================

    static constexpr int MAX_ELEMENTS       = MAX_CITIES;
    static constexpr int MAX_COMPONENTS     = 1;
    static constexpr int MAX_COMPONENT_SIZE = MAX_CITIES;
    static constexpr int NUM_DESTROY_OPS    = 4;  // random, worst, shaw, cluster
    static constexpr int NUM_REPAIR_OPS     = 3;  // greedy, regret-2, random
    static constexpr bool MINIMIZE          = true;
    static constexpr int SHARED_MEM_BYTES   = 4 * 256 * sizeof(float);

    // ================================================================
    // EVALUATION
    // ================================================================

    __device__ static float evaluate(const Solution& sol, const ProblemData& data) {
        if (sol.tour_size <= 1) return 0.0f;

        float total = 0.0f;
        for (int i = 0; i < sol.tour_size; i++) {
            int from = sol.tour[i];
            int to = sol.tour[(i + 1) % sol.tour_size];
            total += data.distances[from * data.pitch + to];
        }
        return total;
    }

    __device__ static uint32_t check_feasibility(const Solution& sol, const ProblemData& data) {
        return (sol.tour_size == data.num_cities) ? 0 : 1;
    }

    // ================================================================
    // ELEMENT ACCESS
    // ================================================================

    __device__ static int get_num_elements(const ProblemData& data) {
        return data.num_cities;
    }

    __device__ static int get_num_components(const Solution& sol, const ProblemData& data) {
        return 1;
    }

    __device__ static int get_element_component(const Solution& sol, int city) {
        return (sol.position[city] >= 0) ? 0 : -1;
    }

    __device__ static int get_element_position(const Solution& sol, int city) {
        return sol.position[city];
    }

    __device__ static int get_component_size(const Solution& sol, int component) {
        return sol.tour_size;
    }

    __device__ static bool is_assigned(const Solution& sol, int city) {
        return sol.position[city] >= 0;
    }

    // ================================================================
    // ELEMENT MUTATION
    // ================================================================

    __device__ static void remove_element(Solution& sol, int city, const ProblemData& data) {
        int pos = sol.position[city];
        if (pos < 0) return;

        // Shift left to fill gap
        for (int i = pos; i < sol.tour_size - 1; i++) {
            sol.tour[i] = sol.tour[i + 1];
            sol.position[sol.tour[i]] = i;
        }
        sol.tour_size--;
        sol.position[city] = -1;
    }

    __device__ static bool insert_element(Solution& sol, int city,
                                           int component, int position,
                                           const ProblemData& data) {
        if (position > sol.tour_size) return false;

        // Shift right to make room
        for (int i = sol.tour_size; i > position; i--) {
            sol.tour[i] = sol.tour[i - 1];
            sol.position[sol.tour[i]] = i;
        }
        sol.tour[position] = city;
        sol.position[city] = position;
        sol.tour_size++;
        return true;
    }

    // ================================================================
    // COST FUNCTIONS
    // ================================================================

    __device__ static float removal_cost(const Solution& sol, int city,
                                          const ProblemData& data) {
        int pos = sol.position[city];
        if (pos < 0 || sol.tour_size <= 2) return 0.0f;

        int prev = sol.tour[(pos - 1 + sol.tour_size) % sol.tour_size];
        int next = sol.tour[(pos + 1) % sol.tour_size];

        float old_cost = data.distances[prev * data.pitch + city] +
                         data.distances[city * data.pitch + next];
        float new_cost = data.distances[prev * data.pitch + next];
        return old_cost - new_cost;  // Positive = good to remove
    }

    __device__ static float insertion_cost(const Solution& sol, int city,
                                            int component, int position,
                                            const ProblemData& data) {
        if (sol.tour_size == 0) return 0.0f;

        int prev, next;
        if (sol.tour_size == 1) {
            prev = sol.tour[0];
            next = sol.tour[0];
        } else {
            prev = (position > 0) ? sol.tour[position - 1]
                                  : sol.tour[sol.tour_size - 1];
            next = (position < sol.tour_size) ? sol.tour[position]
                                              : sol.tour[0];
        }

        float old_edge = data.distances[prev * data.pitch + next];
        float new_edges = data.distances[prev * data.pitch + city] +
                          data.distances[city * data.pitch + next];
        return new_edges - old_edge;
    }

    __device__ static float relatedness(const Solution& sol, int a, int b,
                                         const ProblemData& data) {
        return data.distances[a * data.pitch + b];
    }

    __device__ static float element_distance(int a, int b, const ProblemData& data) {
        return data.distances[a * data.pitch + b];
    }

    __device__ static int num_insertion_positions(const Solution& sol, int city,
                                                   int component,
                                                   const ProblemData& data) {
        return sol.tour_size + 1;
    }

    // ================================================================
    // OPERATOR DISPATCH
    // ================================================================

    __device__ static void destroy(int op_id, Solution& sol,
                                    int* removed, int& num_removed,
                                    int removal_size,
                                    const ProblemData& data,
                                    alns::XorShift128& rng) {
        switch (op_id) {
            case 0: alns::generic_random_destroy<TSPConfig>(
                        sol, removed, num_removed, removal_size, data, rng); break;
            case 1: alns::generic_worst_destroy<TSPConfig>(
                        sol, removed, num_removed, removal_size, data, rng); break;
            case 2: alns::generic_shaw_destroy<TSPConfig>(
                        sol, removed, num_removed, removal_size, data, rng); break;
            case 3: alns::generic_cluster_destroy<TSPConfig>(
                        sol, removed, num_removed, removal_size, data, rng); break;
        }
    }

    __device__ static void repair(int op_id, Solution& sol,
                                   const int* removed, int num_removed,
                                   const ProblemData& data,
                                   alns::XorShift128& rng) {
        switch (op_id) {
            case 0: alns::generic_greedy_repair<TSPConfig>(
                        sol, removed, num_removed, data, rng); break;
            case 1: alns::generic_regret_repair<TSPConfig>(
                        sol, removed, num_removed, 2, data, rng); break;
            case 2: alns::generic_random_repair<TSPConfig>(
                        sol, removed, num_removed, data, rng); break;
        }
    }

    // Parallel repair using all threads
    __device__ static void repair_parallel(int op_id, Solution& sol,
                                            const int* removed, int num_removed,
                                            const ProblemData& data,
                                            alns::XorShift128& rng,
                                            int tid, int num_threads,
                                            float* shared_mem) {
        switch (op_id) {
            case 0:
                alns::generic_greedy_repair_parallel<TSPConfig>(
                    sol, removed, num_removed, data, rng,
                    tid, num_threads, shared_mem);
                break;
            default:
                // Regret and random run on thread 0
                if (tid == 0) {
                    repair(op_id, sol, removed, num_removed, data, rng);
                }
                __syncthreads();
                break;
        }
    }

    // ================================================================
    // HOST FUNCTIONS
    // ================================================================

    static ProblemData load_problem(const std::string& path) {
        ProblemData data;

        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        // Try to detect format: TSPLIB or CSV distance matrix
        std::string first_line;
        std::getline(file, first_line);
        file.seekg(0);

        if (first_line.find("NAME") != std::string::npos ||
            first_line.find("TYPE") != std::string::npos) {
            // TSPLIB format
            load_tsplib(file, data);
        } else {
            // CSV distance matrix
            load_csv_matrix(file, data);
        }

        file.close();
        return data;
    }

    static void upload_problem_data(const ProblemData& host_data, ProblemData** d_data) {
        // Allocate device distance matrix
        size_t matrix_size = host_data.num_cities * host_data.num_cities * sizeof(float);

        ProblemData h_copy = host_data;

        ALNS_CUDA_CHECK(cudaMalloc(&h_copy.distances, matrix_size));
        ALNS_CUDA_CHECK(cudaMemcpy(h_copy.distances, host_data.distances,
                                    matrix_size, cudaMemcpyHostToDevice));

        // Allocate and copy ProblemData struct to device
        ALNS_CUDA_CHECK(cudaMalloc(d_data, sizeof(ProblemData)));
        ALNS_CUDA_CHECK(cudaMemcpy(*d_data, &h_copy, sizeof(ProblemData), cudaMemcpyHostToDevice));
    }

    static HostSolution create_initial_solution(const ProblemData& data) {
        // Nearest neighbor heuristic
        HostSolution sol;
        int n = data.num_cities;
        sol.tour.resize(n);

        std::vector<bool> visited(n, false);

        // Start from city 0
        sol.tour[0] = 0;
        visited[0] = true;

        for (int i = 1; i < n; i++) {
            int last = sol.tour[i - 1];
            float best_dist = FLT_MAX;
            int best_city = -1;

            for (int c = 0; c < n; c++) {
                if (!visited[c]) {
                    float d = data.distances[last * data.pitch + c];
                    if (d < best_dist) {
                        best_dist = d;
                        best_city = c;
                    }
                }
            }

            sol.tour[i] = best_city;
            visited[best_city] = true;
        }

        // Compute total distance
        sol.total_distance = 0.0f;
        for (int i = 0; i < n; i++) {
            int from = sol.tour[i];
            int to = sol.tour[(i + 1) % n];
            sol.total_distance += data.distances[from * data.pitch + to];
        }

        return sol;
    }

    static float host_evaluate(const HostSolution& sol, const ProblemData& data) {
        return sol.total_distance;
    }

    static Solution to_device_solution(const HostSolution& host, const ProblemData& data) {
        Solution sol;
        std::memset(&sol, 0, sizeof(Solution));

        for (int i = 0; i < data.num_cities; i++) {
            sol.position[i] = -1;
        }

        for (int i = 0; i < (int)host.tour.size(); i++) {
            sol.tour[i] = host.tour[i];
            sol.position[host.tour[i]] = i;
        }
        sol.tour_size = host.tour.size();
        sol.objective = host.total_distance;
        sol.feasibility_flags = 0;

        return sol;
    }

    static HostSolution from_device_solution(const Solution& sol, const ProblemData& data) {
        HostSolution host;
        host.tour.resize(sol.tour_size);
        for (int i = 0; i < sol.tour_size; i++) {
            host.tour[i] = sol.tour[i];
        }
        host.total_distance = sol.objective;
        return host;
    }

    static void output_solution(const HostSolution& sol, const ProblemData& data,
                                const std::string& path) {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"total_distance\": " << sol.total_distance << ",\n";
        file << "  \"num_cities\": " << sol.tour.size() << ",\n";
        file << "  \"tour\": [";
        for (size_t i = 0; i < sol.tour.size(); i++) {
            if (i > 0) file << ", ";
            file << sol.tour[i];
        }
        file << "]\n}\n";
        file.close();
    }

private:
    // ================================================================
    // FILE PARSERS
    // ================================================================

    static void load_tsplib(std::ifstream& file, ProblemData& data) {
        std::string line;
        int dimension = 0;
        std::string edge_weight_type;
        std::vector<float> coords_x, coords_y;

        while (std::getline(file, line)) {
            if (line.find("DIMENSION") != std::string::npos) {
                sscanf(line.c_str(), "%*[^0-9]%d", &dimension);
            } else if (line.find("EDGE_WEIGHT_TYPE") != std::string::npos) {
                if (line.find("EUC_2D") != std::string::npos) edge_weight_type = "EUC_2D";
                else if (line.find("ATT") != std::string::npos) edge_weight_type = "ATT";
                else if (line.find("EXPLICIT") != std::string::npos) edge_weight_type = "EXPLICIT";
            } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                coords_x.resize(dimension);
                coords_y.resize(dimension);
                for (int i = 0; i < dimension; i++) {
                    int id;
                    float x, y;
                    file >> id >> x >> y;
                    coords_x[id - 1] = x;
                    coords_y[id - 1] = y;
                }
                break;
            } else if (line.find("EDGE_WEIGHT_SECTION") != std::string::npos) {
                // Read explicit matrix
                data.num_cities = dimension;
                data.pitch = dimension;
                data.distances = new float[dimension * dimension];
                for (int i = 0; i < dimension; i++) {
                    for (int j = 0; j < dimension; j++) {
                        file >> data.distances[i * dimension + j];
                    }
                }
                return;
            }
        }

        if (dimension == 0) {
            throw std::runtime_error("Could not parse TSPLIB file");
        }

        // Compute distance matrix from coordinates
        data.num_cities = dimension;
        data.pitch = dimension;
        data.distances = new float[dimension * dimension];

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i == j) {
                    data.distances[i * dimension + j] = 0.0f;
                } else {
                    float dx = coords_x[i] - coords_x[j];
                    float dy = coords_y[i] - coords_y[j];

                    if (edge_weight_type == "ATT") {
                        float r = sqrtf((dx * dx + dy * dy) / 10.0f);
                        int t = static_cast<int>(r + 0.5f);
                        data.distances[i * dimension + j] = (t < r) ? t + 1.0f : t;
                    } else {
                        data.distances[i * dimension + j] = roundf(sqrtf(dx * dx + dy * dy));
                    }
                }
            }
        }
    }

    static void load_csv_matrix(std::ifstream& file, ProblemData& data) {
        std::vector<std::vector<float>> rows;
        std::string line;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::vector<float> row;
            std::stringstream ss(line);
            std::string val;
            while (std::getline(ss, val, ',')) {
                row.push_back(std::stof(val));
            }
            rows.push_back(row);
        }

        int n = rows.size();
        data.num_cities = n;
        data.pitch = n;
        data.distances = new float[n * n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data.distances[i * n + j] = rows[i][j];
            }
        }
    }
};
