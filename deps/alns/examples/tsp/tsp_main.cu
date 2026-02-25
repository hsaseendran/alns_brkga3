// ============================================================================
// TSP SOLVER - Using Generic ALNS Framework
// ============================================================================
// Usage: ./alns_tsp <tsplib_or_csv_file> [options]
//   -i N    Max iterations (default: 50000)
//   -t N    Max time in seconds (default: 60)
//   -g N    Number of GPUs (-1 = auto, default: -1)
//   -s N    Solutions per GPU (default: 32)
//   -o FILE Output file (default: tsp_solution.json)
//   -q      Quiet mode

#include <iostream>
#include <string>
#include <cstring>

#include "tsp_config.cuh"
#include <alns/gpu/multi_gpu.hpp>

int main(int argc, char* argv[]) {
    // Defaults
    std::string input_file;
    std::string output_file = "tsp_solution.json";
    alns::ALNSRuntimeConfig config;
    config.max_iterations = 50000;
    config.max_time_seconds = 60.0;
    config.cooling_rate = 0.9998f;
    config.initial_temp_factor = 0.05f;
    config.max_removal_fraction = 0.3f;
    config.min_removal = 3;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            input_file = argv[i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            config.max_iterations = std::stoll(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            config.max_time_seconds = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            config.num_gpus = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            config.solutions_per_gpu = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-q") == 0) {
            config.verbose = false;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: " << argv[0] << " <tsplib_or_csv_file> [options]\n"
                      << "  -i N    Max iterations (default: 50000)\n"
                      << "  -t N    Max time seconds (default: 60)\n"
                      << "  -g N    Number of GPUs (-1=auto)\n"
                      << "  -s N    Solutions per GPU (default: 32)\n"
                      << "  -o FILE Output file\n"
                      << "  -q      Quiet\n";
            return 0;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: No input file specified. Use -h for help.\n";
        return 1;
    }

    // Load problem
    std::cout << "Loading problem from: " << input_file << std::endl;
    TSPConfig::ProblemData data = TSPConfig::load_problem(input_file);
    std::cout << "Cities: " << data.num_cities << std::endl;

    // Solve
    alns::MultiGPUSolver<TSPConfig> solver(config);
    TSPConfig::HostSolution result = solver.solve(data);

    // Output
    TSPConfig::output_solution(result, data, output_file);
    std::cout << "Solution written to: " << output_file << std::endl;

    auto stats = solver.getStatistics();
    std::cout << "\nFinal tour distance: " << result.total_distance << std::endl;

    // Cleanup host distance matrix
    delete[] data.distances;

    return 0;
}
