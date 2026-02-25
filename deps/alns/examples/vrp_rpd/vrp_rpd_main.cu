// ============================================================================
// VRP-RPD SOLVER - Using Generic ALNS Framework
// ============================================================================
// Usage: ./alns_vrp_rpd <distance_matrix.csv> <processing_times.csv> [options]
//   -a N    Number of agents (default: 5)
//   -k N    Resources per agent (default: 2)
//   -i N    Max iterations (default: 25000)
//   -t N    Max time in seconds (default: 300)
//   -g N    Number of GPUs (-1 = auto, default: -1)
//   -s N    Solutions per GPU (default: 32)
//   -o FILE Output file (default: vrp_rpd_solution.json)
//   -q      Quiet mode

#include <iostream>
#include <string>
#include <cstring>

#include "vrp_rpd_config.cuh"
#include <alns/gpu/multi_gpu.hpp>

int main(int argc, char* argv[]) {
    // Defaults
    std::string distance_file;
    std::string processing_file;
    std::string output_file = "vrp_rpd_solution.json";
    int num_agents = 5;
    int resources_per_agent = 2;

    alns::ALNSRuntimeConfig config;
    config.max_iterations = 25000;
    config.max_time_seconds = 300.0;
    config.cooling_rate = 0.99975f;
    config.initial_temp_factor = 0.30f;
    config.max_removal_fraction = 0.05f;
    config.min_removal = 4;
    config.stagnation_threshold = 2000;
    config.reheat_factor = 0.5f;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (distance_file.empty())
                distance_file = argv[i];
            else if (processing_file.empty())
                processing_file = argv[i];
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            num_agents = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            resources_per_agent = std::stoi(argv[++i]);
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
            std::cout << "Usage: " << argv[0]
                      << " <distance_matrix.csv> <processing_times.csv> [options]\n"
                      << "  -a N    Number of agents (default: 5)\n"
                      << "  -k N    Resources per agent (default: 2)\n"
                      << "  -i N    Max iterations (default: 25000)\n"
                      << "  -t N    Max time seconds (default: 300)\n"
                      << "  -g N    Number of GPUs (-1=auto)\n"
                      << "  -s N    Solutions per GPU (default: 32)\n"
                      << "  -o FILE Output file\n"
                      << "  -q      Quiet\n";
            return 0;
        }
    }

    if (distance_file.empty()) {
        std::cerr << "Error: No distance matrix file specified. Use -h for help.\n";
        return 1;
    }

    // Load problem
    VRPRPDConfig::ProblemData data;
    if (!processing_file.empty()) {
        data = VRPRPDConfig::load_problem(distance_file, processing_file,
                                           num_agents, resources_per_agent);
    } else {
        data = VRPRPDConfig::load_problem(distance_file);
        data.num_agents = num_agents;
        data.resources_per_agent = resources_per_agent;
    }

    // Solve
    alns::MultiGPUSolver<VRPRPDConfig> solver(config);
    VRPRPDConfig::HostSolution result = solver.solve(data);

    // Output
    VRPRPDConfig::output_solution(result, data, output_file);
    std::cout << "Solution written to: " << output_file << std::endl;

    auto stats = solver.getStatistics();
    std::cout << "\nFinal makespan: " << result.makespan << std::endl;

    // Cleanup
    delete[] data.travel_times;
    delete[] data.processing_times;

    return 0;
}
