#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// ============================================================================
// Solver Configuration
// ============================================================================

struct SolverConfig {
    // ALNS parameters
    double alns_time = 30.0;
    int alns_gpus = -1;             // -1 = auto-detect
    int alns_solutions_per_gpu = 32;

    // BRKGA parameters
    int brkga_gens = 1000;
    int brkga_gpus = -1;            // -1 = auto-detect
    int brkga_pops_per_gpu = 3;
    int brkga_pop_size = 1024;
    int brkga_seed = 42;

    // VRP-RPD specific
    int vrprpd_agents = 5;
    int vrprpd_resources = 2;
    std::string processing_times_file;

    // General
    bool cold_start = false;
    bool verbose = true;
    std::string output_file;
};

// ============================================================================
// Solver Result
// ============================================================================

struct SolverResult {
    std::string problem;
    std::string instance;

    // ALNS phase
    float alns_objective = 0;
    double alns_time_s = 0;

    // BRKGA phase
    float brkga_initial = 0;     // objective after warm-start injection
    float brkga_final = 0;
    int brkga_generations = 0;
    double brkga_time_s = 0;

    // Final
    float final_objective = 0;
    double total_time_s = 0;

    // Problem-specific solution payload (JSON fragment)
    std::string solution_json;
};

// ============================================================================
// CLI Argument Parser
// ============================================================================

struct CliArgs {
    std::string problem;
    std::string instance;
    SolverConfig config;
    bool show_help = false;
};

inline void printUsage(const char* prog) {
    std::printf(
        "Usage: %s <problem> <instance> [options]\n"
        "\n"
        "Problems:\n"
        "  tsp      Traveling Salesman Problem (TSPLIB or CSV format)\n"
        "  cvrp     Capacitated Vehicle Routing Problem (.vrp format)\n"
        "  vrprpd   Vehicle Routing with Release & Pickup-Delivery (CSV)\n"
        "\n"
        "Options:\n"
        "  --alns-time <sec>       ALNS time limit (default: 30)\n"
        "  --alns-gpus <n>         GPUs for ALNS phase (-1=auto, default: -1)\n"
        "  --alns-spg <n>          ALNS solutions per GPU (default: 32)\n"
        "  --brkga-gens <n>        BRKGA generations (default: 1000)\n"
        "  --brkga-gpus <n>        GPUs for BRKGA phase (-1=auto, default: -1)\n"
        "  --brkga-pops <n>        BRKGA populations per GPU (default: 3)\n"
        "  --brkga-pop-size <n>    BRKGA population size (default: 1024)\n"
        "  --seed <n>              Random seed (default: 42)\n"
        "  --output <file>         Write JSON results to file\n"
        "  --cold                  Skip ALNS, run BRKGA only (cold start)\n"
        "  -v, --verbose           Verbose output (default)\n"
        "  -q, --quiet             Suppress progress output\n"
        "  -h, --help              Show this help\n"
        "\n"
        "VRP-RPD specific:\n"
        "  --agents <n>            Number of agents (default: 5)\n"
        "  --resources <n>         Resources per agent (default: 2)\n"
        "  --proc-times <file>     Processing times CSV file\n"
        "\n"
        "Examples:\n"
        "  %s tsp data/tsp/a280.tsp --alns-time 30 --brkga-gens 1000\n"
        "  %s cvrp data/cvrp/X-n101-k25.vrp --alns-time 60\n"
        "  %s vrprpd data/vrprpd/distance_matrix/berlin52_matrix.csv --agents 5\n",
        prog, prog, prog, prog);
}

inline CliArgs parseCliArgs(int argc, char** argv) {
    CliArgs args;

    if (argc < 2) {
        args.show_help = true;
        return args;
    }

    // Collect positional args
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];

        if (a == "-h" || a == "--help") {
            args.show_help = true;
            return args;
        } else if (a == "--alns-time" && i + 1 < argc) {
            args.config.alns_time = std::atof(argv[++i]);
        } else if (a == "--alns-gpus" && i + 1 < argc) {
            args.config.alns_gpus = std::atoi(argv[++i]);
        } else if (a == "--alns-spg" && i + 1 < argc) {
            args.config.alns_solutions_per_gpu = std::atoi(argv[++i]);
        } else if (a == "--brkga-gens" && i + 1 < argc) {
            args.config.brkga_gens = std::atoi(argv[++i]);
        } else if (a == "--brkga-gpus" && i + 1 < argc) {
            args.config.brkga_gpus = std::atoi(argv[++i]);
        } else if (a == "--brkga-pops" && i + 1 < argc) {
            args.config.brkga_pops_per_gpu = std::atoi(argv[++i]);
        } else if (a == "--brkga-pop-size" && i + 1 < argc) {
            args.config.brkga_pop_size = std::atoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            args.config.brkga_seed = std::atoi(argv[++i]);
        } else if (a == "--output" && i + 1 < argc) {
            args.config.output_file = argv[++i];
        } else if (a == "--cold") {
            args.config.cold_start = true;
        } else if (a == "-v" || a == "--verbose") {
            args.config.verbose = true;
        } else if (a == "-q" || a == "--quiet") {
            args.config.verbose = false;
        } else if (a == "--agents" && i + 1 < argc) {
            args.config.vrprpd_agents = std::atoi(argv[++i]);
        } else if (a == "--resources" && i + 1 < argc) {
            args.config.vrprpd_resources = std::atoi(argv[++i]);
        } else if (a == "--proc-times" && i + 1 < argc) {
            args.config.processing_times_file = argv[++i];
        } else if (a[0] != '-') {
            positional.push_back(a);
        }
    }

    if (positional.size() >= 1) args.problem = positional[0];
    if (positional.size() >= 2) args.instance = positional[1];

    return args;
}

// ============================================================================
// Output Helpers
// ============================================================================

inline void printBanner(const SolverResult& r, const SolverConfig& cfg) {
    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("  ALNS-BRKGA Hybrid Solver\n");
    std::printf("  Problem:  %s\n", r.problem.c_str());
    std::printf("  Instance: %s\n", r.instance.c_str());
    std::printf("  Mode:     %s\n", cfg.cold_start ? "COLD START (BRKGA only)" : "WARM START (ALNS -> BRKGA)");
    std::printf("============================================================\n\n");
}

inline void printResults(const SolverResult& r) {
    std::printf("\n");
    std::printf("============================================================\n");
    std::printf("  RESULTS\n");
    if (r.alns_time_s > 0) {
        std::printf("  ALNS solution:     %.2f  (%.1fs)\n", r.alns_objective, r.alns_time_s);
    }
    std::printf("  BRKGA initial:     %.2f\n", r.brkga_initial);
    std::printf("  BRKGA final:       %.2f  (%d gens, %.1fs)\n",
                r.brkga_final, r.brkga_generations, r.brkga_time_s);
    if (r.brkga_initial > 0 && r.brkga_final > 0) {
        float improvement = (1.0f - r.brkga_final / r.brkga_initial) * 100.0f;
        std::printf("  BRKGA improvement: %.2f%%\n", improvement);
    }
    std::printf("  Final objective:   %.2f\n", r.final_objective);
    std::printf("  Total time:        %.1fs\n", r.total_time_s);
    std::printf("============================================================\n\n");
}

inline void writeJsonOutput(const std::string& path, const SolverResult& r) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "Warning: Cannot write JSON to %s\n", path.c_str());
        return;
    }

    float improvement = 0;
    if (r.brkga_initial > 0 && r.brkga_final > 0)
        improvement = (1.0f - r.brkga_final / r.brkga_initial) * 100.0f;

    f << "{\n";
    f << "  \"solver\": \"alns_brkga\",\n";
    f << "  \"version\": \"1.0\",\n";
    f << "  \"problem\": \"" << r.problem << "\",\n";
    f << "  \"instance\": \"" << r.instance << "\",\n";

    if (r.alns_time_s > 0) {
        f << "  \"alns\": {\n";
        f << "    \"objective\": " << r.alns_objective << ",\n";
        f << "    \"time_seconds\": " << r.alns_time_s << "\n";
        f << "  },\n";
    }

    f << "  \"brkga\": {\n";
    f << "    \"warm_start_objective\": " << r.brkga_initial << ",\n";
    f << "    \"final_objective\": " << r.brkga_final << ",\n";
    f << "    \"generations\": " << r.brkga_generations << ",\n";
    f << "    \"time_seconds\": " << r.brkga_time_s << "\n";
    f << "  },\n";

    f << "  \"result\": {\n";
    f << "    \"objective\": " << r.final_objective << ",\n";
    f << "    \"total_time_seconds\": " << r.total_time_s << ",\n";
    f << "    \"improvement_pct\": " << improvement << "\n";
    f << "  }";

    if (!r.solution_json.empty()) {
        f << ",\n  \"solution\": " << r.solution_json << "\n";
    } else {
        f << "\n";
    }

    f << "}\n";
    f.close();

    std::printf("JSON results written to: %s\n", path.c_str());
}
