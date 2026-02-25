// ============================================================================
// ALNS-BRKGA Hybrid Solver — Main Entry Point
//
// A GPU-accelerated combinatorial optimization solver that combines:
//   Phase 1: ALNS (Adaptive Large Neighborhood Search) for fast initial solution
//   Phase 2: BRKGA (Biased Random-Key Genetic Algorithm) for refinement
//
// Supported problems:
//   tsp    — Traveling Salesman Problem
//   cvrp   — Capacitated Vehicle Routing Problem
//   vrprpd — Vehicle Routing with Release & Pickup-Delivery
// ============================================================================

#include "solver_common.hpp"
#include "tsp_solver.hpp"
#include "cvrp_solver.hpp"
#include "vrprpd_solver.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    auto args = parseCliArgs(argc, argv);

    if (args.show_help || args.problem.empty() || args.instance.empty()) {
        printUsage(argv[0]);
        return args.show_help ? 0 : 1;
    }

    // Prepare result with basic info
    SolverResult result;
    result.problem = args.problem;

    // Extract instance filename for display
    auto slash = args.instance.find_last_of("/\\");
    result.instance = (slash != std::string::npos)
        ? args.instance.substr(slash + 1) : args.instance;

    // Print banner
    printBanner(result, args.config);

    // Dispatch to problem-specific solver
    if (args.problem == "tsp") {
        result = solveTsp(args.instance, args.config);
    } else if (args.problem == "cvrp") {
        result = solveCvrp(args.instance, args.config);
    } else if (args.problem == "vrprpd") {
        result = solveVrpRpd(args.instance, args.config);
    } else {
        std::fprintf(stderr, "Error: Unknown problem type '%s'\n", args.problem.c_str());
        std::fprintf(stderr, "Supported: tsp, cvrp, vrprpd\n");
        return 1;
    }

    // Use short filename for display/JSON
    auto s = result.instance.find_last_of("/\\");
    if (s != std::string::npos) result.instance = result.instance.substr(s + 1);

    // Print results summary
    printResults(result);

    // Write JSON output if requested
    if (!args.config.output_file.empty()) {
        writeJsonOutput(args.config.output_file, result);
    }

    return 0;
}
