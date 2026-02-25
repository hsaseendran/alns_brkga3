#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace cvrp {

struct CvrpInstance {
    std::uint32_t dimension = 0;    // total nodes (depot + clients)
    std::uint32_t n_clients = 0;    // dimension - 1
    std::uint32_t capacity = 0;     // vehicle capacity
    std::uint32_t depot = 0;        // depot node index (0-based)

    std::vector<float> distances;   // [dimension * dimension] flattened
    std::vector<std::uint32_t> demands; // [dimension] (depot demand = 0)

    std::string name;
    float optimal = -1.0f;

    float dist(std::uint32_t i, std::uint32_t j) const {
        return distances[i * dimension + j];
    }
};

// Parse TSPLIB CVRP format (.vrp files)
inline CvrpInstance loadCVRPLIB(const std::string& filepath) {
    CvrpInstance inst;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CVRP file: " + filepath);
    }

    // Extract name from filepath
    auto pos = filepath.find_last_of("/\\");
    inst.name = (pos != std::string::npos) ? filepath.substr(pos + 1) : filepath;

    std::vector<double> x, y;
    std::string line;

    while (std::getline(file, line)) {
        // Trim whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();

        if (line.find("DIMENSION") != std::string::npos && line.find(':') != std::string::npos) {
            inst.dimension = std::stoi(line.substr(line.find(':') + 1));
        } else if (line.find("CAPACITY") != std::string::npos && line.find(':') != std::string::npos) {
            inst.capacity = std::stoi(line.substr(line.find(':') + 1));
        } else if (line.find("Optimal value") != std::string::npos ||
                   line.find("Optimal Value") != std::string::npos ||
                   line.find("optimal value") != std::string::npos) {
            // Try to extract optimal from COMMENT line
            auto cpos = line.find_last_of(':');
            if (cpos != std::string::npos) {
                std::string val = line.substr(cpos + 1);
                // Remove trailing )
                val.erase(std::remove(val.begin(), val.end(), ')'), val.end());
                try { inst.optimal = std::stof(val); } catch (...) {}
            }
        } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            x.resize(inst.dimension);
            y.resize(inst.dimension);
            for (std::uint32_t i = 0; i < inst.dimension; ++i) {
                int id; double cx, cy;
                file >> id >> cx >> cy;
                x[id - 1] = cx;  // convert 1-indexed to 0-indexed
                y[id - 1] = cy;
            }
        } else if (line.find("DEMAND_SECTION") != std::string::npos) {
            inst.demands.resize(inst.dimension);
            for (std::uint32_t i = 0; i < inst.dimension; ++i) {
                int id; std::uint32_t d;
                file >> id >> d;
                inst.demands[id - 1] = d;
            }
        } else if (line.find("DEPOT_SECTION") != std::string::npos) {
            int depot_id;
            file >> depot_id;
            if (depot_id > 0) {
                inst.depot = depot_id - 1;  // convert to 0-indexed
            }
        }
    }

    inst.n_clients = inst.dimension - 1;

    // Build Euclidean distance matrix (EUC_2D: round to nearest integer)
    inst.distances.resize(inst.dimension * inst.dimension);
    for (std::uint32_t i = 0; i < inst.dimension; ++i) {
        for (std::uint32_t j = 0; j < inst.dimension; ++j) {
            if (i == j) {
                inst.distances[i * inst.dimension + j] = 0.0f;
            } else {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                inst.distances[i * inst.dimension + j] =
                    static_cast<float>(static_cast<int>(std::sqrt(dx*dx + dy*dy) + 0.5));
            }
        }
    }

    // Try to load optimal cost from .sol file
    if (inst.optimal <= 0) {
        std::string sol_path = filepath;
        auto dot = sol_path.rfind('.');
        if (dot != std::string::npos) {
            sol_path = sol_path.substr(0, dot) + ".sol";
            std::ifstream sol_file(sol_path);
            if (sol_file.is_open()) {
                std::string sol_line;
                while (std::getline(sol_file, sol_line)) {
                    if (sol_line.find("Cost") != std::string::npos) {
                        std::istringstream iss(sol_line);
                        std::string label;
                        float cost;
                        if (iss >> label >> cost) {
                            inst.optimal = cost;
                        }
                    }
                }
            }
        }
    }

    return inst;
}

} // namespace cvrp
