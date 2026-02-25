#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

namespace tsp {

struct TspInstance {
    std::uint32_t n = 0;               // number of cities
    std::vector<float> distances;       // [n x n] distance matrix
    std::string name;
    float optimal = -1.0f;              // known optimal, -1 if unknown
};

// Parse a TSPLIB format file (.tsp)
// Supports: EUC_2D, ATT, CEIL_2D, GEO edge weight types with NODE_COORD_SECTION
inline TspInstance loadTSPLIB(const std::string& filepath) {
    TspInstance inst;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open TSP file: " + filepath);
    }

    std::string line;
    std::string edge_weight_type;
    std::vector<float> x, y;
    bool reading_coords = false;
    bool reading_edge_weights = false;
    std::string display_data_type;

    while (std::getline(file, line)) {
        // Trim
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

        if (line.find("NAME") == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                inst.name = line.substr(pos + 1);
                inst.name.erase(0, inst.name.find_first_not_of(" \t"));
            }
        } else if (line.find("DIMENSION") == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                inst.n = std::stoul(line.substr(pos + 1));
            }
        } else if (line.find("EDGE_WEIGHT_TYPE") == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                edge_weight_type = line.substr(pos + 1);
                edge_weight_type.erase(0, edge_weight_type.find_first_not_of(" \t"));
                edge_weight_type.erase(edge_weight_type.find_last_not_of(" \t") + 1);
            }
        } else if (line == "NODE_COORD_SECTION") {
            reading_coords = true;
            reading_edge_weights = false;
            x.resize(inst.n);
            y.resize(inst.n);
        } else if (line == "EDGE_WEIGHT_SECTION") {
            reading_edge_weights = true;
            reading_coords = false;
            inst.distances.resize(inst.n * inst.n, 0.0f);
        } else if (line == "EOF" || line == "DISPLAY_DATA_SECTION") {
            reading_coords = false;
            reading_edge_weights = false;
        } else if (reading_coords) {
            std::istringstream iss(line);
            int id;
            float cx, cy;
            iss >> id >> cx >> cy;
            if (id >= 1 && id <= static_cast<int>(inst.n)) {
                x[id - 1] = cx;
                y[id - 1] = cy;
            }
        } else if (reading_edge_weights) {
            // Read explicit edge weights (full matrix, upper triangular, etc.)
            std::istringstream iss(line);
            float val;
            static std::uint32_t ew_idx = 0;
            while (iss >> val) {
                std::uint32_t r = ew_idx / inst.n;
                std::uint32_t c = ew_idx % inst.n;
                if (r < inst.n && c < inst.n) {
                    inst.distances[r * inst.n + c] = val;
                }
                ew_idx++;
            }
        }
    }

    // Compute distance matrix from coordinates if not explicitly given
    if (inst.distances.empty() && !x.empty()) {
        inst.distances.resize(inst.n * inst.n, 0.0f);

        for (std::uint32_t i = 0; i < inst.n; ++i) {
            for (std::uint32_t j = i + 1; j < inst.n; ++j) {
                float dx = x[i] - x[j];
                float dy = y[i] - y[j];
                float dist;

                if (edge_weight_type == "EUC_2D") {
                    dist = std::round(std::sqrt(dx * dx + dy * dy));
                } else if (edge_weight_type == "CEIL_2D") {
                    dist = std::ceil(std::sqrt(dx * dx + dy * dy));
                } else if (edge_weight_type == "ATT") {
                    float rij = std::sqrt((dx * dx + dy * dy) / 10.0f);
                    int tij = static_cast<int>(std::round(rij));
                    dist = (tij < rij) ? tij + 1.0f : static_cast<float>(tij);
                } else if (edge_weight_type == "GEO") {
                    // Geographic distance (TSPLIB convention)
                    auto toRad = [](float coord) {
                        int deg = static_cast<int>(coord);
                        float min = coord - deg;
                        return 3.141592653589793f * (deg + 5.0f * min / 3.0f) / 180.0f;
                    };
                    float lat_i = toRad(x[i]), lon_i = toRad(y[i]);
                    float lat_j = toRad(x[j]), lon_j = toRad(y[j]);
                    float RRR = 6378.388f;
                    float q1 = std::cos(lon_i - lon_j);
                    float q2 = std::cos(lat_i - lat_j);
                    float q3 = std::cos(lat_i + lat_j);
                    dist = static_cast<float>(
                        static_cast<int>(RRR * std::acos(0.5f * ((1.0f + q1) * q2 - (1.0f - q1) * q3)) + 1.0f));
                } else {
                    // Default: Euclidean
                    dist = std::sqrt(dx * dx + dy * dy);
                }

                inst.distances[i * inst.n + j] = dist;
                inst.distances[j * inst.n + i] = dist;
            }
        }
    }

    return inst;
}

} // namespace tsp
