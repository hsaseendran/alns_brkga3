#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>

namespace scp {

struct ScpInstance {
    std::uint32_t n_elements = 0;   // universe size |U|
    std::uint32_t n_sets = 0;       // number of sets

    std::vector<float> costs;       // [n_sets] cost of each set

    // CSR representation of set membership:
    //   set_covers[set_offsets[s] .. set_offsets[s+1]) = elements covered by set s
    std::vector<std::uint32_t> set_covers;
    std::vector<std::uint32_t> set_offsets;

    std::string name;
    float optimal = -1.0f;
};

// Parse OR-Library SCP format
// Format:
//   Line 1: n_elements n_sets
//   Next lines: cost of each set (may span multiple lines)
//   Then for each element (1..n_elements):
//     number_of_sets_covering_element
//     set indices (1-indexed)
//
// We convert from element-major (OR-Library) to set-major (CSR) for GPU.
inline ScpInstance loadORLibrary(const std::string& filepath) {
    ScpInstance inst;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open SCP file: " + filepath);
    }

    // Extract name from filepath
    auto pos = filepath.find_last_of("/\\");
    inst.name = (pos != std::string::npos) ? filepath.substr(pos + 1) : filepath;

    // Read dimensions
    file >> inst.n_elements >> inst.n_sets;

    // Read costs
    inst.costs.resize(inst.n_sets);
    for (std::uint32_t s = 0; s < inst.n_sets; ++s) {
        file >> inst.costs[s];
    }

    // Read element-to-set mapping (OR-Library format)
    // We build the inverse: set-to-element (CSR) for the GPU decoder.
    std::vector<std::vector<std::uint32_t>> set_to_elements(inst.n_sets);

    for (std::uint32_t e = 0; e < inst.n_elements; ++e) {
        std::uint32_t num_covering;
        file >> num_covering;
        for (std::uint32_t i = 0; i < num_covering; ++i) {
            std::uint32_t s;
            file >> s;
            s--;  // Convert from 1-indexed to 0-indexed
            set_to_elements[s].push_back(e);
        }
    }

    // Build CSR
    inst.set_offsets.resize(inst.n_sets + 1);
    inst.set_offsets[0] = 0;
    for (std::uint32_t s = 0; s < inst.n_sets; ++s) {
        inst.set_offsets[s + 1] = inst.set_offsets[s]
                                  + static_cast<std::uint32_t>(set_to_elements[s].size());
    }

    inst.set_covers.resize(inst.set_offsets[inst.n_sets]);
    for (std::uint32_t s = 0; s < inst.n_sets; ++s) {
        std::copy(set_to_elements[s].begin(), set_to_elements[s].end(),
                  inst.set_covers.begin() + inst.set_offsets[s]);
    }

    return inst;
}

} // namespace scp
