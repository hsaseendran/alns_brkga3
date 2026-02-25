#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdlib>

namespace json_reader {

// Minimal JSON reader: reads a file, extracts a named integer or float array.
// No external dependencies. Only supports flat JSON with arrays of numbers.

inline std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Find the position of a JSON key's value (after the colon).
// Returns std::string::npos if not found.
inline std::size_t findKey(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    auto pos = json.find(pattern);
    if (pos == std::string::npos) return std::string::npos;
    pos = json.find(':', pos + pattern.size());
    if (pos == std::string::npos) return std::string::npos;
    return pos + 1;
}

// Extract an array of integers from JSON: "key": [1, 2, 3]
inline std::vector<int> readIntArray(const std::string& json, const std::string& key) {
    auto pos = findKey(json, key);
    if (pos == std::string::npos)
        throw std::runtime_error("JSON key not found: " + key);

    pos = json.find('[', pos);
    if (pos == std::string::npos)
        throw std::runtime_error("No array found for key: " + key);

    auto end = json.find(']', pos);
    if (end == std::string::npos)
        throw std::runtime_error("Unterminated array for key: " + key);

    std::string arr = json.substr(pos + 1, end - pos - 1);
    std::vector<int> result;
    std::istringstream iss(arr);
    std::string token;
    while (std::getline(iss, token, ',')) {
        // Skip whitespace
        auto start = token.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        result.push_back(std::atoi(token.c_str() + start));
    }
    return result;
}

// Extract an array of floats from JSON: "key": [1.0, 2.5, 3.7]
inline std::vector<float> readFloatArray(const std::string& json, const std::string& key) {
    auto pos = findKey(json, key);
    if (pos == std::string::npos)
        throw std::runtime_error("JSON key not found: " + key);

    pos = json.find('[', pos);
    if (pos == std::string::npos)
        throw std::runtime_error("No array found for key: " + key);

    auto end = json.find(']', pos);
    if (end == std::string::npos)
        throw std::runtime_error("Unterminated array for key: " + key);

    std::string arr = json.substr(pos + 1, end - pos - 1);
    std::vector<float> result;
    std::istringstream iss(arr);
    std::string token;
    while (std::getline(iss, token, ',')) {
        auto start = token.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        result.push_back(std::strtof(token.c_str() + start, nullptr));
    }
    return result;
}

// Extract a scalar float value from JSON: "key": 123.45
inline float readFloat(const std::string& json, const std::string& key) {
    auto pos = findKey(json, key);
    if (pos == std::string::npos)
        throw std::runtime_error("JSON key not found: " + key);

    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
           json[pos] == '\n' || json[pos] == '\r'))
        ++pos;

    return std::strtof(json.c_str() + pos, nullptr);
}

// Extract a nested array of arrays: "key": [[1,2],[3,4]]
// Returns vector of vector<int>
inline std::vector<std::vector<int>> readNestedIntArray(const std::string& json,
                                                         const std::string& key) {
    auto pos = findKey(json, key);
    if (pos == std::string::npos)
        throw std::runtime_error("JSON key not found: " + key);

    // Find outer array
    pos = json.find('[', pos);
    if (pos == std::string::npos)
        throw std::runtime_error("No array found for key: " + key);

    std::vector<std::vector<int>> result;
    std::size_t i = pos + 1;

    while (i < json.size()) {
        // Skip whitespace
        while (i < json.size() && (json[i] == ' ' || json[i] == '\t' ||
               json[i] == '\n' || json[i] == '\r' || json[i] == ','))
            ++i;

        if (i >= json.size() || json[i] == ']') break;

        if (json[i] == '[') {
            auto inner_end = json.find(']', i);
            if (inner_end == std::string::npos)
                throw std::runtime_error("Unterminated inner array for key: " + key);

            std::string inner = json.substr(i + 1, inner_end - i - 1);
            std::vector<int> row;
            std::istringstream iss(inner);
            std::string token;
            while (std::getline(iss, token, ',')) {
                auto start = token.find_first_not_of(" \t\n\r");
                if (start == std::string::npos) continue;
                row.push_back(std::atoi(token.c_str() + start));
            }
            result.push_back(std::move(row));
            i = inner_end + 1;
        } else {
            ++i;
        }
    }
    return result;
}

} // namespace json_reader
