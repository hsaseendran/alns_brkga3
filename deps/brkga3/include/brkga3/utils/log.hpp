#pragma once

#include <cstdio>

namespace brkga3 {
namespace log {

enum class Level : int {
    NONE  = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    DEBUG = 4
};

// Set default log level at compile time via -DBRKGA3_LOG_LEVEL=N
#ifndef BRKGA3_LOG_LEVEL
#define BRKGA3_LOG_LEVEL 2  // WARN
#endif

inline constexpr Level kLogLevel = static_cast<Level>(BRKGA3_LOG_LEVEL);

template <typename... Args>
inline void error([[maybe_unused]] const char* fmt, [[maybe_unused]] Args... args) {
    if constexpr (kLogLevel >= Level::ERROR) {
        std::fprintf(stderr, "[BRKGA3 ERROR] ");
        std::fprintf(stderr, fmt, args...);
        std::fprintf(stderr, "\n");
    }
}

template <typename... Args>
inline void warn([[maybe_unused]] const char* fmt, [[maybe_unused]] Args... args) {
    if constexpr (kLogLevel >= Level::WARN) {
        std::fprintf(stderr, "[BRKGA3 WARN]  ");
        std::fprintf(stderr, fmt, args...);
        std::fprintf(stderr, "\n");
    }
}

template <typename... Args>
inline void info([[maybe_unused]] const char* fmt, [[maybe_unused]] Args... args) {
    if constexpr (kLogLevel >= Level::INFO) {
        std::fprintf(stderr, "[BRKGA3 INFO]  ");
        std::fprintf(stderr, fmt, args...);
        std::fprintf(stderr, "\n");
    }
}

template <typename... Args>
inline void debug([[maybe_unused]] const char* fmt, [[maybe_unused]] Args... args) {
    if constexpr (kLogLevel >= Level::DEBUG) {
        std::fprintf(stderr, "[BRKGA3 DEBUG] ");
        std::fprintf(stderr, fmt, args...);
        std::fprintf(stderr, "\n");
    }
}

} // namespace log
} // namespace brkga3
