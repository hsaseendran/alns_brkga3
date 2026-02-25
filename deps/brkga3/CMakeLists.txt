cmake_minimum_required(VERSION 3.18)
project(brkga3 LANGUAGES CXX CUDA)

# ============================================================================
# Standards
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture; override with -DCMAKE_CUDA_ARCHITECTURES=XX
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ============================================================================
# CUDA Toolkit (provides CUB, cuRAND)
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Core library
# ============================================================================
add_library(brkga3
    src/config.cpp
    src/brkga.cu
    src/population.cu
    src/island_manager.cu
    src/kernels/evolve.cu
    src/kernels/transpose.cu
    src/kernels/sort.cu
)

target_include_directories(brkga3
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(brkga3
    PUBLIC
        CUDA::cudart
        CUDA::curand
)

set_target_properties(brkga3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Optional: verbose build info
target_compile_options(brkga3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-Wall,-Wextra
    >
)

# Debug/Release flags
target_compile_options(brkga3 PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        -G -lineinfo
    >
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        -O3 --use_fast_math
    >
)

# Log level
target_compile_definitions(brkga3 PUBLIC
    $<$<CONFIG:Debug>:BRKGA3_LOG_LEVEL=4>
    $<$<CONFIG:Release>:BRKGA3_LOG_LEVEL=2>
)

# ============================================================================
# Examples
# ============================================================================
option(BRKGA3_BUILD_EXAMPLES "Build example applications" ON)

if(BRKGA3_BUILD_EXAMPLES)
    # Common include path for shared example utilities (json_reader, etc.)
    set(EXAMPLES_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples/common)

    # TSP
    add_executable(brkga3_tsp
        examples/tsp/main.cu
    )
    target_include_directories(brkga3_tsp PRIVATE ${EXAMPLES_COMMON_DIR})
    target_link_libraries(brkga3_tsp PRIVATE brkga3)

    # SCP
    add_executable(brkga3_scp
        examples/scp/main.cu
    )
    target_include_directories(brkga3_scp PRIVATE ${EXAMPLES_COMMON_DIR})
    target_link_libraries(brkga3_scp PRIVATE brkga3)

    # CVRP
    add_executable(brkga3_cvrp
        examples/cvrp/main.cu
    )
    target_include_directories(brkga3_cvrp PRIVATE ${EXAMPLES_COMMON_DIR})
    target_link_libraries(brkga3_cvrp PRIVATE brkga3)
endif()

# ============================================================================
# Tests (optional, requires GTest)
# ============================================================================
option(BRKGA3_BUILD_TESTS "Build unit tests" OFF)

if(BRKGA3_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()

# ============================================================================
# Benchmarks (optional)
# ============================================================================
option(BRKGA3_BUILD_BENCHMARKS "Build benchmarks" OFF)

if(BRKGA3_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
