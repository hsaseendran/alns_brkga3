cmake_minimum_required(VERSION 3.18)
project(alns_brkga LANGUAGES CXX CUDA)

# ============================================================================
# Standards
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

# ============================================================================
# External project paths
# ============================================================================
set(ALNS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/alns"
    CACHE PATH "Path to ALNS framework root")
set(BRKGA3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/brkga3"
    CACHE PATH "Path to BRKGA3 framework root")

# Build BRKGA3 core library (skip examples/tests/benchmarks)
set(BRKGA3_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BRKGA3_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BRKGA3_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
add_subdirectory(${BRKGA3_DIR} ${CMAKE_BINARY_DIR}/brkga3_build)

# ============================================================================
# Main executable
# ============================================================================
add_executable(alns_brkga
    src/main.cu
    src/tsp_solver.cu
    src/cvrp_solver.cu
    src/vrprpd_solver.cu
)

target_include_directories(alns_brkga PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ALNS_DIR}/include                     # ALNS framework headers
    ${ALNS_DIR}/examples/tsp                # TSPConfig
    ${ALNS_DIR}/examples/vrp_rpd            # VRPRPDConfig
    ${BRKGA3_DIR}/include                   # BRKGA3 public headers
    ${BRKGA3_DIR}/examples/tsp              # TSP decoder + instance loader
    ${BRKGA3_DIR}/examples/cvrp             # CVRP decoder + instance loader
    ${BRKGA3_DIR}/examples/common           # json_reader.hpp
)

target_link_libraries(alns_brkga PRIVATE
    brkga3
    CUDA::cudart
    CUDA::curand
)

set_target_properties(alns_brkga PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(alns_brkga PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        -O3
        -Xcompiler=-Wall
    >
)
