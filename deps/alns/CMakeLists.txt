cmake_minimum_required(VERSION 3.18)
project(ALNS_Framework LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 89 90)

# Framework is header-only (templates)
add_library(alns_framework INTERFACE)
target_include_directories(alns_framework INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ============================================================================
# TSP Example
# ============================================================================
option(BUILD_TSP "Build TSP example solver" ON)
if(BUILD_TSP)
    add_executable(alns_tsp
        examples/tsp/tsp_main.cu)
    target_link_libraries(alns_tsp alns_framework)
    set_target_properties(alns_tsp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(alns_tsp PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>)
endif()

# ============================================================================
# VRP-RPD Example
# ============================================================================
option(BUILD_VRP_RPD "Build VRP-RPD example solver" ON)
if(BUILD_VRP_RPD)
    add_executable(alns_vrp_rpd
        examples/vrp_rpd/vrp_rpd_main.cu)
    target_link_libraries(alns_vrp_rpd alns_framework)
    set_target_properties(alns_vrp_rpd PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(alns_vrp_rpd PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>)
endif()
