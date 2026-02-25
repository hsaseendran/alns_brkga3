# TSP example is built from the top-level CMakeLists.txt
# This file exists for standalone build support if needed.

if(NOT TARGET brkga3)
    find_package(brkga3 REQUIRED)
endif()

add_executable(brkga3_tsp main.cu)
target_link_libraries(brkga3_tsp PRIVATE brkga3)
