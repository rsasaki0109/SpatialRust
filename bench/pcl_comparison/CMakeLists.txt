cmake_minimum_required(VERSION 3.20)

project(spatialrust_pcl_bench LANGUAGES CXX)

find_package(PCL CONFIG REQUIRED COMPONENTS common io filters features search kdtree octree)

add_executable(pcl_bench pcl_bench.cpp)
target_compile_features(pcl_bench PRIVATE cxx_std_17)
target_include_directories(pcl_bench PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(pcl_bench PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(pcl_bench PRIVATE ${PCL_LIBRARIES})
