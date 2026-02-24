cmake_minimum_required(VERSION 3.10)
project(SORTTracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(sort_tracker SHARED src/sort_tracker.cpp)
target_include_directories(sort_tracker PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sort_tracker Eigen3::Eigen nlohmann_json::nlohmann_json)

