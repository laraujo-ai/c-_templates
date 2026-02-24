cmake_minimum_required(VERSION 3.10)
project(CppVideoPipeline)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)

find_package(CUDA QUIET)

if(NOT DEFINED ONNXRUNTIME_LIB OR NOT DEFINED ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR
        "ONNXRUNTIME_LIB and ONNXRUNTIME_INCLUDE_DIR must be set.\n"
        "Example: cmake .. -DONNXRUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so "
        "-DONNXRUNTIME_INCLUDE_DIR=/opt/onnxruntime/include")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

set(SPDLOG_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lib)

include_directories(${SPDLOG_INCLUDE_DIR})

add_subdirectory(src/common)
add_subdirectory(src/components)
