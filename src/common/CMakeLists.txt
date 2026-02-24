add_library(common SHARED
    src/benchmark_metrics.cpp
)

target_include_directories(common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SPDLOG_INCLUDE_DIR}
)

target_link_libraries(common PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
)
