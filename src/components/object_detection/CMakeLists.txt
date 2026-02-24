add_library(object_detection SHARED
    src/yolox_detector.cpp
)

target_link_libraries(object_detection PUBLIC
    common
    ${OpenCV_LIBS}
    onnxruntime
    pthread
    ${CUDA_LIBRARIES}
)

target_include_directories(object_detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
)

option(BUILD_OBJECT_DETECTION_TEST "Build object detection tests" ON)
if(BUILD_OBJECT_DETECTION_TEST)
    add_subdirectory(tests)
endif()
