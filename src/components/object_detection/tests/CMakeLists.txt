add_executable(test_object_detection
    test_object_detection.cpp
    ../../../../lib/catch2/catch_amalgamated.cpp
)

target_include_directories(test_object_detection PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/include
    ${CMAKE_SOURCE_DIR}/src/components/object_detection
    ${CMAKE_SOURCE_DIR}/lib/catch2
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_object_detection
    object_detection
    ${OpenCV_LIBS}
)

enable_testing()
add_test(NAME ObjectDetectionTest COMMAND test_object_detection)
