add_executable(test_stream_handler
    test_stream_handler.cpp
    ../../../../lib/catch2/catch_amalgamated.cpp
)

target_include_directories(test_stream_handler PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common/include
    ${CMAKE_SOURCE_DIR}/src/components/stream_handler
    ${CMAKE_SOURCE_DIR}/lib/catch2
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_stream_handler
    stream_handler
    ${OpenCV_LIBS}
)

enable_testing()
add_test(NAME StreamHandlerTest COMMAND test_stream_handler)
