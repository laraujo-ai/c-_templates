add_library(stream_handler SHARED src/vision_stream_handlers.cpp)

target_link_libraries(stream_handler PUBLIC
    common
    ${GSTREAMER_LIBRARIES}
    ${GST_APP_LIBRARIES}
    pthread
)

target_include_directories(stream_handler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
)

target_compile_options(stream_handler PRIVATE
    ${GSTREAMER_CFLAGS_OTHER}
    ${GST_APP_CFLAGS_OTHER}
)

option(BUILD_STREAM_HANDLER_TESTS "Build stream handler tests" ON)
if(BUILD_STREAM_HANDLER_TESTS)
    add_subdirectory(tests)
endif()
