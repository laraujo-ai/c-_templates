add_executable(test_tracker test_tracker.cpp ../../../../lib/catch2/catch2amalgamated.cpp)

target_link_libraries(test_tracker PRIVATE tracker)

target_include_directories(test_tracker PRIVATE ../../..)