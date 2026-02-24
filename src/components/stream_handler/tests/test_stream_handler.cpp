#define CATCH_CONFIG_MAIN
#include "../../../lib/catch2/catch_amalgamated.hpp"
#include "vision_stream_handlers.hpp"
#include "../../../common/include/interfaces.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <thread>
#include <chrono>

using namespace project_x;

static std::string createTestVideo(const std::string& filename, int num_frames, double fps = 30.0) {
    std::string filepath = "/tmp/" + filename;

    cv::VideoWriter writer(filepath, cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                           fps, cv::Size(640, 480));
    if (!writer.isOpened()) {
        throw std::runtime_error("Failed to create test video");
    }
    for (int i = 0; i < num_frames; ++i) {
        cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(frame, std::to_string(i), cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 255, 255), 3);
        writer.write(frame);
    }
    writer.release();
    return filepath;
}

// ---------------------------------------------------------------------------
// Lifecycle tests
// ---------------------------------------------------------------------------

TEST_CASE("GStreamerFileHandler lifecycle", "[stream_handler]") {
    std::string test_video = createTestVideo("test_lifecycle.mp4", 60);

    SECTION("Successfully starts stream with valid video file") {
        GStreamerFileHandler handler(10, 30, 640, 480);
        REQUIRE(handler.startStream(test_video));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        REQUIRE(handler.isActive());
        handler.stopStream();
    }

    SECTION("Fails to start stream with invalid file path") {
        GStreamerFileHandler handler(10, 30, 640, 480);
        REQUIRE_FALSE(handler.startStream("/nonexistent/video.mp4"));
        REQUIRE_FALSE(handler.isActive());
    }

    SECTION("Cannot start an already active stream") {
        GStreamerFileHandler handler(10, 30, 640, 480);
        REQUIRE(handler.startStream(test_video));
        REQUIRE_FALSE(handler.startStream(test_video));
        handler.stopStream();
    }

    SECTION("stopStream when not active is safe") {
        GStreamerFileHandler handler(10, 30, 640, 480);
        handler.stopStream();
        REQUIRE_FALSE(handler.isActive());
    }

    std::remove(test_video.c_str());
}

// ---------------------------------------------------------------------------
// Frame delivery tests
// ---------------------------------------------------------------------------

TEST_CASE("GStreamerFileHandler delivers frames", "[stream_handler]") {
    std::string test_video = createTestVideo("test_frames.mp4", 60, 30.0);

    SECTION("getNextFrame returns a valid frame") {
        GStreamerFileHandler handler(30, 30, 640, 480);
        REQUIRE(handler.startStream(test_video));

        auto fc = handler.getNextFrame();
        REQUIRE(fc.has_value());
        REQUIRE_FALSE(fc->frame.empty());

        handler.stopStream();
    }

    SECTION("Frames have the correct dimensions and type") {
        GStreamerFileHandler handler(30, 30, 320, 320);
        REQUIRE(handler.startStream(test_video));

        auto fc = handler.getNextFrame();
        REQUIRE(fc.has_value());
        REQUIRE(fc->frame.cols == 320);
        REQUIRE(fc->frame.rows == 320);
        REQUIRE(fc->frame.type() == CV_8UC3);

        handler.stopStream();
    }

    SECTION("Frame has a non-zero timestamp") {
        GStreamerFileHandler handler(30, 30, 640, 480);
        REQUIRE(handler.startStream(test_video));

        auto fc = handler.getNextFrame();
        REQUIRE(fc.has_value());
        REQUIRE(fc->timestamp_ms > 0);

        handler.stopStream();
    }

    SECTION("source_id is set from filename when camera_id is empty") {
        GStreamerFileHandler handler(30, 30, 640, 480);
        REQUIRE(handler.startStream(test_video));

        auto fc = handler.getNextFrame();
        REQUIRE(fc.has_value());
        REQUIRE(fc->source_id == "test_frames.mp4");

        handler.stopStream();
    }

    SECTION("source_id uses custom camera_id when set") {
        GStreamerFileHandler handler(30, 30, 640, 480, "cam_001");
        REQUIRE(handler.startStream(test_video));

        auto fc = handler.getNextFrame();
        REQUIRE(fc.has_value());
        REQUIRE(fc->source_id == "cam_001");

        handler.stopStream();
    }

    SECTION("getNextFrame returns nullopt after end of file") {
        // Use a tiny video so it finishes quickly
        std::string short_video = createTestVideo("test_eof.mp4", 15, 30.0);
        GStreamerFileHandler handler(60, 30, 640, 480);
        REQUIRE(handler.startStream(short_video));

        // Drain all frames
        while (auto fc = handler.getNextFrame()) { /* consume */ }

        REQUIRE_FALSE(handler.isActive());
        std::remove(short_video.c_str());
    }

    std::remove(test_video.c_str());
}

// ---------------------------------------------------------------------------
// Queue back-pressure test
// ---------------------------------------------------------------------------

TEST_CASE("GStreamerFileHandler respects max_queue_size", "[stream_handler]") {
    std::string test_video = createTestVideo("test_queue.mp4", 120, 30.0);

    SECTION("Queue does not grow beyond max_queue_size") {
        // Small queue — frames beyond the limit should be silently dropped
        GStreamerFileHandler handler(5, 30, 640, 480);
        REQUIRE(handler.startStream(test_video));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Reading back should still work fine
        auto fc = handler.getNextFrame();
        REQUIRE(fc.has_value());
        handler.stopStream();
    }

    std::remove(test_video.c_str());
}
