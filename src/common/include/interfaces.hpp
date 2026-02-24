#pragma once

#include <optional>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace project_x {

// ---------------------------------------------------------------------------
// Stream codec selection
// ---------------------------------------------------------------------------
enum class StreamCodec { H264, H265 };

// ---------------------------------------------------------------------------
// Detection result from any object detector
// ---------------------------------------------------------------------------
struct Detection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

// ---------------------------------------------------------------------------
// Single decoded frame + metadata from a stream handler
// ---------------------------------------------------------------------------
struct FrameContainer {
    cv::Mat frame;
    uint64_t timestamp_ms{0};
    std::string source_id;
};

// ---------------------------------------------------------------------------
// Base interface for all object tracklets
// ---------------------------------------------------------------------------
class BaseTracklet {
public:
    virtual ~BaseTracklet() = default;
    virtual void update(const Eigen::Vector4d& bbox, double conf) = 0;
    virtual Eigen::Vector4d predict() = 0;
    virtual Eigen::Vector4d get_state() const = 0;
};

// ---------------------------------------------------------------------------
// Base interface for all stream handlers
// ---------------------------------------------------------------------------
class IStreamHandler {
public:
    virtual ~IStreamHandler() = default;

    /**
     * @brief Starts the stream from the given source (URL or file path).
     * @return True if the stream started successfully.
     */
    virtual bool startStream(const std::string& source) = 0;

    /**
     * @brief Stops the stream and releases all resources.
     */
    virtual void stopStream() = 0;

    /**
     * @brief Blocks until the next frame is available and returns it.
     * @return FrameContainer, or std::nullopt when the stream has ended.
     */
    virtual std::optional<FrameContainer> getNextFrame() = 0;

    /**
     * @brief Returns true while the stream is running.
     */
    virtual bool isActive() const = 0;
};

} // namespace project_x
