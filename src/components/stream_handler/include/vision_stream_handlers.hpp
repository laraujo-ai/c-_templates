#pragma once

#include "../../../common/include/interfaces.hpp"
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>

namespace project_x {

/**
 * @brief RTSP stream handler using GStreamer with NVIDIA hardware acceleration.
 *
 * Decodes frames from an RTSP source and pushes them into a bounded queue.
 * Reconnection is handled automatically up to MAX_RECONNECT_ATTEMPTS with
 * exponential back-off.
 */
class GStreamerRTSPHandler : public IStreamHandler {
private:
    std::string rtsp_url_;
    std::string camera_id_;
    std::atomic<bool> is_active_;
    std::thread capture_thread_;
    std::queue<FrameContainer> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    int max_queue_size_;
    int target_fps_;
    int target_width_;
    int target_height_;
    StreamCodec stream_codec_;

    GstElement* pipeline_;
    GstElement* appsink_;
    GstBus* bus_;
    GMainLoop* main_loop_;
    std::thread gst_thread_;

    std::atomic<bool> lost_connection_{false};
    std::atomic<int> reconnect_attempts_{0};
    std::atomic<bool> should_reconnect_{true};
    std::chrono::steady_clock::time_point last_frame_time_;
    std::mutex frame_time_mutex_;

    int MAX_RECONNECT_ATTEMPTS_;
    int FRAME_TIMEOUT_MS_;

    void gstreamerLoop();
    void captureLoop();
    bool initializeGStreamer();
    void cleanupGStreamer();
    void stopPipeline();
    bool reconnect();

    std::string buildNvidiaHardwarePipeline() const;
    std::string getDepayElement() const;
    std::string getParserElement() const;

    /**
     * @brief Wraps a decoded frame in a FrameContainer and enqueues it.
     */
    void processFrame(cv::Mat&& frame, uint64_t timestamp_ms);

    static GstFlowReturn onNewSample(GstElement* appsink, gpointer user_data);
    static gboolean onBusMessage(GstBus* bus, GstMessage* message, gpointer user_data);
    void handlePipelineError(GstMessage* message);
    void handlePipelineWarning(GstMessage* message);
    void handlePipelineInfo(GstMessage* message);

public:
    /**
     * @param max_queue_size          Maximum frames buffered before the oldest is dropped.
     * @param target_fps              Frames per second to decode.
     * @param target_width            Output frame width.
     * @param target_height           Output frame height.
     * @param codec                   Stream codec (H264 / H265).
     * @param camera_id               Source identifier; auto-generated from URL if empty.
     * @param MAX_RECONNECT_ATTEMPTS  Max reconnection attempts before giving up.
     * @param FRAME_TIMEOUT_MS        Milliseconds without a frame before reconnect.
     */
    GStreamerRTSPHandler(int max_queue_size = 10,
                         int target_fps = 30, int target_width = 640, int target_height = 640,
                         StreamCodec codec = StreamCodec::H264,
                         const std::string& camera_id = "",
                         int MAX_RECONNECT_ATTEMPTS = 5, int FRAME_TIMEOUT_MS = 5000);

    ~GStreamerRTSPHandler();

    bool startStream(const std::string& rtsp_url) override;
    void stopStream() override;
    std::optional<FrameContainer> getNextFrame() override;
    bool isActive() const override;

    void setCameraId(const std::string& camera_id) { camera_id_ = camera_id; }
    void setMaxQueueSize(int size) { max_queue_size_ = size; }
    void setTargetResolution(int width, int height) { target_width_ = width; target_height_ = height; }
    void setTargetFPS(int fps) { target_fps_ = fps; }
    void setStreamCodec(StreamCodec codec) { stream_codec_ = codec; }
};

/**
 * @brief File stream handler using GStreamer with NVIDIA hardware acceleration.
 *
 * Decodes frames from a local video file and pushes them into a bounded queue.
 * The handler becomes inactive automatically when the end of the file is reached.
 */
class GStreamerFileHandler : public IStreamHandler {
private:
    std::string file_path_;
    std::string camera_id_;
    std::atomic<bool> is_active_;
    std::thread capture_thread_;
    std::queue<FrameContainer> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    int max_queue_size_;
    int target_fps_;
    int target_width_;
    int target_height_;

    GstElement* pipeline_;
    GstElement* appsink_;
    GstBus* bus_;
    GMainLoop* main_loop_;
    std::thread gst_thread_;

    uint64_t file_start_time_ms_;

    void gstreamerLoop();
    void captureLoop();
    bool initializeGStreamer();
    void cleanupGStreamer();
    std::string buildNvidiaHardwarePipeline() const;

    void processFrame(cv::Mat&& frame, uint64_t timestamp_ms);

    static GstFlowReturn onNewSample(GstElement* appsink, gpointer user_data);
    static gboolean onBusMessage(GstBus* bus, GstMessage* message, gpointer user_data);
    static gboolean onEndOfStream(GstBus* bus, GstMessage* message, gpointer user_data);
    void handlePipelineError(GstMessage* message);
    void handlePipelineWarning(GstMessage* message);
    void handlePipelineInfo(GstMessage* message);

public:
    /**
     * @param max_queue_size  Maximum frames buffered before the oldest is dropped.
     * @param target_fps      Frames per second to decode.
     * @param target_width    Output frame width.
     * @param target_height   Output frame height.
     * @param camera_id       Source identifier; auto-generated from filename if empty.
     */
    GStreamerFileHandler(int max_queue_size = 10,
                         int target_fps = 30, int target_width = 640, int target_height = 640,
                         const std::string& camera_id = "");

    ~GStreamerFileHandler();

    bool startStream(const std::string& file_path) override;
    void stopStream() override;
    std::optional<FrameContainer> getNextFrame() override;
    bool isActive() const override;

    void setCameraId(const std::string& camera_id) { camera_id_ = camera_id; }
    void setMaxQueueSize(int size) { max_queue_size_ = size; }
    void setTargetResolution(int width, int height) { target_width_ = width; target_height_ = height; }
    void setTargetFPS(int fps) { target_fps_ = fps; }
};

} // namespace project_x
