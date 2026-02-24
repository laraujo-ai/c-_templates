#include "../include/vision_stream_handlers.hpp"
#include "../../../common/include/logger.hpp"
#include <iostream>
#include <chrono>
#include <sstream>
#include <mutex>

namespace {
    std::mutex anchor_mutex;
    bool anchor_set = false;
    std::chrono::time_point<std::chrono::system_clock> anchor_system_time;
    uint64_t anchor_pts = 0;
}

GstPadProbeReturn timestamp_anchor_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);

    if (buffer->pts != GST_CLOCK_TIME_NONE) {
        std::lock_guard<std::mutex> lock(anchor_mutex);
        bool is_reset = false;
        if (anchor_set && buffer->pts < anchor_pts) {
            uint64_t drop = anchor_pts - buffer->pts;
            if (drop > GST_SECOND) {
                is_reset = true;
            }
        }
        if (!anchor_set || is_reset) {
            anchor_system_time = std::chrono::system_clock::now();
            anchor_pts = buffer->pts;
            anchor_set = true;

            LOG_INFO("ANCHOR SET/RESET! Base PTS: {}, UTC Time (ms): {}",
                     GST_TIME_AS_MSECONDS(anchor_pts),
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         anchor_system_time.time_since_epoch()).count());
        }
    }

    return GST_PAD_PROBE_OK;
}

namespace project_x {

// ---------------------------------------------------------------------------
// GStreamerRTSPHandler
// ---------------------------------------------------------------------------

GStreamerRTSPHandler::GStreamerRTSPHandler(int max_queue_size,
                                           int target_fps, int target_width, int target_height,
                                           StreamCodec codec, const std::string& camera_id,
                                           int MAX_RECONNECT_ATTEMPTS, int FRAME_TIMEOUT_MS)
    : is_active_(false), max_queue_size_(max_queue_size),
      target_fps_(target_fps), target_width_(target_width), target_height_(target_height),
      stream_codec_(codec), camera_id_(camera_id),
      MAX_RECONNECT_ATTEMPTS_(MAX_RECONNECT_ATTEMPTS), FRAME_TIMEOUT_MS_(FRAME_TIMEOUT_MS),
      pipeline_(nullptr), appsink_(nullptr), bus_(nullptr), main_loop_(nullptr) {

    last_frame_time_ = std::chrono::steady_clock::now();
}

GStreamerRTSPHandler::~GStreamerRTSPHandler() {
    stopStream();
}

bool GStreamerRTSPHandler::startStream(const std::string& rtsp_url) {
    if (is_active_) return false;

    rtsp_url_ = rtsp_url;
    if (camera_id_.empty()) {
        camera_id_ = "rtsp_camera_" + std::to_string(std::hash<std::string>{}(rtsp_url) % 10000);
    }

    if (!initializeGStreamer()) {
        LOG_ERROR("GStreamer initialization failed");
        return false;
    }

    lost_connection_ = false;
    reconnect_attempts_ = 0;
    should_reconnect_ = true;
    {
        std::lock_guard<std::mutex> lock(frame_time_mutex_);
        last_frame_time_ = std::chrono::steady_clock::now();
    }

    is_active_ = true;
    gst_thread_ = std::thread(&GStreamerRTSPHandler::gstreamerLoop, this);
    capture_thread_ = std::thread(&GStreamerRTSPHandler::captureLoop, this);

    return true;
}

void GStreamerRTSPHandler::stopStream() {
    if (!is_active_) return;

    is_active_ = false;
    if (main_loop_ && g_main_loop_is_running(main_loop_)) {
        g_main_loop_quit(main_loop_);
    }
    if (gst_thread_.joinable()) gst_thread_.join();
    if (capture_thread_.joinable()) capture_thread_.join();

    cleanupGStreamer();
    queue_cv_.notify_all();
}

std::optional<FrameContainer> GStreamerRTSPHandler::getNextFrame() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return !frame_queue_.empty() || !is_active_; });

    if (frame_queue_.empty()) return std::nullopt;

    FrameContainer fc = std::move(frame_queue_.front());
    frame_queue_.pop();
    return fc;
}

bool GStreamerRTSPHandler::isActive() const {
    return is_active_;
}

std::string GStreamerRTSPHandler::buildNvidiaHardwarePipeline() const {
    std::stringstream ss;
    ss << "rtspsrc location=\"" << rtsp_url_ << "\" "
       << "latency=50 protocols=tcp ntp-sync=true ! "
       << getDepayElement() << " name=depay ! "
       << getParserElement() << " ! "
       << "nvv4l2decoder enable-max-performance=1 disable-dpb=true ! "
       << "nvvidconv ! "
       << "videorate ! "
       << "video/x-raw, width=" << target_width_ << ", height=" << target_height_
       << ", framerate=" << target_fps_ << "/1 ! "
       << "videoconvert ! "
       << "video/x-raw, format=BGR ! "
       << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true";
    return ss.str();
}

std::string GStreamerRTSPHandler::getDepayElement() const {
    switch (stream_codec_) {
        case StreamCodec::H265: return "rtph265depay";
        default:                return "rtph264depay";
    }
}

std::string GStreamerRTSPHandler::getParserElement() const {
    switch (stream_codec_) {
        case StreamCodec::H265: return "h265parse";
        default:                return "h264parse";
    }
}

bool GStreamerRTSPHandler::initializeGStreamer() {
    GError* error = nullptr;
    std::string pipeline_str = buildNvidiaHardwarePipeline();

    LOG_INFO("Initializing GStreamer pipeline for camera: {}", camera_id_);
    LOG_INFO("Pipeline: {}", pipeline_str);

    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline_ || error) {
        LOG_ERROR("GStreamer pipeline creation failed: {}", error ? error->message : "Unknown");
        if (error) g_error_free(error);
        return false;
    }

    GstElement *depay = gst_bin_get_by_name(GST_BIN(pipeline_), "depay");
    if (depay) {
        GstPad *sink_pad = gst_element_get_static_pad(depay, "sink");
        if (sink_pad) {
            {
                std::lock_guard<std::mutex> lock(anchor_mutex);
                anchor_set = false;
                anchor_pts = 0;
            }
            gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER, timestamp_anchor_probe, NULL, NULL);
            gst_object_unref(sink_pad);
        }
        gst_object_unref(depay);
    }

    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
        LOG_ERROR("Appsink element not found");
        return false;
    }

    g_object_set(appsink_, "emit-signals", TRUE, "sync", FALSE, "max-buffers", 2, "drop", TRUE, nullptr);
    g_signal_connect(appsink_, "new-sample", G_CALLBACK(onNewSample), this);

    bus_ = gst_element_get_bus(pipeline_);
    gst_bus_add_signal_watch(bus_);
    g_signal_connect(bus_, "message::error",         G_CALLBACK(onBusMessage), this);
    g_signal_connect(bus_, "message::warning",       G_CALLBACK(onBusMessage), this);
    g_signal_connect(bus_, "message::info",          G_CALLBACK(onBusMessage), this);
    g_signal_connect(bus_, "message::state-changed", G_CALLBACK(onBusMessage), this);

    main_loop_ = g_main_loop_new(nullptr, FALSE);
    return true;
}

void GStreamerRTSPHandler::cleanupGStreamer() {
    if (bus_) {
        gst_bus_remove_signal_watch(bus_);
        gst_object_unref(bus_);
        bus_ = nullptr;
    }
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    if (appsink_) {
        gst_object_unref(appsink_);
        appsink_ = nullptr;
    }
    if (main_loop_) {
        g_main_loop_unref(main_loop_);
        main_loop_ = nullptr;
    }
}

void GStreamerRTSPHandler::stopPipeline() {
    if (main_loop_ && g_main_loop_is_running(main_loop_)) {
        g_main_loop_quit(main_loop_);
    }
    cleanupGStreamer();
}

bool GStreamerRTSPHandler::reconnect() {
    int attempts = reconnect_attempts_.load();
    int backoff_seconds = 1 << std::min(attempts, 4);

    LOG_INFO("Attempting reconnection for camera '{}' (attempt {}/{}, waiting {}s)",
             camera_id_, attempts + 1, MAX_RECONNECT_ATTEMPTS_, backoff_seconds);

    std::this_thread::sleep_for(std::chrono::seconds(backoff_seconds));

    stopPipeline();
    if (gst_thread_.joinable()) gst_thread_.join();

    if (!initializeGStreamer()) {
        LOG_ERROR("Failed to initialize GStreamer during reconnection for camera '{}'", camera_id_);
        return false;
    }

    gst_thread_ = std::thread(&GStreamerRTSPHandler::gstreamerLoop, this);
    {
        std::lock_guard<std::mutex> lock(frame_time_mutex_);
        last_frame_time_ = std::chrono::steady_clock::now();
    }
    LOG_INFO("Successfully reconnected camera '{}'", camera_id_);
    return true;
}

void GStreamerRTSPHandler::gstreamerLoop() {
    if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (main_loop_) g_main_loop_run(main_loop_);
}

void GStreamerRTSPHandler::captureLoop() {
    while (is_active_ && should_reconnect_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (lost_connection_) {
            if (reconnect()) {
                lost_connection_ = false;
                reconnect_attempts_ = 0;
            } else {
                reconnect_attempts_++;
                if (reconnect_attempts_ >= MAX_RECONNECT_ATTEMPTS_) {
                    LOG_ERROR("Camera '{}' failed to reconnect. Marking inactive.", camera_id_);
                    is_active_ = false;
                    queue_cv_.notify_all();
                    break;
                }
            }
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(frame_time_mutex_);
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - last_frame_time_).count();
            if (elapsed > FRAME_TIMEOUT_MS_) {
                LOG_WARN("Camera '{}' timeout ({}ms), connection lost", camera_id_, elapsed);
                lost_connection_ = true;
            }
        }
    }
}

void GStreamerRTSPHandler::processFrame(cv::Mat&& frame, uint64_t timestamp_ms) {
    if (!is_active_) return;

    {
        std::lock_guard<std::mutex> lock(frame_time_mutex_);
        last_frame_time_ = std::chrono::steady_clock::now();
    }

    FrameContainer fc;
    fc.frame = std::move(frame);
    fc.timestamp_ms = timestamp_ms;
    fc.source_id = camera_id_;

    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (static_cast<int>(frame_queue_.size()) < max_queue_size_) {
        frame_queue_.push(std::move(fc));
        queue_cv_.notify_one();
    }
}

GstFlowReturn GStreamerRTSPHandler::onNewSample(GstElement* appsink, gpointer user_data) {
    GStreamerRTSPHandler* handler = static_cast<GStreamerRTSPHandler*>(user_data);

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    if (!sample) return GST_FLOW_ERROR;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps*   caps   = gst_sample_get_caps(sample);

    if (buffer && caps) {
        uint64_t absolute_timestamp_ms = 0;
        GstReferenceTimestampMeta *meta = gst_buffer_get_reference_timestamp_meta(buffer, NULL);
        if (meta) {
            absolute_timestamp_ms = GST_TIME_AS_MSECONDS(meta->timestamp);
        }

        if (absolute_timestamp_ms == 0) {
            bool has_anchor = false;
            uint64_t current_anchor_pts = 0;
            std::chrono::time_point<std::chrono::system_clock> current_anchor_time;
            {
                std::lock_guard<std::mutex> lock(anchor_mutex);
                if (anchor_set) {
                    has_anchor = true;
                    current_anchor_pts = anchor_pts;
                    current_anchor_time = anchor_system_time;
                }
            }

            if (has_anchor && buffer->pts != GST_CLOCK_TIME_NONE && buffer->pts >= current_anchor_pts) {
                uint64_t pts_diff_ms = GST_TIME_AS_MSECONDS(buffer->pts - current_anchor_pts);
                auto anchor_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_anchor_time.time_since_epoch()).count();
                absolute_timestamp_ms = anchor_ms + pts_diff_ms;
            } else {
                absolute_timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            }
        }

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            GstStructure* structure = gst_caps_get_structure(caps, 0);
            int width, height;
            gst_structure_get_int(structure, "width", &width);
            gst_structure_get_int(structure, "height", &height);

            cv::Mat frame_view(height, width, CV_8UC3, map.data);
            cv::Mat frame_copy = frame_view.clone();
            gst_buffer_unmap(buffer, &map);

            handler->processFrame(std::move(frame_copy), absolute_timestamp_ms);
        }
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

gboolean GStreamerRTSPHandler::onBusMessage(GstBus* bus, GstMessage* message, gpointer user_data) {
    GStreamerRTSPHandler* handler = static_cast<GStreamerRTSPHandler*>(user_data);

    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR:
            handler->handlePipelineError(message);
            break;
        case GST_MESSAGE_WARNING:
            handler->handlePipelineWarning(message);
            break;
        case GST_MESSAGE_INFO:
            handler->handlePipelineInfo(message);
            break;
        case GST_MESSAGE_STATE_CHANGED: {
            if (GST_MESSAGE_SRC(message) == GST_OBJECT(handler->pipeline_)) {
                GstState old_state, new_state, pending_state;
                gst_message_parse_state_changed(message, &old_state, &new_state, &pending_state);
                LOG_INFO("Pipeline state changed from {} to {} (camera: {})",
                         gst_element_state_get_name(old_state),
                         gst_element_state_get_name(new_state),
                         handler->camera_id_);
            }
            break;
        }
        default:
            break;
    }
    return TRUE;
}

void GStreamerRTSPHandler::handlePipelineError(GstMessage* message) {
    GError* error = nullptr;
    gchar* debug_info = nullptr;
    gst_message_parse_error(message, &error, &debug_info);

    LOG_ERROR("GStreamer error [{}]: {} | {}",
              rtsp_url_,
              error ? error->message : "Unknown",
              debug_info ? debug_info : "");

    if (error) g_error_free(error);
    if (debug_info) g_free(debug_info);
    lost_connection_ = true;
}

void GStreamerRTSPHandler::handlePipelineWarning(GstMessage* message) {
    GError* warning = nullptr;
    gchar* debug_info = nullptr;
    gst_message_parse_warning(message, &warning, &debug_info);

    LOG_WARN("GStreamer warning: {}",  warning ? warning->message : "Unknown");

    if (warning) g_error_free(warning);
    if (debug_info) g_free(debug_info);
}

void GStreamerRTSPHandler::handlePipelineInfo(GstMessage* message) {
    GError* info = nullptr;
    gchar* debug_info = nullptr;
    gst_message_parse_info(message, &info, &debug_info);
    if (info) g_error_free(info);
    if (debug_info) g_free(debug_info);
}

// ---------------------------------------------------------------------------
// GStreamerFileHandler
// ---------------------------------------------------------------------------

GStreamerFileHandler::GStreamerFileHandler(int max_queue_size,
                                           int target_fps, int target_width, int target_height,
                                           const std::string& camera_id)
    : is_active_(false), max_queue_size_(max_queue_size),
      target_fps_(target_fps), target_width_(target_width), target_height_(target_height),
      camera_id_(camera_id),
      pipeline_(nullptr), appsink_(nullptr), bus_(nullptr), main_loop_(nullptr),
      file_start_time_ms_(0) {

    gst_init(nullptr, nullptr);
}

GStreamerFileHandler::~GStreamerFileHandler() {
    stopStream();
}

bool GStreamerFileHandler::startStream(const std::string& file_path) {
    if (is_active_) return false;

    file_path_ = file_path;

    if (camera_id_.empty()) {
        size_t last_slash = file_path.find_last_of("/\\");
        camera_id_ = (last_slash != std::string::npos) ? file_path.substr(last_slash + 1) : file_path;
    }

    if (!initializeGStreamer()) {
        LOG_ERROR("GStreamer initialization failed for file: {}", file_path);
        return false;
    }

    is_active_ = true;
    gst_thread_     = std::thread(&GStreamerFileHandler::gstreamerLoop, this);
    capture_thread_ = std::thread(&GStreamerFileHandler::captureLoop, this);

    return true;
}

void GStreamerFileHandler::stopStream() {
    if (!is_active_) return;

    is_active_ = false;

    if (main_loop_ && g_main_loop_is_running(main_loop_)) {
        g_main_loop_quit(main_loop_);
    }
    if (gst_thread_.joinable())     gst_thread_.join();
    if (capture_thread_.joinable()) capture_thread_.join();

    cleanupGStreamer();
    queue_cv_.notify_all();
}

std::optional<FrameContainer> GStreamerFileHandler::getNextFrame() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return !frame_queue_.empty() || !is_active_; });

    if (frame_queue_.empty()) return std::nullopt;

    FrameContainer fc = std::move(frame_queue_.front());
    frame_queue_.pop();
    return fc;
}

bool GStreamerFileHandler::isActive() const {
    return is_active_;
}

std::string GStreamerFileHandler::buildNvidiaHardwarePipeline() const {
    std::stringstream ss;
    ss << "filesrc location=\"" << file_path_ << "\" ! "
       << "decodebin ! "
       << "nvvideoconvert ! "
       << "videorate ! "
       << "video/x-raw,width=" << target_width_ << ",height=" << target_height_
       << ",framerate=" << target_fps_ << "/1 ! "
       << "videoconvert ! "
       << "video/x-raw,format=BGR ! "
       << "appsink name=sink emit-signals=true sync=false max-buffers=2 drop=false";
    return ss.str();
}

bool GStreamerFileHandler::initializeGStreamer() {
    GError* error = nullptr;
    std::string pipeline_str = buildNvidiaHardwarePipeline();

    pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline_ || error) {
        LOG_ERROR("GStreamer pipeline failed: {}", error ? error->message : "Unknown");
        if (error) g_error_free(error);
        return false;
    }

    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
        LOG_ERROR("Appsink element not found");
        return false;
    }

    g_object_set(appsink_, "emit-signals", TRUE, "sync", FALSE, "max-buffers", 2, "drop", FALSE, nullptr);
    g_signal_connect(appsink_, "new-sample", G_CALLBACK(onNewSample), this);

    bus_ = gst_element_get_bus(pipeline_);
    gst_bus_add_signal_watch(bus_);
    g_signal_connect(bus_, "message::error",   G_CALLBACK(onBusMessage),   this);
    g_signal_connect(bus_, "message::warning", G_CALLBACK(onBusMessage),   this);
    g_signal_connect(bus_, "message::info",    G_CALLBACK(onBusMessage),   this);
    g_signal_connect(bus_, "message::eos",     G_CALLBACK(onEndOfStream),  this);

    main_loop_ = g_main_loop_new(nullptr, FALSE);

    file_start_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    return true;
}

void GStreamerFileHandler::cleanupGStreamer() {
    if (bus_) {
        gst_bus_remove_signal_watch(bus_);
        gst_object_unref(bus_);
        bus_ = nullptr;
    }
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
    if (appsink_) {
        gst_object_unref(appsink_);
        appsink_ = nullptr;
    }
    if (main_loop_) {
        g_main_loop_unref(main_loop_);
        main_loop_ = nullptr;
    }
}

void GStreamerFileHandler::gstreamerLoop() {
    if (pipeline_) gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (main_loop_) g_main_loop_run(main_loop_);
}

void GStreamerFileHandler::captureLoop() {
    while (is_active_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void GStreamerFileHandler::processFrame(cv::Mat&& frame, uint64_t timestamp_ms) {
    if (!is_active_) return;

    FrameContainer fc;
    fc.frame = std::move(frame);
    fc.timestamp_ms = timestamp_ms;
    fc.source_id = camera_id_;

    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (static_cast<int>(frame_queue_.size()) < max_queue_size_) {
        frame_queue_.push(std::move(fc));
        queue_cv_.notify_one();
    }
}

GstFlowReturn GStreamerFileHandler::onNewSample(GstElement* appsink, gpointer user_data) {
    GStreamerFileHandler* handler = static_cast<GStreamerFileHandler*>(user_data);

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    if (!sample) return GST_FLOW_ERROR;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps*   caps   = gst_sample_get_caps(sample);

    if (buffer && caps) {
        GstClockTime pts = GST_BUFFER_PTS(buffer);
        uint64_t absolute_timestamp_ms = handler->file_start_time_ms_ + GST_TIME_AS_MSECONDS(pts);

        GstMapInfo map;
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            GstStructure* structure = gst_caps_get_structure(caps, 0);
            int width, height;
            gst_structure_get_int(structure, "width", &width);
            gst_structure_get_int(structure, "height", &height);

            cv::Mat frame_view(height, width, CV_8UC3, map.data);
            cv::Mat frame_copy = frame_view.clone();
            gst_buffer_unmap(buffer, &map);

            handler->processFrame(std::move(frame_copy), absolute_timestamp_ms);
        }
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

gboolean GStreamerFileHandler::onBusMessage(GstBus* bus, GstMessage* message, gpointer user_data) {
    GStreamerFileHandler* handler = static_cast<GStreamerFileHandler*>(user_data);

    switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR:
            handler->handlePipelineError(message);
            break;
        case GST_MESSAGE_WARNING:
            handler->handlePipelineWarning(message);
            break;
        case GST_MESSAGE_INFO:
            handler->handlePipelineInfo(message);
            break;
        default:
            break;
    }
    return TRUE;
}

gboolean GStreamerFileHandler::onEndOfStream(GstBus* bus, GstMessage* message, gpointer user_data) {
    GStreamerFileHandler* handler = static_cast<GStreamerFileHandler*>(user_data);

    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
        handler->is_active_ = false;
        handler->queue_cv_.notify_all();

        if (handler->main_loop_ && g_main_loop_is_running(handler->main_loop_)) {
            g_main_loop_quit(handler->main_loop_);
        }
    }
    return TRUE;
}

void GStreamerFileHandler::handlePipelineError(GstMessage* message) {
    GError* error = nullptr;
    gchar* debug_info = nullptr;
    gst_message_parse_error(message, &error, &debug_info);

    LOG_ERROR("GStreamer file error [{}]: {} | {}",
              file_path_,
              error ? error->message : "Unknown",
              debug_info ? debug_info : "");

    if (error) g_error_free(error);
    if (debug_info) g_free(debug_info);
    is_active_ = false;
}

void GStreamerFileHandler::handlePipelineWarning(GstMessage* message) {
    GError* warning = nullptr;
    gchar* debug_info = nullptr;
    gst_message_parse_warning(message, &warning, &debug_info);
    if (warning) g_error_free(warning);
    if (debug_info) g_free(debug_info);
}

void GStreamerFileHandler::handlePipelineInfo(GstMessage* message) {
    GError* info = nullptr;
    gchar* debug_info = nullptr;
    gst_message_parse_info(message, &info, &debug_info);
    if (info) g_error_free(info);
    if (debug_info) g_free(debug_info);
}

} // namespace project_x
