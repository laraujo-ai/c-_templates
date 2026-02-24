#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

namespace project_x {
/**
 * @brief Thread-safe benchmarking metrics for video analysis pipeline.
 *
 * Tracks performance metrics including frames per second (FPS),
 * objects detected/tracked per second, and VRAM consumption.
 */
class BenchmarkMetrics {
public:
    BenchmarkMetrics();

    /**
     * @brief Records that objects have been detected/tracked.
     * @param object_count Number of objects detected/tracked in this batch.
     */
    void recordObjectsProcessed(size_t object_count);

    /**
     * @brief Records that frames have been processed.
     * @param frame_count Number of frames processed in this batch.
     */
    void recordFramesProcessed(size_t frame_count);

    /**
     * @brief Resets all metrics and restarts the timer.
     */
    void reset();

    /**
     * @brief Gets the elapsed time since metrics started/reset.
     * @return Elapsed time in seconds.
     */
    double getElapsedSeconds() const;

    /**
     * @brief Gets the total number of objects processed.
     * @return Total objects processed.
     */
    size_t getTotalObjects() const;

    /**
     * @brief Gets the total number of frames processed.
     * @return Total frames processed.
     */
    size_t getTotalFrames() const;

    /**
     * @brief Calculates frames processed per second (FPS).
     * @return Frames per second (0.0 if no time elapsed).
     */
    double getFramesPerSecond() const;

    /**
     * @brief Calculates average objects processed per second.
     * @return Objects per second (0.0 if no time elapsed).
     */
    double getObjectsPerSecond() const;

    /**
     * @brief Calculates average objects per frame.
     * @return Average objects per frame (0.0 if no frames processed).
     */
    double getAverageObjectsPerFrame() const;

    /**
     * @brief Generates a formatted summary string of current metrics.
     * @return String containing FPS, objects/sec, and objects/frame.
     */
    std::string getSummary() const;

    /**
     * @brief Gets current CUDA memory usage (VRAM).
     * @return VRAM used in MB, or -1.0 if CUDA is unavailable.
     */
    double getVRAMUsageMB() const;

    /**
     * @brief Gets peak CUDA memory usage (VRAM) since start.
     * @return Peak VRAM used in MB, or -1.0 if CUDA is unavailable.
     */
    double getPeakVRAMUsageMB() const;

    /**
     * @brief Updates the peak VRAM usage if current is higher.
     */
    void updatePeakVRAM();

    std::chrono::steady_clock::time_point getStartTime();

    void setStartTime();

private:
    std::atomic<size_t> total_objects_;
    std::atomic<size_t> total_frames_;
    std::chrono::steady_clock::time_point start_time_;
    mutable std::mutex time_mutex_;
    mutable std::mutex vram_mutex_;
    double peak_vram_mb_;
};

} // namespace project_x