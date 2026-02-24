#include "../include/benchmark_metrics.hpp"
#include <sstream>
#include <iomanip>

// Include CUDA runtime for memory tracking
#include <cuda_runtime.h>

namespace project_x {

BenchmarkMetrics::BenchmarkMetrics()
    : total_objects_(0),
      total_frames_(0),
      start_time_(std::chrono::steady_clock::time_point{}),
      peak_vram_mb_(0.0) {
}

void BenchmarkMetrics::recordObjectsProcessed(size_t object_count) {
    total_objects_.fetch_add(object_count, std::memory_order_relaxed);
}

void BenchmarkMetrics::recordFramesProcessed(size_t frame_count) {
    total_frames_.fetch_add(frame_count, std::memory_order_relaxed);
}

void BenchmarkMetrics::reset() {
    total_objects_.store(0, std::memory_order_relaxed);
    total_frames_.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(time_mutex_);
    start_time_ = std::chrono::steady_clock::now();
}

double BenchmarkMetrics::getElapsedSeconds() const {
    std::lock_guard<std::mutex> lock(time_mutex_);
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    return duration.count() / 1000.0;
}

size_t BenchmarkMetrics::getTotalObjects() const {
    return total_objects_.load(std::memory_order_relaxed);
}

size_t BenchmarkMetrics::getTotalFrames() const {
    return total_frames_.load(std::memory_order_relaxed);
}

double BenchmarkMetrics::getFramesPerSecond() const {
    double elapsed = getElapsedSeconds();
    if (elapsed < 0.001) {
        return 0.0;
    }
    return getTotalFrames() / elapsed;
}

double BenchmarkMetrics::getObjectsPerSecond() const {
    double elapsed = getElapsedSeconds();
    if (elapsed < 0.001) {
        return 0.0;
    }
    return getTotalObjects() / elapsed;
}

double BenchmarkMetrics::getAverageObjectsPerFrame() const {
    size_t total_frames = getTotalFrames();
    if (total_frames == 0) {
        return 0.0;
    }
    return static_cast<double>(getTotalObjects()) / total_frames;
}

std::string BenchmarkMetrics::getSummary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "FPS: " << getFramesPerSecond()
        << " | Objects/sec: " << getObjectsPerSecond()
        << " | Avg Objects/Frame: " << getAverageObjectsPerFrame()
        << " | Total Objects: " << getTotalObjects()
        << " | Total Frames: " << getTotalFrames()
        << " | Elapsed: " << getElapsedSeconds() << "s";

    double vram = getVRAMUsageMB();
    if (vram >= 0.0) {
        oss << " | VRAM: " << vram << " MB"
            << " | Peak VRAM: " << getPeakVRAMUsageMB() << " MB";
    }

    return oss.str();
}

double BenchmarkMetrics::getVRAMUsageMB() const {
    size_t free_mem = 0;
    size_t total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);

    if (err != cudaSuccess) {
        return -1.0;
    }

    size_t used_mem = total_mem - free_mem;
    return static_cast<double>(used_mem) / (1024.0 * 1024.0);
}

double BenchmarkMetrics::getPeakVRAMUsageMB() const {
    std::lock_guard<std::mutex> lock(vram_mutex_);
    return peak_vram_mb_;
}

void BenchmarkMetrics::updatePeakVRAM() {
    double current_vram = getVRAMUsageMB();
    if (current_vram >= 0.0) {
        std::lock_guard<std::mutex> lock(vram_mutex_);
        if (current_vram > peak_vram_mb_) {
            peak_vram_mb_ = current_vram;
        }
    }
}
void BenchmarkMetrics::setStartTime(){
    this->start_time_ = std::chrono::steady_clock::now();
}

std::chrono::steady_clock::time_point BenchmarkMetrics::getStartTime()
{
    return this->start_time_;
};


} // namespace project_x