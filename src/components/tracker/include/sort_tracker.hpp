#ifndef SORT_TRACKER_H
#define SORT_TRACKER_H

#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <ctime>

#include "../../../common/include/interfaces.hpp"

namespace project_x
{
constexpr int MAX_HISTORY_SIZE = 10;

class GeneralTracklet;

/**
 * @brief Kalman filter for state estimation in object tracking.
 */
class KalmanFilter {
public:
    /**
     * @brief Constructs Kalman filter with specified dimensions.
     * @param dim_x State vector dimension
     * @param dim_z Measurement vector dimension
     */
    KalmanFilter(int dim_x, int dim_z);

    /**
     * @brief Performs prediction step.
     */
    void predict();

    /**
     * @brief Performs update step with measurement.
     * @param z Measurement vector
     */
    void update(const Eigen::VectorXd& z);

    Eigen::MatrixXd F;
    Eigen::MatrixXd H;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd P;
    Eigen::VectorXd x;

private:
    int dim_x_;
    int dim_z_;
    Eigen::MatrixXd I;
};

/**
 * @brief Tracklet for maintaining object track state.
 */
class GeneralTracklet : public project_x::BaseTracklet {
public:
    /**
     * @brief Constructs tracklet with initial detection.
     * @param bbox Bounding box in xyxy format
     * @param conf Confidence score
     * @param label Class label
     */
    GeneralTracklet(const Eigen::Vector4d& bbox, double conf, int label);

    void update(const Eigen::Vector4d& bbox, double conf) override;
    Eigen::Vector4d predict() override;
    Eigen::Vector4d get_state() const override;

    /**
     * @brief Converts tracklet state to JSON representation.
     * @return JSON object with tracklet state
     */
    nlohmann::json to_json() const;

    int time_since_update;
    std::string id;
    std::vector<Eigen::Vector4d> history;
    int hits;
    int hit_streak;
    int age;
    int64_t tracker_id;
    double conf;
    int label;

private:
    std::unique_ptr<KalmanFilter> kf;
};

/**
 * @brief SORT (Simple Online and Realtime Tracking) tracker implementation.
 */
class SortTracker {
public:
    /**
     * @brief Constructs SORT tracker with specified parameters.
     * @param max_age Maximum frames to keep track without update
     * @param min_hits Minimum hits before track is confirmed
     * @param iou_threshold IoU threshold for matching detections to tracks
     */
    SortTracker(int max_age = 1, int min_hits = 3, double iou_threshold = 0.3);

    /**
     * @brief Updates tracker with new detections.
     * @param dets Vector of detections in current frame
     * @return Vector of JSON objects representing active tracks
     */
    std::vector<nlohmann::json> track(const std::vector<project_x::Detection>& dets);

private:
    int max_age_;
    int min_hits_;
    double iou_threshold_;
    std::vector<std::unique_ptr<GeneralTracklet>> trackers_;
    int frame_count_;
};

/**
 * @brief Generates ULID (Universally Unique Lexicographically Sortable Identifier).
 * @return ULID string
 */
std::string generate_ulid();

/**
 * @brief Converts bounding box to measurement format.
 * @param bbox Bounding box in xyxy format
 * @return Measurement vector
 */
Eigen::Vector4d convert_bbox_to_z(const Eigen::Vector4d& bbox);

/**
 * @brief Converts state vector to bounding box format.
 * @param x State vector
 * @return Bounding box in xyxy format
 */
Eigen::Vector4d convert_x_to_bbox(const Eigen::VectorXd& x);

/**
 * @brief Computes IoU between multiple bounding box pairs.
 * @param bb_test Test bounding boxes
 * @param bb_gt Ground truth bounding boxes
 * @return IoU matrix
 */
Eigen::MatrixXd iou_batch(const Eigen::MatrixXd& bb_test, const Eigen::MatrixXd& bb_gt);

/**
 * @brief Associates detections to existing trackers using Hungarian algorithm.
 * @param detections Current frame detections
 * @param trackers Predicted tracker states
 * @param iou_threshold IoU threshold for matching
 * @return Tuple of matched pairs, unmatched detections, and unmatched trackers
 */
std::tuple<std::vector<std::pair<int,int>>, std::vector<int>, std::vector<int>>
    associate_detections_to_trackers(const std::vector<project_x::Detection>& detections,
                                      const std::vector<Eigen::Vector4d>& trackers,
                                      double iou_threshold);

/**
 * @brief Solves linear assignment problem using Hungarian algorithm.
 * @param cost_matrix Cost matrix for assignment
 * @return Vector of matched index pairs
 */
std::vector<std::pair<int,int>> linear_assignment(const Eigen::MatrixXd& cost_matrix);

}

#endif
