#include "../include/sort_tracker.hpp"

namespace project_x
{

void hasTrackletMoved(const std::vector<Eigen::Vector4d>& tracklet_history, bool& hasMoved)
{
    size_t historySize = tracklet_history.size();
    if (historySize < 2) {
        hasMoved = true;
        return;
    }

    auto last_box = tracklet_history[historySize - 1];
    auto llast_box = tracklet_history[historySize - 2];
    double MOVEMENT_THRESHOLD = 1.0;

    auto sideways_movement = abs(last_box[0] - llast_box[0]) >MOVEMENT_THRESHOLD || abs(last_box[2] - llast_box[2]) >MOVEMENT_THRESHOLD;
    auto vertical_movement = abs(last_box[1] - llast_box[1]) >MOVEMENT_THRESHOLD || abs(last_box[3] - llast_box[3] ) >MOVEMENT_THRESHOLD;

    hasMoved = (sideways_movement || vertical_movement);
}       

KalmanFilter::KalmanFilter(int dim_x, int dim_z)
    : dim_x_(dim_x), dim_z_(dim_z) {

    F = Eigen::MatrixXd::Identity(dim_x, dim_x);
    H = Eigen::MatrixXd::Zero(dim_z, dim_x);
    Q = Eigen::MatrixXd::Identity(dim_x, dim_x);
    R = Eigen::MatrixXd::Identity(dim_z, dim_z);
    P = Eigen::MatrixXd::Identity(dim_x, dim_x);
    x = Eigen::VectorXd::Zero(dim_x);
    I = Eigen::MatrixXd::Identity(dim_x, dim_x);
}

void KalmanFilter::predict() {
    x = F * x;
    P = F * P * F.transpose() + Q;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    Eigen::VectorXd y = z - H * x;
    Eigen::MatrixXd S = H * P * H.transpose() + R;
    Eigen::MatrixXd K = P * H.transpose() * S.inverse();
    x = x + K * y;
    P = (I - K * H) * P;
}

std::string generate_ulid() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 35);

    const char* chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string ulid;

    ulid += std::to_string(ms);

    for (int i = 0; i < 10; ++i) {
        ulid += chars[dis(gen)];
    }

    return ulid;
}

Eigen::Vector4d convert_bbox_to_z(const Eigen::Vector4d& bbox) {
    double w = bbox[2] - bbox[0];
    double h = bbox[3] - bbox[1];
    double x = bbox[0] + w / 2.0;
    double y = bbox[1] + h / 2.0;
    double s = w * h;
    double r = w / h;

    Eigen::Vector4d z;
    z << x, y, s, r;
    return z;
}

Eigen::Vector4d convert_x_to_bbox(const Eigen::VectorXd& x) {
    double w = std::sqrt(x[2] * x[3]);
    double h = x[2] / w;

    Eigen::Vector4d bbox;
    bbox << x[0] - w/2.0, x[1] - h/2.0, x[0] + w/2.0, x[1] + h/2.0;
    return bbox;
}

Eigen::MatrixXd iou_batch(const Eigen::MatrixXd& bb_test, const Eigen::MatrixXd& bb_gt) {
    int n_test = bb_test.rows();
    int n_gt = bb_gt.rows();
    Eigen::MatrixXd iou(n_test, n_gt);

    for (int i = 0; i < n_test; ++i) {
        for (int j = 0; j < n_gt; ++j) {
            double xx1 = std::max(bb_test(i, 0), bb_gt(j, 0));
            double yy1 = std::max(bb_test(i, 1), bb_gt(j, 1));
            double xx2 = std::min(bb_test(i, 2), bb_gt(j, 2));
            double yy2 = std::min(bb_test(i, 3), bb_gt(j, 3));

            double w = std::max(0.0, xx2 - xx1);
            double h = std::max(0.0, yy2 - yy1);
            double wh = w * h;

            double area_test = (bb_test(i, 2) - bb_test(i, 0)) * (bb_test(i, 3) - bb_test(i, 1));
            double area_gt = (bb_gt(j, 2) - bb_gt(j, 0)) * (bb_gt(j, 3) - bb_gt(j, 1));

            iou(i, j) = wh / (area_test + area_gt - wh);
        }
    }

    return iou;
}

std::vector<std::pair<int,int>> linear_assignment(const Eigen::MatrixXd& cost_matrix) {
    int rows = cost_matrix.rows();
    int cols = cost_matrix.cols();

    std::vector<int> row_match(rows, -1);
    std::vector<int> col_match(cols, -1);

    std::vector<std::tuple<double, int, int>> costs;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            costs.push_back({cost_matrix(i, j), i, j});
        }
    }

    std::sort(costs.begin(), costs.end());

    std::vector<std::pair<int,int>> matches;
    for (const auto& [cost, i, j] : costs) {
        if (row_match[i] == -1 && col_match[j] == -1) {
            row_match[i] = j;
            col_match[j] = i;
            matches.push_back({i, j});
        }
    }

    return matches;
}

std::tuple<std::vector<std::pair<int,int>>, std::vector<int>, std::vector<int>>
associate_detections_to_trackers(const std::vector<project_x::Detection>& detections,
                                  const std::vector<Eigen::Vector4d>& trackers,
                                  double iou_threshold) {

    if (trackers.empty()) {
        std::vector<int> unmatched_dets;
        for (size_t i = 0; i < detections.size(); ++i) {
            unmatched_dets.push_back(i);
        }
        return std::make_tuple(std::vector<std::pair<int, int>>(), unmatched_dets, std::vector<int>());
    }

    Eigen::MatrixXd det_mat(detections.size(), 4);
    for (size_t i = 0; i < detections.size(); ++i) {
        det_mat.row(i) << detections[i].x1, detections[i].y1, detections[i].x2, detections[i].y2;
    }

    Eigen::MatrixXd trk_mat(trackers.size(), 4);
    for (size_t i = 0; i < trackers.size(); ++i) {
        trk_mat.row(i) = trackers[i];
    }

    Eigen::MatrixXd iou_matrix = iou_batch(det_mat, trk_mat);

    std::vector<std::pair<int,int>> matched_indices;

    if (iou_matrix.rows() > 0 && iou_matrix.cols() > 0) {
        matched_indices = linear_assignment(-iou_matrix);
    }

    std::vector<bool> matched_det(detections.size(), false);
    std::vector<bool> matched_trk(trackers.size(), false);

    std::vector<std::pair<int,int>> matches;
    for (const auto& [d, t] : matched_indices) {
        if (iou_matrix(d, t) >= iou_threshold) {
            matches.push_back({d, t});
            matched_det[d] = true;
            matched_trk[t] = true;
        }
    }

    std::vector<int> unmatched_detections;
    for (size_t d = 0; d < detections.size(); ++d) {
        if (!matched_det[d]) {
            unmatched_detections.push_back(d);
        }
    }

    std::vector<int> unmatched_trackers;
    for (size_t t = 0; t < trackers.size(); ++t) {
        if (!matched_trk[t]) {
            unmatched_trackers.push_back(t);
        }
    }

    return {matches, unmatched_detections, unmatched_trackers};
}

GeneralTracklet::GeneralTracklet(const Eigen::Vector4d& bbox, double conf, int label)
    : time_since_update(0), hits(0), hit_streak(0), age(0),
      conf(conf), label(label) {

    kf = std::make_unique<KalmanFilter>(7, 4);

    kf->F << 1, 0, 0, 0, 1, 0, 0,
             0, 1, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 0, 0, 1,
             0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 1;

    kf->H << 1, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0;

    kf->R.block(2, 2, 2, 2) *= 10.0;
    kf->P.block(4, 4, 3, 3) *= 1000.0;
    kf->P *= 10.0;
    kf->Q(6, 6) *= 0.01;
    kf->Q.block(4, 4, 3, 3) *= 0.01;

    Eigen::Vector4d z = convert_bbox_to_z(bbox);
    kf->x.head(4) = z;

    id = generate_ulid();
    auto now = std::chrono::system_clock::now();
    tracker_id = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
}

void GeneralTracklet::update(const Eigen::Vector4d& bbox, double conf) {
    time_since_update = 0;
    // history.clear();
    hits++;
    hit_streak++;

    Eigen::Vector4d z = convert_bbox_to_z(bbox);
    kf->update(z);
    this->conf = conf;
}

Eigen::Vector4d GeneralTracklet::predict() {
    if ((kf->x[6] + kf->x[2]) <= 0) {
        kf->x[6] *= 0.0;
    }

    kf->predict();
    age++;

    if (time_since_update > 0) {
        hit_streak = 0;
    }

    time_since_update++;

    if (history.size() >= MAX_HISTORY_SIZE) {
        history.erase(history.begin());
    }

    Eigen::Vector4d bbox = convert_x_to_bbox(kf->x);
    history.push_back(bbox);
    return history.back();
}

Eigen::Vector4d GeneralTracklet::get_state() const {
    return convert_x_to_bbox(kf->x);
}

nlohmann::json GeneralTracklet::to_json() const {
    Eigen::Vector4d bbox = get_state();
    bool hasMoved = false;
    hasTrackletMoved(this->history, hasMoved);
    return {
        {"BoundingBox", {(int)bbox[0], (int)bbox[1], (int)bbox[2], (int)bbox[3]}},
        {"ULID", id},
        {"TrackerId", tracker_id},
        {"Confidence", std::round(conf * 100.0) / 100.0},
        {"Label", label},
        {"hasMoved", hasMoved}
    };
}

SortTracker::SortTracker(int max_age, int min_hits, double iou_threshold)
    : max_age_(max_age), min_hits_(min_hits), iou_threshold_(iou_threshold),
      frame_count_(0) {
}

std::vector<nlohmann::json> SortTracker::track(const std::vector<project_x::Detection>& dets) {
    frame_count_++;

    std::vector<Eigen::Vector4d> trks;
    std::vector<int> to_del;

    for (size_t t = 0; t < trackers_.size(); ++t) {
        Eigen::Vector4d pos = trackers_[t]->predict();

        if (std::isnan(pos[0]) || std::isnan(pos[1]) || std::isnan(pos[2]) || std::isnan(pos[3])) {
            to_del.push_back(t);
        } else {
            trks.push_back(pos);
        }
    }

    for (int i = to_del.size() - 1; i >= 0; --i) {
        trackers_.erase(trackers_.begin() + to_del[i]);
    }

    auto [matched, unmatched_dets, unmatched_trks] =
        associate_detections_to_trackers(dets, trks, iou_threshold_);

    for (const auto& [d, t] : matched) {
        Eigen::Vector4d bbox;
        bbox << dets[d].x1, dets[d].y1, dets[d].x2, dets[d].y2;
        trackers_[t]->update(bbox, dets[d].score);
    }

    for (int i : unmatched_dets) {
        Eigen::Vector4d bbox;
        bbox << dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2;
        trackers_.push_back(std::make_unique<GeneralTracklet>(bbox, dets[i].score, dets[i].class_id));
    }

    std::vector<nlohmann::json> results;
    std::vector<int> to_remove;

    for (size_t i = 0; i < trackers_.size(); ++i) {
        auto& trk = trackers_[i];

        if (trk->time_since_update < 1 &&
            (trk->hit_streak >= min_hits_ || frame_count_ <= min_hits_)) {
            results.push_back(trk->to_json());
        }

        if (trk->time_since_update > max_age_) {
            to_remove.push_back(i);
        }
    }

    for (int i = to_remove.size() - 1; i >= 0; --i) {
        trackers_.erase(trackers_.begin() + to_remove[i]);
    }

    return results;
}

}
