#include "yolox_detector.hpp"

namespace project_x {
    
YOLOXDetector::YOLOXDetector(const std::string& model_path, int num_threads,std::string& trt_cache_path, const std::vector<int>& classes)
    : IBaseModel<cv::Mat, std::vector<Detection>>(model_path, num_threads, trt_cache_path),
      score_threshold_(0.25f),
      nms_threshold_(0.45f),
      classes_(classes),
      ratio_(1.0f)
{
    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();

    target_h_ = static_cast<int>(input_shape_[2]);
    target_w_ = static_cast<int>(input_shape_[3]);
}

std::vector<Detection> YOLOXDetector::detect(const cv::Mat& image, float score_thr, float nms_thr) {
    std::lock_guard<std::mutex> lock(detector_mutex_);
    score_threshold_ = score_thr;
    nms_threshold_ = nms_thr;
    return this->run(image);
}


std::vector<Ort::Value> YOLOXDetector::preprocess(const cv::Mat& input) {
    auto [input_data, ratio] = preprocessImage(input);
    ratio_ = ratio;

    std::vector<int64_t> input_shape = {1, 3, target_h_, target_w_};
    std::vector<Ort::Value> tensors;

    // Ensure the old buffer is destroyed BEFORE creating the new tensor
    // This prevents the tensor from pointing to freed memory
    input_data_fp32_.clear();
    input_data_fp32_ = std::move(input_data);

    auto tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_data_fp32_.data(),
        input_data_fp32_.size(),
        input_shape.data(),
        input_shape.size()
    );
    tensors.push_back(std::move(tensor));
    return tensors;
}

std::pair<std::vector<float>, float> YOLOXDetector::preprocessImage(const cv::Mat& ori_frame) {
    cv::Mat padded_img(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));

    float r = std::min(static_cast<float>(target_h_) / ori_frame.rows,
                       static_cast<float>(target_w_) / ori_frame.cols);

    int resized_h = static_cast<int>(ori_frame.rows * r);
    int resized_w = static_cast<int>(ori_frame.cols * r);
    cv::Mat resized_img;
    cv::resize(ori_frame, resized_img, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    resized_img.copyTo(padded_img(cv::Rect(0, 0, resized_w, resized_h)));

    cv::Mat float_img;
    padded_img.convertTo(float_img, CV_32F);

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    std::vector<float> input_data(3 * target_h_ * target_w_);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_data.data() + c * target_h_ * target_w_,
                    channels[c].data,
                    target_h_ * target_w_ * sizeof(float));
    }

    return {input_data, r};
}

std::vector<Detection> YOLOXDetector::postprocess(std::vector<Ort::Value>& output_tensors) {
    auto postprocess_total_start = std::chrono::steady_clock::now();

    if (output_tensors.size() < 4) {
        throw std::runtime_error("Expected 4 output tensors from model with embedded decode");
    }

    auto boxes_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
    size_t num_predictions = boxes_shape[1];  

    auto scores_shape = output_tensors[3].GetTensorTypeAndShapeInfo().GetShape();
    size_t num_classes = scores_shape[2];  

    std::vector<float> boxes_fp32;
    std::vector<float> scores_fp32;

    float* boxes_ptr = output_tensors[2].GetTensorMutableData<float>();
    float* scores_ptr = output_tensors[3].GetTensorMutableData<float>();
    boxes_fp32.assign(boxes_ptr, boxes_ptr + num_predictions * 4);
    scores_fp32.assign(scores_ptr, scores_ptr + num_predictions * num_classes);
    
    for (size_t i = 0; i < num_predictions; ++i) {
        boxes_fp32[i * 4 + 0] /= ratio_;  
        boxes_fp32[i * 4 + 1] /= ratio_;  
        boxes_fp32[i * 4 + 2] /= ratio_;  
        boxes_fp32[i * 4 + 3] /= ratio_;  
    }

    std::vector<float> valid_boxes;
    std::vector<float> valid_scores;
    std::vector<int> valid_cls_inds;

    for (size_t i = 0; i < num_predictions; ++i) {
        float max_score = -std::numeric_limits<float>::infinity();
        int max_idx = 0;
        for (size_t c = 0; c < num_classes; ++c) {
            float raw_score = scores_fp32[i * num_classes + c];
            if (raw_score > max_score) {
                max_score = raw_score;
                max_idx = static_cast<int>(c);
            }
        }

        if (max_score > score_threshold_) {
            float x1 = boxes_fp32[i * 4 + 0];
            float y1 = boxes_fp32[i * 4 + 1];
            float x2 = boxes_fp32[i * 4 + 2];
            float y2 = boxes_fp32[i * 4 + 3];

            valid_boxes.push_back(x1);
            valid_boxes.push_back(y1);
            valid_boxes.push_back(x2);
            valid_boxes.push_back(y2);
            valid_scores.push_back(max_score);
            valid_cls_inds.push_back(max_idx);
        }
    }

    if (valid_scores.empty()) {
        return std::vector<Detection>();
    }

    std::vector<int> keep = nms(valid_boxes, valid_scores, nms_threshold_);

    std::vector<Detection> detections;
    for (int idx : keep) {
        if (std::find(classes_.begin(), classes_.end(), valid_cls_inds[idx]) == classes_.end()) {
            continue;
        }

        Detection det;
        det.x1 = valid_boxes[idx * 4];
        det.y1 = valid_boxes[idx * 4 + 1];
        det.x2 = valid_boxes[idx * 4 + 2];
        det.y2 = valid_boxes[idx * 4 + 3];
        det.score = valid_scores[idx];
        det.class_id = valid_cls_inds[idx];
        detections.push_back(det);
    }
    return detections;
}

std::vector<int> YOLOXDetector::nms(const std::vector<float>& boxes, const std::vector<float>& scores,
                                     float nms_thr) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });

    std::vector<float> areas(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        float x1 = boxes[i * 4];
        float y1 = boxes[i * 4 + 1];
        float x2 = boxes[i * 4 + 2];
        float y2 = boxes[i * 4 + 3];
        areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1);
    }

    std::vector<int> keep;
    while (!indices.empty()) {
        int idx = indices[0];
        keep.push_back(idx);

        if (indices.size() == 1) break;

        std::vector<int> new_indices;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx2 = indices[i];

            float x1 = std::max(boxes[idx * 4], boxes[idx2 * 4]);
            float y1 = std::max(boxes[idx * 4 + 1], boxes[idx2 * 4 + 1]);
            float x2 = std::min(boxes[idx * 4 + 2], boxes[idx2 * 4 + 2]);
            float y2 = std::min(boxes[idx * 4 + 3], boxes[idx2 * 4 + 3]);

            float w = std::max(0.0f, x2 - x1 + 1);
            float h = std::max(0.0f, y2 - y1 + 1);
            float inter = w * h;

            float iou = inter / (areas[idx] + areas[idx2] - inter);

            if (iou <= nms_thr) {
                new_indices.push_back(idx2);
            }
        }

        indices = new_indices;
    }

    return keep;
}

std::vector<Detection> YOLOXDetector::multiclass_nms_class_agnostic(const std::vector<float>& boxes,
                                                                     const std::vector<std::vector<float>>& scores,
                                                                     float nms_thr, float score_thr) {
    size_t num_boxes = scores.size();
    std::vector<int> cls_inds;
    std::vector<float> cls_scores;

    for (const auto& score_vec : scores) {
        auto max_it = std::max_element(score_vec.begin(), score_vec.end());
        int cls_idx = std::distance(score_vec.begin(), max_it);
        float max_score = *max_it;

        cls_inds.push_back(cls_idx);
        cls_scores.push_back(max_score);
    }

    std::vector<float> valid_boxes;
    std::vector<float> valid_scores;
    std::vector<int> valid_cls_inds;

    for (size_t i = 0; i < num_boxes; ++i) {
        if (cls_scores[i] > score_thr) {
            valid_boxes.push_back(boxes[i * 4]);
            valid_boxes.push_back(boxes[i * 4 + 1]);
            valid_boxes.push_back(boxes[i * 4 + 2]);
            valid_boxes.push_back(boxes[i * 4 + 3]);
            valid_scores.push_back(cls_scores[i]);
            valid_cls_inds.push_back(cls_inds[i]);
        }
    }

    if (valid_scores.empty()) {
        return std::vector<Detection>();
    }

    std::vector<int> keep = nms(valid_boxes, valid_scores, nms_thr);

    std::vector<Detection> detections;
    for (int idx : keep) {
        if(std::find(this->classes_.begin(), this->classes_.end(), valid_cls_inds[idx]) == this->classes_.end())
        {
            continue;
        }
        
        Detection det;
        det.x1 = valid_boxes[idx * 4];
        det.y1 = valid_boxes[idx * 4 + 1];
        det.x2 = valid_boxes[idx * 4 + 2];
        det.y2 = valid_boxes[idx * 4 + 3];
        det.score = valid_scores[idx];
        det.class_id = valid_cls_inds[idx];
        detections.push_back(det);
    }

    return detections;
}
}
