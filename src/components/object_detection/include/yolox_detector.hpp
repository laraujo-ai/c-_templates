#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <limits>
#include <mutex>

#include "../../../common/include/interfaces.hpp"
#include "../../../common/include/base_model.hpp"

namespace project_x {

/**
 * @brief YOLOX object detector using ONNX Runtime.
 */
class YOLOXDetector : public IBaseModel<cv::Mat, std::vector<Detection>> {
    public:
        /**
         * @brief Constructs YOLOX detector with specified parameters.
         * @param model_path Path to ONNX model file
         * @param num_threads Number of inference threads
         * @param classes Vector of class IDs to detect
         */
        YOLOXDetector(const std::string& model_path, int num_threads, std::string& trt_cache_path, const std::vector<int>& classes);

        ~YOLOXDetector() = default;

        /**
         * @brief Performs object detection on an image.
         * @param image Input image
         * @param score_thr Confidence score threshold
         * @param nms_thr NMS IoU threshold
         * @return Vector of detected objects
         */
        std::vector<Detection> detect(const cv::Mat& image, float score_thr = 0.25f, float nms_thr = 0.45f);

    protected:
        std::vector<Ort::Value> preprocess(const cv::Mat& input) override;
        std::vector<Detection> postprocess(std::vector<Ort::Value>& output_tensors) override;

    private:
        /**
         * @brief Preprocesses image for model input.
         * @param ori_frame Original input frame
         * @return Pair of preprocessed data and resize ratio
         */
        std::pair<std::vector<float>, float> preprocessImage(const cv::Mat& ori_frame);

        /**
         * @brief Performs non-maximum suppression.
         * @param boxes Bounding boxes
         * @param scores Confidence scores
         * @param nms_thr NMS IoU threshold
         * @return Indices of boxes to keep
         */
        std::vector<int> nms(const std::vector<float>& boxes, const std::vector<float>& scores, float nms_thr);

        /**
         * @brief Performs class-agnostic multiclass NMS.
         * @param boxes Bounding boxes
         * @param scores Per-class confidence scores
         * @param nms_thr NMS threshold
         * @param score_thr Confidence threshold
         * @return Vector of detections after NMS
         */
        std::vector<Detection> multiclass_nms_class_agnostic(const std::vector<float>& boxes,
                                                            const std::vector<std::vector<float>>& scores,
                                                            float nms_thr, float score_thr);

        std::vector<int64_t> input_shape_;
        int target_h_;
        int target_w_;

        float score_threshold_;
        float nms_threshold_;
        float ratio_;
        std::vector<int> classes_;

        std::vector<Ort::Float16_t> input_data_fp16_;
        std::vector<float> input_data_fp32_;

        mutable std::mutex detector_mutex_;
};

}
