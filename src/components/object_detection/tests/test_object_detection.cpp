#include "../../../lib/catch2/catch_amalgamated.hpp"
#include "yolox_detector.hpp"
#include <opencv2/opencv.hpp>

namespace project_x
{
    const std::string TEST_MODEL_PATH = "/workspace/models/detector.onnx";
    std::string trt_cache_path = "./trt_cache";
    TEST_CASE("Model loading")
    {
        SECTION("Model loads correctly with valid path")
        {
            REQUIRE_NOTHROW(YOLOXDetector(TEST_MODEL_PATH, 2, trt_cache_path, {0}));
        }

        SECTION("Crash with invalid model path")
        {
            REQUIRE_THROWS(YOLOXDetector("/nonexistent/model.onnx", 2, trt_cache_path, {0}));
        }
    }

    TEST_CASE("Basic detection")
    {
        YOLOXDetector detector(TEST_MODEL_PATH, 2, trt_cache_path, {0});
        cv::Mat test_img(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));

        SECTION("Detection returns valid results")
        {
            auto detections = detector.detect(test_img);

            for (const auto& det : detections) {
                REQUIRE(det.score >= 0.0f);
                REQUIRE(det.score <= 1.0f);
                REQUIRE(det.x2 >= det.x1);
                REQUIRE(det.y2 >= det.y1);
            }
        }

        SECTION("Higher threshold reduces detections")
        {
            auto low_thr = detector.detect(test_img, 0.1f, 0.45f);
            auto high_thr = detector.detect(test_img, 0.8f, 0.45f);
            REQUIRE(high_thr.size() <= low_thr.size());
        }
    }

    TEST_CASE("Class filtering")
    {
        SECTION("Single class filter")
        {
            YOLOXDetector detector(TEST_MODEL_PATH, 2, trt_cache_path, {0}); 
            cv::Mat img(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
            auto detections = detector.detect(img);

            for (const auto& det : detections) {
                REQUIRE(det.class_id == 0);
            }
        }

        SECTION("Empty class filter returns nothing")
        {
            YOLOXDetector detector(TEST_MODEL_PATH, 2, trt_cache_path, {});
            cv::Mat img(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
            auto detections = detector.detect(img);
            REQUIRE(detections.empty());
        }
    }

    TEST_CASE("Edge cases")
    {
        YOLOXDetector detector(TEST_MODEL_PATH, 2, trt_cache_path, {0});

        SECTION("Empty image throws")
        {
            cv::Mat empty;
            REQUIRE_THROWS(detector.detect(empty));
        }

        SECTION("Different image sizes work")
        {
            cv::Mat small(240, 320, CV_8UC3, cv::Scalar(128, 128, 128));
            cv::Mat large(1080, 1920, CV_8UC3, cv::Scalar(128, 128, 128));

            REQUIRE_NOTHROW(detector.detect(small));
            REQUIRE_NOTHROW(detector.detect(large));
        }
    }
}