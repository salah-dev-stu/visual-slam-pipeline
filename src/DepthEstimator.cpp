#include "DepthEstimator.h"
#include "Config.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <iostream>

DepthEstimator::DepthEstimator()
    : available_(false) {}

DepthEstimator::~DepthEstimator() = default;

bool DepthEstimator::init(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MiDaS");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        } catch (...) {
        }

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);
        available_ = true;
        return true;
    } catch (const std::exception& e) {
        available_ = false;
        return false;
    }
}

cv::Mat DepthEstimator::estimate(const cv::Mat& image) {
    if (!available_) return cv::Mat();

    int orig_h = image.rows;
    int orig_w = image.cols;
    int sz = Config::MIDAS_INPUT_SIZE;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(sz, sz));

    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

    cv::Mat channels[3];
    cv::split(float_img, channels);
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std_val[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean[c]) / std_val[c];
    }

    std::vector<float> input_data(3 * sz * sz);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < sz; y++) {
            for (int x = 0; x < sz; x++) {
                input_data[c * sz * sz + y * sz + x] = channels[c].at<float>(y, x);
            }
        }
    }

    std::vector<int64_t> input_shape = {1, 3, sz, sz};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                  input_names, &input_tensor, 1,
                                  output_names, 1);

    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    float* out_data = outputs[0].GetTensorMutableData<float>();

    int out_h, out_w;
    if (out_shape.size() == 3) {
        out_h = out_shape[1];
        out_w = out_shape[2];
    } else if (out_shape.size() == 2) {
        out_h = out_shape[0];
        out_w = out_shape[1];
    } else {
        // Assume square
        int total = 1;
        for (auto s : out_shape) total *= s;
        out_h = out_w = (int)std::sqrt(total);
    }

    cv::Mat depth_small(out_h, out_w, CV_32F, out_data);

    cv::Mat depth;
    cv::resize(depth_small, depth, cv::Size(orig_w, orig_h));

    // MiDaS outputs inverse depth. Convert and normalize to [0, 1]
    double min_val, max_val;
    cv::minMaxLoc(depth, &min_val, &max_val);
    if (max_val - min_val > 1e-6) {
        depth = (depth - min_val) / (max_val - min_val);
    }

    return depth;
}
