#ifndef DEPTH_ESTIMATOR_H
#define DEPTH_ESTIMATOR_H

#include <opencv2/core.hpp>
#include <string>
#include <memory>

namespace Ort { class Session; class Env; }

class DepthEstimator {
public:
    DepthEstimator();
    ~DepthEstimator();

    bool init(const std::string& model_path);

    cv::Mat estimate(const cv::Mat& image);

    bool is_available() const { return available_; }

private:
    bool available_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};

#endif
