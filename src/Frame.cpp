#include "Frame.h"
#include "FeatureExtractor.h"
#include "DepthEstimator.h"
#include "Config.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <regex>
#include <cmath>

Frame::Frame()
    : id_(-1), timestamp_(0.0), processed_(false), is_keyframe_(false), has_real_depth_(false) {
    R_ = cv::Mat::eye(3, 3, CV_64F);
    t_ = cv::Mat::zeros(3, 1, CV_64F);
}

Frame::Frame(int id, const std::string& image_path, double timestamp)
    : id_(id), image_path_(image_path), timestamp_(timestamp),
      processed_(false), is_keyframe_(false), has_real_depth_(false) {
    R_ = cv::Mat::eye(3, 3, CV_64F);
    t_ = cv::Mat::zeros(3, 1, CV_64F);

    image_ = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image_.empty()) {
        return;
    }
    cv::cvtColor(image_, gray_, cv::COLOR_BGR2GRAY);
}

void Frame::detect_features(FeatureExtractor& extractor) {
    if (gray_.empty()) return;
    extractor.extract(image_, keypoints_, descriptors_);
    map_point_indices_.assign(keypoints_.size(), -1);
    processed_ = true;
}

void Frame::estimate_depth(DepthEstimator& estimator) {
    if (image_.empty() || has_real_depth_) return;  // Skip MiDaS if real depth loaded
    depth_map_ = estimator.estimate(image_);
}

void Frame::load_depth_image(const std::string& depth_path) {
    cv::Mat depth_raw = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    if (depth_raw.empty()) return;

    depth_raw.convertTo(depth_map_, CV_32F, 1.0 / 5000.0);
    depth_map_.setTo(0, depth_raw == 0);
    has_real_depth_ = true;
}

void Frame::compute_global_descriptor() {
    if (descriptors_.empty()) {
        global_descriptor_ = cv::Mat();
        return;
    }

    if (descriptors_.type() == CV_32F) {
        cv::reduce(descriptors_, global_descriptor_, 0, cv::REDUCE_AVG);
        double norm = cv::norm(global_descriptor_);
        if (norm > 1e-8) global_descriptor_ /= norm;
        return;
    }

    int n = descriptors_.rows;
    int bytes = descriptors_.cols;  // 32 for ORB (256 bits)
    int bits = bytes * 8;

    global_descriptor_ = cv::Mat::zeros(1, bits, CV_32F);
    float* gd = global_descriptor_.ptr<float>(0);

    for (int r = 0; r < n; r++) {
        const uchar* row = descriptors_.ptr<uchar>(r);
        for (int b = 0; b < bytes; b++) {
            uchar byte = row[b];
            for (int bit = 0; bit < 8; bit++) {
                if (byte & (1 << bit)) {
                    gd[b * 8 + bit] += 1.0f;
                }
            }
        }
    }

    for (int i = 0; i < bits; i++) {
        gd[i] /= (float)n;
    }

    double norm = cv::norm(global_descriptor_);
    if (norm > 1e-8) {
        global_descriptor_ /= norm;
    }
}

cv::Mat Frame::get_pose() const {
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R_.copyTo(T(cv::Rect(0, 0, 3, 3)));
    t_.copyTo(T(cv::Rect(3, 0, 1, 3)));
    return T;
}

void Frame::set_pose(const cv::Mat& R, const cv::Mat& t) {
    R.copyTo(R_);
    t.copyTo(t_);
}

double Frame::parse_timestamp(const std::string& filename) {
    std::regex re(R"((\d+\.\d+)\.png)");
    std::smatch match;
    if (std::regex_search(filename, match, re)) {
        return std::stod(match[1].str());
    }
    return 0.0;
}

cv::Mat Frame::draw_keypoints() const {
    cv::Mat output;
    cv::drawKeypoints(image_, keypoints_, output, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DEFAULT);
    return output;
}
