#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace Ort { class Session; class Env; class SessionOptions; class MemoryInfo; }

class FeatureExtractor {
public:
    FeatureExtractor();
    ~FeatureExtractor();

    bool init(const std::string& model_path);

    void extract(const cv::Mat& image,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors);

    bool using_superpoint() const { return use_superpoint_; }

    // Feature cache for deterministic GPU results
    void set_cache_path(const std::string& path);
    bool load_cache();
    bool save_cache();
    bool cache_active() const { return cache_loaded_; }
    int cache_size() const { return (int)cache_.size(); }

private:
    void extract_superpoint(const cv::Mat& gray,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::Mat& descriptors);

    void extract_orb(const cv::Mat& gray,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::Mat& descriptors);

    void nms(const cv::Mat& heatmap, int radius,
             std::vector<cv::KeyPoint>& keypoints, int max_keypoints);

    bool use_superpoint_;
    cv::Ptr<cv::ORB> orb_;

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    // Cache members
    struct CachedFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };
    std::string cache_path_;
    bool cache_loaded_ = false;
    int extract_counter_ = 0;
    std::unordered_map<int, CachedFeatures> cache_;
};

#endif
