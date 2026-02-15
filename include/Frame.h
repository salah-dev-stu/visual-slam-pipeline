#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>

class FeatureExtractor;
class DepthEstimator;

class Frame {
public:
    Frame();
    Frame(int id, const std::string& image_path, double timestamp = 0.0);

    void detect_features(FeatureExtractor& extractor);
    void estimate_depth(DepthEstimator& estimator);
    void load_depth_image(const std::string& depth_path);
    void compute_global_descriptor();

    bool has_real_depth() const { return has_real_depth_; }

    cv::Mat get_rotation() const { return R_.clone(); }
    cv::Mat get_translation() const { return t_.clone(); }
    cv::Mat get_pose() const;

    void set_rotation(const cv::Mat& R) { R.copyTo(R_); }
    void set_translation(const cv::Mat& t) { t.copyTo(t_); }
    void set_pose(const cv::Mat& R, const cv::Mat& t);

    int id() const { return id_; }
    double timestamp() const { return timestamp_; }
    const cv::Mat& image() const { return image_; }
    const cv::Mat& gray() const { return gray_; }
    const std::vector<cv::KeyPoint>& keypoints() const { return keypoints_; }
    const cv::Mat& descriptors() const { return descriptors_; }
    const cv::Mat& depth_map() const { return depth_map_; }
    void set_depth_map(const cv::Mat& depth) { depth_map_ = depth.clone(); has_real_depth_ = true; }
    const cv::Mat& global_descriptor() const { return global_descriptor_; }
    bool is_processed() const { return processed_; }

    bool is_keyframe() const { return is_keyframe_; }
    void set_keyframe(bool kf) { is_keyframe_ = kf; }

    std::vector<int>& map_point_indices() { return map_point_indices_; }
    const std::vector<int>& map_point_indices() const { return map_point_indices_; }

    static double parse_timestamp(const std::string& filename);
    cv::Mat draw_keypoints() const;

private:
    int id_;
    double timestamp_;
    std::string image_path_;

    cv::Mat image_;
    cv::Mat gray_;

    std::vector<cv::KeyPoint> keypoints_;
    cv::Mat descriptors_;
    cv::Mat depth_map_;
    cv::Mat global_descriptor_;

    cv::Mat R_;
    cv::Mat t_;

    bool processed_;
    bool is_keyframe_;
    bool has_real_depth_;
    std::vector<int> map_point_indices_;
};

#endif
