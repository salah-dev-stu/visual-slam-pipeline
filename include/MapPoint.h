#ifndef MAP_POINT_H
#define MAP_POINT_H

#include <opencv2/core.hpp>
#include <vector>
#include <utility>

class MapPoint {
public:
    MapPoint();
    MapPoint(int id, const cv::Point3d& position, const cv::Mat& descriptor);

    int id() const { return id_; }

    cv::Point3d position() const { return position_; }
    void set_position(const cv::Point3d& pos) { position_ = pos; }

    const cv::Mat& descriptor() const { return descriptor_; }

    void add_observation(int frame_id, int keypoint_idx);
    const std::vector<std::pair<int, int>>& observations() const { return observations_; }
    int observation_count() const { return (int)observations_.size(); }

    bool is_valid() const { return valid_; }
    void set_valid(bool v) { valid_ = v; }

    // ORB-SLAM3 style tracking
    void increase_visible(int n = 1) { visible_count_ += n; }
    void increase_found(int n = 1) { found_count_ += n; }
    float get_found_ratio() const { return visible_count_ > 0 ? (float)found_count_ / visible_count_ : 0.0f; }
    int visible_count() const { return visible_count_; }
    int found_count() const { return found_count_; }

    void set_first_kf_id(int id) { first_kf_id_ = id; }
    int first_kf_id() const { return first_kf_id_; }

private:
    int id_;
    cv::Point3d position_;
    std::vector<std::pair<int, int>> observations_;
    cv::Mat descriptor_;
    bool valid_;

    // Visibility tracking (ORB-SLAM3 style)
    int visible_count_ = 0;
    int found_count_ = 0;
    int first_kf_id_ = 0;
};

#endif
