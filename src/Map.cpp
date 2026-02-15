#include "Map.h"

Map::Map() {}

void Map::add_frame(std::shared_ptr<Frame> frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    frames_.push_back(frame);
}

void Map::add_map_point(const MapPoint& mp) {
    std::lock_guard<std::mutex> lock(mutex_);
    map_points_.push_back(mp);
}

std::shared_ptr<Frame> Map::get_frame(int id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& f : frames_) {
        if (f->id() == id) return f;
    }
    return nullptr;
}

std::vector<std::shared_ptr<Frame>> Map::get_all_frames() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_;
}

int Map::frame_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (int)frames_.size();
}

std::vector<std::shared_ptr<Frame>> Map::get_keyframes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Frame>> kfs;
    for (const auto& f : frames_) {
        if (f->is_keyframe()) kfs.push_back(f);
    }
    return kfs;
}

void Map::add_display_point(const cv::Point3d& pt) {
    display_points_.push_back(pt);
}

std::vector<cv::Point3d> Map::get_all_display_points() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return display_points_;
}

std::vector<cv::Point3d> Map::get_all_point_positions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<cv::Point3d> pts;
    pts.reserve(map_points_.size());
    for (const auto& mp : map_points_) {
        if (mp.is_valid()) {
            pts.push_back(mp.position());
        }
    }
    return pts;
}

std::vector<cv::Point3d> Map::get_trajectory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<cv::Point3d> traj;
    traj.reserve(frames_.size());
    for (const auto& f : frames_) {
        cv::Mat t = f->get_translation();
        traj.emplace_back(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    }
    return traj;
}

std::vector<cv::Mat> Map::get_all_poses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<cv::Mat> poses;
    poses.reserve(frames_.size());
    for (const auto& f : frames_) {
        poses.push_back(f->get_pose());
    }
    return poses;
}
