#ifndef MAP_H
#define MAP_H

#include "Frame.h"
#include "MapPoint.h"
#include <vector>
#include <memory>
#include <mutex>

class Map {
public:
    Map();

    void add_frame(std::shared_ptr<Frame> frame);
    void add_map_point(const MapPoint& mp);

    std::shared_ptr<Frame> get_frame(int id) const;
    std::vector<std::shared_ptr<Frame>> get_all_frames() const;
    int frame_count() const;

    std::vector<std::shared_ptr<Frame>> get_keyframes() const;

    std::vector<cv::Point3d> get_all_point_positions() const;
    std::vector<cv::Point3d> get_all_display_points() const;
    void add_display_point(const cv::Point3d& pt);
    std::vector<cv::Point3d> get_trajectory() const;
    std::vector<cv::Mat> get_all_poses() const;

    std::vector<MapPoint>& map_points() { return map_points_; }
    const std::vector<MapPoint>& map_points() const { return map_points_; }
    std::vector<std::shared_ptr<Frame>>& frames_direct() { return frames_; }

    std::mutex& mutex() { return mutex_; }

private:
    std::vector<std::shared_ptr<Frame>> frames_;
    std::vector<MapPoint> map_points_;
    std::vector<cv::Point3d> display_points_;
    mutable std::mutex mutex_;
};

#endif
