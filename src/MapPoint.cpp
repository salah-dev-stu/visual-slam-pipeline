#include "MapPoint.h"

MapPoint::MapPoint()
    : id_(-1), position_(0, 0, 0), valid_(false) {}

MapPoint::MapPoint(int id, const cv::Point3d& position, const cv::Mat& descriptor)
    : id_(id), position_(position), valid_(true) {
    if (!descriptor.empty()) {
        descriptor_ = descriptor.clone();
    }
}

void MapPoint::add_observation(int frame_id, int keypoint_idx) {
    observations_.emplace_back(frame_id, keypoint_idx);
}
