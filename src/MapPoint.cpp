#include "MapPoint.h"

/// Default constructor: creates an invalid map point at the origin.
MapPoint::MapPoint()
    : id_(-1), position_(0, 0, 0), valid_(false) {}

/// Constructs a valid map point with a 3D position and optional descriptor.
MapPoint::MapPoint(int id, const cv::Point3d& position, const cv::Mat& descriptor)
    : id_(id), position_(position), valid_(true) {
    if (!descriptor.empty()) {
        descriptor_ = descriptor.clone();
    }
}

/// Records that this point was observed as keypoint keypoint_idx in the given frame.
void MapPoint::add_observation(int frame_id, int keypoint_idx) {
    observations_.emplace_back(frame_id, keypoint_idx);
}
