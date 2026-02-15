#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <opencv2/core.hpp>
#include <vector>
#include <memory>

class Frame;
class Map;

struct LoopConstraint {
    int from_id;     // keyframe frame id
    int to_id;       // keyframe frame id
    cv::Mat R_rel;   // relative rotation (from -> to)
    cv::Mat t_rel;   // relative translation (from -> to)
    double trans_sigma;
    double rot_sigma;
};

class Optimizer {
public:
    Optimizer();

    static cv::Point2d project_point(const cv::Point3d& pw,
                                     const cv::Mat& R_world,
                                     const cv::Mat& t_world,
                                     const cv::Mat& K);

    std::pair<double, double> optimize_pose(
        std::shared_ptr<Frame> frame,
        const std::vector<cv::Point3d>& points_3d,
        const std::vector<cv::Point2f>& points_2d,
        const cv::Mat& K);

    std::pair<double, double> local_bundle_adjustment(
        Map& map,
        const cv::Mat& K,
        int window_size = 10);

    void correct_loop(Map& map, int loop_start_id, int loop_end_id,
                      const cv::Mat& R_correction, const cv::Mat& t_correction);

    int pose_graph_optimize(
        Map& map,
        const std::vector<LoopConstraint>& loop_constraints,
        const cv::Mat& gravity_world,
        double initial_height,
        bool has_height_prior);
};

#endif
