#ifndef SLAM_H
#define SLAM_H

#include "Frame.h"
#include "FeatureExtractor.h"
#include "DepthEstimator.h"
#include "Map.h"
#include "Optimizer.h"
#include "LoopCloser.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <memory>
#include <utility>

class Slam {
public:
    Slam();

    bool init(const std::string& model_dir);
    void set_initial_pose(const cv::Mat& R, const cv::Mat& t);
    void seed_motion(const cv::Mat& direction);
    void compute_gravity_direction();
    bool process_frame(std::shared_ptr<Frame> frame);

    Map& map() { return map_; }
    const Map& map() const { return map_; }

    int frame_count() const { return frame_count_; }
    int last_match_count() const { return last_match_count_; }
    int last_inlier_count() const { return last_inlier_count_; }
    int keyframe_count() const { return keyframe_count_; }
    int map_point_count() const;
    int loop_count() const { return loop_closer_.loop_count(); }

    // Post-hoc pose graph optimization (run after all frames processed)
    void run_posthoc_pgo();

    // RTS backward smoother (run after all frames processed)
    void run_rts_smoother();

    double last_epipolar_error_before() const { return epipolar_error_before_; }
    double last_epipolar_error_after() const { return epipolar_error_after_; }
    double last_reproj_error_before() const { return reproj_error_before_; }
    double last_reproj_error_after() const { return reproj_error_after_; }

    bool last_was_pnp() const { return last_pnp_; }
    bool last_was_loop() const { return last_loop_; }

    // Match visualization data
    std::shared_ptr<Frame> ref_frame() const { return ref_frame_; }
    const std::vector<cv::DMatch>& last_matches_before() const { return last_matches_before_; }
    const std::vector<cv::DMatch>& last_matches_after() const { return last_matches_after_; }

    FeatureExtractor& feature_extractor() { return feature_extractor_; }
    DepthEstimator& depth_estimator() { return depth_estimator_; }

    // Loop closure edges for viewer
    std::vector<std::pair<cv::Point3d, cv::Point3d>> get_loop_edges() const;

    struct AccelSample {
        double timestamp;
        double ax, ay, az;
    };
    void set_accelerometer_data(const std::vector<AccelSample>& data);

private:
    std::vector<cv::DMatch> match_features(const cv::Mat& desc1, const cv::Mat& desc2,
                                              std::vector<cv::DMatch>* raw_matches_out = nullptr);

    void extract_matched_points(const std::vector<cv::KeyPoint>& kp1,
                                const std::vector<cv::KeyPoint>& kp2,
                                const std::vector<cv::DMatch>& matches,
                                std::vector<cv::Point2f>& pts1,
                                std::vector<cv::Point2f>& pts2);

    bool estimate_motion(const std::vector<cv::Point2f>& pts1,
                         const std::vector<cv::Point2f>& pts2,
                         cv::Mat& R, cv::Mat& t, cv::Mat& mask);

    double compute_epipolar_error(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  const cv::Mat& F);

    void triangulate_points(std::shared_ptr<Frame> frame1,
                            std::shared_ptr<Frame> frame2,
                            const std::vector<cv::DMatch>& matches);

    bool is_keyframe(std::shared_ptr<Frame> frame, int match_count);

    void run_pnp(std::shared_ptr<Frame> frame);

    void create_points_from_depth(std::shared_ptr<Frame> frame);

    int track_local_map(std::shared_ptr<Frame> frame);

    bool estimate_motion_3d3d(const std::vector<cv::Point2f>& pts1,
                               const std::vector<cv::Point2f>& pts2,
                               std::shared_ptr<Frame> ref_frame,
                               std::shared_ptr<Frame> cur_frame,
                               cv::Mat& R_out, cv::Mat& t_out);

    double estimate_scale_from_depth(const std::vector<cv::Point2f>& pts1,
                                      const std::vector<cv::Point2f>& pts2,
                                      const cv::Mat& R_rel, const cv::Mat& t_rel,
                                      std::shared_ptr<Frame> current_frame);

    double estimate_scale_single_depth(const std::vector<cv::Point2f>& pts1,
                                        const std::vector<cv::Point2f>& pts2,
                                        const cv::Mat& R_rel, const cv::Mat& t_rel,
                                        const cv::Mat& depth1);

private:
    FeatureExtractor feature_extractor_;
    DepthEstimator depth_estimator_;
    Map map_;
    Optimizer optimizer_;
    LoopCloser loop_closer_;

    cv::Mat K_;
    cv::Mat R_world_;
    cv::Mat t_world_;
    cv::Mat last_translation_;  // for keyframe decisions

    std::shared_ptr<Frame> last_frame_;
    std::shared_ptr<Frame> last_keyframe_;
    std::shared_ptr<Frame> ref_frame_;  // reference frame for Essential matrix (keyframe when available)

    int frame_count_;
    int keyframe_count_;
    int last_match_count_;
    int last_inlier_count_;

    double epipolar_error_before_;
    double epipolar_error_after_;
    double reproj_error_before_;
    double reproj_error_after_;

    bool last_pnp_;
    bool last_loop_;

    std::vector<cv::DMatch> last_matches_before_;
    std::vector<cv::DMatch> last_matches_after_;

    std::vector<std::pair<int, int>> loop_edges_;

    double last_good_scale_ = -1.0;

    cv::Mat R_prev_;
    cv::Mat t_prev_;
    bool have_prev_pose_ = false;

    std::vector<AccelSample> accel_data_;
    bool is_frame_stationary(double timestamp) const;
    cv::Mat gravity_world_;  // gravity unit vector in world frame (from accelerometer)
    double initial_height_ = 0.0;  // initial height along gravity direction (absolute anchor)
    bool has_initial_height_ = false;
    bool was_stationary_ = false;
    int pnp_recovery_cooldown_ = 0;

    cv::Ptr<cv::FlannBasedMatcher> matcher_l2_;
    cv::Ptr<cv::BFMatcher> matcher_hamming_;

    std::vector<LoopConstraint> loop_constraints_;  // accumulated for PGO

    // EKF state (6D: position + velocity)
    cv::Mat ekf_x_;          // 6x1 state [px,py,pz, vx,vy,vz]
    cv::Mat ekf_P_;          // 6x6 covariance
    bool ekf_initialized_ = false;
    double last_frame_time_ = 0;

    // EKF methods
    void ekf_initialize(const cv::Mat& pos, double timestamp);
    void ekf_predict(double dt);
    void ekf_update_visual(const cv::Mat& z_pos, double sigma_vis);
    void ekf_update_height(double h_measured, double sigma_h);

    // EKF snapshots for RTS backward smoother
    struct EKFSnapshot {
        cv::Mat x_pred;   // 6x1 predicted state (after predict, before updates)
        cv::Mat P_pred;   // 6x6 predicted covariance
        cv::Mat x_filt;   // 6x1 filtered state (after all updates + step clamp)
        cv::Mat P_filt;   // 6x6 filtered covariance (after updates, before step clamp)
        double dt;
        int frame_id;     // map frame index for pose update
    };
    std::vector<EKFSnapshot> ekf_snapshots_;
};

#endif
