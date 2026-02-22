#include "Slam.h"
#include "Config.h"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>


/// Initializes SLAM state: camera intrinsics, identity pose, and feature matchers.
Slam::Slam()
    : frame_count_(0), keyframe_count_(0),
      last_match_count_(0), last_inlier_count_(0),
      epipolar_error_before_(0), epipolar_error_after_(0),
      reproj_error_before_(0), reproj_error_after_(0),
      last_pnp_(false), last_loop_(false) {

    K_ = Config::getCameraMatrix();
    R_world_ = cv::Mat::eye(3, 3, CV_64F);
    t_world_ = cv::Mat::zeros(3, 1, CV_64F);

    matcher_l2_ = cv::FlannBasedMatcher::create();
    matcher_hamming_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
}

/// Loads the SuperPoint and MiDaS ONNX models from the given directory.
bool Slam::init(const std::string& model_dir) {
    feature_extractor_.init(model_dir + "/superpoint_v1.onnx");
    depth_estimator_.init(model_dir + "/midas_v21_small_256.onnx");
    return true;
}

/// Sets the initial camera pose in world coordinates.
void Slam::set_initial_pose(const cv::Mat& R, const cv::Mat& t) {
    R.copyTo(R_world_);
    t.copyTo(t_world_);
}

/// Seeds the motion model with a known translation direction.
void Slam::seed_motion(const cv::Mat& direction) {
    last_translation_ = direction.clone();
}

/// Returns the number of valid 3D map points.
int Slam::map_point_count() const {
    auto pts = map_.get_all_point_positions();
    return (int)pts.size();
}

/// Returns loop closure edges as pairs of 3D positions for visualization.
std::vector<std::pair<cv::Point3d, cv::Point3d>> Slam::get_loop_edges() const {
    std::vector<std::pair<cv::Point3d, cv::Point3d>> edges;
    for (const auto& e : loop_edges_) {
        auto f1 = map_.get_frame(e.first);
        auto f2 = map_.get_frame(e.second);
        if (f1 && f2) {
            cv::Mat t1 = f1->get_translation();
            cv::Mat t2 = f2->get_translation();
            edges.emplace_back(
                cv::Point3d(t1.at<double>(0), t1.at<double>(1), t1.at<double>(2)),
                cv::Point3d(t2.at<double>(0), t2.at<double>(1), t2.at<double>(2)));
        }
    }
    return edges;
}

/// Estimates the translation scale using depth from both reference and current frames.
/// Back-projects matched 2D points to 3D using Kinect depth, then solves for the scale
/// factor s in: P2 = R_rel * P1 + s * t_rel. Uses IQR-based outlier rejection
/// and returns the median of filtered scale estimates.
/// Falls back to single-depth estimation if the current frame lacks depth.
double Slam::estimate_scale_from_depth(const std::vector<cv::Point2f>& pts1,
                                        const std::vector<cv::Point2f>& pts2,
                                        const cv::Mat& R_rel, const cv::Mat& t_rel,
                                        std::shared_ptr<Frame> current_frame) {
    auto ref = ref_frame_ ? ref_frame_ : last_frame_;
    if (!ref || !ref->has_real_depth() || ref->depth_map().empty())
        return -1.0;

    const cv::Mat& depth1 = ref->depth_map();

    auto cur = current_frame;
    if (!cur || !cur->has_real_depth() || cur->depth_map().empty()) {
        return estimate_scale_single_depth(pts1, pts2, R_rel, t_rel, depth1);
    }

    const cv::Mat& depth2 = cur->depth_map();
    double fx = K_.at<double>(0, 0), fy = K_.at<double>(1, 1);
    double cx = K_.at<double>(0, 2), cy = K_.at<double>(1, 2);

    double tx = t_rel.at<double>(0), ty = t_rel.at<double>(1), tz = t_rel.at<double>(2);

    std::vector<double> scales;
    scales.reserve(pts1.size());

    for (size_t i = 0; i < pts1.size(); i++) {
        int px1 = (int)std::round(pts1[i].x);
        int py1 = (int)std::round(pts1[i].y);
        int px2 = (int)std::round(pts2[i].x);
        int py2 = (int)std::round(pts2[i].y);

        if (px1 < 0 || px1 >= depth1.cols || py1 < 0 || py1 >= depth1.rows) continue;
        if (px2 < 0 || px2 >= depth2.cols || py2 < 0 || py2 >= depth2.rows) continue;

        float d1 = depth1.at<float>(py1, px1);
        float d2 = depth2.at<float>(py2, px2);

        if (d1 <= Config::DEPTH_MIN || d1 > Config::DEPTH_MAX) continue;
        if (d2 <= Config::DEPTH_MIN || d2 > Config::DEPTH_MAX) continue;

        cv::Mat P1 = (cv::Mat_<double>(3, 1) <<
            (pts1[i].x - cx) * d1 / fx,
            (pts1[i].y - cy) * d1 / fy,
            (double)d1);

        cv::Mat P2 = (cv::Mat_<double>(3, 1) <<
            (pts2[i].x - cx) * d2 / fx,
            (pts2[i].y - cy) * d2 / fy,
            (double)d2);

        // s = (P2 - R_rel * P1) dot t_rel  (since ||t_rel|| = 1)
        cv::Mat diff = P2 - R_rel * P1;
        double s = diff.at<double>(0) * tx + diff.at<double>(1) * ty + diff.at<double>(2) * tz;

        if (s > 0.001 && s < 50.0) {
            scales.push_back(s);
        }
    }

    if (scales.size() < 10) {
        return estimate_scale_single_depth(pts1, pts2, R_rel, t_rel, depth1);
    }

    // IQR-based outlier rejection
    std::sort(scales.begin(), scales.end());
    int q1_idx = scales.size() / 4;
    int q3_idx = 3 * scales.size() / 4;
    double q1 = scales[q1_idx], q3 = scales[q3_idx];
    double iqr = q3 - q1;
    double lo = q1 - 1.5 * iqr, hi = q3 + 1.5 * iqr;

    std::vector<double> filtered;
    for (double s : scales) {
        if (s >= lo && s <= hi) filtered.push_back(s);
    }

    if (filtered.empty()) {
        std::sort(scales.begin(), scales.end());
        return scales[scales.size() / 2];
    }

    std::sort(filtered.begin(), filtered.end());
    double result = filtered[filtered.size() / 2];

    return result;
}

/// Estimates scale using only the reference frame's depth map (single-view fallback).
/// For each matched point, back-projects from ref depth and solves for scale using
/// the reprojection constraint in x and y directions independently.
double Slam::estimate_scale_single_depth(const std::vector<cv::Point2f>& pts1,
                                          const std::vector<cv::Point2f>& pts2,
                                          const cv::Mat& R_rel, const cv::Mat& t_rel,
                                          const cv::Mat& depth1) {
    double fx = K_.at<double>(0, 0), fy = K_.at<double>(1, 1);
    double cx = K_.at<double>(0, 2), cy = K_.at<double>(1, 2);

    std::vector<double> scales;
    scales.reserve(pts1.size());

    for (size_t i = 0; i < pts1.size(); i++) {
        int px1 = (int)std::round(pts1[i].x);
        int py1 = (int)std::round(pts1[i].y);
        if (px1 < 0 || px1 >= depth1.cols || py1 < 0 || py1 >= depth1.rows) continue;

        float d1 = depth1.at<float>(py1, px1);
        if (d1 <= Config::DEPTH_MIN || d1 > Config::DEPTH_MAX) continue;

        double X1 = (pts1[i].x - cx) * d1 / fx;
        double Y1 = (pts1[i].y - cy) * d1 / fy;
        double Z1 = d1;
        cv::Mat P1 = (cv::Mat_<double>(3, 1) << X1, Y1, Z1);

        cv::Mat P2_no_t = R_rel * P1;
        double Rx = P2_no_t.at<double>(0), Ry = P2_no_t.at<double>(1), Rz = P2_no_t.at<double>(2);
        double tx = t_rel.at<double>(0), ty = t_rel.at<double>(1), tz = t_rel.at<double>(2);

        double a = (pts2[i].x - cx) / fx;
        double denom_x = tx - a * tz;
        if (std::abs(denom_x) > 1e-4) {
            double s = (a * Rz - Rx) / denom_x;
            if (s > 0.001 && s < 100.0) scales.push_back(s);
        }

        double b = (pts2[i].y - cy) / fy;
        double denom_y = ty - b * tz;
        if (std::abs(denom_y) > 1e-4) {
            double s = (b * Rz - Ry) / denom_y;
            if (s > 0.001 && s < 100.0) scales.push_back(s);
        }
    }

    if (scales.size() < 10) return -1.0;
    std::sort(scales.begin(), scales.end());
    return scales[scales.size() / 2];
}

/// Estimates relative pose via 3D-3D rigid body alignment using RANSAC + SVD.
/// Back-projects matched 2D points from both frames using Kinect depth to get 3D-3D
/// correspondences, then finds the best rigid transform (R, t) via RANSAC with
/// 3-point SVD-based minimal solver. Refines with all inliers after RANSAC.
/// Preferred over Essential matrix when depth is available (metric, no scale ambiguity).
bool Slam::estimate_motion_3d3d(const std::vector<cv::Point2f>& pts1,
                                 const std::vector<cv::Point2f>& pts2,
                                 std::shared_ptr<Frame> ref_frame,
                                 std::shared_ptr<Frame> cur_frame,
                                 cv::Mat& R_out, cv::Mat& t_out) {
    if (!ref_frame || !ref_frame->has_real_depth() || ref_frame->depth_map().empty()) {
        return false;
    }
    if (!cur_frame || !cur_frame->has_real_depth() || cur_frame->depth_map().empty()) {
        return false;
    }

    const cv::Mat& depth1 = ref_frame->depth_map();
    const cv::Mat& depth2 = cur_frame->depth_map();
    double fx = K_.at<double>(0, 0), fy = K_.at<double>(1, 1);
    double cx = K_.at<double>(0, 2), cy = K_.at<double>(1, 2);

    // Back-project matched 2D points to 3D using depth
    std::vector<cv::Mat> pts3d_1, pts3d_2;
    pts3d_1.reserve(pts1.size());
    pts3d_2.reserve(pts1.size());

    for (size_t i = 0; i < pts1.size(); i++) {
        int px1 = (int)std::round(pts1[i].x);
        int py1 = (int)std::round(pts1[i].y);
        int px2 = (int)std::round(pts2[i].x);
        int py2 = (int)std::round(pts2[i].y);

        if (px1 < 0 || px1 >= depth1.cols || py1 < 0 || py1 >= depth1.rows) continue;
        if (px2 < 0 || px2 >= depth2.cols || py2 < 0 || py2 >= depth2.rows) continue;

        float d1 = depth1.at<float>(py1, px1);
        float d2 = depth2.at<float>(py2, px2);

        if (d1 <= Config::DEPTH_MIN || d1 > Config::DEPTH_MAX) continue;
        if (d2 <= Config::DEPTH_MIN || d2 > Config::DEPTH_MAX) continue;

        cv::Mat P1 = (cv::Mat_<double>(3, 1) <<
            (pts1[i].x - cx) * d1 / fx,
            (pts1[i].y - cy) * d1 / fy,
            (double)d1);
        cv::Mat P2 = (cv::Mat_<double>(3, 1) <<
            (pts2[i].x - cx) * d2 / fx,
            (pts2[i].y - cy) * d2 / fy,
            (double)d2);

        pts3d_1.push_back(P1);
        pts3d_2.push_back(P2);
    }

    int N = (int)pts3d_1.size();
    if (N < 10) {
        return false;
    }

    // RANSAC with SVD-based rigid transform (3-point minimal solver)
    const int MAX_ITER = Config::RANSAC_3D3D_ITERATIONS;
    const double INLIER_THRESH = Config::RANSAC_3D3D_INLIER_THRESH;

    int best_inliers = 0;
    cv::Mat best_R, best_t;

    std::mt19937 rng(42 + frame_count_);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Sample 3 random correspondences
        int i0 = rng() % N;
        int i1, i2;
        do { i1 = rng() % N; } while (i1 == i0);
        do { i2 = rng() % N; } while (i2 == i0 || i2 == i1);

        cv::Mat c1 = (pts3d_1[i0] + pts3d_1[i1] + pts3d_1[i2]) / 3.0;
        cv::Mat c2 = (pts3d_2[i0] + pts3d_2[i1] + pts3d_2[i2]) / 3.0;

        // Cross-covariance H = sum((P1_i - c1) * (P2_i - c2)^T)
        cv::Mat H = (pts3d_1[i0] - c1) * (pts3d_2[i0] - c2).t()
                  + (pts3d_1[i1] - c1) * (pts3d_2[i1] - c2).t()
                  + (pts3d_1[i2] - c1) * (pts3d_2[i2] - c2).t();

        cv::Mat W, U, Vt;
        cv::SVD::compute(H, W, U, Vt);
        cv::Mat V = Vt.t();

        cv::Mat R_cand = V * U.t();
        if (cv::determinant(R_cand) < 0) {
            V.col(2) *= -1;
            R_cand = V * U.t();
        }

        cv::Mat t_cand = c2 - R_cand * c1;

        int inliers = 0;
        for (int j = 0; j < N; j++) {
            cv::Mat diff = pts3d_2[j] - (R_cand * pts3d_1[j] + t_cand);
            if (cv::norm(diff) < INLIER_THRESH) {
                inliers++;
            }
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_R = R_cand.clone();
            best_t = t_cand.clone();
        }
    }

    if (best_inliers < 10) {
        return false;
    }

    // Refine with all RANSAC inliers
    std::vector<int> inlier_idx;
    inlier_idx.reserve(best_inliers);
    cv::Mat c1 = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat c2 = cv::Mat::zeros(3, 1, CV_64F);

    for (int j = 0; j < N; j++) {
        cv::Mat diff = pts3d_2[j] - (best_R * pts3d_1[j] + best_t);
        if (cv::norm(diff) < INLIER_THRESH) {
            c1 += pts3d_1[j];
            c2 += pts3d_2[j];
            inlier_idx.push_back(j);
        }
    }

    int inlier_count = (int)inlier_idx.size();
    c1 /= inlier_count;
    c2 /= inlier_count;

    cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
    for (int j : inlier_idx) {
        H += (pts3d_1[j] - c1) * (pts3d_2[j] - c2).t();
    }

    cv::Mat W, U, Vt;
    cv::SVD::compute(H, W, U, Vt);
    cv::Mat V = Vt.t();

    R_out = V * U.t();
    if (cv::determinant(R_out) < 0) {
        V.col(2) *= -1;
        R_out = V * U.t();
    }

    t_out = c2 - R_out * c1;

    // Sanity checks: reject excessive or negligible translation
    double t_norm = cv::norm(t_out);
    if (t_norm > Config::RANSAC_3D3D_MAX_TRANSLATION) {
        return false;
    }
    if (t_norm < 0.0001) {
        return false;
    }

    double det = cv::determinant(R_out);
    if (std::abs(det - 1.0) > 0.01) {
        return false;
    }

    return true;
}

/// Tracks existing map points by projecting them into the current frame and matching
/// descriptors within a spatial search window. Uses a grid-based spatial index for
/// efficient 2D lookup. Updates frame's map_point_indices with matched point IDs.
int Slam::track_local_map(std::shared_ptr<Frame> frame) {
    auto& indices = frame->map_point_indices();
    const auto& kps = frame->keypoints();
    const cv::Mat& descs = frame->descriptors();
    if (kps.empty() || descs.empty()) return 0;

    int nkp = (int)kps.size();
    std::vector<double> best_desc_dist(nkp, 1e9);

    // Build grid index for keypoints for efficient spatial search
    const int CELL_SIZE = Config::TRACK_GRID_CELL_SIZE;
    const int GRID_W = (Config::IMAGE_WIDTH + CELL_SIZE - 1) / CELL_SIZE;
    const int GRID_H = (Config::IMAGE_HEIGHT + CELL_SIZE - 1) / CELL_SIZE;
    std::vector<std::vector<int>> grid(GRID_W * GRID_H);

    for (int ki = 0; ki < nkp; ki++) {
        int gx = std::min((int)(kps[ki].pt.x / CELL_SIZE), GRID_W - 1);
        int gy = std::min((int)(kps[ki].pt.y / CELL_SIZE), GRID_H - 1);
        if (gx >= 0 && gy >= 0) {
            grid[gy * GRID_W + gx].push_back(ki);
        }
    }

    cv::Mat R = frame->get_rotation();
    cv::Mat t = frame->get_translation();
    cv::Mat R_cam = R.t();
    cv::Mat t_cam = -R_cam * t;

    const double SEARCH_RADIUS = Config::TRACK_SEARCH_RADIUS;
    const double SEARCH_RADIUS_SQ = SEARCH_RADIUS * SEARCH_RADIUS;
    const double DESC_THRESHOLD = Config::TRACK_DESC_THRESHOLD;
    int tracked = 0;

    std::lock_guard<std::mutex> lock(map_.mutex());
    auto& mps = map_.map_points();

    for (int mp_id = 0; mp_id < (int)mps.size(); mp_id++) {
        if (!mps[mp_id].is_valid()) continue;
        if (mps[mp_id].descriptor().empty()) continue;

        // Project map point to image coordinates
        const cv::Point3d& pos = mps[mp_id].position();
        double px = R_cam.at<double>(0,0)*pos.x + R_cam.at<double>(0,1)*pos.y + R_cam.at<double>(0,2)*pos.z + t_cam.at<double>(0);
        double py = R_cam.at<double>(1,0)*pos.x + R_cam.at<double>(1,1)*pos.y + R_cam.at<double>(1,2)*pos.z + t_cam.at<double>(1);
        double pz = R_cam.at<double>(2,0)*pos.x + R_cam.at<double>(2,1)*pos.y + R_cam.at<double>(2,2)*pos.z + t_cam.at<double>(2);

        if (pz < Config::DEPTH_MIN || pz > Config::TRIANG_MAX_DEPTH) continue;

        double u = Config::FX * px / pz + Config::CX;
        double v = Config::FY * py / pz + Config::CY;

        if (u < 0 || u >= Config::IMAGE_WIDTH || v < 0 || v >= Config::IMAGE_HEIGHT) continue;

        // Search in nearby grid cells for descriptor match
        int gx0 = std::max(0, (int)((u - SEARCH_RADIUS) / CELL_SIZE));
        int gy0 = std::max(0, (int)((v - SEARCH_RADIUS) / CELL_SIZE));
        int gx1 = std::min(GRID_W - 1, (int)((u + SEARCH_RADIUS) / CELL_SIZE));
        int gy1 = std::min(GRID_H - 1, (int)((v + SEARCH_RADIUS) / CELL_SIZE));

        int best_ki = -1;
        double best_dist = DESC_THRESHOLD;

        const cv::Mat& mp_desc = mps[mp_id].descriptor();

        for (int gy = gy0; gy <= gy1; gy++) {
            for (int gx = gx0; gx <= gx1; gx++) {
                for (int ki : grid[gy * GRID_W + gx]) {
                    double dx = u - kps[ki].pt.x;
                    double dy = v - kps[ki].pt.y;
                    if (dx*dx + dy*dy > SEARCH_RADIUS_SQ) continue;

                    double desc_dist = cv::norm(mp_desc, descs.row(ki), cv::NORM_L2);
                    if (desc_dist < best_dist) {
                        best_dist = desc_dist;
                        best_ki = ki;
                    }
                }
            }
        }

        if (best_ki >= 0 && best_dist < best_desc_dist[best_ki]) {
            indices[best_ki] = mp_id;
            best_desc_dist[best_ki] = best_dist;
            mps[mp_id].add_observation(frame->id(), best_ki);
            tracked++;
        }
    }

    return tracked;
}

/// Invalidates map points that reproject with large errors (>20px) in the given frame.
/// Used after keyframe creation to remove grossly inconsistent points.
void Slam::cull_map_points(std::shared_ptr<Frame> frame) {
    std::lock_guard<std::mutex> lock(map_.mutex());
    auto& mps = map_.map_points();
    const auto& indices = frame->map_point_indices();
    cv::Mat R_cur = frame->get_rotation();
    cv::Mat t_cur = frame->get_translation();
    cv::Mat R_cam = R_cur.t();
    cv::Mat t_cam = -R_cam * t_cur;
    double fx = K_.at<double>(0,0), fy = K_.at<double>(1,1);
    double cx = K_.at<double>(0,2), cy = K_.at<double>(1,2);
    for (int i = 0; i < (int)indices.size(); i++) {
        int mp_id = indices[i];
        if (mp_id >= 0 && mp_id < (int)mps.size() && mps[mp_id].is_valid()) {
            cv::Point3d pw = mps[mp_id].position();
            cv::Mat Pw = (cv::Mat_<double>(3,1) << pw.x, pw.y, pw.z);
            cv::Mat pc = R_cam * Pw + t_cam;
            double z = pc.at<double>(2);
            if (z < Config::DEPTH_MIN) { mps[mp_id].set_valid(false); continue; }
            double u = fx * pc.at<double>(0) / z + cx;
            double v = fy * pc.at<double>(1) / z + cy;
            double dx = u - frame->keypoints()[i].pt.x;
            double dy = v - frame->keypoints()[i].pt.y;
            if (dx*dx + dy*dy > 400.0) {
                mps[mp_id].set_valid(false);
            }
        }
    }
}

/// Unified PnP solver: runs solvePnPRansac on the given 3D-2D correspondences,
/// converts the result from camera frame to world frame (R_world, t_world).
/// Returns success=false if PnP fails or inlier count is below min_inliers.
Slam::PnPResult Slam::solve_pnp(const std::vector<cv::Point3f>& obj_pts,
                                  const std::vector<cv::Point2f>& img_pts,
                                  int ransac_iters, int min_inliers) {
    PnPResult result;
    result.success = false;
    result.inlier_count = 0;

    if ((int)obj_pts.size() < min_inliers) return result;

    cv::Mat rvec, tvec, inliers;
    bool ok = cv::solvePnPRansac(obj_pts, img_pts, K_, cv::Mat(),
                                  rvec, tvec, false,
                                  ransac_iters, (float)Config::PNP_RANSAC_THRESHOLD,
                                  0.99, inliers);
    if (!ok || inliers.rows < min_inliers) return result;

    cv::Mat R_cam;
    cv::Rodrigues(rvec, R_cam);

    result.success = true;
    result.R_world = R_cam.t();
    result.t_world = -R_cam.t() * tvec;
    result.inlier_count = inliers.rows;
    return result;
}

/// Attempts PnP-based tracking recovery when feature matching with the reference
/// frame fails. Matches current frame descriptors against all valid map points
/// using FLANN, then solves PnP to recover the camera pose.
/// Returns: 1 = recovered, 0 = not needed (enough matches), -1 = failed
int Slam::try_pnp_recovery(std::shared_ptr<Frame> frame) {
    if (pnp_recovery_cooldown_ > 0) pnp_recovery_cooldown_--;

    if (last_match_count_ >= Config::MIN_MATCHES) return 0;

    // In cooldown period after a recent recovery — skip
    if (pnp_recovery_cooldown_ > 0) {
        last_frame_ = frame;
        return -1;
    }

    // Collect all valid map points and their descriptors for matching
    std::vector<cv::Point3f> all_obj_pts;
    cv::Mat all_mp_descs;
    {
        std::lock_guard<std::mutex> lock(map_.mutex());
        const auto& mps = map_.map_points();
        for (int mi = 0; mi < (int)mps.size(); mi++) {
            if (!mps[mi].is_valid() || mps[mi].descriptor().empty()) continue;
            cv::Point3d p = mps[mi].position();
            all_obj_pts.emplace_back((float)p.x, (float)p.y, (float)p.z);
            all_mp_descs.push_back(mps[mi].descriptor());
        }
    }

    if (all_mp_descs.rows >= 50 && !frame->descriptors().empty()) {
        std::vector<cv::Point3f> obj_pts;
        std::vector<cv::Point2f> img_pts;

        // Match current frame features against global map descriptors
        auto flann = cv::FlannBasedMatcher::create();
        std::vector<std::vector<cv::DMatch>> knn;
        flann->knnMatch(frame->descriptors(), all_mp_descs, knn, 2);

        for (const auto& m : knn) {
            if (m.size() >= 2 && m[0].distance < Config::FLANN_RATIO_THRESHOLD * m[1].distance) {
                obj_pts.push_back(all_obj_pts[m[0].trainIdx]);
                img_pts.push_back(frame->keypoints()[m[0].queryIdx].pt);
            }
        }

        if ((int)obj_pts.size() >= 20) {
            auto pnp = solve_pnp(obj_pts, img_pts, 300, 15);

            if (pnp.success) {
                double jump = cv::norm(pnp.t_world - t_world_);

                if (jump < Config::PNP_RECOVERY_MAX_JUMP) {
                    // Blend recovered pose with current estimate
                    double blend = (jump < 0.1) ? Config::PNP_RECOVERY_BLEND_CLOSE : Config::PNP_RECOVERY_BLEND_FAR;
                    R_world_ = pnp.R_world.clone();
                    t_world_ = (1.0 - blend) * t_world_ + blend * pnp.t_world;
                    frame->set_pose(R_world_, t_world_);
                    map_.add_frame(frame);

                    frame->set_keyframe(true);
                    keyframe_count_++;
                    create_points_from_depth(frame);
                    last_keyframe_ = frame;
                    last_frame_ = frame;
                    frame_count_++;
                    // Reset EKF state to recovered position
                    if (ekf_initialized_) {
                        for (int i = 0; i < 3; i++)
                            ekf_x_.at<double>(i) = t_world_.at<double>(i);
                        for (int i = 3; i < 6; i++)
                            ekf_x_.at<double>(i) = 0;
                    }
                    last_frame_time_ = frame->timestamp();
                    pnp_recovery_cooldown_ = 10;
                    return 1;  // Recovery succeeded
                }
            }
        }
    }

    last_frame_ = frame;
    return -1;  // Recovery failed
}

/// Handles processing when the camera is detected as stationary (via accelerometer).
/// Maintains current position, refines orientation via PnP with tracked map points,
/// and resets EKF velocity to zero. Returns true if the frame was stationary.
bool Slam::process_stationary_frame(std::shared_ptr<Frame> frame,
                                     const std::vector<cv::DMatch>& good_matches) {
    bool frame_stationary = is_frame_stationary(frame->timestamp());
    if (!frame_stationary || frame_count_ <= 5) return false;

    frame->set_pose(R_world_, t_world_);
    map_.add_frame(frame);

    int tracked = track_local_map(frame);

    // Refine rotation using tracked map points via PnP
    if (tracked >= 10) {
        std::vector<cv::Point3f> obj_pts;
        std::vector<cv::Point2f> img_pts;
        {
            std::lock_guard<std::mutex> lock(map_.mutex());
            const auto& mps = map_.map_points();
            const auto& indices = frame->map_point_indices();
            for (int i = 0; i < (int)indices.size(); i++) {
                int mp_id = indices[i];
                if (mp_id >= 0 && mp_id < (int)mps.size() && mps[mp_id].is_valid()) {
                    cv::Point3d p = mps[mp_id].position();
                    obj_pts.emplace_back((float)p.x, (float)p.y, (float)p.z);
                    img_pts.push_back(frame->keypoints()[i].pt);
                }
            }
        }
        auto pnp = solve_pnp(obj_pts, img_pts, 100, 10);
        if (pnp.success) {
            R_world_ = pnp.R_world;
            frame->set_pose(R_world_, t_world_);
        }
    }

    // Check if rotation has drifted enough to warrant a new keyframe
    if (last_keyframe_) {
        cv::Mat R_kf = last_keyframe_->get_rotation();
        cv::Mat R_diff = R_world_.t() * R_kf;
        cv::Mat rvec_diff;
        cv::Rodrigues(R_diff, rvec_diff);
        double angle = cv::norm(rvec_diff);
        if (angle > 0.25) {
            frame->set_keyframe(true);
            keyframe_count_++;
            create_points_from_depth(frame);
            last_keyframe_ = frame;
        }
    }

    last_frame_ = frame;
    last_match_count_ = (int)good_matches.size();
    last_inlier_count_ = last_match_count_;
    frame_count_++;
    was_stationary_ = true;

    // Zero out velocity during stationary periods
    last_translation_ = cv::Mat::zeros(3, 1, CV_64F);

    if (ekf_initialized_) {
        ekf_x_.at<double>(3) = 0;
        ekf_x_.at<double>(4) = 0;
        ekf_x_.at<double>(5) = 0;
        ekf_x_.at<double>(0) = t_world_.at<double>(0);
        ekf_x_.at<double>(1) = t_world_.at<double>(1);
        ekf_x_.at<double>(2) = t_world_.at<double>(2);
        for (int i = 3; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                ekf_P_.at<double>(i, j) = 0;
                ekf_P_.at<double>(j, i) = 0;
            }
            ekf_P_.at<double>(i, i) = 1e-4;
        }
    }
    last_frame_time_ = frame->timestamp();

    return true;
}

/// Common keyframe initialization: triangulates new map points with the previous
/// keyframe, creates depth-based map points, runs optional local BA, and culls
/// map points with large reprojection errors.
void Slam::setup_new_keyframe(std::shared_ptr<Frame> frame) {
    if (last_keyframe_) {
        auto kf_matches = match_features(last_keyframe_->descriptors(), frame->descriptors());
        if ((int)kf_matches.size() >= Config::MIN_MATCHES) {
            triangulate_points(last_keyframe_, frame, kf_matches);
        }
    }

    create_points_from_depth(frame);

    if (Config::ENABLE_LOCAL_BA) {
        cv::Mat t_before = frame->get_translation().clone();
        auto [ba_err_before, ba_err_after] = optimizer_.local_bundle_adjustment(map_, K_, 10);
        if (ba_err_after < ba_err_before && ba_err_after > 0) {
            cv::Mat t_after = frame->get_translation();
            double ba_jump = cv::norm(t_after - t_before);
            if (ba_jump < Config::BA_MAX_JUMP) {
                R_world_ = frame->get_rotation().clone();
                t_world_ = frame->get_translation().clone();
            } else {
                frame->set_pose(R_world_, t_world_);
            }
        }
    }

    cull_map_points(frame);
}

/// Detects loop closures by matching the current frame's global descriptor against
/// previous keyframes, then verifies via PnP on 2D-3D correspondences near the
/// matched frame. If verified, creates a PGO constraint for post-hoc optimization.
void Slam::handle_loop_closure(std::shared_ptr<Frame> frame) {
    LoopResult lr = loop_closer_.detect(frame, map_, K_);
    if (!lr.detected) return;

    last_loop_ = true;
    loop_edges_.emplace_back(lr.matched_frame_id, frame->id());

    // Build 2D-3D correspondences from map points observed near the matched frame
    std::vector<cv::Point3f> lc_obj_pts;
    std::vector<cv::Point2f> lc_img_pts;
    {
        std::lock_guard<std::mutex> lock(map_.mutex());
        const auto& mps = map_.map_points();

        cv::Mat mp_descs;
        std::vector<int> mp_ids_vec;
        for (int mi = 0; mi < (int)mps.size(); mi++) {
            if (!mps[mi].is_valid() || mps[mi].descriptor().empty()) continue;
            const auto& obs = mps[mi].observations();
            bool near_lc = false;
            for (const auto& ob : obs) {
                if (std::abs(ob.first - lr.matched_frame_id) < Config::LC_NEARBY_FRAME_RANGE) {
                    near_lc = true;
                    break;
                }
            }
            if (!near_lc) continue;
            mp_descs.push_back(mps[mi].descriptor());
            mp_ids_vec.push_back(mi);
        }

        if (mp_descs.rows >= 20 && !frame->descriptors().empty()) {
            auto flann = cv::FlannBasedMatcher::create();
            std::vector<std::vector<cv::DMatch>> knn;
            flann->knnMatch(frame->descriptors(), mp_descs, knn, 2);

            for (const auto& m : knn) {
                if (m.size() >= 2 && m[0].distance < Config::FLANN_RATIO_THRESHOLD * m[1].distance) {
                    int mp_id = mp_ids_vec[m[0].trainIdx];
                    cv::Point3d p = mps[mp_id].position();
                    lc_obj_pts.emplace_back((float)p.x, (float)p.y, (float)p.z);
                    lc_img_pts.push_back(frame->keypoints()[m[0].queryIdx].pt);
                }
            }
        }
    }

    // Verify loop closure via PnP and create PGO constraint
    auto pnp = solve_pnp(lc_obj_pts, lc_img_pts, 300, 15);
    if (!pnp.success) return;

    double jump = cv::norm(pnp.t_world - t_world_);
    if (jump >= Config::LC_MAX_JUMP || jump <= Config::LC_MIN_JUMP) return;

    auto matched_frame = map_.get_frame(lr.matched_frame_id);
    if (!matched_frame) return;

    cv::Mat R_from = matched_frame->get_rotation();
    cv::Mat t_from = matched_frame->get_translation();

    LoopConstraint lc_constraint;
    lc_constraint.from_id = lr.matched_frame_id;
    lc_constraint.to_id = frame->id();
    lc_constraint.R_rel = R_from.t() * pnp.R_world;
    lc_constraint.t_rel = R_from.t() * (pnp.t_world - t_from);
    lc_constraint.trans_sigma = Config::PGO_LC_TRANS_SIGMA;
    lc_constraint.rot_sigma = Config::PGO_LC_ROT_SIGMA;
    loop_constraints_.push_back(lc_constraint);
}

/// Main SLAM processing pipeline. For each incoming frame:
/// 1. Extracts features (SuperPoint or ORB)
/// 2. Matches against reference frame (keyframe when available)
/// 3. Attempts PnP recovery if matching fails
/// 4. Handles stationary frames via accelerometer
/// 5. Estimates motion (3D-3D RANSAC preferred, Essential matrix fallback)
/// 6. Fuses pose estimate with EKF (constant-velocity model + height prior)
/// 7. Tracks local map points and refines pose via PnP
/// 8. Creates keyframes, triangulates new map points, detects loop closures
bool Slam::process_frame(std::shared_ptr<Frame> frame) {
    if (!frame || frame->image().empty()) return false;

    last_pnp_ = false;
    last_loop_ = false;
    last_matches_before_.clear();
    last_matches_after_.clear();

    // --- Feature extraction ---
    frame->detect_features(feature_extractor_);

    if (frame->keypoints().size() < (size_t)Config::MIN_MATCHES) {
        last_frame_ = frame;
        return false;
    }

    // --- First frame initialization ---
    if (!last_frame_) {
        frame->set_pose(R_world_, t_world_);
        frame->set_keyframe(true);
        map_.add_frame(frame);
        last_frame_ = frame;
        last_keyframe_ = frame;
        keyframe_count_++;
        frame_count_++;
        return true;
    }

    // --- Feature matching against reference frame ---
    ref_frame_ = (last_keyframe_ && !last_keyframe_->descriptors().empty()) ? last_keyframe_ : last_frame_;

    std::vector<cv::DMatch> raw_matches;
    std::vector<cv::DMatch> good_matches = match_features(
        ref_frame_->descriptors(), frame->descriptors(), &raw_matches);
    last_match_count_ = (int)good_matches.size();
    last_matches_before_ = raw_matches;

    // If keyframe matching is weak, try promoting last_frame as a bridge keyframe
    if (last_match_count_ < Config::MIN_MATCHES && last_frame_ && last_frame_ != ref_frame_) {
        auto temp_matches = match_features(last_frame_->descriptors(), frame->descriptors());

        if ((int)temp_matches.size() >= Config::MIN_MATCHES) {
            if (!last_frame_->is_keyframe()) {
                last_frame_->set_keyframe(true);
                keyframe_count_++;

                if (last_keyframe_) {
                    auto bridge_matches = match_features(
                        last_keyframe_->descriptors(), last_frame_->descriptors());
                    if ((int)bridge_matches.size() >= Config::MIN_MATCHES) {
                        triangulate_points(last_keyframe_, last_frame_, bridge_matches);
                    }
                }
                create_points_from_depth(last_frame_);
                last_keyframe_ = last_frame_;
            }

            ref_frame_ = last_keyframe_;
            raw_matches.clear();
            good_matches = match_features(ref_frame_->descriptors(), frame->descriptors(), &raw_matches);
            last_match_count_ = (int)good_matches.size();
            last_matches_before_ = raw_matches;
        }
    }

    // --- PnP recovery when tracking is lost ---
    int pnp_result = try_pnp_recovery(frame);
    if (pnp_result == 1) return true;
    if (pnp_result == -1) return false;

    // --- Geometric verification via fundamental matrix ---
    std::vector<cv::Point2f> pts1, pts2;
    extract_matched_points(ref_frame_->keypoints(), frame->keypoints(),
                           good_matches, pts1, pts2);

    cv::Mat F_mask;
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                                        3.0, Config::RANSAC_PROB, F_mask);

    if (!F.empty()) {
        epipolar_error_before_ = compute_epipolar_error(pts1, pts2, F);
    }

    if (!F.empty() && !F_mask.empty()) {
        std::vector<cv::Point2f> pts1_in, pts2_in;
        std::vector<cv::DMatch> inlier_matches;
        for (int i = 0; i < (int)pts1.size(); i++) {
            if (F_mask.at<uchar>(i)) {
                pts1_in.push_back(pts1[i]);
                pts2_in.push_back(pts2[i]);
                inlier_matches.push_back(good_matches[i]);
            }
        }
        if (!pts1_in.empty()) {
            epipolar_error_after_ = compute_epipolar_error(pts1_in, pts2_in, F);
        }
        pts1 = pts1_in;
        pts2 = pts2_in;
        good_matches = inlier_matches;
    }

    last_matches_after_ = good_matches;

    // --- Stationary frame handling ---
    if (process_stationary_frame(frame, good_matches)) return true;

    // Post-stationary transition: refresh reference and re-match
    if (was_stationary_ && last_frame_) {
        was_stationary_ = false;
        if (!last_frame_->is_keyframe()) {
            last_frame_->set_keyframe(true);
            keyframe_count_++;
            create_points_from_depth(last_frame_);
            last_keyframe_ = last_frame_;
        }
        ref_frame_ = last_keyframe_;
        raw_matches.clear();
        good_matches = match_features(ref_frame_->descriptors(), frame->descriptors(), &raw_matches);
        last_match_count_ = (int)good_matches.size();
        last_matches_before_ = raw_matches;

        extract_matched_points(ref_frame_->keypoints(), frame->keypoints(),
                               good_matches, pts1, pts2);

        if (pts1.size() >= 8) {
            cv::Mat F2 = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC, 3.0, Config::RANSAC_PROB, F_mask);
            if (!F2.empty() && !F_mask.empty()) {
                std::vector<cv::Point2f> pts1_in, pts2_in;
                std::vector<cv::DMatch> inlier_matches;
                for (int i = 0; i < (int)pts1.size(); i++) {
                    if (F_mask.at<uchar>(i)) {
                        pts1_in.push_back(pts1[i]);
                        pts2_in.push_back(pts2[i]);
                        inlier_matches.push_back(good_matches[i]);
                    }
                }
                pts1 = pts1_in;
                pts2 = pts2_in;
                good_matches = inlier_matches;
            }
        }
        last_matches_after_ = good_matches;
    }

    // --- Motion estimation: 3D-3D preferred, Essential matrix fallback ---
    cv::Mat R_3d, t_3d;
    bool use_3d3d = estimate_motion_3d3d(pts1, pts2, ref_frame_, frame, R_3d, t_3d);

    cv::Mat R_ref = ref_frame_->get_rotation();
    cv::Mat t_ref = ref_frame_->get_translation();
    cv::Mat R_new, t_new;

    if (use_3d3d) {
        // Direct metric pose from 3D-3D alignment
        R_new = R_ref * R_3d.t();
        t_new = t_ref - R_new * t_3d;
    } else {
        // Essential matrix decomposition + depth-based scale estimation
        cv::Mat R_rel, t_rel, E_mask;
        bool motion_ok = estimate_motion(pts1, pts2, R_rel, t_rel, E_mask);

        if (!motion_ok) {
            last_frame_ = frame;
            return false;
        }

        double scale = estimate_scale_from_depth(pts1, pts2, R_rel, t_rel, frame);
        if (scale <= 0) {
            scale = (last_good_scale_ > 0) ? last_good_scale_ : Config::MOTION_SCALE;
        } else {
            last_good_scale_ = scale;
        }
        R_new = R_ref * R_rel.t();
        cv::Mat t_motion = R_new * (scale * t_rel);
        t_new = t_ref - t_motion;
    }

    // --- EKF predict + update ---
    {
        if (!ekf_initialized_) {
            ekf_initialize(t_world_, frame->timestamp());
        }

        double dt = frame->timestamp() - last_frame_time_;
        double ekf_dt = dt;
        if (dt > 0 && dt < 1.0) {
            ekf_predict(dt);
        }

        cv::Mat x_pred_snap = ekf_x_.clone();
        cv::Mat P_pred_snap = ekf_P_.clone();

        // Use tighter measurement noise for 3D-3D (metric) vs E-matrix (noisy scale)
        double sigma_vis = use_3d3d ? Config::EKF_SIGMA_VIS_3D3D : Config::EKF_SIGMA_VIS_EMAT;

        cv::Mat ekf_pred_pos = ekf_x_(cv::Range(0, 3), cv::Range(0, 1));
        double innovation = cv::norm(t_new - ekf_pred_pos);

        // Innovation gating: increase measurement noise for large innovations
        if (innovation < Config::EKF_INNOV_GATE) {
            ekf_update_visual(t_new, sigma_vis);
        } else {
            ekf_update_visual(t_new, innovation * 0.5);
        }

        // Height constraint from accelerometer-derived gravity direction
        if (!gravity_world_.empty() && has_initial_height_) {
            ekf_update_height(initial_height_, Config::EKF_SIGMA_HEIGHT);
        }

        cv::Mat P_filt_snap = ekf_P_.clone();

        cv::Mat ekf_pos = ekf_x_(cv::Range(0, 3), cv::Range(0, 1)).clone();
        cv::Mat delta_t = ekf_pos - t_world_;
        double step = cv::norm(delta_t);

        // Step clamp: prevent unreasonably large single-frame jumps
        if (step > Config::EKF_MAX_STEP && step > 1e-6) {
            delta_t = delta_t * (Config::EKF_MAX_STEP / step);
            ekf_pos = t_world_ + delta_t;
            for (int i = 0; i < 3; i++) ekf_x_.at<double>(i) = ekf_pos.at<double>(i);
            double dt_frame = std::max(0.01, frame->timestamp() - last_frame_time_);
            for (int i = 0; i < 3; i++)
                ekf_x_.at<double>(i + 3) = delta_t.at<double>(i) / dt_frame;
        }

        last_translation_ = delta_t.clone();
        t_new = ekf_pos;

        // Store EKF snapshot for RTS backward smoother
        EKFSnapshot snap;
        snap.x_pred = x_pred_snap;
        snap.P_pred = P_pred_snap;
        snap.x_filt = ekf_x_.clone();
        snap.P_filt = P_filt_snap;
        snap.dt = ekf_dt;
        snap.frame_id = (int)map_.frames_direct().size();
        ekf_snapshots_.push_back(snap);
    }

    last_frame_time_ = frame->timestamp();

    R_world_ = R_new;
    t_world_ = t_new;

    frame->set_pose(R_world_, t_world_);
    map_.add_frame(frame);

    // --- Local map tracking + PnP refinement ---
    int tracked = track_local_map(frame);
    refine_pose_via_local_pnp(frame, tracked);

    // --- Proactive keyframe: force when match count is getting low ---
    if (!frame->is_keyframe() && last_match_count_ < Config::MIN_MATCHES * 2) {
        int frames_since_kf = frame->id() - last_keyframe_->id();
        if (frames_since_kf >= 5) {
            frame->set_keyframe(true);
            keyframe_count_++;
            setup_new_keyframe(frame);
            last_keyframe_ = frame;
        }
    }

    // --- Regular keyframe decision ---
    if (is_keyframe(frame, last_match_count_)) {
        frame->set_keyframe(true);
        keyframe_count_++;
        setup_new_keyframe(frame);

        // Periodic PnP refinement against global map
        if (keyframe_count_ % Config::PNP_INTERVAL == 0) {
            run_pnp(frame);
        }

        // --- Loop closure detection and PGO constraint generation ---
        if (keyframe_count_ % Config::LC_CHECK_INTERVAL == 0) {
            handle_loop_closure(frame);
        }

        // --- Map point visibility tracking (ORB-SLAM3-style) ---
        {
            std::lock_guard<std::mutex> lock(map_.mutex());
            auto& mps = map_.map_points();
            for (auto& mp : mps) {
                if (!mp.is_valid()) continue;
                cv::Point2d proj = Optimizer::project_point(mp.position(), R_world_, t_world_, K_);
                if (proj.x >= 0 && proj.x < Config::IMAGE_WIDTH &&
                    proj.y >= 0 && proj.y < Config::IMAGE_HEIGHT) {
                    mp.increase_visible();
                    for (int ki = 0; ki < (int)frame->keypoints().size(); ki++) {
                        double dx = proj.x - frame->keypoints()[ki].pt.x;
                        double dy = proj.y - frame->keypoints()[ki].pt.y;
                        if (dx*dx + dy*dy < Config::TRACK_VISIBILITY_RADIUS * Config::TRACK_VISIBILITY_RADIUS) {
                            mp.increase_found();
                            break;
                        }
                    }
                }
            }
        }

        // Periodic culling of poorly-tracked map points by found ratio
        if (keyframe_count_ % 3 == 0) {
            std::lock_guard<std::mutex> lock(map_.mutex());
            auto& mps = map_.map_points();
            for (auto& mp : mps) {
                if (!mp.is_valid()) continue;
                int age = keyframe_count_ - mp.first_kf_id();
                if (age >= 3 && mp.visible_count() > 0) {
                    if (mp.get_found_ratio() < Config::CULL_FOUND_RATIO_YOUNG) {
                        mp.set_valid(false);
                    }
                }
                if (age >= 5 && mp.observation_count() <= 2 && mp.get_found_ratio() < Config::CULL_FOUND_RATIO_OLD) {
                    mp.set_valid(false);
                }
            }
        }

        last_keyframe_ = frame;
    }

    last_frame_ = frame;
    frame_count_++;

    return true;
}

/// Matches descriptors using FLANN (L2 for float/SuperPoint) or BFMatcher (Hamming
/// for binary/ORB). Applies Lowe's ratio test for L2 or distance threshold for Hamming.
/// Optionally outputs all raw matches before filtering.
std::vector<cv::DMatch> Slam::match_features(const cv::Mat& desc1, const cv::Mat& desc2,
                                              std::vector<cv::DMatch>* raw_matches_out) {
    std::vector<cv::DMatch> good_matches;
    if (desc1.empty() || desc2.empty()) return good_matches;

    bool is_float = (desc1.type() == CV_32F);

    if (is_float) {
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher_l2_->knnMatch(desc1, desc2, knn_matches, 2);

        for (const auto& m : knn_matches) {
            if (m.size() >= 2) {
                if (raw_matches_out) raw_matches_out->push_back(m[0]);
                if (m[0].distance < Config::L2_RATIO_THRESHOLD * m[1].distance) {
                    good_matches.push_back(m[0]);
                }
            }
        }
    } else {
        std::vector<cv::DMatch> matches;
        matcher_hamming_->match(desc1, desc2, matches);

        for (const auto& m : matches) {
            if (raw_matches_out) raw_matches_out->push_back(m);
            if (m.distance < Config::DISTANCE_THRESHOLD) {
                good_matches.push_back(m);
            }
        }
    }

    return good_matches;
}

/// Extracts 2D point coordinates from keypoints using match indices.
void Slam::extract_matched_points(const std::vector<cv::KeyPoint>& kp1,
                                   const std::vector<cv::KeyPoint>& kp2,
                                   const std::vector<cv::DMatch>& matches,
                                   std::vector<cv::Point2f>& pts1,
                                   std::vector<cv::Point2f>& pts2) {
    pts1.clear();
    pts2.clear();
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
}

/// Decomposes the Essential matrix to recover relative rotation and translation.
/// Uses RANSAC to find the Essential matrix, then cv::recoverPose for decomposition.
/// Returns false if insufficient inliers or degenerate configuration.
bool Slam::estimate_motion(const std::vector<cv::Point2f>& pts1,
                             const std::vector<cv::Point2f>& pts2,
                             cv::Mat& R, cv::Mat& t, cv::Mat& mask) {
    if (pts1.size() < 5) return false;

    cv::Mat E = cv::findEssentialMat(pts1, pts2, K_,
                                      cv::RANSAC, Config::RANSAC_PROB,
                                      Config::RANSAC_THRESHOLD, mask);
    if (E.empty()) return false;

    last_inlier_count_ = cv::countNonZero(mask);
    if (last_inlier_count_ < Config::MIN_INLIERS) return false;

    int inliers = cv::recoverPose(E, pts1, pts2, K_, R, t, mask);
    if (inliers < Config::MIN_INLIERS) return false;

    double det = cv::determinant(R);
    if (std::abs(det - 1.0) > 0.01) return false;

    return true;
}

/// Computes the mean symmetric epipolar distance: d(x2, F*x1) for all point pairs.
/// Used to evaluate geometric consistency of matches before and after filtering.
double Slam::compute_epipolar_error(const std::vector<cv::Point2f>& pts1,
                                     const std::vector<cv::Point2f>& pts2,
                                     const cv::Mat& F) {
    if (F.empty() || pts1.empty()) return 0;

    double total_error = 0;
    int count = 0;

    for (size_t i = 0; i < pts1.size(); i++) {
        cv::Mat x1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, 1.0);

        cv::Mat Fx1 = F * x1;
        double num = std::abs(x2.dot(Fx1));
        double denom = std::sqrt(Fx1.at<double>(0) * Fx1.at<double>(0) +
                                  Fx1.at<double>(1) * Fx1.at<double>(1));
        if (denom > 1e-10) {
            total_error += num / denom;
            count++;
        }
    }

    return count > 0 ? total_error / count : 0;
}

/// Triangulates new 3D map points from matched features between two keyframes.
/// Uses DLT triangulation, then validates via depth range, reprojection error,
/// and distance from camera. When Kinect depth is available, replaces triangulated
/// depth with measured depth for higher accuracy.
void Slam::triangulate_points(std::shared_ptr<Frame> frame1,
                                std::shared_ptr<Frame> frame2,
                                const std::vector<cv::DMatch>& matches) {
    cv::Mat R1 = frame1->get_rotation();
    cv::Mat t1 = frame1->get_translation();
    cv::Mat R1_cam = R1.t();
    cv::Mat t1_cam = -R1_cam * t1;

    cv::Mat R2 = frame2->get_rotation();
    cv::Mat t2 = frame2->get_translation();
    cv::Mat R2_cam = R2.t();
    cv::Mat t2_cam = -R2_cam * t2;

    cv::Mat RT1, RT2;
    cv::hconcat(R1_cam, t1_cam, RT1);
    cv::hconcat(R2_cam, t2_cam, RT2);

    cv::Mat P1 = K_ * RT1;
    cv::Mat P2 = K_ * RT2;

    std::vector<cv::Point2f> pts1, pts2;
    std::vector<int> match_indices;
    for (int i = 0; i < (int)matches.size(); i++) {
        pts1.push_back(frame1->keypoints()[matches[i].queryIdx].pt);
        pts2.push_back(frame2->keypoints()[matches[i].trainIdx].pt);
        match_indices.push_back(i);
    }

    if (pts1.size() < 5) return;

    cv::Mat pts4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, pts4D);

    int added = 0;
    std::lock_guard<std::mutex> lock(map_.mutex());
    int next_id = (int)map_.map_points().size();

    bool use_real_depth = frame2->has_real_depth() && !frame2->depth_map().empty();

    for (int i = 0; i < pts4D.cols; i++) {
        float w = pts4D.at<float>(3, i);
        if (std::abs(w) < 1e-6) continue;

        cv::Point3d pt(pts4D.at<float>(0, i) / w,
                       pts4D.at<float>(1, i) / w,
                       pts4D.at<float>(2, i) / w);

        // Override with Kinect depth when available (more accurate than triangulation)
        if (use_real_depth) {
            int px = (int)std::round(pts2[i].x);
            int py = (int)std::round(pts2[i].y);
            if (px >= 0 && px < frame2->depth_map().cols &&
                py >= 0 && py < frame2->depth_map().rows) {
                float z_real = frame2->depth_map().at<float>(py, px);
                if (z_real > Config::DEPTH_MIN && z_real < Config::DEPTH_MAX) {
                    double x_cam = (pts2[i].x - Config::CX) * z_real / Config::FX;
                    double y_cam = (pts2[i].y - Config::CY) * z_real / Config::FY;
                    cv::Mat p_cam = (cv::Mat_<double>(3, 1) << x_cam, y_cam, (double)z_real);
                    cv::Mat p_world = R2 * p_cam + t2;
                    pt = cv::Point3d(p_world.at<double>(0),
                                     p_world.at<double>(1),
                                     p_world.at<double>(2));
                }
            }
        }

        // Validate: point must be in front of both cameras within depth range
        cv::Mat pt_cam1 = R1_cam * (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z) + t1_cam;
        cv::Mat pt_cam2 = R2_cam * (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z) + t2_cam;

        double z1 = pt_cam1.at<double>(2);
        double z2 = pt_cam2.at<double>(2);

        if (z1 < Config::TRIANG_MIN_DEPTH || z1 > Config::TRIANG_MAX_DEPTH) continue;
        if (z2 < Config::TRIANG_MIN_DEPTH || z2 > Config::TRIANG_MAX_DEPTH) continue;

        // Validate reprojection error in both views
        cv::Point2d reproj2 = Optimizer::project_point(pt, R2, t2, K_);
        double dx2 = reproj2.x - pts2[i].x;
        double dy2 = reproj2.y - pts2[i].y;
        if (std::sqrt(dx2 * dx2 + dy2 * dy2) > Config::TRIANG_MAX_REPROJ_ERROR) continue;

        cv::Point2d reproj1 = Optimizer::project_point(pt, R1, t1, K_);
        double dx1 = reproj1.x - pts1[i].x;
        double dy1 = reproj1.y - pts1[i].y;
        if (std::sqrt(dx1 * dx1 + dy1 * dy1) > Config::TRIANG_MAX_REPROJ_ERROR) continue;

        double dist_from_cam = cv::norm(pt - cv::Point3d(t2.at<double>(0), t2.at<double>(1), t2.at<double>(2)));
        if (dist_from_cam > Config::TRIANG_MAX_CAM_DIST) continue;

        int kp2_idx = matches[match_indices[i]].trainIdx;
        cv::Mat desc;
        if (!frame2->descriptors().empty()) {
            desc = frame2->descriptors().row(kp2_idx).clone();
        }

        MapPoint mp(next_id, pt, desc);
        mp.set_first_kf_id(keyframe_count_);
        mp.add_observation(frame1->id(), matches[match_indices[i]].queryIdx);
        mp.add_observation(frame2->id(), kp2_idx);

        map_.map_points().push_back(mp);
        map_.add_display_point(pt);

        frame1->map_point_indices()[matches[match_indices[i]].queryIdx] = next_id;
        frame2->map_point_indices()[kp2_idx] = next_id;

        next_id++;
        added++;
    }
}

/// Determines if a frame should be a keyframe based on frame gap and match count.
bool Slam::is_keyframe(std::shared_ptr<Frame> frame, int match_count) {
    if (!last_keyframe_) return true;

    int frame_gap = frame->id() - last_keyframe_->id();
    if (frame_gap < Config::KF_MIN_FRAME_GAP) return false;

    if (match_count < Config::KF_MIN_MATCHES) return false;

    return true;
}

/// Computes reprojection error before and after PnP-based pose refinement
/// using locally tracked map points. Blends PnP result with current pose
/// using an adaptive weight based on inlier ratio.
void Slam::refine_pose_via_local_pnp(std::shared_ptr<Frame> frame, int tracked) {
    // Compute reprojection error before PnP refinement
    {
        double fx = K_.at<double>(0,0), fy = K_.at<double>(1,1);
        double cx = K_.at<double>(0,2), cy = K_.at<double>(1,2);
        cv::Mat Rc = R_world_.t();
        cv::Mat tc = -Rc * t_world_;
        double sum = 0;
        int cnt = 0;
        std::lock_guard<std::mutex> lock(map_.mutex());
        const auto& mps = map_.map_points();
        const auto& indices = frame->map_point_indices();
        for (int i = 0; i < (int)indices.size(); i++) {
            int mp_id = indices[i];
            if (mp_id < 0 || mp_id >= (int)mps.size() || !mps[mp_id].is_valid()) continue;
            cv::Point3d pw = mps[mp_id].position();
            cv::Mat p = (cv::Mat_<double>(3,1) << pw.x, pw.y, pw.z);
            cv::Mat pc = Rc * p + tc;
            double z = pc.at<double>(2);
            if (z < 0.01) continue;
            double u = fx * pc.at<double>(0) / z + cx;
            double v = fy * pc.at<double>(1) / z + cy;
            double dx = u - frame->keypoints()[i].pt.x;
            double dy = v - frame->keypoints()[i].pt.y;
            sum += std::sqrt(dx*dx + dy*dy);
            cnt++;
        }
        reproj_error_before_ = cnt > 0 ? sum / cnt : 0.0;
        reproj_error_after_ = reproj_error_before_;
    }

    // PnP refinement using tracked map points
    if (tracked >= 10) {
        std::vector<cv::Point3f> obj_pts;
        std::vector<cv::Point2f> img_pts;
        {
            std::lock_guard<std::mutex> lock(map_.mutex());
            const auto& mps = map_.map_points();
            const auto& indices = frame->map_point_indices();
            for (int i = 0; i < (int)indices.size(); i++) {
                int mp_id = indices[i];
                if (mp_id >= 0 && mp_id < (int)mps.size() && mps[mp_id].is_valid()) {
                    cv::Point3d p = mps[mp_id].position();
                    obj_pts.emplace_back((float)p.x, (float)p.y, (float)p.z);
                    img_pts.push_back(frame->keypoints()[i].pt);
                }
            }
        }

        cv::Mat R_prev_pose = R_world_.clone();
        cv::Mat t_prev_pose = t_world_.clone();
        auto pnp = solve_pnp(obj_pts, img_pts, 100, 10);
        if (pnp.success) {
            double jump = cv::norm(pnp.t_world - t_world_);
            if (jump < Config::PNP_REFINE_MAX_JUMP) {
                double inlier_ratio = (double)pnp.inlier_count / (double)obj_pts.size();

                // Adaptive blend: higher inlier ratio → trust PnP more
                double blend = std::min(0.5, 0.3 + 0.2 * std::max(0.0, std::min(1.0, (inlier_ratio - 0.5) / 0.5)));

                cv::Mat t_blended = (1.0 - blend) * t_world_ + blend * pnp.t_world;

                cv::Mat rvec_cur, rvec_pnp_r;
                cv::Rodrigues(R_world_, rvec_cur);
                cv::Rodrigues(pnp.R_world, rvec_pnp_r);
                cv::Mat rvec_blended = (1.0 - blend) * rvec_cur + blend * rvec_pnp_r;
                cv::Mat R_blended;
                cv::Rodrigues(rvec_blended, R_blended);

                R_world_ = R_blended;
                t_world_ = t_blended;
                frame->set_pose(R_world_, t_world_);

                // Compute reprojection error improvement
                double fx = K_.at<double>(0,0), fy = K_.at<double>(1,1);
                double cx = K_.at<double>(0,2), cy = K_.at<double>(1,2);
                auto compute_reproj = [&](const cv::Mat& R_w, const cv::Mat& t_w) {
                    cv::Mat Rc = R_w.t();
                    cv::Mat tc = -Rc * t_w;
                    double sum = 0;
                    int cnt = 0;
                    for (int pi = 0; pi < (int)obj_pts.size(); pi++) {
                        cv::Mat pw = (cv::Mat_<double>(3,1) << obj_pts[pi].x, obj_pts[pi].y, obj_pts[pi].z);
                        cv::Mat pc = Rc * pw + tc;
                        double z = pc.at<double>(2);
                        if (z < 0.01) continue;
                        double u = fx * pc.at<double>(0) / z + cx;
                        double v = fy * pc.at<double>(1) / z + cy;
                        double dx = u - img_pts[pi].x;
                        double dy = v - img_pts[pi].y;
                        sum += std::sqrt(dx*dx + dy*dy);
                        cnt++;
                    }
                    return cnt > 0 ? sum / cnt : 0.0;
                };
                reproj_error_before_ = compute_reproj(R_prev_pose, t_prev_pose);
                reproj_error_after_ = compute_reproj(R_world_, t_world_);
            }
        }
    }
}

/// Refines camera pose by blending current estimate with PnP solution from
/// tracked map points. Uses adaptive blending based on inlier ratio.
void Slam::run_pnp(std::shared_ptr<Frame> frame) {
    std::vector<cv::Point3f> obj_pts;
    std::vector<cv::Point2f> img_pts;

    {
        std::lock_guard<std::mutex> lock(map_.mutex());
        const auto& mps = map_.map_points();
        const auto& indices = frame->map_point_indices();

        for (int i = 0; i < (int)indices.size(); i++) {
            int mp_id = indices[i];
            if (mp_id >= 0 && mp_id < (int)mps.size() && mps[mp_id].is_valid()) {
                cv::Point3d p = mps[mp_id].position();
                obj_pts.emplace_back((float)p.x, (float)p.y, (float)p.z);
                img_pts.push_back(frame->keypoints()[i].pt);
            }
        }
    }

    auto pnp = solve_pnp(obj_pts, img_pts, 100, Config::PNP_MIN_POINTS);
    if (!pnp.success) return;

    cv::Mat t_cur = frame->get_translation();
    double jump_dist = cv::norm(pnp.t_world - t_cur);
    if (jump_dist > Config::PNP_PERIODIC_MAX_JUMP) return;

    // Blend PnP result with current pose
    double blend = Config::PNP_PERIODIC_BLEND;

    cv::Mat t_blended = (1.0 - blend) * t_cur + blend * pnp.t_world;

    cv::Mat R_cur = frame->get_rotation();
    cv::Mat rvec_cur, rvec_new_r;
    cv::Rodrigues(R_cur, rvec_cur);
    cv::Rodrigues(pnp.R_world, rvec_new_r);
    cv::Mat rvec_blended = (1.0 - blend) * rvec_cur + blend * rvec_new_r;
    cv::Mat R_blended;
    cv::Rodrigues(rvec_blended, R_blended);

    R_world_ = R_blended.clone();
    t_world_ = t_blended.clone();

    frame->set_pose(R_world_, t_world_);

    last_pnp_ = true;
}

/// Creates 3D map points from Kinect depth at keypoint locations that don't
/// already have an associated map point from triangulation.
void Slam::create_points_from_depth(std::shared_ptr<Frame> frame) {
    if (!frame->has_real_depth() || frame->depth_map().empty()) return;

    cv::Mat R = frame->get_rotation();
    cv::Mat t = frame->get_translation();
    const cv::Mat& depth = frame->depth_map();
    const auto& keypoints = frame->keypoints();
    auto& indices = frame->map_point_indices();

    std::lock_guard<std::mutex> lock(map_.mutex());
    int next_id = (int)map_.map_points().size();
    int added = 0;

    for (int i = 0; i < (int)keypoints.size(); i++) {
        // Skip keypoints that already have a map point from triangulation
        if (indices[i] >= 0) continue;

        float u = keypoints[i].pt.x;
        float v = keypoints[i].pt.y;
        int px = (int)std::round(u);
        int py = (int)std::round(v);

        if (px < 0 || px >= depth.cols || py < 0 || py >= depth.rows) continue;

        float z = depth.at<float>(py, px);
        if (z <= Config::DEPTH_MIN || z > Config::TRIANG_MAX_CAM_DIST) continue;

        // Back-project pixel to 3D camera coordinates, then transform to world
        double x_cam = (u - Config::CX) * z / Config::FX;
        double y_cam = (v - Config::CY) * z / Config::FY;
        cv::Mat p_cam = (cv::Mat_<double>(3, 1) << x_cam, y_cam, (double)z);

        cv::Mat p_world = R * p_cam + t;
        cv::Point3d pt(p_world.at<double>(0), p_world.at<double>(1), p_world.at<double>(2));

        cv::Mat desc;
        if (!frame->descriptors().empty()) {
            desc = frame->descriptors().row(i).clone();
        }

        MapPoint mp(next_id, pt, desc);
        mp.set_first_kf_id(keyframe_count_);
        mp.add_observation(frame->id(), i);

        map_.map_points().push_back(mp);
        map_.add_display_point(pt);
        indices[i] = next_id;

        next_id++;
        added++;
    }
}

/// Stores accelerometer data for stationary detection and gravity estimation.
void Slam::set_accelerometer_data(const std::vector<AccelSample>& data) {
    accel_data_ = data;
}

/// Estimates the gravity direction in world frame from accelerometer data.
/// Averages all accelerometer readings, transforms to world frame, and snaps
/// to the nearest axis. Sets the initial height for the height constraint.
void Slam::compute_gravity_direction() {
    if (accel_data_.empty()) return;

    double ax_sum = 0, ay_sum = 0, az_sum = 0;
    for (const auto& s : accel_data_) {
        ax_sum += s.ax;
        ay_sum += s.ay;
        az_sum += s.az;
    }
    int n = (int)accel_data_.size();
    cv::Mat g_cam = (cv::Mat_<double>(3, 1) << ax_sum / n, ay_sum / n, az_sum / n);

    // Transform to world frame and snap to nearest axis
    gravity_world_ = R_world_ * g_cam;
    double norm = cv::norm(gravity_world_);
    if (norm > 1e-6) gravity_world_ /= norm;

    int max_axis = 0;
    double max_val = 0;
    for (int i = 0; i < 3; i++) {
        double v = std::abs(gravity_world_.at<double>(i));
        if (v > max_val) { max_val = v; max_axis = i; }
    }
    double sign = gravity_world_.at<double>(max_axis) > 0 ? 1.0 : -1.0;
    gravity_world_ = cv::Mat::zeros(3, 1, CV_64F);
    gravity_world_.at<double>(max_axis) = sign;

    initial_height_ = t_world_.dot(gravity_world_);
    has_initial_height_ = true;
}

/// Detects whether the camera is stationary at the given timestamp by analyzing
/// accelerometer magnitude variance in a 200ms window. Low variance indicates
/// the robot is not moving.
bool Slam::is_frame_stationary(double timestamp) const {
    if (accel_data_.empty()) return false;

    const double window = 0.1;  // +/-100ms
    const double threshold = 0.15;  // accel std threshold

    int lo = 0, hi = (int)accel_data_.size() - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (accel_data_[mid].timestamp < timestamp - window) lo = mid + 1;
        else hi = mid;
    }

    std::vector<double> mags;
    for (int i = lo; i < (int)accel_data_.size() && accel_data_[i].timestamp <= timestamp + window; i++) {
        double ax = accel_data_[i].ax, ay = accel_data_[i].ay, az = accel_data_[i].az;
        mags.push_back(std::sqrt(ax*ax + ay*ay + az*az));
    }

    if (mags.size() < 5) return false;

    double mean = 0;
    for (double m : mags) mean += m;
    mean /= mags.size();

    double var = 0;
    for (double m : mags) var += (m - mean) * (m - mean);
    var /= mags.size();

    return std::sqrt(var) < threshold;
}

/// Initializes the 6-DOF EKF state (position + velocity) at the given position.
void Slam::ekf_initialize(const cv::Mat& pos, double timestamp) {
    ekf_x_ = cv::Mat::zeros(6, 1, CV_64F);
    pos.copyTo(ekf_x_(cv::Range(0, 3), cv::Range(0, 1)));

    ekf_P_ = cv::Mat::zeros(6, 6, CV_64F);
    for (int i = 0; i < 3; i++) ekf_P_.at<double>(i, i) = 0.001;
    for (int i = 3; i < 6; i++) ekf_P_.at<double>(i, i) = 0.01;

    last_frame_time_ = timestamp;
    ekf_initialized_ = true;
}

/// EKF prediction step: propagates state using constant-velocity model with
/// velocity decay. Process noise Q is derived from piecewise-constant acceleration.
void Slam::ekf_predict(double dt) {
    if (!ekf_initialized_ || dt <= 0) return;

    double decay = Config::EKF_VEL_DECAY;

    // Propagate state: position += velocity * dt, velocity *= decay
    for (int i = 0; i < 3; i++) {
        ekf_x_.at<double>(i) += ekf_x_.at<double>(i + 3) * dt;
        ekf_x_.at<double>(i + 3) *= decay;
    }

    // State transition matrix F
    cv::Mat F = cv::Mat::eye(6, 6, CV_64F);
    for (int i = 0; i < 3; i++) {
        F.at<double>(i, i + 3) = dt;
        F.at<double>(i + 3, i + 3) = decay;
    }

    // Process noise Q from piecewise-constant acceleration model
    double sigma_a = Config::EKF_PROCESS_ACCEL;
    cv::Mat Q = cv::Mat::zeros(6, 6, CV_64F);
    for (int i = 0; i < 3; i++) {
        Q.at<double>(i, i) = 0.25 * dt * dt * dt * dt * sigma_a * sigma_a;
        Q.at<double>(i + 3, i + 3) = dt * dt * sigma_a * sigma_a;
        Q.at<double>(i, i + 3) = 0.5 * dt * dt * dt * sigma_a * sigma_a;
        Q.at<double>(i + 3, i) = Q.at<double>(i, i + 3);
    }

    ekf_P_ = F * ekf_P_ * F.t() + Q;
}

/// EKF update step for visual odometry measurement (3D position).
/// Uses Joseph form for numerical stability: P = (I-KH)*P*(I-KH)' + K*R*K'
void Slam::ekf_update_visual(const cv::Mat& z_pos, double sigma_vis) {
    if (!ekf_initialized_) return;

    // Observation matrix: we observe position directly (first 3 states)
    cv::Mat H = cv::Mat::zeros(3, 6, CV_64F);
    for (int i = 0; i < 3; i++) H.at<double>(i, i) = 1.0;

    cv::Mat R = cv::Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < 3; i++) R.at<double>(i, i) = sigma_vis * sigma_vis;

    cv::Mat y = z_pos - H * ekf_x_;         // Innovation
    cv::Mat S = H * ekf_P_ * H.t() + R;     // Innovation covariance
    cv::Mat K = ekf_P_ * H.t() * S.inv();   // Kalman gain
    ekf_x_ = ekf_x_ + K * y;

    // Joseph form covariance update
    cv::Mat I = cv::Mat::eye(6, 6, CV_64F);
    cv::Mat IKH = I - K * H;
    ekf_P_ = IKH * ekf_P_ * IKH.t() + K * R * K.t();
}

/// EKF update step for height constraint along gravity direction.
/// Constrains the camera height to remain near the initial height (robot on flat floor).
void Slam::ekf_update_height(double h_target, double sigma_h) {
    if (!ekf_initialized_ || gravity_world_.empty()) return;

    // H maps state to height along gravity: h = g^T * position
    cv::Mat H = cv::Mat::zeros(1, 6, CV_64F);
    for (int i = 0; i < 3; i++) H.at<double>(0, i) = gravity_world_.at<double>(i);

    cv::Mat R_h = (cv::Mat_<double>(1, 1) << sigma_h * sigma_h);

    double h_pred = 0;
    for (int i = 0; i < 3; i++) h_pred += gravity_world_.at<double>(i) * ekf_x_.at<double>(i);
    cv::Mat y_h = (cv::Mat_<double>(1, 1) << h_target - h_pred);

    cv::Mat S = H * ekf_P_ * H.t() + R_h;
    cv::Mat K = ekf_P_ * H.t() * S.inv();
    ekf_x_ = ekf_x_ + K * y_h;

    cv::Mat I = cv::Mat::eye(6, 6, CV_64F);
    cv::Mat IKH = I - K * H;
    ekf_P_ = IKH * ekf_P_ * IKH.t() + K * R_h * K.t();
}

/// Runs post-hoc pose graph optimization with g2o using accumulated loop constraints
/// and the height prior. Called after all frames are processed.
void Slam::run_posthoc_pgo() {
    if (!has_initial_height_ && loop_constraints_.empty()) {
        return;
    }
    optimizer_.pose_graph_optimize(
        map_, loop_constraints_, gravity_world_,
        initial_height_, has_initial_height_);
}

/// Rauch-Tung-Striebel (RTS) backward smoother. Uses stored EKF snapshots
/// (predicted and filtered states/covariances) to optimally smooth the entire
/// trajectory after all frames are processed. Applies smoothed positions to
/// frame poses in the map.
void Slam::run_rts_smoother() {
    int N = (int)ekf_snapshots_.size();
    if (N < 3) {
        return;
    }

    double decay = Config::EKF_VEL_DECAY;

    // Initialize: last smoothed state = last filtered state
    std::vector<cv::Mat> x_smooth(N), P_smooth(N);
    x_smooth[N-1] = ekf_snapshots_[N-1].x_filt.clone();
    P_smooth[N-1] = ekf_snapshots_[N-1].P_filt.clone();

    // Backward pass: compute smoother gain and propagate corrections
    for (int k = N - 2; k >= 0; k--) {
        double dt = ekf_snapshots_[k+1].dt;

        // Reconstruct transition matrix F for this time step
        cv::Mat F = cv::Mat::eye(6, 6, CV_64F);
        for (int i = 0; i < 3; i++) {
            F.at<double>(i, i + 3) = dt;
            F.at<double>(i + 3, i + 3) = decay;
        }

        // Smoother gain: C_k = P_filt[k] * F^T * inv(P_pred[k+1])
        cv::Mat Ft = F.t();
        cv::Mat P_pred_inv = ekf_snapshots_[k+1].P_pred.inv(cv::DECOMP_SVD);
        cv::Mat C = ekf_snapshots_[k].P_filt * Ft * P_pred_inv;

        // Smoothed state and covariance
        x_smooth[k] = ekf_snapshots_[k].x_filt
                     + C * (x_smooth[k+1] - ekf_snapshots_[k+1].x_pred);
        P_smooth[k] = ekf_snapshots_[k].P_filt
                     + C * (P_smooth[k+1] - ekf_snapshots_[k+1].P_pred) * C.t();
    }

    // Apply smoothed positions to frame poses
    auto& frames = map_.frames_direct();
    int updated = 0;
    for (int k = 0; k < N; k++) {
        int fid = ekf_snapshots_[k].frame_id;
        if (fid >= 0 && fid < (int)frames.size()) {
            cv::Mat pos_smooth = x_smooth[k](cv::Range(0, 3), cv::Range(0, 1)).clone();
            frames[fid]->set_pose(frames[fid]->get_rotation(), pos_smooth);
            updated++;
        }
    }

    (void)updated;
}
