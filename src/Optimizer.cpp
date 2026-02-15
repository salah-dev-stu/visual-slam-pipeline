#include "Optimizer.h"
#include "Frame.h"
#include "Map.h"
#include "Config.h"
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <climits>

// g2o headers for pose graph optimization
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/core/base_unary_edge.h>

Optimizer::Optimizer() {}

cv::Point2d Optimizer::project_point(const cv::Point3d& pw,
                                      const cv::Mat& R_world,
                                      const cv::Mat& t_world,
                                      const cv::Mat& K) {
    cv::Mat R_cam = R_world.t();
    cv::Mat t_cam = -R_cam * t_world;

    cv::Mat Pw = (cv::Mat_<double>(3, 1) << pw.x, pw.y, pw.z);
    cv::Mat pc = R_cam * Pw + t_cam;

    double z = pc.at<double>(2);
    if (z < 1e-6) return cv::Point2d(-1, -1);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    double u = fx * pc.at<double>(0) / z + cx;
    double v = fy * pc.at<double>(1) / z + cy;

    return cv::Point2d(u, v);
}

std::pair<double, double> Optimizer::optimize_pose(
    std::shared_ptr<Frame> frame,
    const std::vector<cv::Point3d>& points_3d,
    const std::vector<cv::Point2f>& points_2d,
    const cv::Mat& K) {

    if (points_3d.size() < 3 || points_3d.size() != points_2d.size()) {
        return {0, 0};
    }

    int n = (int)points_3d.size();

    cv::Mat R = frame->get_rotation();
    cv::Mat t = frame->get_translation();

    // Convert to rvec, tvec parameterization
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);
    cv::Mat tvec = t.clone();

    double error_before = 0;
    for (int i = 0; i < n; i++) {
        cv::Point2d proj = project_point(points_3d[i], R, t, K);
        double dx = proj.x - points_2d[i].x;
        double dy = proj.y - points_2d[i].y;
        error_before += dx * dx + dy * dy;
    }
    error_before = std::sqrt(error_before / n);

    double lambda = Config::OPT_LM_LAMBDA;

    for (int iter = 0; iter < Config::OPT_MAX_ITERATIONS; iter++) {
        cv::Mat R_cur;
        cv::Rodrigues(rvec, R_cur);
        cv::Mat t_cur = tvec.clone();

        cv::Mat J = cv::Mat::zeros(2 * n, 6, CV_64F);
        cv::Mat r = cv::Mat::zeros(2 * n, 1, CV_64F);

        for (int i = 0; i < n; i++) {
            cv::Point2d proj = project_point(points_3d[i], R_cur, t_cur, K);
            r.at<double>(2 * i, 0) = proj.x - points_2d[i].x;
            r.at<double>(2 * i + 1, 0) = proj.y - points_2d[i].y;

            double eps = 1e-6;
            for (int j = 0; j < 6; j++) {
                cv::Mat rvec_p = rvec.clone();
                cv::Mat tvec_p = tvec.clone();
                if (j < 3) {
                    rvec_p.at<double>(j) += eps;
                } else {
                    tvec_p.at<double>(j - 3) += eps;
                }
                cv::Mat R_p;
                cv::Rodrigues(rvec_p, R_p);
                cv::Point2d proj_p = project_point(points_3d[i], R_p, tvec_p, K);

                J.at<double>(2 * i, j) = (proj_p.x - proj.x) / eps;
                J.at<double>(2 * i + 1, j) = (proj_p.y - proj.y) / eps;
            }
        }

        cv::Mat JtJ = J.t() * J;
        cv::Mat Jtr = J.t() * r;

        for (int i = 0; i < 6; i++) {
            JtJ.at<double>(i, i) += lambda;
        }

        cv::Mat delta;
        bool solved = cv::solve(JtJ, -Jtr, delta, cv::DECOMP_CHOLESKY);
        if (!solved) {
            lambda *= 10;
            continue;
        }

        cv::Mat rvec_new = rvec + delta(cv::Range(0, 3), cv::Range(0, 1));
        cv::Mat tvec_new = tvec + delta(cv::Range(3, 6), cv::Range(0, 1));

        cv::Mat R_new;
        cv::Rodrigues(rvec_new, R_new);
        double error_new = 0;
        for (int i = 0; i < n; i++) {
            cv::Point2d proj = project_point(points_3d[i], R_new, tvec_new, K);
            double dx = proj.x - points_2d[i].x;
            double dy = proj.y - points_2d[i].y;
            error_new += dx * dx + dy * dy;
        }
        error_new = std::sqrt(error_new / n);

        double current_error = 0;
        for (int i = 0; i < n; i++) {
            cv::Point2d proj = project_point(points_3d[i], R_cur, t_cur, K);
            double dx = proj.x - points_2d[i].x;
            double dy = proj.y - points_2d[i].y;
            current_error += dx * dx + dy * dy;
        }
        current_error = std::sqrt(current_error / n);

        if (error_new < current_error) {
            rvec = rvec_new;
            tvec = tvec_new;
            lambda /= 2;
        } else {
            lambda *= 10;
        }

        if (std::abs(current_error - error_new) < Config::OPT_CONVERGENCE) {
            break;
        }
    }

    cv::Mat R_opt;
    cv::Rodrigues(rvec, R_opt);
    frame->set_pose(R_opt, tvec);

    double error_after = 0;
    for (int i = 0; i < n; i++) {
        cv::Point2d proj = project_point(points_3d[i], R_opt, tvec, K);
        double dx = proj.x - points_2d[i].x;
        double dy = proj.y - points_2d[i].y;
        error_after += dx * dx + dy * dy;
    }
    error_after = std::sqrt(error_after / n);

    return {error_before, error_after};
}

std::pair<double, double> Optimizer::local_bundle_adjustment(
    Map& map, const cv::Mat& K, int window_size) {

    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
    const double cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);
    const double HUBER_DELTA = 5.0;  // pixels

    struct Obs {
        int kf_idx, pt_idx;
        double obs_u, obs_v;
    };

    std::vector<std::shared_ptr<Frame>> keyframes;
    std::vector<int> mp_global_ids;  // local pt_idx -> global map_point id
    std::vector<cv::Point3d> points;
    std::vector<Obs> observations;
    std::unordered_map<int, int> mp_id_to_local;

    {
        std::lock_guard<std::mutex> lock(map.mutex());
        auto& all_frames = map.frames_direct();
        auto& mps = map.map_points();

        for (auto& f : all_frames) {
            if (f->is_keyframe()) keyframes.push_back(f);
        }

        int N_total = (int)keyframes.size();
        int start = std::max(0, N_total - window_size);
        keyframes = std::vector<std::shared_ptr<Frame>>(
            keyframes.begin() + start, keyframes.end());

        if ((int)keyframes.size() < 2) return {0, 0};

        for (int ki = 0; ki < (int)keyframes.size(); ki++) {
            const auto& indices = keyframes[ki]->map_point_indices();
            const auto& kps = keyframes[ki]->keypoints();

            for (int kpi = 0; kpi < (int)indices.size(); kpi++) {
                int mp_id = indices[kpi];
                if (mp_id < 0 || mp_id >= (int)mps.size()) continue;
                if (!mps[mp_id].is_valid()) continue;

                auto it = mp_id_to_local.find(mp_id);
                int pt_idx;
                if (it == mp_id_to_local.end()) {
                    pt_idx = (int)points.size();
                    mp_id_to_local[mp_id] = pt_idx;
                    mp_global_ids.push_back(mp_id);
                    points.push_back(mps[mp_id].position());
                } else {
                    pt_idx = it->second;
                }

                observations.push_back({ki, pt_idx, (double)kps[kpi].pt.x, (double)kps[kpi].pt.y});
            }
        }
    }

    int N = (int)keyframes.size();
    int M = (int)points.size();
    int num_obs = (int)observations.size();

    if (num_obs < 20 || M < 10) return {0, 0};

    std::vector<cv::Mat> rvecs(N), tvecs(N);
    for (int i = 0; i < N; i++) {
        cv::Mat R = keyframes[i]->get_rotation();
        cv::Rodrigues(R, rvecs[i]);
        tvecs[i] = keyframes[i]->get_translation().clone();
    }

    std::vector<std::vector<int>> point_observers(M);
    for (const auto& obs : observations) {
        auto& pobs = point_observers[obs.pt_idx];
        bool found = false;
        for (int v : pobs) { if (v == obs.kf_idx) { found = true; break; } }
        if (!found) pobs.push_back(obs.kf_idx);
    }

    auto project_fn = [&](const std::vector<cv::Mat>& rv, const std::vector<cv::Mat>& tv,
                          const std::vector<cv::Point3d>& pts, int ki, int pi) -> cv::Point2d {
        cv::Mat R;
        cv::Rodrigues(rv[ki], R);
        const double* Rd = (const double*)R.data;
        const double* td = (const double*)tv[ki].data;
        double px = pts[pi].x - td[0];
        double py = pts[pi].y - td[1];
        double pz = pts[pi].z - td[2];
        // R_cam = R^T, so pc = R^T * (Pw - t)
        double cx_ = Rd[0]*px + Rd[3]*py + Rd[6]*pz;
        double cy_ = Rd[1]*px + Rd[4]*py + Rd[7]*pz;
        double cz_ = Rd[2]*px + Rd[5]*py + Rd[8]*pz;
        if (cz_ < 1e-6) return cv::Point2d(-1, -1);
        return cv::Point2d(fx * cx_ / cz_ + cx, fy * cy_ / cz_ + cy);
    };

    double error_before = 0;
    for (const auto& obs : observations) {
        auto proj = project_fn(rvecs, tvecs, points, obs.kf_idx, obs.pt_idx);
        if (proj.x < 0) continue;
        double dx = proj.x - obs.obs_u, dy = proj.y - obs.obs_v;
        error_before += dx * dx + dy * dy;
    }
    error_before = std::sqrt(error_before / num_obs);

    double lambda = 1e-4;
    const int MAX_ITER = 15;
    int pose_dim = 6 * N;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        struct PoseCache {
            cv::Mat R;       // 3x3
            double Rd[9];    // R data (row-major)
            double td[3];    // tvec data
        };
        std::vector<PoseCache> pcache(N);
        for (int i = 0; i < N; i++) {
            cv::Rodrigues(rvecs[i], pcache[i].R);
            const double* src = (const double*)pcache[i].R.data;
            std::memcpy(pcache[i].Rd, src, 9 * sizeof(double));
            const double* ts = (const double*)tvecs[i].data;
            std::memcpy(pcache[i].td, ts, 3 * sizeof(double));
        }

        std::vector<cv::Mat> Hpp(N), bp_vec(N);
        for (int i = 0; i < N; i++) {
            Hpp[i] = cv::Mat::zeros(6, 6, CV_64F);
            bp_vec[i] = cv::Mat::zeros(6, 1, CV_64F);
        }

        std::vector<cv::Mat> Hmm(M), bm_vec(M);
        for (int j = 0; j < M; j++) {
            Hmm[j] = cv::Mat::zeros(3, 3, CV_64F);
            bm_vec[j] = cv::Mat::zeros(3, 1, CV_64F);
        }

        std::unordered_map<int64_t, cv::Mat> Hpm_map;

        double total_cost = 0;

        for (const auto& obs : observations) {
            int ki = obs.kf_idx;
            int pi = obs.pt_idx;
            const double* Rd = pcache[ki].Rd;
            const double* td = pcache[ki].td;

            double dx_ = points[pi].x - td[0];
            double dy_ = points[pi].y - td[1];
            double dz_ = points[pi].z - td[2];
            double X = Rd[0]*dx_ + Rd[3]*dy_ + Rd[6]*dz_;
            double Y = Rd[1]*dx_ + Rd[4]*dy_ + Rd[7]*dz_;
            double Z = Rd[2]*dx_ + Rd[5]*dy_ + Rd[8]*dz_;

            if (Z < 1e-6) continue;

            double inv_z = 1.0 / Z;
            double inv_z2 = inv_z * inv_z;
            double u_proj = fx * X * inv_z + cx;
            double v_proj = fy * Y * inv_z + cy;

            double ru = u_proj - obs.obs_u;
            double rv = v_proj - obs.obs_v;
            double r_norm = std::sqrt(ru * ru + rv * rv);

            double w = 1.0;
            if (r_norm > HUBER_DELTA) w = HUBER_DELTA / r_norm;
            double sw = std::sqrt(w);  // scale Jacobian and residual

            total_cost += w * (ru * ru + rv * rv);

            double ru_w = ru * sw;
            double rv_w = rv * sw;

            double dp00 = fx * inv_z;
            double dp02 = -fx * X * inv_z2;
            double dp11 = fy * inv_z;
            double dp12 = -fy * Y * inv_z2;

            double Jm[2][3];
            for (int c = 0; c < 3; c++) {
                double rc0 = Rd[c * 3 + 0];  // R_cam(0,c)
                double rc1 = Rd[c * 3 + 1];  // R_cam(1,c)
                double rc2 = Rd[c * 3 + 2];  // R_cam(2,c)
                Jm[0][c] = (dp00 * rc0 + dp02 * rc2) * sw;
                Jm[1][c] = (dp11 * rc1 + dp12 * rc2) * sw;
            }

            double Jt[2][3];
            for (int c = 0; c < 3; c++) {
                Jt[0][c] = -Jm[0][c];  // negative of point Jacobian
                Jt[1][c] = -Jm[1][c];
            }

            double Jr[2][3];
            const double eps = 1e-6;
            for (int d = 0; d < 3; d++) {
                cv::Mat rv_p = rvecs[ki].clone();
                rv_p.at<double>(d) += eps;
                cv::Mat R_p;
                cv::Rodrigues(rv_p, R_p);
                const double* Rp = (const double*)R_p.data;
                double Xp = Rp[0]*dx_ + Rp[3]*dy_ + Rp[6]*dz_;
                double Yp = Rp[1]*dx_ + Rp[4]*dy_ + Rp[7]*dz_;
                double Zp = Rp[2]*dx_ + Rp[5]*dy_ + Rp[8]*dz_;
                if (Zp < 1e-6) { Jr[0][d] = 0; Jr[1][d] = 0; continue; }
                double up = fx * Xp / Zp + cx;
                double vp = fy * Yp / Zp + cy;
                Jr[0][d] = (up - u_proj) / eps * sw;
                Jr[1][d] = (vp - v_proj) / eps * sw;
            }

            double Jp[2][6];
            for (int c = 0; c < 3; c++) {
                Jp[0][c] = Jr[0][c];
                Jp[1][c] = Jr[1][c];
                Jp[0][c + 3] = Jt[0][c];
                Jp[1][c + 3] = Jt[1][c];
            }

            double* Hpp_data = (double*)Hpp[ki].data;
            for (int r = 0; r < 6; r++) {
                for (int c = r; c < 6; c++) {
                    double val = Jp[0][r] * Jp[0][c] + Jp[1][r] * Jp[1][c];
                    Hpp_data[r * 6 + c] += val;
                    if (r != c) Hpp_data[c * 6 + r] += val;
                }
            }
            double* bp_data = (double*)bp_vec[ki].data;
            for (int r = 0; r < 6; r++) {
                bp_data[r] += Jp[0][r] * ru_w + Jp[1][r] * rv_w;
            }

            double* Hmm_data = (double*)Hmm[pi].data;
            for (int r = 0; r < 3; r++) {
                for (int c = r; c < 3; c++) {
                    double val = Jm[0][r] * Jm[0][c] + Jm[1][r] * Jm[1][c];
                    Hmm_data[r * 3 + c] += val;
                    if (r != c) Hmm_data[c * 3 + r] += val;
                }
            }
            double* bm_data = (double*)bm_vec[pi].data;
            for (int r = 0; r < 3; r++) {
                bm_data[r] += Jm[0][r] * ru_w + Jm[1][r] * rv_w;
            }

            int64_t key = (int64_t)ki * M + pi;
            auto hpm_it = Hpm_map.find(key);
            if (hpm_it == Hpm_map.end()) {
                cv::Mat hpm = cv::Mat::zeros(6, 3, CV_64F);
                double* hd = (double*)hpm.data;
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 3; c++) {
                        hd[r * 3 + c] = Jp[0][r] * Jm[0][c] + Jp[1][r] * Jm[1][c];
                    }
                }
                Hpm_map[key] = hpm;
            } else {
                double* hd = (double*)hpm_it->second.data;
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 3; c++) {
                        hd[r * 3 + c] += Jp[0][r] * Jm[0][c] + Jp[1][r] * Jm[1][c];
                    }
                }
            }
        }

        for (int i = 0; i < N; i++) {
            for (int d = 0; d < 6; d++) {
                Hpp[i].at<double>(d, d) += 1e10;
            }
        }

        // S = Hpp - Σ_j Hpm_j * Hmm_j^{-1} * Hpm_j^T
        cv::Mat S = cv::Mat::zeros(pose_dim, pose_dim, CV_64F);
        cv::Mat bp_schur = cv::Mat::zeros(pose_dim, 1, CV_64F);

        for (int i = 0; i < N; i++) {
            Hpp[i].copyTo(S(cv::Rect(6 * i, 6 * i, 6, 6)));
            bp_vec[i].copyTo(bp_schur(cv::Rect(0, 6 * i, 1, 6)));
        }

        for (int d = 0; d < pose_dim; d++) {
            S.at<double>(d, d) *= (1.0 + lambda);
        }

        std::vector<cv::Mat> Hmm_inv(M);
        for (int j = 0; j < M; j++) {
            cv::Mat Hmm_d = Hmm[j].clone();
            for (int d = 0; d < 3; d++) {
                Hmm_d.at<double>(d, d) *= (1.0 + lambda);
            }

            cv::Mat inv;
            double det = cv::determinant(Hmm_d);
            if (std::abs(det) < 1e-20) {
                Hmm_inv[j] = cv::Mat::zeros(3, 3, CV_64F);
                continue;
            }
            cv::invert(Hmm_d, inv, cv::DECOMP_CHOLESKY);
            Hmm_inv[j] = inv;

            const auto& observers = point_observers[j];
            if (observers.empty()) continue;

            struct HpmEntry { int ki; cv::Mat mat; };
            std::vector<HpmEntry> hpm_entries;
            for (int ki : observers) {
                int64_t key = (int64_t)ki * M + j;
                auto it = Hpm_map.find(key);
                if (it != Hpm_map.end()) {
                    hpm_entries.push_back({ki, it->second});
                }
            }

            for (const auto& ea : hpm_entries) {
                cv::Mat Hpm_a_Hinv = ea.mat * inv;  // 6x3 * 3x3 = 6x3

                cv::Mat bp_contrib = Hpm_a_Hinv * bm_vec[j];  // 6x1
                bp_schur(cv::Rect(0, 6 * ea.ki, 1, 6)) -= bp_contrib;

                for (const auto& eb : hpm_entries) {
                    cv::Mat S_contrib = Hpm_a_Hinv * eb.mat.t();  // 6x3 * 3x6 = 6x6
                    S(cv::Rect(6 * eb.ki, 6 * ea.ki, 6, 6)) -= S_contrib;
                }
            }
        }

        cv::Mat dp;
        bool solved = cv::solve(S, -bp_schur, dp, cv::DECOMP_CHOLESKY);
        if (!solved) {
            solved = cv::solve(S, -bp_schur, dp, cv::DECOMP_SVD);
            if (!solved) {
                lambda *= 10;
                continue;
            }
        }

        // dm[j] = Hmm_inv[j] * (-bm[j] - Σ_i Hpm[i,j]^T * dp_i)
        std::vector<cv::Point3d> points_new(M);
        for (int j = 0; j < M; j++) {
            cv::Mat rhs = -bm_vec[j].clone();
            for (int ki : point_observers[j]) {
                int64_t key = (int64_t)ki * M + j;
                auto it = Hpm_map.find(key);
                if (it == Hpm_map.end()) continue;
                rhs -= it->second.t() * dp(cv::Rect(0, 6 * ki, 1, 6));  // 3x6 * 6x1 = 3x1
            }
            cv::Mat dm = Hmm_inv[j] * rhs;
            points_new[j].x = points[j].x + dm.at<double>(0);
            points_new[j].y = points[j].y + dm.at<double>(1);
            points_new[j].z = points[j].z + dm.at<double>(2);
        }

        std::vector<cv::Mat> rvecs_new(N), tvecs_new(N);
        for (int i = 0; i < N; i++) {
            rvecs_new[i] = rvecs[i] + dp(cv::Rect(0, 6 * i, 1, 3));
            tvecs_new[i] = tvecs[i] + dp(cv::Rect(0, 6 * i + 3, 1, 3));
        }

        double new_cost = 0;
        for (const auto& obs : observations) {
            auto proj = project_fn(rvecs_new, tvecs_new, points_new, obs.kf_idx, obs.pt_idx);
            if (proj.x < 0) { new_cost += 100.0; continue; }
            double du = proj.x - obs.obs_u, dv = proj.y - obs.obs_v;
            double rn = std::sqrt(du * du + dv * dv);
            double w = (rn > HUBER_DELTA) ? HUBER_DELTA / rn : 1.0;
            new_cost += w * (du * du + dv * dv);
        }

        if (new_cost < total_cost) {
            rvecs = std::move(rvecs_new);
            tvecs = std::move(tvecs_new);
            points = std::move(points_new);
            lambda = std::max(1e-7, lambda * 0.5);

            double rel_change = (total_cost - new_cost) / (total_cost + 1e-10);
            if (rel_change < 1e-4) break;
        } else {
            lambda *= 5.0;
            if (lambda > 1e6) break;
        }
    }

    double error_after = 0;
    for (const auto& obs : observations) {
        auto proj = project_fn(rvecs, tvecs, points, obs.kf_idx, obs.pt_idx);
        if (proj.x < 0) continue;
        double dx = proj.x - obs.obs_u, dy = proj.y - obs.obs_v;
        error_after += dx * dx + dy * dy;
    }
    error_after = std::sqrt(error_after / num_obs);

    {
        std::lock_guard<std::mutex> lock(map.mutex());
        auto& mps = map.map_points();

        for (int i = 1; i < N; i++) {
            cv::Mat R_opt;
            cv::Rodrigues(rvecs[i], R_opt);
            keyframes[i]->set_pose(R_opt, tvecs[i]);
        }

        for (int j = 0; j < M; j++) {
            int gid = mp_global_ids[j];
            if (gid >= 0 && gid < (int)mps.size() && mps[gid].is_valid()) {
                mps[gid].set_position(points[j]);
            }
        }
    }

    return {error_before, error_after};
}

// Height prior edge for PGO
class EdgeHeightPrior : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeHeightPrior(const Eigen::Vector3d& gravity_dir) : gravity_dir_(gravity_dir) {}

    void computeError() override {
        const g2o::VertexSE3* v = static_cast<const g2o::VertexSE3*>(_vertices[0]);
        Eigen::Vector3d pos = v->estimate().translation();
        _error(0) = gravity_dir_.dot(pos) - _measurement;
    }

    bool read(std::istream&) override { return true; }
    bool write(std::ostream&) const override { return true; }

private:
    Eigen::Vector3d gravity_dir_;
};

static Eigen::Isometry3d cvToIsometry(const cv::Mat& R, const cv::Mat& t) {
    Eigen::Matrix3d R_eigen;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_eigen(i, j) = R.at<double>(i, j);
    Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear() = R_eigen;
    iso.translation() = t_eigen;
    return iso;
}

static void isometryToCv(const Eigen::Isometry3d& iso, cv::Mat& R, cv::Mat& t) {
    R = cv::Mat::eye(3, 3, CV_64F);
    t = cv::Mat::zeros(3, 1, CV_64F);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            R.at<double>(i, j) = iso.linear()(i, j);
        t.at<double>(i) = iso.translation()(i);
    }
}

int Optimizer::pose_graph_optimize(
    Map& map,
    const std::vector<LoopConstraint>& loop_constraints,
    const cv::Mat& gravity_world,
    double initial_height,
    bool has_height_prior) {

    std::vector<std::shared_ptr<Frame>> keyframes;
    {
        std::lock_guard<std::mutex> lock(map.mutex());
        auto& frames = map.frames_direct();
        for (auto& f : frames) {
            if (f->is_keyframe()) keyframes.push_back(f);
        }
    }

    if ((int)keyframes.size() < 3) return 0;

    std::unordered_map<int, int> frame_id_to_kf_idx;
    for (int i = 0; i < (int)keyframes.size(); i++) {
        frame_id_to_kf_idx[keyframes[i]->id()] = i;
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolver;
    typedef g2o::LinearSolverEigen<BlockSolver::PoseMatrixType> LinearSolver;

    g2o::SparseOptimizer optimizer;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolver>(std::make_unique<LinearSolver>())
    );
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    int N = (int)keyframes.size();
    std::vector<Eigen::Isometry3d> old_poses(N);

    for (int i = 0; i < N; i++) {
        cv::Mat R = keyframes[i]->get_rotation();
        cv::Mat t = keyframes[i]->get_translation();
        old_poses[i] = cvToIsometry(R, t);

        auto vertex = new g2o::VertexSE3();
        vertex->setId(i);
        vertex->setEstimate(old_poses[i]);
        if (i == 0) vertex->setFixed(true);  // Anchor first keyframe
        optimizer.addVertex(vertex);
    }

    int edge_id = 0;
    Eigen::Matrix<double, 6, 6> odom_info = Eigen::Matrix<double, 6, 6>::Zero();
    for (int i = 0; i < 3; i++)
        odom_info(i, i) = 1.0 / (Config::PGO_ODOM_TRANS_SIGMA * Config::PGO_ODOM_TRANS_SIGMA);
    for (int i = 3; i < 6; i++)
        odom_info(i, i) = 1.0 / (Config::PGO_ODOM_ROT_SIGMA * Config::PGO_ODOM_ROT_SIGMA);

    for (int i = 0; i < N - 1; i++) {
        auto edge = new g2o::EdgeSE3();
        edge->setId(edge_id++);
        edge->setVertex(0, optimizer.vertex(i));
        edge->setVertex(1, optimizer.vertex(i + 1));

        Eigen::Isometry3d T_rel = old_poses[i].inverse() * old_poses[i + 1];
        edge->setMeasurement(T_rel);
        edge->setInformation(odom_info);
        optimizer.addEdge(edge);
    }

    int loop_edges_added = 0;

    for (const auto& lc : loop_constraints) {
        auto it_from = frame_id_to_kf_idx.find(lc.from_id);
        auto it_to = frame_id_to_kf_idx.find(lc.to_id);
        if (it_from == frame_id_to_kf_idx.end() || it_to == frame_id_to_kf_idx.end()) continue;

        int idx_from = it_from->second;
        int idx_to = it_to->second;

        Eigen::Matrix3d R_eigen;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                R_eigen(r, c) = lc.R_rel.at<double>(r, c);
        Eigen::Vector3d t_eigen(lc.t_rel.at<double>(0), lc.t_rel.at<double>(1), lc.t_rel.at<double>(2));

        Eigen::Isometry3d T_rel = Eigen::Isometry3d::Identity();
        T_rel.linear() = R_eigen;
        T_rel.translation() = t_eigen;

        Eigen::Matrix<double, 6, 6> lc_info = Eigen::Matrix<double, 6, 6>::Zero();
        for (int i = 0; i < 3; i++)
            lc_info(i, i) = 1.0 / (lc.trans_sigma * lc.trans_sigma);
        for (int i = 3; i < 6; i++)
            lc_info(i, i) = 1.0 / (lc.rot_sigma * lc.rot_sigma);

        auto edge = new g2o::EdgeSE3();
        edge->setId(edge_id++);
        edge->setVertex(0, optimizer.vertex(idx_from));
        edge->setVertex(1, optimizer.vertex(idx_to));
        edge->setMeasurement(T_rel);
        edge->setInformation(lc_info);
        optimizer.addEdge(edge);
        loop_edges_added++;
    }

    if (has_height_prior && !gravity_world.empty()) {
        Eigen::Vector3d g_dir(gravity_world.at<double>(0),
                              gravity_world.at<double>(1),
                              gravity_world.at<double>(2));

        Eigen::Matrix<double, 1, 1> h_info;
        h_info(0, 0) = 1.0 / (Config::PGO_HEIGHT_SIGMA * Config::PGO_HEIGHT_SIGMA);

        for (int i = 0; i < N; i++) {
            auto edge = new EdgeHeightPrior(g_dir);
            edge->setId(edge_id++);
            edge->setVertex(0, optimizer.vertex(i));
            edge->setMeasurement(initial_height);
            edge->setInformation(h_info);
            optimizer.addEdge(edge);
        }
    }

    if (loop_edges_added == 0 && !has_height_prior) return 0;

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    {
        std::lock_guard<std::mutex> lock(map.mutex());

        std::vector<Eigen::Isometry3d> new_poses(N);
        for (int i = 0; i < N; i++) {
            auto v = static_cast<g2o::VertexSE3*>(optimizer.vertex(i));
            new_poses[i] = v->estimate();
        }

        for (int i = 0; i < N; i++) {
            cv::Mat R_opt, t_opt;
            isometryToCv(new_poses[i], R_opt, t_opt);
            keyframes[i]->set_pose(R_opt, t_opt);
        }

        auto& frames = map.frames_direct();
        for (auto& f : frames) {
            if (f->is_keyframe()) continue;

            int fid = f->id();
            int prev_kf_idx = -1, next_kf_idx = -1;
            for (int i = 0; i < N; i++) {
                if (keyframes[i]->id() <= fid) prev_kf_idx = i;
                if (keyframes[i]->id() > fid && next_kf_idx < 0) next_kf_idx = i;
            }

            if (prev_kf_idx < 0) continue;
            if (next_kf_idx < 0) next_kf_idx = prev_kf_idx;

            Eigen::Vector3d dt_prev = new_poses[prev_kf_idx].translation() - old_poses[prev_kf_idx].translation();
            Eigen::Vector3d dt_next = new_poses[next_kf_idx].translation() - old_poses[next_kf_idx].translation();

            double alpha = 0.0;
            if (prev_kf_idx != next_kf_idx) {
                int prev_id = keyframes[prev_kf_idx]->id();
                int next_id = keyframes[next_kf_idx]->id();
                alpha = (double)(fid - prev_id) / (double)(next_id - prev_id);
            }

            Eigen::Vector3d dt_interp = (1.0 - alpha) * dt_prev + alpha * dt_next;

            cv::Mat t_old = f->get_translation();
            cv::Mat t_new = t_old.clone();
            t_new.at<double>(0) += dt_interp(0);
            t_new.at<double>(1) += dt_interp(1);
            t_new.at<double>(2) += dt_interp(2);
            f->set_translation(t_new);
        }

        auto& mps = map.map_points();
        for (auto& mp : mps) {
            if (!mp.is_valid()) continue;
            const auto& obs = mp.observations();
            if (obs.empty()) continue;

            int obs_frame_id = obs[0].first;
            int kf_idx = -1;

            auto it = frame_id_to_kf_idx.find(obs_frame_id);
            if (it != frame_id_to_kf_idx.end()) {
                kf_idx = it->second;
            } else {
                int best_dist = INT_MAX;
                for (int i = 0; i < N; i++) {
                    int dist = std::abs(keyframes[i]->id() - obs_frame_id);
                    if (dist < best_dist) {
                        best_dist = dist;
                        kf_idx = i;
                    }
                }
            }

            if (kf_idx >= 0) {
                Eigen::Isometry3d delta = new_poses[kf_idx] * old_poses[kf_idx].inverse();
                cv::Point3d pos = mp.position();
                Eigen::Vector3d pw(pos.x, pos.y, pos.z);
                Eigen::Vector3d pw_new = delta * pw;
                mp.set_position(cv::Point3d(pw_new(0), pw_new(1), pw_new(2)));
            }
        }
    }

    return loop_edges_added;
}

void Optimizer::correct_loop(Map& map, int loop_start_id, int loop_end_id,
                              const cv::Mat& R_correction, const cv::Mat& t_correction) {
    std::lock_guard<std::mutex> lock(map.mutex());

    int range = loop_end_id - loop_start_id;
    if (range <= 0) return;

    auto& frames = map.frames_direct();
    for (auto& frame : frames) {
        int fid = frame->id();
        if (fid < loop_start_id || fid > loop_end_id) continue;

        double alpha = (double)(fid - loop_start_id) / range;
        cv::Mat t_corr = alpha * t_correction;
        cv::Mat t_old = frame->get_translation();
        frame->set_translation(t_old + t_corr);
    }

    auto& points = map.map_points();
    for (auto& mp : points) {
        if (!mp.is_valid()) continue;
        const auto& obs = mp.observations();
        if (obs.empty()) continue;
        int obs_frame = obs[0].first;
        if (obs_frame >= loop_start_id && obs_frame <= loop_end_id) {
            double alpha = (double)(obs_frame - loop_start_id) / range;
            cv::Point3d pos = mp.position();
            pos.x += alpha * t_correction.at<double>(0);
            pos.y += alpha * t_correction.at<double>(1);
            pos.z += alpha * t_correction.at<double>(2);
            mp.set_position(pos);
        }
    }
}
