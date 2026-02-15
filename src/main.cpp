#include "Frame.h"
#include "Slam.h"
#include "Viewer.h"
#include "Config.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <cmath>
#include <iomanip>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <nanoflann.hpp>

namespace fs = std::filesystem;

struct ImageInfo {
    double timestamp;
    std::string rgb_path;
    std::string depth_path;
};

struct DepthInfo {
    double timestamp;
    std::string path;
};

std::vector<DepthInfo> load_depth_list(const std::string& dataset_path) {
    std::vector<DepthInfo> depths;
    std::string depth_txt = dataset_path + "depth.txt";
    std::ifstream ifs(depth_txt);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            DepthInfo info;
            std::string rel_path;
            if (iss >> info.timestamp >> rel_path) {
                info.path = dataset_path + rel_path;
                depths.push_back(info);
            }
        }
    }
    std::sort(depths.begin(), depths.end(),
              [](const DepthInfo& a, const DepthInfo& b) {
                  return a.timestamp < b.timestamp;
              });
    return depths;
}

std::string find_closest_depth(double rgb_ts, const std::vector<DepthInfo>& depths,
                                double max_diff = 0.02) {
    if (depths.empty()) return "";
    int lo = 0, hi = (int)depths.size() - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (depths[mid].timestamp < rgb_ts) lo = mid + 1;
        else hi = mid;
    }
    double best_diff = std::abs(depths[lo].timestamp - rgb_ts);
    int best = lo;
    if (lo > 0) {
        double diff2 = std::abs(depths[lo - 1].timestamp - rgb_ts);
        if (diff2 < best_diff) { best_diff = diff2; best = lo - 1; }
    }
    if (best_diff <= max_diff) return depths[best].path;
    return "";
}

std::vector<ImageInfo> load_from_associations(const std::string& dataset_path) {
    std::vector<ImageInfo> images;
    std::string assoc_file = dataset_path + "associations.txt";
    std::ifstream ifs(assoc_file);
    if (!ifs.is_open()) return images;

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double rgb_ts, depth_ts;
        std::string rgb_rel, depth_rel;
        if (iss >> rgb_ts >> rgb_rel >> depth_ts >> depth_rel) {
            ImageInfo info;
            info.timestamp = rgb_ts;
            info.rgb_path = dataset_path + rgb_rel;
            info.depth_path = dataset_path + depth_rel;
            images.push_back(info);
        }
    }

    std::sort(images.begin(), images.end(),
              [](const ImageInfo& a, const ImageInfo& b) { return a.timestamp < b.timestamp; });
    return images;
}

struct AccelSample {
    double timestamp;
    double ax, ay, az;
};

std::vector<AccelSample> load_accelerometer(const std::string& dataset_path) {
    std::vector<AccelSample> data;
    std::string accel_file = dataset_path + "accelerometer.txt";
    std::ifstream ifs(accel_file);
    if (!ifs.is_open()) {
        return data;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        AccelSample s;
        if (iss >> s.timestamp >> s.ax >> s.ay >> s.az) {
            data.push_back(s);
        }
    }
    std::sort(data.begin(), data.end(),
              [](const AccelSample& a, const AccelSample& b) { return a.timestamp < b.timestamp; });
    return data;
}

std::vector<ImageInfo> load_image_list(const std::string& dataset_path) {
    // Try associations.txt first (guaranteed RGB-depth pairing)
    auto images = load_from_associations(dataset_path);
    if (!images.empty()) {
        int with_depth = 0;
        for (const auto& img : images) {
            if (!img.depth_path.empty()) with_depth++;
        }
        return images;
    }

    images.clear();

    auto depths = load_depth_list(dataset_path);

    std::string rgb_txt = dataset_path + "rgb.txt";
    std::ifstream ifs(rgb_txt);

    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            ImageInfo info;
            std::string rel_path;
            if (iss >> info.timestamp >> rel_path) {
                info.rgb_path = dataset_path + rel_path;
                info.depth_path = find_closest_depth(info.timestamp, depths);
                images.push_back(info);
            }
        }
        ifs.close();
    }

    if (images.empty()) {
        // Fallback: scan rgb/ directory
        std::string search_dir = dataset_path;
        if (fs::exists(dataset_path + "rgb/")) {
            search_dir = dataset_path + "rgb/";
        }
        for (const auto& entry : fs::directory_iterator(search_dir)) {
            if (entry.path().extension() == ".png" ||
                entry.path().extension() == ".jpg") {
                ImageInfo info;
                info.timestamp = Frame::parse_timestamp(entry.path().filename().string());
                info.rgb_path = entry.path().string();
                info.depth_path = find_closest_depth(info.timestamp, depths);
                images.push_back(info);
            }
        }
    }

    std::sort(images.begin(), images.end(),
              [](const ImageInfo& a, const ImageInfo& b) {
                  return a.timestamp < b.timestamp;
              });

    int with_depth = 0;
    for (const auto& img : images) {
        if (!img.depth_path.empty()) with_depth++;
    }
    return images;
}

// Ground truth
struct GTPose {
    double timestamp;
    double tx, ty, tz;
    double qx, qy, qz, qw;
};

std::vector<GTPose> load_ground_truth(const std::string& dataset_path) {
    std::vector<GTPose> gt;
    std::ifstream ifs(dataset_path + "groundtruth.txt");
    if (!ifs.is_open()) return gt;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        GTPose p;
        if (iss >> p.timestamp >> p.tx >> p.ty >> p.tz >> p.qx >> p.qy >> p.qz >> p.qw) {
            gt.push_back(p);
        }
    }
    std::sort(gt.begin(), gt.end(),
              [](const GTPose& a, const GTPose& b) { return a.timestamp < b.timestamp; });
    return gt;
}

GTPose find_closest_gt(double ts, const std::vector<GTPose>& gt) {
    int lo = 0, hi = (int)gt.size() - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (gt[mid].timestamp < ts) lo = mid + 1;
        else hi = mid;
    }
    if (lo > 0 && std::abs(gt[lo-1].timestamp - ts) < std::abs(gt[lo].timestamp - ts))
        return gt[lo-1];
    return gt[lo];
}

struct AlignmentResult {
    double scale;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    double ate_rmse;
    std::vector<cv::Point3d> aligned_trajectory;
    std::vector<cv::Point3d> gt_trajectory;
};

AlignmentResult compute_ate(const std::vector<std::pair<double, cv::Point3d>>& est_poses,
                             const std::vector<GTPose>& gt_all) {
    AlignmentResult result;
    result.ate_rmse = -1;
    result.scale = 1.0;

    if (est_poses.size() < 3 || gt_all.empty()) return result;

    std::vector<Eigen::Vector3d> est_pts, gt_pts;
    for (const auto& ep : est_poses) {
        GTPose gp = find_closest_gt(ep.first, gt_all);
        if (std::abs(gp.timestamp - ep.first) > 0.05) continue;
        est_pts.emplace_back(ep.second.x, ep.second.y, ep.second.z);
        gt_pts.emplace_back(gp.tx, gp.ty, gp.tz);
    }

    int n = (int)est_pts.size();
    if (n < 3) return result;

    Eigen::Vector3d est_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d gt_mean = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; i++) {
        est_mean += est_pts[i];
        gt_mean += gt_pts[i];
    }
    est_mean /= n;
    gt_mean /= n;

    std::vector<Eigen::Vector3d> est_c(n), gt_c(n);
    for (int i = 0; i < n; i++) {
        est_c[i] = est_pts[i] - est_mean;
        gt_c[i] = gt_pts[i] - gt_mean;
    }

    double sigma_est = 0, sigma_gt = 0;
    for (int i = 0; i < n; i++) {
        sigma_est += est_c[i].squaredNorm();
        sigma_gt += gt_c[i].squaredNorm();
    }
    sigma_est /= n;
    sigma_gt /= n;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int i = 0; i < n; i++) {
        H += gt_c[i] * est_c[i].transpose();
    }
    H /= n;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Vector3d S = svd.singularValues();

    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if ((U * V.transpose()).determinant() < 0) {
        D(2, 2) = -1;
    }

    result.R = U * D * V.transpose();
    result.scale = (S.asDiagonal() * D).trace() / sigma_est;
    result.t = gt_mean - result.scale * result.R * est_mean;

    double sum_sq = 0;
    result.aligned_trajectory.resize(n);
    result.gt_trajectory.resize(n);
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d aligned = result.scale * result.R * est_pts[i] + result.t;
        result.aligned_trajectory[i] = cv::Point3d(aligned.x(), aligned.y(), aligned.z());
        result.gt_trajectory[i] = cv::Point3d(gt_pts[i].x(), gt_pts[i].y(), gt_pts[i].z());
        sum_sq += (aligned - gt_pts[i]).squaredNorm();
    }
    result.ate_rmse = std::sqrt(sum_sq / n);

    return result;
}

// RPE
struct RPEResult {
    double rpe_trans_rmse;   // translation RPE RMSE (m/step)
    double rpe_trans_mean;   // translation RPE mean
    double rpe_trans_max;    // translation RPE max
    double rpe_rot_rmse;     // rotation RPE RMSE (deg/step)
    double rpe_rot_mean;
    int num_pairs;
};

RPEResult compute_rpe(const std::vector<std::pair<double, cv::Point3d>>& est_poses,
                       const std::vector<GTPose>& gt_all,
                       int delta = 1) {
    RPEResult result = {-1, -1, -1, -1, -1, 0};
    int n = (int)est_poses.size();
    if (n < delta + 1 || gt_all.empty()) return result;

    double sum_trans_sq = 0, sum_trans = 0, max_trans = 0;
    double sum_rot_sq = 0, sum_rot = 0;
    int count = 0;

    for (int i = 0; i < n - delta; i++) {
        GTPose g1 = find_closest_gt(est_poses[i].first, gt_all);
        GTPose g2 = find_closest_gt(est_poses[i + delta].first, gt_all);
        if (std::abs(g1.timestamp - est_poses[i].first) > 0.05) continue;
        if (std::abs(g2.timestamp - est_poses[i + delta].first) > 0.05) continue;

        double gt_dx = g2.tx - g1.tx;
        double gt_dy = g2.ty - g1.ty;
        double gt_dz = g2.tz - g1.tz;

        double est_dx = est_poses[i + delta].second.x - est_poses[i].second.x;
        double est_dy = est_poses[i + delta].second.y - est_poses[i].second.y;
        double est_dz = est_poses[i + delta].second.z - est_poses[i].second.z;

        double err_x = est_dx - gt_dx;
        double err_y = est_dy - gt_dy;
        double err_z = est_dz - gt_dz;
        double err_trans = std::sqrt(err_x*err_x + err_y*err_y + err_z*err_z);

        sum_trans_sq += err_trans * err_trans;
        sum_trans += err_trans;
        if (err_trans > max_trans) max_trans = err_trans;

        count++;
    }

    if (count == 0) return result;
    result.rpe_trans_rmse = std::sqrt(sum_trans_sq / count);
    result.rpe_trans_mean = sum_trans / count;
    result.rpe_trans_max = max_trans;
    result.num_pairs = count;
    return result;
}

void save_trajectory(const std::string& path,
                     const std::vector<std::pair<double, cv::Point3d>>& poses) {
    std::ofstream ofs(path);
    ofs << std::fixed << std::setprecision(6);
    for (const auto& p : poses) {
        ofs << p.first << " " << p.second.x << " " << p.second.y << " " << p.second.z
            << " 0 0 0 1" << std::endl;
    }
}

void rotation_to_quaternion(const cv::Mat& R, double& qx, double& qy, double& qz, double& qw) {
    double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
    if (trace > 0) {
        double s = 0.5 / std::sqrt(trace + 1.0);
        qw = 0.25 / s;
        qx = (R.at<double>(2,1) - R.at<double>(1,2)) * s;
        qy = (R.at<double>(0,2) - R.at<double>(2,0)) * s;
        qz = (R.at<double>(1,0) - R.at<double>(0,1)) * s;
    } else if (R.at<double>(0,0) > R.at<double>(1,1) && R.at<double>(0,0) > R.at<double>(2,2)) {
        double s = 2.0 * std::sqrt(1.0 + R.at<double>(0,0) - R.at<double>(1,1) - R.at<double>(2,2));
        qw = (R.at<double>(2,1) - R.at<double>(1,2)) / s;
        qx = 0.25 * s;
        qy = (R.at<double>(0,1) + R.at<double>(1,0)) / s;
        qz = (R.at<double>(0,2) + R.at<double>(2,0)) / s;
    } else if (R.at<double>(1,1) > R.at<double>(2,2)) {
        double s = 2.0 * std::sqrt(1.0 + R.at<double>(1,1) - R.at<double>(0,0) - R.at<double>(2,2));
        qw = (R.at<double>(0,2) - R.at<double>(2,0)) / s;
        qx = (R.at<double>(0,1) + R.at<double>(1,0)) / s;
        qy = 0.25 * s;
        qz = (R.at<double>(1,2) + R.at<double>(2,1)) / s;
    } else {
        double s = 2.0 * std::sqrt(1.0 + R.at<double>(2,2) - R.at<double>(0,0) - R.at<double>(1,1));
        qw = (R.at<double>(1,0) - R.at<double>(0,1)) / s;
        qx = (R.at<double>(0,2) + R.at<double>(2,0)) / s;
        qy = (R.at<double>(1,2) + R.at<double>(2,1)) / s;
        qz = 0.25 * s;
    }
}

void save_trajectory_full(const std::string& path,
                          const std::vector<std::shared_ptr<Frame>>& frames) {
    std::ofstream ofs(path);
    ofs << std::fixed << std::setprecision(6);
    for (const auto& f : frames) {
        cv::Mat R = f->get_rotation();
        cv::Mat t = f->get_translation();
        double qx, qy, qz, qw;
        rotation_to_quaternion(R, qx, qy, qz, qw);
        ofs << f->timestamp() << " "
            << t.at<double>(0) << " " << t.at<double>(1) << " " << t.at<double>(2) << " "
            << qx << " " << qy << " " << qz << " " << qw << std::endl;
    }
}

std::vector<cv::Point3d> adaptive_downsample(const std::vector<cv::Point3d>& pts,
                                              const std::vector<cv::Point3d>& traj) {
    if (pts.empty()) return pts;
    if (traj.empty()) return pts;

    std::vector<cv::Point3d> traj_sub;
    int step = std::max(1, (int)traj.size() / 50);
    for (size_t i = 0; i < traj.size(); i += step) traj_sub.push_back(traj[i]);

    struct VoxelHash {
        size_t operator()(const std::tuple<int,int,int>& k) const {
            auto h1 = std::hash<int>{}(std::get<0>(k));
            auto h2 = std::hash<int>{}(std::get<1>(k));
            auto h3 = std::hash<int>{}(std::get<2>(k));
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
    std::unordered_map<std::tuple<int,int,int>, cv::Point3d, VoxelHash> grid;

    for (const auto& p : pts) {
        double min_dist_sq = 1e18;
        for (const auto& tp : traj_sub) {
            double dx = p.x - tp.x, dy = p.y - tp.y, dz = p.z - tp.z;
            double d = dx*dx + dy*dy + dz*dz;
            if (d < min_dist_sq) min_dist_sq = d;
        }
        double dist = std::sqrt(min_dist_sq);

        double voxel;
        if (dist < 1.5)       voxel = 0.01;   // very near: 1cm
        else if (dist < 3.0)  voxel = 0.03;   // near: 3cm
        else if (dist < 5.0)  voxel = 0.08;   // medium: 8cm
        else                  voxel = 0.2;     // far: 20cm

        auto key = std::make_tuple(
            (int)std::floor(p.x / voxel),
            (int)std::floor(p.y / voxel),
            (int)std::floor(p.z / voxel));
        grid.emplace(key, p);
    }

    std::vector<cv::Point3d> result;
    result.reserve(grid.size());
    for (const auto& kv : grid) result.push_back(kv.second);
    return result;
}

struct EigenPointCloud {
    std::vector<Eigen::Vector3d> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx][dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree3D = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, EigenPointCloud>,
    EigenPointCloud, 3>;

std::vector<cv::Point3d> statistical_outlier_removal(const std::vector<cv::Point3d>& pts,
                                                       int k_neighbors = 20,
                                                       double std_ratio = 1.0) {
    int n = (int)pts.size();
    if (n <= k_neighbors) return pts;

    EigenPointCloud cloud;
    cloud.pts.resize(n);
    for (int i = 0; i < n; i++)
        cloud.pts[i] = Eigen::Vector3d(pts[i].x, pts[i].y, pts[i].z);
    KDTree3D tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    int k_query = k_neighbors + 1; // +1 because query point is its own nearest neighbor
    std::vector<uint32_t> indices(k_query);
    std::vector<double> dists_sq(k_query);
    std::vector<double> mean_dists(n, 0.0);

    for (int i = 0; i < n; i++) {
        tree.knnSearch(cloud.pts[i].data(), k_query, indices.data(), dists_sq.data());
        double sum = 0;
        int count = 0;
        for (int ki = 0; ki < k_query; ki++) {
            if (indices[ki] == (uint32_t)i) continue; // skip self
            sum += std::sqrt(dists_sq[ki]);
            count++;
            if (count >= k_neighbors) break;
        }
        mean_dists[i] = (count > 0) ? sum / count : 0;
    }

    double global_mean = 0;
    for (double d : mean_dists) global_mean += d;
    global_mean /= n;

    double global_var = 0;
    for (double d : mean_dists) global_var += (d - global_mean) * (d - global_mean);
    global_var /= n;
    double global_std = std::sqrt(global_var);

    double threshold = global_mean + std_ratio * global_std;

    std::vector<cv::Point3d> result;
    result.reserve(n);
    for (int i = 0; i < n; i++) {
        if (mean_dists[i] <= threshold) {
            result.push_back(pts[i]);
        }
    }

    return result;
}

std::vector<cv::Point3d> surface_aware_filter(const std::vector<cv::Point3d>& pts,
                                                int k_neighbors = 25,
                                                double min_anisotropy = 0.3) {
    int n = (int)pts.size();
    if (n <= k_neighbors) return pts;

    EigenPointCloud cloud;
    cloud.pts.resize(n);
    for (int i = 0; i < n; i++)
        cloud.pts[i] = Eigen::Vector3d(pts[i].x, pts[i].y, pts[i].z);
    KDTree3D tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    int k_query = k_neighbors + 1;
    std::vector<uint32_t> indices(k_query);
    std::vector<double> dists_sq(k_query);
    std::vector<bool> keep(n, false);

    for (int i = 0; i < n; i++) {
        tree.knnSearch(cloud.pts[i].data(), k_query, indices.data(), dists_sq.data());

        double cx = 0, cy = 0, cz = 0;
        int count = 0;
        for (int ki = 0; ki < k_query && count < k_neighbors; ki++) {
            if (indices[ki] == (uint32_t)i) continue;
            cx += pts[indices[ki]].x;
            cy += pts[indices[ki]].y;
            cz += pts[indices[ki]].z;
            count++;
        }
        if (count == 0) continue;
        cx /= count; cy /= count; cz /= count;

        double cov[3][3] = {};
        count = 0;
        for (int ki = 0; ki < k_query && count < k_neighbors; ki++) {
            if (indices[ki] == (uint32_t)i) continue;
            double dx = pts[indices[ki]].x - cx;
            double dy = pts[indices[ki]].y - cy;
            double dz = pts[indices[ki]].z - cz;
            cov[0][0] += dx*dx; cov[0][1] += dx*dy; cov[0][2] += dx*dz;
            cov[1][1] += dy*dy; cov[1][2] += dy*dz;
            cov[2][2] += dz*dz;
            count++;
        }
        cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];

        cv::Mat covMat(3, 3, CV_64F);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                covMat.at<double>(r, c) = cov[r][c];

        cv::Mat eigenvalues;
        cv::eigen(covMat, eigenvalues);
        double l1 = eigenvalues.at<double>(0);  // largest
        double l3 = eigenvalues.at<double>(2);  // smallest

        if (l1 < 1e-12) continue;

        double anisotropy = 1.0 - (l3 / l1);

        if (anisotropy >= min_anisotropy) {
            keep[i] = true;
        }
    }

    std::vector<cv::Point3d> result;
    result.reserve(n);
    int kept = 0;
    for (int i = 0; i < n; i++) {
        if (keep[i]) { result.push_back(pts[i]); kept++; }
    }

    return result;
}

std::vector<Eigen::Vector3d> estimate_normals(const EigenPointCloud& cloud,
                                                const KDTree3D& tree,
                                                int k = 20) {
    int n = (int)cloud.pts.size();
    std::vector<Eigen::Vector3d> normals(n);
    std::vector<uint32_t> indices(k);
    std::vector<double> dists(k);

    for (int i = 0; i < n; i++) {
        size_t found = tree.knnSearch(cloud.pts[i].data(), k, indices.data(), dists.data());
        if (found < 3) { normals[i] = Eigen::Vector3d::UnitZ(); continue; }

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (size_t j = 0; j < found; j++)
            centroid += cloud.pts[indices[j]];
        centroid /= (double)found;

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (size_t j = 0; j < found; j++) {
            Eigen::Vector3d d = cloud.pts[indices[j]] - centroid;
            cov += d * d.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
        normals[i] = eig.eigenvectors().col(0); // smallest eigenvalue
    }
    return normals;
}

std::vector<cv::Point3d> compute_normals(const std::vector<cv::Point3d>& pts, int k = 20) {
    EigenPointCloud cloud;
    cloud.pts.resize(pts.size());
    for (size_t i = 0; i < pts.size(); i++)
        cloud.pts[i] = Eigen::Vector3d(pts[i].x, pts[i].y, pts[i].z);
    KDTree3D tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();
    auto enormals = estimate_normals(cloud, tree, k);
    std::vector<cv::Point3d> result(pts.size());
    for (size_t i = 0; i < pts.size(); i++)
        result[i] = cv::Point3d(enormals[i].x(), enormals[i].y(), enormals[i].z());
    return result;
}

void compute_mesh(const std::vector<cv::Point3d>& pts,
                  const std::vector<cv::Point3d>& normals,
                  std::vector<cv::Point3d>& tri_verts,
                  std::vector<cv::Point3d>& tri_normals,
                  double max_edge = 0.12) {
    tri_verts.clear();
    tri_normals.clear();
    if (pts.size() < 3) return;

    EigenPointCloud cloud;
    cloud.pts.resize(pts.size());
    for (size_t i = 0; i < pts.size(); i++)
        cloud.pts[i] = Eigen::Vector3d(pts[i].x, pts[i].y, pts[i].z);
    KDTree3D tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    const int k = 20;
    const double normal_thresh = 0.5;
    const double max_edge_sq = max_edge * max_edge;
    const double max_ang_gap = 1.2;  // ~70 degrees — skip fan across large angular gaps

    tri_verts.reserve(pts.size() * 10 * 3);
    tri_normals.reserve(pts.size() * 10 * 3);

    std::vector<uint32_t> indices(k);
    std::vector<double> dists(k);

    for (size_t i = 0; i < pts.size(); i++) {
        size_t found = tree.knnSearch(cloud.pts[i].data(), k, indices.data(), dists.data());

        Eigen::Vector3d ni(normals[i].x, normals[i].y, normals[i].z);
        Eigen::Vector3d pi = cloud.pts[i];

        // Build tangent basis
        double ax = std::abs(ni.x()), ay = std::abs(ni.y()), az = std::abs(ni.z());
        Eigen::Vector3d up;
        if (ax <= ay && ax <= az) up = Eigen::Vector3d(1,0,0);
        else if (ay <= az) up = Eigen::Vector3d(0,1,0);
        else up = Eigen::Vector3d(0,0,1);
        Eigen::Vector3d u = ni.cross(up).normalized();
        Eigen::Vector3d v = ni.cross(u);

        // Collect valid coplanar neighbors within max_edge
        struct Neigh { uint32_t idx; double angle; };
        std::vector<Neigh> valid;
        for (size_t j = 0; j < found; j++) {
            if (indices[j] == (uint32_t)i) continue;
            if (dists[j] > max_edge_sq) continue;
            Eigen::Vector3d nj(normals[indices[j]].x, normals[indices[j]].y, normals[indices[j]].z);
            if (std::abs(ni.dot(nj)) < normal_thresh) continue;

            Eigen::Vector3d d = cloud.pts[indices[j]] - pi;
            double angle = std::atan2(d.dot(v), d.dot(u));
            valid.push_back({indices[j], angle});
        }
        if (valid.size() < 2) continue;

        // Sort by angle around center
        std::sort(valid.begin(), valid.end(),
                  [](const Neigh& a, const Neigh& b) { return a.angle < b.angle; });

        // Create triangle fan for consecutive neighbors
        for (size_t j = 0; j < valid.size(); j++) {
            size_t j_next = (j + 1) % valid.size();
            uint32_t bi = valid[j].idx, ci = valid[j_next].idx;

            // Skip large angular gaps (would span across empty space)
            double ang_diff = valid[j_next].angle - valid[j].angle;
            if (j_next == 0) ang_diff += 2.0 * M_PI;  // wrap-around
            if (ang_diff > max_ang_gap) continue;

            double edge_sq = (cloud.pts[bi] - cloud.pts[ci]).squaredNorm();
            if (edge_sq > max_edge_sq * 2.5) continue;

            tri_verts.push_back(pts[i]);
            tri_verts.push_back(pts[bi]);
            tri_verts.push_back(pts[ci]);
            tri_normals.push_back(normals[i]);
            tri_normals.push_back(normals[bi]);
            tri_normals.push_back(normals[ci]);
        }
    }
}

Eigen::Vector3d project_to_mls(const Eigen::Vector3d& query,
                                 const EigenPointCloud& cloud,
                                 const std::vector<Eigen::Vector3d>& normals,
                                 const KDTree3D& tree,
                                 double radius_sq,
                                 double h_sq) {
    std::vector<nanoflann::ResultItem<uint32_t, double>> matches;
    matches.reserve(64);
    nanoflann::SearchParameters params;
    params.sorted = false;
    tree.radiusSearch(query.data(), radius_sq, matches, params);

    if (matches.size() < 6) return query;

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    double wsum = 0;
    for (auto& m : matches) {
        double w = std::exp(-m.second / h_sq);
        centroid += w * cloud.pts[m.first];
        wsum += w;
    }
    centroid /= wsum;

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (auto& m : matches) {
        double w = std::exp(-m.second / h_sq);
        Eigen::Vector3d d = cloud.pts[m.first] - centroid;
        cov += w * d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    Eigen::Vector3d plane_normal = eig.eigenvectors().col(0);
    Eigen::Vector3d u_axis = eig.eigenvectors().col(2);
    Eigen::Vector3d v_axis = eig.eigenvectors().col(1);

    int nn = (int)matches.size();
    Eigen::MatrixXd A(nn, 6);
    Eigen::VectorXd b(nn);
    Eigen::VectorXd W(nn);

    for (int j = 0; j < nn; j++) {
        Eigen::Vector3d p = cloud.pts[matches[j].first] - centroid;
        double u = p.dot(u_axis);
        double v = p.dot(v_axis);
        double h = p.dot(plane_normal);
        double w = std::exp(-matches[j].second / h_sq);

        A(j, 0) = 1; A(j, 1) = u; A(j, 2) = v;
        A(j, 3) = u*u; A(j, 4) = u*v; A(j, 5) = v*v;
        b(j) = h;
        W(j) = w;
    }

    Eigen::MatrixXd WA = W.asDiagonal() * A;
    Eigen::VectorXd coeffs = (WA.transpose() * A).ldlt().solve(WA.transpose() * b);

    Eigen::Vector3d q_local = query - centroid;
    double u_q = q_local.dot(u_axis);
    double v_q = q_local.dot(v_axis);
    double h_q = coeffs(0) + coeffs(1)*u_q + coeffs(2)*v_q
               + coeffs(3)*u_q*u_q + coeffs(4)*u_q*v_q + coeffs(5)*v_q*v_q;

    return centroid + u_q * u_axis + v_q * v_axis + h_q * plane_normal;
}

std::vector<cv::Point3d> densify_surfaces(const std::vector<cv::Point3d>& pts,
                                            double search_radius = 0.12,
                                            double fill_step = 0.05,
                                            double normal_thresh = 0.95) {
    if (pts.size() < 20) return pts;


    EigenPointCloud cloud;
    cloud.pts.resize(pts.size());
    for (size_t i = 0; i < pts.size(); i++)
        cloud.pts[i] = Eigen::Vector3d(pts[i].x, pts[i].y, pts[i].z);

    KDTree3D tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    auto normals = estimate_normals(cloud, tree, 20);

    double radius_sq = search_radius * search_radius;

    std::vector<Eigen::Vector3d> new_pts;
    new_pts.reserve(cloud.pts.size() * 3);
    for (auto& p : cloud.pts) new_pts.push_back(p); // keep originals

    int filled = 0;
    for (size_t i = 0; i < cloud.pts.size(); i++) {
        const Eigen::Vector3d& pi = cloud.pts[i];
        const Eigen::Vector3d& ni = normals[i];

        std::vector<nanoflann::ResultItem<uint32_t, double>> matches;
        matches.reserve(64);
        tree.radiusSearch(pi.data(), radius_sq, matches);

        // Filter to coplanar neighbors (same surface: normals nearly parallel)
        std::vector<Eigen::Vector3d> coplanar;
        coplanar.push_back(pi);
        for (auto& m : matches) {
            if (m.first == (uint32_t)i) continue;
            double dot = std::abs(ni.dot(normals[m.first]));
            if (dot >= normal_thresh) {
                coplanar.push_back(cloud.pts[m.first]);
            }
        }

        if (coplanar.size() < 4) continue;

        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (auto& p : coplanar) centroid += p;
        centroid /= (double)coplanar.size();

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (auto& p : coplanar) {
            Eigen::Vector3d d = p - centroid;
            cov += d * d.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);

        double l1 = eig.eigenvalues()(2); // largest
        double l3 = eig.eigenvalues()(0); // smallest
        if (l1 < 1e-12) continue;
        if (l3 / l1 > 0.15) continue; // not flat enough, skip

        Eigen::Vector3d plane_n = eig.eigenvectors().col(0); // normal
        Eigen::Vector3d u_axis = eig.eigenvectors().col(2);  // primary spread
        Eigen::Vector3d v_axis = eig.eigenvectors().col(1);  // secondary spread

        double u_min = 1e9, u_max = -1e9, v_min = 1e9, v_max = -1e9;
        for (auto& p : coplanar) {
            Eigen::Vector3d d = p - centroid;
            double u = d.dot(u_axis);
            double v = d.dot(v_axis);
            u_min = std::min(u_min, u); u_max = std::max(u_max, u);
            v_min = std::min(v_min, v); v_max = std::max(v_max, v);
        }

        for (double u = u_min; u <= u_max; u += fill_step) {
            for (double v = v_min; v <= v_max; v += fill_step) {
                Eigen::Vector3d candidate = centroid + u * u_axis + v * v_axis;

                uint32_t idx; double dist_sq;
                tree.knnSearch(candidate.data(), 1, &idx, &dist_sq);
                if (dist_sq < fill_step * fill_step * 0.2) continue;

                bool near_existing = false;
                for (auto& p : coplanar) {
                    if ((candidate - p).squaredNorm() < radius_sq * 0.5) {
                        near_existing = true;
                        break;
                    }
                }
                if (!near_existing) continue;

                new_pts.push_back(candidate);
                filled++;
            }
        }
    }

    double voxel = fill_step * 0.7;
    struct VoxelHash {
        size_t operator()(const std::tuple<int,int,int>& k) const {
            return std::hash<int>{}(std::get<0>(k)) ^
                  (std::hash<int>{}(std::get<1>(k)) << 1) ^
                  (std::hash<int>{}(std::get<2>(k)) << 2);
        }
    };
    std::unordered_map<std::tuple<int,int,int>, Eigen::Vector3d, VoxelHash> grid;
    for (auto& p : new_pts) {
        auto key = std::make_tuple((int)std::floor(p.x() / voxel),
                                    (int)std::floor(p.y() / voxel),
                                    (int)std::floor(p.z() / voxel));
        grid.emplace(key, p);
    }

    std::vector<cv::Point3d> result;
    result.reserve(grid.size());
    for (auto& kv : grid)
        result.emplace_back(kv.second.x(), kv.second.y(), kv.second.z());

    return result;
}

struct SharedState {
    std::atomic<bool> processing_done{false};
    std::atomic<bool> should_stop{false};

    // Background point cloud filter thread state
    std::mutex filter_mutex;
    std::condition_variable filter_cv;
    std::atomic<bool> filter_request{false};
    std::atomic<bool> filter_running{false};
    std::atomic<bool> filter_shutdown{false};
    std::vector<cv::Point3d> filter_input_pts;
    std::vector<cv::Point3d> filter_input_traj;
    std::vector<cv::Point3d> filter_output_pts;
    std::atomic<bool> filter_result_ready{false};

    // Dense cloud SOR in background
    std::vector<cv::Point3d> dense_input;
    std::vector<cv::Point3d> dense_output;
    std::atomic<bool> dense_filter_ready{false};
};

Viewer* g_viewer = nullptr;

void point_cloud_filter_thread(SharedState& state) {
    while (!state.filter_shutdown) {
        std::unique_lock<std::mutex> lock(state.filter_mutex);
        state.filter_cv.wait(lock, [&] {
            return state.filter_request.load() || state.filter_shutdown.load();
        });
        if (state.filter_shutdown) break;

        state.filter_request = false;
        state.filter_running = true;

        // Copy input under lock
        auto pts = std::move(state.filter_input_pts);
        auto traj = std::move(state.filter_input_traj);
        lock.unlock();

        if (!traj.empty() && !pts.empty()) {
            double tx_min = 1e9, tx_max = -1e9, ty_min = 1e9, ty_max = -1e9, tz_min = 1e9, tz_max = -1e9;
            for (const auto& tp : traj) {
                tx_min = std::min(tx_min, tp.x); tx_max = std::max(tx_max, tp.x);
                ty_min = std::min(ty_min, tp.y); ty_max = std::max(ty_max, tp.y);
                tz_min = std::min(tz_min, tp.z); tz_max = std::max(tz_max, tp.z);
            }
            double margin = 3.0;
            std::vector<cv::Point3d> bbox_filtered;
            bbox_filtered.reserve(pts.size());
            for (const auto& p : pts) {
                if (p.x >= tx_min - margin && p.x <= tx_max + margin &&
                    p.y >= ty_min - margin && p.y <= ty_max + margin &&
                    p.z >= tz_min - margin && p.z <= tz_max + margin) {
                    bbox_filtered.push_back(p);
                }
            }
            pts = std::move(bbox_filtered);
        }

        state.filter_result_ready = false;
        state.filter_running = false;
    }
}

void slam_processing_thread(const std::vector<ImageInfo>& images,
                             Slam& slam,
                             Viewer& viewer,
                             SharedState& state,
                             const std::vector<GTPose>& gt_all,
                             const std::string& dataset_path,
                             const std::string& model_dir,
                             bool run_poisson) {

    // Load models in background thread so viewer renders immediately
    slam.init(model_dir);

    std::string cache_path = model_dir + "/sp_cache.bin";
    slam.feature_extractor().set_cache_path(cache_path);
    bool cache_was_loaded = slam.feature_extractor().load_cache();
    (void)cache_was_loaded;

    auto accel_data = load_accelerometer(dataset_path);
    if (!accel_data.empty()) {
        std::vector<Slam::AccelSample> slam_accel(accel_data.size());
        for (size_t i = 0; i < accel_data.size(); i++) {
            slam_accel[i].timestamp = accel_data[i].timestamp;
            slam_accel[i].ax = accel_data[i].ax;
            slam_accel[i].ay = accel_data[i].ay;
            slam_accel[i].az = accel_data[i].az;
        }
        slam.set_accelerometer_data(slam_accel);
    }

    slam.set_initial_pose(cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F));
    slam.compute_gravity_direction();

    cv::Point3d origin_offset(0, 0, 0);

    if (!gt_all.empty()) {
        std::vector<cv::Point3d> gt_traj;
        gt_traj.reserve(gt_all.size());
        for (const auto& g : gt_all) {
            gt_traj.emplace_back(g.tx - origin_offset.x, g.ty - origin_offset.y, g.tz - origin_offset.z);
        }
        viewer.update_ground_truth(gt_traj);
    }

    int last_filter_kf_count = 0;

    std::vector<cv::Point3d> dense_cloud;
    dense_cloud.reserve(500000);
    constexpr int DENSE_PIXEL_STEP = 8;     // subsample pixels
    constexpr double DENSE_MAX_DEPTH = 5.0; // meters
    constexpr double DENSE_VOXEL = 0.02;    // 2cm voxel grid for dedup
    constexpr double DENSE_VOXEL_INV = 1.0 / DENSE_VOXEL;
    auto voxel_hash = [](const std::tuple<int,int,int>& v) {
        size_t h = 0;
        h ^= std::hash<int>()(std::get<0>(v)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(std::get<1>(v)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(std::get<2>(v)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    };
    std::unordered_set<std::tuple<int,int,int>, decltype(voxel_hash)> dense_voxels(0, voxel_hash);

    for (size_t i = 0; i < images.size() && !state.should_stop && !viewer.should_restart(); ++i) {
        auto frame = std::make_shared<Frame>(i, images[i].rgb_path, images[i].timestamp);

        if (frame->image().empty()) continue;

        if (i % Config::FRAME_STEP != 0) continue;

        if (!images[i].depth_path.empty()) {
            frame->load_depth_image(images[i].depth_path);
        }

        bool processed = slam.process_frame(frame);

        if (processed) {
            auto raw_traj = slam.map().get_trajectory();
            for (auto& p : raw_traj) {
                p.x -= origin_offset.x; p.y -= origin_offset.y; p.z -= origin_offset.z;
            }
            viewer.update_trajectory(raw_traj);

            if (frame->has_real_depth()) {
                cv::Mat R = frame->get_rotation();
                cv::Mat t = frame->get_translation();
                const cv::Mat& depth = frame->depth_map();

                for (int v = 0; v < depth.rows; v += DENSE_PIXEL_STEP) {
                    for (int u = 0; u < depth.cols; u += DENSE_PIXEL_STEP) {
                        float z = depth.at<float>(v, u);
                        if (z <= 0 || z >= DENSE_MAX_DEPTH) continue;

                        double x_cam = (u - Config::CX) * z / Config::FX;
                        double y_cam = (v - Config::CY) * z / Config::FY;

                        double px = R.at<double>(0,0)*x_cam + R.at<double>(0,1)*y_cam + R.at<double>(0,2)*z + t.at<double>(0) - origin_offset.x;
                        double py = R.at<double>(1,0)*x_cam + R.at<double>(1,1)*y_cam + R.at<double>(1,2)*z + t.at<double>(1) - origin_offset.y;
                        double pz = R.at<double>(2,0)*x_cam + R.at<double>(2,1)*y_cam + R.at<double>(2,2)*z + t.at<double>(2) - origin_offset.z;

                        auto vk = std::make_tuple(
                            (int)std::floor(px * DENSE_VOXEL_INV),
                            (int)std::floor(py * DENSE_VOXEL_INV),
                            (int)std::floor(pz * DENSE_VOXEL_INV));
                        if (dense_voxels.insert(vk).second) {
                            dense_cloud.emplace_back(px, py, pz);
                        }
                    }
                }

                if (slam.frame_count() % 5 == 0) {
                    viewer.update_map_points(dense_cloud);
                }
            }

            if (state.dense_filter_ready.load()) {
                std::lock_guard<std::mutex> lk(state.filter_mutex);
                if (!state.dense_output.empty()) {
                    dense_cloud = state.dense_output;
                    // Rebuild voxel set from cleaned cloud
                    dense_voxels.clear();
                    for (const auto& p : dense_cloud) {
                        dense_voxels.insert(std::make_tuple(
                            (int)std::floor(p.x * DENSE_VOXEL_INV),
                            (int)std::floor(p.y * DENSE_VOXEL_INV),
                            (int)std::floor(p.z * DENSE_VOXEL_INV)));
                    }
                    viewer.update_map_points(dense_cloud);
                }
                state.dense_filter_ready = false;
            }

            int cur_kf = slam.keyframe_count();
            if (cur_kf > last_filter_kf_count + 9 && !state.filter_running.load()) {
                last_filter_kf_count = cur_kf;
                {
                    std::lock_guard<std::mutex> lk(state.filter_mutex);
                    state.dense_input = dense_cloud;
                }
                state.filter_request = true;
                state.filter_cv.notify_one();
            }

            viewer.update_loop_edges(slam.get_loop_edges());
            auto poses = slam.map().get_all_poses();
            for (auto& pose : poses) {
                pose.at<double>(0,3) -= origin_offset.x;
                pose.at<double>(1,3) -= origin_offset.y;
                pose.at<double>(2,3) -= origin_offset.z;
            }
            viewer.update_poses(poses);

            {
                auto sparse_pts = slam.map().get_all_display_points();
                for (auto& p : sparse_pts) {
                    p.x -= origin_offset.x; p.y -= origin_offset.y; p.z -= origin_offset.z;
                }
                if (sparse_pts.size() > 50) {
                    auto traj = slam.map().get_trajectory();
                    if (!traj.empty()) {
                        double tx_min = 1e9, tx_max = -1e9, ty_min = 1e9, ty_max = -1e9, tz_min = 1e9, tz_max = -1e9;
                        for (const auto& tp : traj) {
                            tx_min = std::min(tx_min, tp.x); tx_max = std::max(tx_max, tp.x);
                            ty_min = std::min(ty_min, tp.y); ty_max = std::max(ty_max, tp.y);
                            tz_min = std::min(tz_min, tp.z); tz_max = std::max(tz_max, tp.z);
                        }
                        double margin = 3.0;
                        std::vector<cv::Point3d> bbox_pts;
                        bbox_pts.reserve(sparse_pts.size());
                        for (const auto& p : sparse_pts) {
                            if (p.x >= tx_min - margin && p.x <= tx_max + margin &&
                                p.y >= ty_min - margin && p.y <= ty_max + margin &&
                                p.z >= tz_min - margin && p.z <= tz_max + margin) {
                                bbox_pts.push_back(p);
                            }
                        }
                        sparse_pts = bbox_pts;
                    }
                    sparse_pts = surface_aware_filter(statistical_outlier_removal(sparse_pts, 25, 0.6), 25, 0.4);
                    sparse_pts = densify_surfaces(sparse_pts);
                }
                viewer.update_sparse_points(sparse_pts);
            }
        }

        cv::Mat display = frame->draw_keypoints();

        std::string info1 = "Frame: " + std::to_string(i + 1) + "/" +
                            std::to_string(images.size()) +
                            " | Matches: " + std::to_string(slam.last_match_count()) +
                            " | Inliers: " + std::to_string(slam.last_inlier_count());
        cv::putText(display, info1, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        std::string info2 = "MapPts: " + std::to_string(slam.map_point_count()) +
                            " | KFs: " + std::to_string(slam.keyframe_count()) +
                            " | Loops: " + std::to_string(slam.loop_count());
        cv::putText(display, info2, cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        char err_buf[256];
        snprintf(err_buf, sizeof(err_buf),
                 "EpiErr: %.2f->%.2f | ReprojErr: %.2f->%.2f",
                 slam.last_epipolar_error_before(), slam.last_epipolar_error_after(),
                 slam.last_reproj_error_before(), slam.last_reproj_error_after());
        cv::putText(display, std::string(err_buf), cv::Point(10, 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        std::string info4;
        if (slam.last_was_pnp()) info4 += "[PnP] ";
        if (slam.last_was_loop()) info4 += "[LOOP CLOSED] ";
        if (frame->is_keyframe()) info4 += "[KF] ";
        if (!info4.empty()) {
            cv::putText(display, info4, cv::Point(10, 100),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }

        std::string feat_type = slam.feature_extractor().using_superpoint() ?
                                "[SuperPoint]" : "[ORB]";
        std::string depth_type;
        if (frame->has_real_depth())
            depth_type = "[TUM Depth]";
        else if (slam.depth_estimator().is_available())
            depth_type = "[MiDaS]";
        else
            depth_type = "[No Depth]";
        cv::putText(display, feat_type + " " + depth_type, cv::Point(10, display.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 128, 0), 1);

        auto ref = slam.ref_frame();
        const auto& matches_before = slam.last_matches_before();
        const auto& matches_after = slam.last_matches_after();
        if (ref && !ref->image().empty() && !frame->image().empty()
            && !matches_before.empty()) {
            int nkp1 = (int)ref->keypoints().size();
            int nkp2 = (int)frame->keypoints().size();
            auto valid_matches = [&](const std::vector<cv::DMatch>& m) {
                for (const auto& dm : m)
                    if (dm.queryIdx < 0 || dm.queryIdx >= nkp1 ||
                        dm.trainIdx < 0 || dm.trainIdx >= nkp2) return false;
                return true;
            };

            if (valid_matches(matches_before) && valid_matches(matches_after)) {
                cv::Mat match_before_img, match_after_img;
                cv::drawMatches(ref->image(), ref->keypoints(),
                                frame->image(), frame->keypoints(),
                                matches_before, match_before_img,
                                cv::Scalar(0, 255, 0), cv::Scalar::all(-1),
                                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::drawMatches(ref->image(), ref->keypoints(),
                                frame->image(), frame->keypoints(),
                                matches_after, match_after_img,
                                cv::Scalar(0, 255, 0), cv::Scalar::all(-1),
                                std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                cv::putText(match_before_img,
                            "All matches (raw): " + std::to_string(matches_before.size()),
                            cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
                cv::putText(match_after_img,
                            "After filtering (ratio test + RANSAC): " + std::to_string(matches_after.size()),
                            cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);

                cv::Mat match_display;
                cv::vconcat(match_before_img, match_after_img, match_display);
                double scale = 640.0 / match_display.cols;
                cv::resize(match_display, match_display, cv::Size(), scale, scale);
                cv::imshow("Matches: Before vs After Filtering", match_display);
            }
        }

        if (g_viewer) {
            g_viewer->update_image(display);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    slam.run_rts_smoother();

    std::vector<std::pair<double, cv::Point3d>> est_poses;
    {
        auto all_frames = slam.map().get_all_frames();
        for (const auto& f : all_frames) {
            cv::Mat t = f->get_translation();
            est_poses.emplace_back(f->timestamp(),
                cv::Point3d(t.at<double>(0), t.at<double>(1), t.at<double>(2)));
        }
    }

    if (!cache_was_loaded && slam.feature_extractor().cache_size() > 0) {
        slam.feature_extractor().save_cache();
    }

    if (!dense_cloud.empty()) {
        viewer.update_map_points(dense_cloud);
    }

    state.processing_done = true;

    if (!gt_all.empty() && !est_poses.empty()) {
        auto align = compute_ate(est_poses, gt_all);
        if (align.ate_rmse >= 0) {
            std::cout << "ATE RMSE: " << align.ate_rmse << " m" << std::endl;
        }
    }

    // Final PLY save using raw (unaligned) points
    {
        auto raw_traj = slam.map().get_trajectory();
        auto raw_pts = slam.map().get_all_display_points();

        {
            std::vector<cv::Point3d> filtered_pts;
            filtered_pts.reserve(raw_pts.size());
            if (!raw_traj.empty()) {
                double tx_min = 1e9, tx_max = -1e9;
                double ty_min = 1e9, ty_max = -1e9;
                double tz_min = 1e9, tz_max = -1e9;
                for (const auto& tp : raw_traj) {
                    tx_min = std::min(tx_min, tp.x); tx_max = std::max(tx_max, tp.x);
                    ty_min = std::min(ty_min, tp.y); ty_max = std::max(ty_max, tp.y);
                    tz_min = std::min(tz_min, tp.z); tz_max = std::max(tz_max, tp.z);
                }
                double margin = 3.0;
                for (const auto& p : raw_pts) {
                    if (p.x >= tx_min - margin && p.x <= tx_max + margin &&
                        p.y >= ty_min - margin && p.y <= ty_max + margin &&
                        p.z >= tz_min - margin && p.z <= tz_max + margin) {
                        filtered_pts.push_back(p);
                    }
                }
            } else {
                filtered_pts = raw_pts;
            }
            filtered_pts = surface_aware_filter(statistical_outlier_removal(filtered_pts, 25, 0.6), 25, 0.4);
            filtered_pts = densify_surfaces(filtered_pts);

            viewer.update_sparse_points(filtered_pts);

            if (!dense_cloud.empty()) {
                viewer.update_map_points(dense_cloud);
            } else {
                viewer.update_map_points(filtered_pts);
            }

                std::string ply_path = "estimated_map.ply";
                {
                    EigenPointCloud ecloud;
                    ecloud.pts.resize(filtered_pts.size());
                    for (size_t pi = 0; pi < filtered_pts.size(); pi++)
                        ecloud.pts[pi] = Eigen::Vector3d(filtered_pts[pi].x, filtered_pts[pi].y, filtered_pts[pi].z);
                    KDTree3D etree(3, ecloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                    etree.buildIndex();
                    auto enormals = estimate_normals(ecloud, etree, 20);

                    std::ofstream ply_ofs(ply_path);
                    if (ply_ofs.is_open()) {
                        ply_ofs << "ply\n";
                        ply_ofs << "format ascii 1.0\n";
                        ply_ofs << "element vertex " << filtered_pts.size() << "\n";
                        ply_ofs << "property float x\nproperty float y\nproperty float z\n";
                        ply_ofs << "property float nx\nproperty float ny\nproperty float nz\n";
                        ply_ofs << "end_header\n";
                        ply_ofs << std::fixed << std::setprecision(6);
                        for (size_t pi = 0; pi < filtered_pts.size(); pi++) {
                            ply_ofs << filtered_pts[pi].x << " " << filtered_pts[pi].y << " " << filtered_pts[pi].z
                                    << " " << enormals[pi].x() << " " << enormals[pi].y() << " " << enormals[pi].z() << "\n";
                        }
                        ply_ofs.close();
                    }
                }

                if (run_poisson) {
                    std::string poisson_bin = "../external/PoissonRecon/Bin/Linux/PoissonRecon";
                    std::string trimmer_bin = "../external/PoissonRecon/Bin/Linux/SurfaceTrimmer";
                    std::string mesh_path = "poisson_mesh.ply";
                    std::string trimmed_path = "poisson_trimmed.ply";

                    // depth 8 = fine detail, pointWeight keeps mesh tight to data
                    std::string cmd = poisson_bin + " --in " + ply_path +
                                      " --out " + mesh_path + " --depth 8 --pointWeight 6 --density --ascii 2>&1";
                    int ret = system(cmd.c_str());

                    if (ret == 0) {
                        // Try trimming (needs density values from --ascii mode)
                        // trim 8 = aggressive, removes all loosely-supported blobs
                        std::string trim_cmd = trimmer_bin + " --in " + mesh_path +
                                               " --out " + trimmed_path + " --trim 8 --ascii 2>&1";
                        int ret2 = system(trim_cmd.c_str());
                        std::string final_mesh = (ret2 == 0) ? trimmed_path : mesh_path;


                        std::ifstream mesh_ifs(final_mesh);
                        if (mesh_ifs.is_open()) {
                            std::string line;
                            int num_verts = 0;
                            int num_props = 0;
                            bool in_header = true;
                            bool past_vertex = false;
                            while (std::getline(mesh_ifs, line) && in_header) {
                                if (line.find("element vertex") != std::string::npos)
                                    sscanf(line.c_str(), "element vertex %d", &num_verts);
                                if (line.find("element face") != std::string::npos)
                                    past_vertex = true;
                                if (!past_vertex && line.find("property") != std::string::npos)
                                    num_props++;
                                if (line == "end_header") in_header = false;
                            }
                            std::vector<cv::Point3d> mesh_pts;
                            mesh_pts.reserve(num_verts);
                            for (int vi = 0; vi < num_verts; vi++) {
                                if (!std::getline(mesh_ifs, line)) break;
                                double x, y, z;
                                if (sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z) >= 3) {
                                    mesh_pts.emplace_back(x, y, z);
                                }
                            }
                            mesh_ifs.close();
                            if (!mesh_pts.empty()) {
                                viewer.update_map_points(mesh_pts);
                            }
                        }
                    } else {
                    }
                }
            }
        }

    if (!dense_cloud.empty()) {
        std::string dense_ply_path = "dense_map.ply";
        std::ofstream dply(dense_ply_path);
        if (dply.is_open()) {
            dply << "ply\nformat ascii 1.0\nelement vertex " << dense_cloud.size()
                 << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
            dply << std::fixed << std::setprecision(6);
            for (const auto& p : dense_cloud) {
                dply << p.x << " " << p.y << " " << p.z << "\n";
            }
            dply.close();
        }
    }

    save_trajectory("estimated_trajectory.txt", est_poses);
    save_trajectory_full("estimated_trajectory_full.txt", slam.map().get_all_frames());
}


int main(int argc, char** argv) {
    std::string dataset_path = Config::DATASET_PATH;
    std::string model_dir = "models";
    bool run_poisson = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--poisson") {
            run_poisson = true;
        } else if (dataset_path == Config::DATASET_PATH) {
            dataset_path = arg;
            if (dataset_path.back() != '/') dataset_path += '/';
        } else {
            model_dir = arg;
        }
    }


    std::vector<ImageInfo> images = load_image_list(dataset_path);
    if (images.empty()) {
        return -1;
    }
    auto gt_all = load_ground_truth(dataset_path);

    Viewer viewer;
    viewer.init();
    g_viewer = &viewer;

    bool running = true;
    while (running) {
        Slam slam;

        SharedState state;
        viewer.clear_restart();

        std::thread filter_thread(point_cloud_filter_thread, std::ref(state));

        std::thread slam_thread(slam_processing_thread, std::cref(images),
                                 std::ref(slam), std::ref(viewer), std::ref(state),
                                 std::cref(gt_all), std::cref(dataset_path),
                                 std::cref(model_dir), run_poisson);

        while (!viewer.should_quit() && !state.processing_done && !viewer.should_restart()) {
            viewer.show_image();
            if (!viewer.render_frame()) break;
        }

        while (!viewer.should_quit() && !viewer.should_restart()) {
            viewer.show_image();
            if (!viewer.render_frame()) break;
        }

        state.should_stop = true;
        if (slam_thread.joinable()) {
            slam_thread.join();
        }

        state.filter_shutdown = true;
        state.filter_cv.notify_one();
        if (filter_thread.joinable()) {
            filter_thread.join();
        }

        if (viewer.should_quit()) {
            running = false;
        } else if (viewer.should_restart()) {
            viewer.update_trajectory({});
            viewer.update_poses({});
            viewer.update_map_points({});
            viewer.update_sparse_points({});
            viewer.update_loop_edges({});
            viewer.reset_interpolation();
        }
    }

    g_viewer = nullptr;
    viewer.shutdown();
    return 0;
}
