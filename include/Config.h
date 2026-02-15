#ifndef CONFIG_H
#define CONFIG_H

#include <opencv2/core.hpp>
#include <string>

namespace Config {

// Image dimensions
constexpr int IMAGE_WIDTH = 640;
constexpr int IMAGE_HEIGHT = 480;

// Camera intrinsics (TUM freiburg2)
constexpr double FX = 525.0;
constexpr double FY = 525.0;
constexpr double CX = 319.5;
constexpr double CY = 239.5;

inline cv::Mat getCameraMatrix() {
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        FX,  0.0, CX,
        0.0, FY,  CY,
        0.0, 0.0, 1.0);
    return K;
}

// Dataset
const std::string DATASET_PATH = "../rgbd_dataset_freiburg2_pioneer_slam3/";

// DL Model paths
const std::string SUPERPOINT_MODEL = "models/superpoint_v1.onnx";
const std::string MIDAS_MODEL = "models/midas_v21_small_256.onnx";

// SuperPoint parameters
constexpr float SP_CONFIDENCE_THRESHOLD = 0.005f;
constexpr int SP_NMS_RADIUS = 4;
constexpr int SP_MAX_KEYPOINTS = 400;

// MiDaS parameters
constexpr int MIDAS_INPUT_SIZE = 256;

// ORB fallback parameters
constexpr int NUM_FEATURES = 3000;
constexpr float SCALE_FACTOR = 1.2f;
constexpr int NUM_LEVELS = 8;

// Matching parameters
constexpr float L2_RATIO_THRESHOLD = 0.75f;
constexpr float DISTANCE_THRESHOLD = 30.0f;
constexpr int MIN_MATCHES = 30;
constexpr int MIN_INLIERS = 15;

// RANSAC
constexpr double RANSAC_PROB = 0.999;
constexpr double RANSAC_THRESHOLD = 1.0;

// Triangulation
constexpr double TRIANG_MAX_REPROJ_ERROR = 3.0;
constexpr double TRIANG_MIN_DEPTH = 0.05;
constexpr double TRIANG_MAX_DEPTH = 50.0;
constexpr double TRIANG_MAX_CAM_DIST = 5.0;

// PnP
constexpr int PNP_INTERVAL = 5;
constexpr int PNP_MIN_POINTS = 10;
constexpr double PNP_RANSAC_THRESHOLD = 8.0;

// Keyframe
constexpr int KF_MIN_FRAME_GAP = 10;
constexpr int KF_MIN_MATCHES = 50;

// Loop closure (feature-matching based, enabled)
constexpr int LC_MIN_FRAME_GAP = 200;
constexpr float LC_SIMILARITY_THRESHOLD = 0.70f;
constexpr int LC_MIN_INLIERS = 30;
constexpr int LC_CHECK_INTERVAL = 200;

// Local BA (disabled: hurts ATE on this sequence due to sparse observations)
constexpr bool ENABLE_LOCAL_BA = false;

// Optimizer (Gauss-Newton with LM damping)
constexpr int OPT_MAX_ITERATIONS = 10;
constexpr double OPT_CONVERGENCE = 1e-6;
constexpr double OPT_LM_LAMBDA = 1e-3;

// Viewer / visualization
constexpr int FRAME_STEP = 3;
constexpr int VIEWER_WIDTH = 1024;
constexpr int VIEWER_HEIGHT = 768;
constexpr double TRAJECTORY_SCALE = 2.0;

// Motion scale
constexpr double MOTION_SCALE = 0.05;

// Pose Graph Optimization
constexpr int PGO_TRIGGER_INTERVAL = 20;
constexpr double PGO_ODOM_TRANS_SIGMA = 0.05;
constexpr double PGO_ODOM_ROT_SIGMA = 0.02;
constexpr double PGO_LC_TRANS_SIGMA = 0.03;
constexpr double PGO_LC_ROT_SIGMA = 0.01;
constexpr double PGO_HEIGHT_SIGMA = 0.005;

// EKF parameters
constexpr double EKF_SIGMA_VIS_3D3D = 0.04;
constexpr double EKF_SIGMA_VIS_EMAT = 0.10;
constexpr double EKF_SIGMA_HEIGHT = 0.01;
constexpr double EKF_PROCESS_ACCEL = 1.0;
constexpr double EKF_VEL_DECAY = 0.95;
constexpr double EKF_INNOV_GATE = 0.3;
constexpr double EKF_MAX_STEP = 0.10;

}

#endif
