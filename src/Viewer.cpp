#include "Viewer.h"
#include "Config.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>

static const float kViewpointX = 0.0f;
static const float kViewpointY = -0.7f;
static const float kViewpointZ = -1.8f;
static const float kViewpointF = 500.0f;

Viewer::Viewer() : initialized_(false), should_quit_(false), should_restart_(false),
                   s_cam_(nullptr), d_cam_(nullptr),
                   has_new_image_(false),
                   key_frame_size_(0.1f), key_frame_line_width_(1.0f),
                   camera_size_(0.15f), camera_line_width_(3.0f),
                   has_interpolated_pose_(false),
                   view_mode_(VIEW_CAMERA),
                   has_initial_focus_(false),
                   focus_x_(0), focus_y_(0), focus_z_(0) {
}

Viewer::~Viewer() {
    shutdown();
}

void Viewer::set_initial_viewpoint(double x, double y, double z) {
    has_initial_focus_ = true;
    focus_x_ = x * Config::TRAJECTORY_SCALE;
    focus_y_ = y * Config::TRAJECTORY_SCALE;
    focus_z_ = z * Config::TRAJECTORY_SCALE;
}

void Viewer::init() {
    if (initialized_) return;

    int width = 1024;
    int height = 768;

    pangolin::CreateWindowAndBind("SLAM System", width, height);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    s_cam_ = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(width, height, kViewpointF, kViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(kViewpointX, kViewpointY, kViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    float aspect = (float)width / height;

    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));

    follow_button_ = new pangolin::Var<bool>("ui.Follow Camera", true, true);  // ON by default (same as Reset View)
    restart_button_ = new pangolin::Var<bool>("ui.Restart", false, false);
    top_view_button_ = new pangolin::Var<bool>("ui.Top View", false, false);
    side_view_button_ = new pangolin::Var<bool>("ui.Side View", false, false);
    reset_view_button_ = new pangolin::Var<bool>("ui.Reset View", false, false);
    show_points_button_ = new pangolin::Var<bool>("ui.Show Dense", false, true);
    show_sparse_button_ = new pangolin::Var<bool>("ui.Show Sparse", true, true);

    d_cam_ = &pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -aspect)
        .SetHandler(new pangolin::Handler3D(*s_cam_));

    initialized_ = true;
}

bool Viewer::render_frame() {
    if (!initialized_ || should_quit_) return false;

    if (pangolin::ShouldQuit()) {
        should_quit_ = true;
        return false;
    }

    if (pangolin::Pushed(*restart_button_)) {
        should_restart_ = true;
    }

    if (pangolin::Pushed(*top_view_button_)) {
        view_mode_ = VIEW_TOP;
        *follow_button_ = true;
        s_cam_->SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000));
        s_cam_->SetModelViewMatrix(pangolin::ModelViewLookAt(0, -40, 0.01, 0, 0, 0, 0.0, 0.0, 1.0));
    }

    if (pangolin::Pushed(*side_view_button_)) {
        view_mode_ = VIEW_SIDE;
        *follow_button_ = true;
        s_cam_->SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, kViewpointF, kViewpointF, 512, 389, 0.1, 1000));
        s_cam_->SetModelViewMatrix(pangolin::ModelViewLookAt(15, 0, 0, 0, 0, 0, 0.0, -1.0, 0.0));
    }

    if (pangolin::Pushed(*reset_view_button_)) {
        view_mode_ = VIEW_CAMERA;
        *follow_button_ = true;
        s_cam_->SetProjectionMatrix(pangolin::ProjectionMatrix(1024, 768, kViewpointF, kViewpointF, 512, 389, 0.1, 1000));
        s_cam_->SetModelViewMatrix(pangolin::ModelViewLookAt(kViewpointX, kViewpointY, kViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.12f, 0.12f, 0.14f, 1.0f);

    d_cam_->Activate(*s_cam_);
    if (*follow_button_) {
        follow_current_pose();
    }

    draw_trajectory();
    draw_camera_poses();

    if (*show_points_button_) {
        draw_map_points();
    }
    if (*show_sparse_button_) {
        draw_sparse_points();
    }


    pangolin::FinishFrame();
    return true;
}

void Viewer::shutdown() {
    if (initialized_) {
        pangolin::DestroyWindow("SLAM System");
        delete s_cam_;
        s_cam_ = nullptr;
        d_cam_ = nullptr;
        initialized_ = false;
    }
}

void Viewer::update_trajectory(const std::vector<cv::Point3d>& trajectory) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    trajectory_ = trajectory;
}

void Viewer::update_poses(const std::vector<cv::Mat>& poses) {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    poses_.clear();
    poses_.reserve(poses.size());
    for (const auto& p : poses) {
        poses_.push_back(p.clone());
    }
}

void Viewer::update_image(const cv::Mat& image) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (!image.empty()) {
        current_image_ = image.clone();
        has_new_image_ = true;
    }
}

void Viewer::update_map_points(const std::vector<cv::Point3d>& points) {
    std::lock_guard<std::mutex> lock(points_mutex_);
    map_points_ = points;
}

void Viewer::update_sparse_points(const std::vector<cv::Point3d>& points) {
    std::lock_guard<std::mutex> lock(points_mutex_);
    sparse_points_ = points;
}

void Viewer::update_loop_edges(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& edges) {
    std::lock_guard<std::mutex> lock(loop_mutex_);
    loop_edges_ = edges;
}

void Viewer::update_ground_truth(const std::vector<cv::Point3d>& gt) {
    std::lock_guard<std::mutex> lock(gt_mutex_);
    ground_truth_ = gt;
}

void Viewer::show_image() {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (has_new_image_ && !current_image_.empty()) {
        cv::imshow("SLAM Video", current_image_);
        cv::waitKey(1);
        has_new_image_ = false;
    }
}

void Viewer::draw_trajectory() {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);

    if (trajectory_.empty()) return;

    glColor3f(1.0f, 1.0f, 1.0f);
    glLineWidth(3.0f);
    glBegin(GL_LINE_STRIP);
    for (const auto& pt : trajectory_) {
        glVertex3d(pt.x * Config::TRAJECTORY_SCALE,
                   pt.y * Config::TRAJECTORY_SCALE,
                   pt.z * Config::TRAJECTORY_SCALE);
    }
    glEnd();

    glColor3f(0.2f, 1.0f, 0.4f);
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    glVertex3d(trajectory_[0].x * Config::TRAJECTORY_SCALE,
               trajectory_[0].y * Config::TRAJECTORY_SCALE,
               trajectory_[0].z * Config::TRAJECTORY_SCALE);
    glEnd();

    glColor3f(1.0f, 0.25f, 0.25f);
    glBegin(GL_POINTS);
    glVertex3d(trajectory_.back().x * Config::TRAJECTORY_SCALE,
               trajectory_.back().y * Config::TRAJECTORY_SCALE,
               trajectory_.back().z * Config::TRAJECTORY_SCALE);
    glEnd();
}

void Viewer::draw_camera_poses() {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);

    if (poses_.empty()) return;

    for (size_t i = 0; i < poses_.size(); ++i) {
        draw_camera_frustum(poses_[i], key_frame_size_, false);
    }

    const cv::Mat& target_pose = poses_.back();
    if (!has_interpolated_pose_) {
        interpolated_pose_ = target_pose.clone();
        has_interpolated_pose_ = true;
    } else {
        const double alpha = 0.15;
        interpolated_pose_ = (1.0 - alpha) * interpolated_pose_ + alpha * target_pose;
    }

    draw_camera_frustum(interpolated_pose_, camera_size_, true);
}

void Viewer::draw_camera_frustum(const cv::Mat& pose, float size, bool current) {
    if (pose.empty() || pose.rows != 4 || pose.cols != 4) return;

    const float w = size;
    const float h = size * 0.75f;
    const float z = size * 0.6f;

    float scale = Config::TRAJECTORY_SCALE;

    GLdouble m[16];
    m[0]  = pose.at<double>(0, 0);
    m[1]  = pose.at<double>(1, 0);
    m[2]  = pose.at<double>(2, 0);
    m[3]  = 0.0;
    m[4]  = pose.at<double>(0, 1);
    m[5]  = pose.at<double>(1, 1);
    m[6]  = pose.at<double>(2, 1);
    m[7]  = 0.0;
    m[8]  = pose.at<double>(0, 2);
    m[9]  = pose.at<double>(1, 2);
    m[10] = pose.at<double>(2, 2);
    m[11] = 0.0;
    m[12] = pose.at<double>(0, 3) * scale;
    m[13] = pose.at<double>(1, 3) * scale;
    m[14] = pose.at<double>(2, 3) * scale;
    m[15] = 1.0;

    glPushMatrix();
    glMultMatrixd(m);

    if (current) {
        glColor3f(1.0f, 0.3f, 0.3f);
        glLineWidth(4.0f);
    } else {
        glColor3f(0.6f, 0.6f, 0.65f);
        glLineWidth(1.0f);
    }

    glBegin(GL_LINES);
    glVertex3f(0, 0, 0); glVertex3f(w, h, z);
    glVertex3f(0, 0, 0); glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0); glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0); glVertex3f(-w, h, z);

    glVertex3f(w, h, z);   glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);  glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);  glVertex3f(w, h, z);
    glVertex3f(-w, -h, z); glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void Viewer::draw_map_points() {
    std::lock_guard<std::mutex> lock(points_mutex_);

    if (map_points_.empty()) return;

    float scale = Config::TRAJECTORY_SCALE;

    double y_min = 1e9, y_max = -1e9;
    for (const auto& pt : map_points_) {
        if (pt.y < y_min) y_min = pt.y;
        if (pt.y > y_max) y_max = pt.y;
    }
    double y_range = y_max - y_min;
    if (y_range < 0.01) y_range = 1.0;

    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (const auto& pt : map_points_) {
        float t = (float)((pt.y - y_min) / y_range);
        float r, g, b;
        if (t < 0.25f) {
            float s = t / 0.25f;
            r = 0.28f - s * 0.10f; g = 0.05f + s * 0.20f; b = 0.48f + s * 0.05f;
        } else if (t < 0.50f) {
            float s = (t - 0.25f) / 0.25f;
            r = 0.18f - s * 0.05f; g = 0.25f + s * 0.23f; b = 0.53f - s * 0.08f;
        } else if (t < 0.75f) {
            float s = (t - 0.50f) / 0.25f;
            r = 0.13f + s * 0.22f; g = 0.48f + s * 0.20f; b = 0.45f - s * 0.22f;
        } else {
            float s = (t - 0.75f) / 0.25f;
            r = 0.35f + s * 0.58f; g = 0.68f + s * 0.20f; b = 0.23f - s * 0.08f;
        }
        glColor3f(r, g, b);
        glVertex3d(pt.x * scale, pt.y * scale, pt.z * scale);
    }
    glEnd();
}

void Viewer::draw_sparse_points() {
    std::lock_guard<std::mutex> lock(points_mutex_);
    if (sparse_points_.empty()) return;

    float scale = Config::TRAJECTORY_SCALE;

    glPointSize(3.0f);
    glBegin(GL_POINTS);
    for (const auto& pt : sparse_points_) {
        glColor3f(0.2f, 0.8f, 1.0f);  // cyan
        glVertex3d(pt.x * scale, pt.y * scale, pt.z * scale);
    }
    glEnd();
}

void Viewer::draw_loop_edges() {
    std::lock_guard<std::mutex> lock(loop_mutex_);

    if (loop_edges_.empty()) return;

    float scale = Config::TRAJECTORY_SCALE;

    glColor3f(1.0f, 0.0f, 0.0f);  // Red
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    for (const auto& edge : loop_edges_) {
        glVertex3d(edge.first.x * scale, edge.first.y * scale, edge.first.z * scale);
        glVertex3d(edge.second.x * scale, edge.second.y * scale, edge.second.z * scale);
    }
    glEnd();
}

void Viewer::draw_ground_truth() {
    std::lock_guard<std::mutex> lock(gt_mutex_);

    if (ground_truth_.empty()) return;

    float scale = Config::TRAJECTORY_SCALE;

    glColor3f(0.0f, 0.6f, 0.0f);  // Green for ground truth
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (const auto& pt : ground_truth_) {
        glVertex3d(pt.x * scale, pt.y * scale, pt.z * scale);
    }
    glEnd();
}

void Viewer::follow_current_pose() {
    std::lock_guard<std::mutex> lock(trajectory_mutex_);

    if (poses_.empty()) return;

    const cv::Mat& pose = poses_.back();
    float scale = Config::TRAJECTORY_SCALE;

    if (view_mode_ == VIEW_CAMERA) {
        pangolin::OpenGlMatrix Twc;
        Twc.m[0]  = pose.at<double>(0, 0);
        Twc.m[1]  = pose.at<double>(1, 0);
        Twc.m[2]  = pose.at<double>(2, 0);
        Twc.m[3]  = 0.0;
        Twc.m[4]  = pose.at<double>(0, 1);
        Twc.m[5]  = pose.at<double>(1, 1);
        Twc.m[6]  = pose.at<double>(2, 1);
        Twc.m[7]  = 0.0;
        Twc.m[8]  = pose.at<double>(0, 2);
        Twc.m[9]  = pose.at<double>(1, 2);
        Twc.m[10] = pose.at<double>(2, 2);
        Twc.m[11] = 0.0;
        Twc.m[12] = pose.at<double>(0, 3) * scale;
        Twc.m[13] = pose.at<double>(1, 3) * scale;
        Twc.m[14] = pose.at<double>(2, 3) * scale;
        Twc.m[15] = 1.0;
        s_cam_->Follow(Twc);
    } else {
        pangolin::OpenGlMatrix Ow;
        Ow.SetIdentity();
        Ow.m[12] = pose.at<double>(0, 3) * scale;
        Ow.m[13] = pose.at<double>(1, 3) * scale;
        Ow.m[14] = pose.at<double>(2, 3) * scale;
        s_cam_->Follow(Ow);
    }
}
