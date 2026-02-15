#ifndef VIEWER_H
#define VIEWER_H

#include <pangolin/pangolin.h>
#include <opencv2/core.hpp>
#include <vector>
#include <mutex>
#include <atomic>
#include <utility>

class Viewer {
public:
    enum ViewMode { VIEW_CAMERA, VIEW_TOP, VIEW_SIDE };

    Viewer();
    ~Viewer();

    void init();
    void set_initial_viewpoint(double x, double y, double z);
    bool render_frame();
    void shutdown();

    void update_trajectory(const std::vector<cv::Point3d>& trajectory);
    void update_poses(const std::vector<cv::Mat>& poses);
    void update_image(const cv::Mat& image);
    void update_map_points(const std::vector<cv::Point3d>& points);
    void update_sparse_points(const std::vector<cv::Point3d>& points);
    void update_loop_edges(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& edges);
    void update_ground_truth(const std::vector<cv::Point3d>& gt);

    bool should_quit() const { return should_quit_; }
    void request_quit() { should_quit_ = true; }
    bool should_restart() const { return should_restart_; }
    void clear_restart() { should_restart_ = false; }
    void reset_interpolation() { has_interpolated_pose_ = false; view_mode_ = VIEW_CAMERA; }

    void show_image();

private:
    void draw_trajectory();
    void draw_camera_poses();
    void draw_camera_frustum(const cv::Mat& pose, float size, bool current = false);
    void draw_map_points();
    void draw_sparse_points();
    void draw_loop_edges();
    void draw_ground_truth();
    void follow_current_pose();

private:
    bool initialized_;
    std::atomic<bool> should_quit_;
    std::atomic<bool> should_restart_;

    std::vector<cv::Point3d> trajectory_;
    std::vector<cv::Mat> poses_;
    mutable std::mutex trajectory_mutex_;

    std::vector<cv::Point3d> map_points_;       // dense cloud
    std::vector<cv::Point3d> sparse_points_;    // triangulated
    mutable std::mutex points_mutex_;

    std::vector<std::pair<cv::Point3d, cv::Point3d>> loop_edges_;
    mutable std::mutex loop_mutex_;

    std::vector<cv::Point3d> ground_truth_;
    mutable std::mutex gt_mutex_;

    cv::Mat current_image_;
    bool has_new_image_;
    mutable std::mutex image_mutex_;

    pangolin::OpenGlRenderState* s_cam_;
    pangolin::View* d_cam_;

    float key_frame_size_;
    float key_frame_line_width_;
    float camera_size_;
    float camera_line_width_;

    cv::Mat interpolated_pose_;
    bool has_interpolated_pose_;

    ViewMode view_mode_;

    pangolin::Var<bool>* follow_button_;
    pangolin::Var<bool>* restart_button_;
    pangolin::Var<bool>* top_view_button_;
    pangolin::Var<bool>* side_view_button_;
    pangolin::Var<bool>* reset_view_button_;
    pangolin::Var<bool>* show_points_button_;
    pangolin::Var<bool>* show_sparse_button_;

    bool has_initial_focus_;
    double focus_x_, focus_y_, focus_z_;
};

#endif
