#include "LoopCloser.h"
#include "Frame.h"
#include "Map.h"
#include "Config.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <cmath>

LoopCloser::LoopCloser() : loop_count_(0) {}

LoopResult LoopCloser::detect(const std::shared_ptr<Frame>& current_frame,
                               const Map& map,
                               const cv::Mat& K) {
    LoopResult result;
    result.detected = false;

    if (current_frame->descriptors().empty()) return result;

    auto keyframes = map.get_keyframes();
    if (keyframes.size() < 2) return result;

    bool is_float = (current_frame->descriptors().type() == CV_32F);

    cv::Ptr<cv::DescriptorMatcher> matcher;
    if (is_float) {
        matcher = cv::FlannBasedMatcher::create();
    } else {
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    }

    int best_inliers = 0;
    std::shared_ptr<Frame> best_match;
    cv::Mat best_R, best_t;
    int best_match_count = 0;

    // Direct feature matching against distant keyframes
    int checked = 0;
    for (const auto& kf : keyframes) {
        if (current_frame->id() - kf->id() < Config::LC_MIN_FRAME_GAP) continue;
        if (kf->descriptors().empty()) continue;

        checked++;
        if (checked % 5 != 0) continue;

        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(current_frame->descriptors(), kf->descriptors(),
                          knn_matches, 2);

        std::vector<cv::DMatch> good_matches;
        float ratio_thresh = is_float ? Config::L2_RATIO_THRESHOLD : 0.8f;
        for (const auto& m : knn_matches) {
            if (m.size() >= 2 && m[0].distance < ratio_thresh * m[1].distance) {
                good_matches.push_back(m[0]);
            }
        }

        if ((int)good_matches.size() < Config::MIN_MATCHES) continue;

        std::vector<cv::Point2f> pts1, pts2;
        for (const auto& m : good_matches) {
            pts1.push_back(current_frame->keypoints()[m.queryIdx].pt);
            pts2.push_back(kf->keypoints()[m.trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC,
                                          Config::RANSAC_PROB, Config::RANSAC_THRESHOLD, mask);
        if (E.empty()) continue;

        int inlier_count = cv::countNonZero(mask);
        if (inlier_count < Config::LC_MIN_INLIERS) continue;

        if (inlier_count > best_inliers) {
            best_inliers = inlier_count;
            best_match = kf;
            best_match_count = (int)good_matches.size();

            cv::Mat R, t;
            cv::recoverPose(E, pts1, pts2, K, R, t, mask);
            best_R = R;
            best_t = t;
        }
    }

    if (best_match && best_inliers >= Config::LC_MIN_INLIERS) {
        result.detected = true;
        result.matched_frame_id = best_match->id();
        result.R_relative = best_R;
        result.t_relative = best_t;
        loop_count_++;

    }

    return result;
}
