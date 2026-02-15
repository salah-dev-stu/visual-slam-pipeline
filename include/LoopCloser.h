#ifndef LOOP_CLOSER_H
#define LOOP_CLOSER_H

#include <opencv2/core.hpp>
#include <vector>
#include <memory>

class Frame;
class Map;

struct LoopResult {
    bool detected;
    int matched_frame_id;
    cv::Mat R_relative;
    cv::Mat t_relative;
};

class LoopCloser {
public:
    LoopCloser();

    LoopResult detect(const std::shared_ptr<Frame>& current_frame,
                      const Map& map,
                      const cv::Mat& K);

    int loop_count() const { return loop_count_; }

private:
    int loop_count_;
};

#endif
