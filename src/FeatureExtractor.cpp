#include "FeatureExtractor.h"
#include "Config.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>

FeatureExtractor::FeatureExtractor()
    : use_superpoint_(false) {
    orb_ = cv::ORB::create(Config::NUM_FEATURES, Config::SCALE_FACTOR, Config::NUM_LEVELS);
}

FeatureExtractor::~FeatureExtractor() = default;

bool FeatureExtractor::init(const std::string& model_path) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        } catch (...) {
            opts.SetIntraOpNumThreads(4);
        }

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);
        use_superpoint_ = true;
        return true;
    } catch (const std::exception& e) {
        use_superpoint_ = false;
        return false;
    }
}

void FeatureExtractor::extract(const cv::Mat& image,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::Mat& descriptors) {
    int idx = extract_counter_++;

    if (cache_loaded_) {
        auto it = cache_.find(idx);
        if (it != cache_.end()) {
            keypoints = it->second.keypoints;
            descriptors = it->second.descriptors.clone();
            return;
        }
    }

    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;

    if (use_superpoint_) {
        extract_superpoint(gray, keypoints, descriptors);
    } else {
        extract_orb(gray, keypoints, descriptors);
    }

    if (!cache_path_.empty()) {
        CachedFeatures cf;
        cf.keypoints = keypoints;
        cf.descriptors = descriptors.clone();
        cache_[idx] = std::move(cf);
    }
}

void FeatureExtractor::extract_superpoint(const cv::Mat& gray,
                                           std::vector<cv::KeyPoint>& keypoints,
                                           cv::Mat& descriptors) {
    int H = gray.rows;
    int W = gray.cols;
    int Hp = ((H + 7) / 8) * 8;
    int Wp = ((W + 7) / 8) * 8;

    cv::Mat input_f;
    gray.convertTo(input_f, CV_32F, 1.0 / 255.0);

    if (Hp != H || Wp != W) {
        cv::Mat padded = cv::Mat::zeros(Hp, Wp, CV_32F);
        input_f.copyTo(padded(cv::Rect(0, 0, W, H)));
        input_f = padded;
    }

    std::vector<float> input_data(Hp * Wp);
    std::memcpy(input_data.data(), input_f.data, Hp * Wp * sizeof(float));

    std::vector<int64_t> input_shape = {1, 1, Hp, Wp};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"image"};
    const char* output_names[] = {"semi", "desc"};

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                  input_names, &input_tensor, 1,
                                  output_names, 2);

    // Parse semi output: 1 x 65 x Hc x Wc
    auto semi_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int Hc = semi_shape[2];
    int Wc = semi_shape[3];
    float* semi_data = outputs[0].GetTensorMutableData<float>();

    cv::Mat heatmap = cv::Mat::zeros(Hp, Wp, CV_32F);

    for (int hc = 0; hc < Hc; hc++) {
        for (int wc = 0; wc < Wc; wc++) {
            float cell[65];
            for (int c = 0; c < 65; c++) {
                cell[c] = semi_data[c * Hc * Wc + hc * Wc + wc];
            }
            float max_val = *std::max_element(cell, cell + 65);
            float sum = 0;
            for (int c = 0; c < 65; c++) {
                cell[c] = std::exp(cell[c] - max_val);
                sum += cell[c];
            }
            for (int c = 0; c < 65; c++) {
                cell[c] /= sum;
            }
            for (int c = 0; c < 64; c++) {
                int y = hc * 8 + c / 8;
                int x = wc * 8 + c % 8;
                if (y < Hp && x < Wp) {
                    heatmap.at<float>(y, x) = cell[c];
                }
            }
        }
    }

    nms(heatmap, Config::SP_NMS_RADIUS, keypoints, Config::SP_MAX_KEYPOINTS);

    keypoints.erase(
        std::remove_if(keypoints.begin(), keypoints.end(),
                       [W, H](const cv::KeyPoint& kp) {
                           return kp.pt.x >= W || kp.pt.y >= H;
                       }),
        keypoints.end());

    if (keypoints.empty()) {
        descriptors = cv::Mat();
        return;
    }

    // Parse desc output: 1 x 256 x Hc x Wc
    float* desc_data = outputs[1].GetTensorMutableData<float>();

    descriptors = cv::Mat((int)keypoints.size(), 256, CV_32F);

    for (int i = 0; i < (int)keypoints.size(); i++) {
        float sx = keypoints[i].pt.x / 8.0f;
        float sy = keypoints[i].pt.y / 8.0f;

        int x0 = std::max(0, std::min((int)std::floor(sx), Wc - 1));
        int y0 = std::max(0, std::min((int)std::floor(sy), Hc - 1));
        int x1 = std::min(x0 + 1, Wc - 1);
        int y1 = std::min(y0 + 1, Hc - 1);

        float wx = sx - x0;
        float wy = sy - y0;

        for (int c = 0; c < 256; c++) {
            float v00 = desc_data[c * Hc * Wc + y0 * Wc + x0];
            float v01 = desc_data[c * Hc * Wc + y0 * Wc + x1];
            float v10 = desc_data[c * Hc * Wc + y1 * Wc + x0];
            float v11 = desc_data[c * Hc * Wc + y1 * Wc + x1];

            float val = (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                        wy * ((1 - wx) * v10 + wx * v11);
            descriptors.at<float>(i, c) = val;
        }

        float norm = 0;
        for (int c = 0; c < 256; c++) {
            float v = descriptors.at<float>(i, c);
            norm += v * v;
        }
        norm = std::sqrt(norm);
        if (norm > 1e-8f) {
            for (int c = 0; c < 256; c++) {
                descriptors.at<float>(i, c) /= norm;
            }
        }
    }
}

void FeatureExtractor::extract_orb(const cv::Mat& gray,
                                    std::vector<cv::KeyPoint>& keypoints,
                                    cv::Mat& descriptors) {
    orb_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
}

void FeatureExtractor::nms(const cv::Mat& heatmap, int radius,
                            std::vector<cv::KeyPoint>& keypoints, int max_keypoints) {
    keypoints.clear();

    struct Candidate {
        float score;
        int x, y;
    };
    std::vector<Candidate> candidates;

    for (int y = 0; y < heatmap.rows; y++) {
        for (int x = 0; x < heatmap.cols; x++) {
            float val = heatmap.at<float>(y, x);
            if (val > Config::SP_CONFIDENCE_THRESHOLD) {
                candidates.push_back({val, x, y});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    cv::Mat suppressed = cv::Mat::zeros(heatmap.rows, heatmap.cols, CV_8U);

    for (const auto& c : candidates) {
        if ((int)keypoints.size() >= max_keypoints) break;
        if (suppressed.at<uint8_t>(c.y, c.x)) continue;

        keypoints.emplace_back(cv::Point2f(c.x, c.y), 8.0f, -1, c.score);

        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int ny = c.y + dy;
                int nx = c.x + dx;
                if (ny >= 0 && ny < heatmap.rows && nx >= 0 && nx < heatmap.cols) {
                    suppressed.at<uint8_t>(ny, nx) = 1;
                }
            }
        }
    }
}

void FeatureExtractor::set_cache_path(const std::string& path) {
    cache_path_ = path;
}

bool FeatureExtractor::load_cache() {
    if (cache_path_.empty()) return false;

    std::ifstream ifs(cache_path_, std::ios::binary);
    if (!ifs.is_open()) return false;

    uint32_t magic, version, num_entries;
    ifs.read(reinterpret_cast<char*>(&magic), 4);
    ifs.read(reinterpret_cast<char*>(&version), 4);
    ifs.read(reinterpret_cast<char*>(&num_entries), 4);

    if (magic != 0x53504346 || version != 1) {  // "SPCF" = SuperPoint Cache File
        return false;
    }

    cache_.clear();
    for (uint32_t i = 0; i < num_entries; i++) {
        int32_t frame_idx;
        int32_t num_kp, desc_rows, desc_cols, desc_type;

        ifs.read(reinterpret_cast<char*>(&frame_idx), 4);
        ifs.read(reinterpret_cast<char*>(&num_kp), 4);

        CachedFeatures cf;
        cf.keypoints.resize(num_kp);
        for (int k = 0; k < num_kp; k++) {
            float x, y, size, angle, response;
            int32_t octave, class_id;
            ifs.read(reinterpret_cast<char*>(&x), 4);
            ifs.read(reinterpret_cast<char*>(&y), 4);
            ifs.read(reinterpret_cast<char*>(&size), 4);
            ifs.read(reinterpret_cast<char*>(&angle), 4);
            ifs.read(reinterpret_cast<char*>(&response), 4);
            ifs.read(reinterpret_cast<char*>(&octave), 4);
            ifs.read(reinterpret_cast<char*>(&class_id), 4);
            cf.keypoints[k] = cv::KeyPoint(x, y, size, angle, response, octave, class_id);
        }

        ifs.read(reinterpret_cast<char*>(&desc_rows), 4);
        ifs.read(reinterpret_cast<char*>(&desc_cols), 4);
        ifs.read(reinterpret_cast<char*>(&desc_type), 4);

        if (desc_rows > 0 && desc_cols > 0) {
            cf.descriptors = cv::Mat(desc_rows, desc_cols, desc_type);
            size_t nbytes = desc_rows * desc_cols * cf.descriptors.elemSize();
            ifs.read(reinterpret_cast<char*>(cf.descriptors.data), nbytes);
        }

        cache_[frame_idx] = std::move(cf);
    }

    cache_loaded_ = true;
    return true;
}

bool FeatureExtractor::save_cache() {
    if (cache_path_.empty() || cache_.empty()) return false;

    std::ofstream ofs(cache_path_, std::ios::binary);
    if (!ofs.is_open()) {
        return false;
    }

    uint32_t magic = 0x53504346;  // "SPCF"
    uint32_t version = 1;
    uint32_t num_entries = (uint32_t)cache_.size();

    ofs.write(reinterpret_cast<const char*>(&magic), 4);
    ofs.write(reinterpret_cast<const char*>(&version), 4);
    ofs.write(reinterpret_cast<const char*>(&num_entries), 4);

    std::vector<int> indices;
    indices.reserve(cache_.size());
    for (const auto& kv : cache_) indices.push_back(kv.first);
    std::sort(indices.begin(), indices.end());

    for (int idx : indices) {
        const auto& cf = cache_.at(idx);
        int32_t frame_idx = idx;
        int32_t num_kp = (int32_t)cf.keypoints.size();

        ofs.write(reinterpret_cast<const char*>(&frame_idx), 4);
        ofs.write(reinterpret_cast<const char*>(&num_kp), 4);

        for (const auto& kp : cf.keypoints) {
            float x = kp.pt.x, y = kp.pt.y, size = kp.size;
            float angle = kp.angle, response = kp.response;
            int32_t octave = kp.octave, class_id = kp.class_id;
            ofs.write(reinterpret_cast<const char*>(&x), 4);
            ofs.write(reinterpret_cast<const char*>(&y), 4);
            ofs.write(reinterpret_cast<const char*>(&size), 4);
            ofs.write(reinterpret_cast<const char*>(&angle), 4);
            ofs.write(reinterpret_cast<const char*>(&response), 4);
            ofs.write(reinterpret_cast<const char*>(&octave), 4);
            ofs.write(reinterpret_cast<const char*>(&class_id), 4);
        }

        int32_t desc_rows = cf.descriptors.rows;
        int32_t desc_cols = cf.descriptors.cols;
        int32_t desc_type = cf.descriptors.type();
        ofs.write(reinterpret_cast<const char*>(&desc_rows), 4);
        ofs.write(reinterpret_cast<const char*>(&desc_cols), 4);
        ofs.write(reinterpret_cast<const char*>(&desc_type), 4);

        if (desc_rows > 0 && desc_cols > 0) {
            size_t nbytes = desc_rows * desc_cols * cf.descriptors.elemSize();
            ofs.write(reinterpret_cast<const char*>(cf.descriptors.data), nbytes);
        }
    }

    return true;
}
