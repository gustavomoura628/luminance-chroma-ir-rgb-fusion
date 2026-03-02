#include "ir_rgb_fusion_cuda/fusion_pipeline.hpp"
#include "ir_rgb_fusion_cuda/cuda_check.hpp"
#include "ir_rgb_fusion_cuda/fusion_kernel.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

static constexpr int kMinMatches = 10;

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

FusionPipeline::FusionPipeline(const Params& params)
    : params_(params),
      start_time_(std::chrono::steady_clock::now())
{
    // Allocate persistent device buffers for affine + color matrix
    CUDA_CHECK(cudaMalloc(&d_M_inv_, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_M_color_, 9 * sizeof(float)));
    upload_color_matrix();

    worker_thread_ = std::thread(&FusionPipeline::warp_worker, this);
}

FusionPipeline::~FusionPipeline() {
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        worker_stop_ = true;
    }
    pending_cv_.notify_one();
    if (worker_thread_.joinable()) worker_thread_.join();

    free_buffers();
    if (d_M_inv_) cudaFree(d_M_inv_);
    if (d_M_color_) cudaFree(d_M_color_);
}

// ---------------------------------------------------------------------------
// GPU buffer management
// ---------------------------------------------------------------------------

void FusionPipeline::ensure_buffers(int roi_h, int roi_w) {
    if (roi_h == buf_roi_h_ && roi_w == buf_roi_w_) return;
    free_buffers();

    buf_roi_h_ = roi_h;
    buf_roi_w_ = roi_w;
    size_t ir_bytes = static_cast<size_t>(roi_h) * roi_w;
    size_t rgb_bytes = ir_bytes * 3;

    CUDA_CHECK(cudaMalloc(&d_ir_, ir_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_, rgb_bytes));
    CUDA_CHECK(cudaMallocHost(&h_output_, rgb_bytes));
}

void FusionPipeline::free_buffers() {
    if (d_ir_) { cudaFree(d_ir_); d_ir_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (h_output_) { cudaFreeHost(h_output_); h_output_ = nullptr; }
    buf_roi_h_ = buf_roi_w_ = 0;
}

void FusionPipeline::upload_color_matrix() {
    CUDA_CHECK(cudaMemcpy(d_M_color_, kMColor, 9 * sizeof(float),
                          cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Warp submission + freeze logic
// ---------------------------------------------------------------------------

void FusionPipeline::submit_warp(const cv::Mat& rgb_gray, const cv::Mat& ir_gray) {
    if (frozen_.load(std::memory_order_relaxed)) return;

    // Auto-freeze check
    if (params_.freeze_mode == "auto") {
        std::lock_guard<std::mutex> lock(warp_mutex_);
        if (M_.has_value()) {
            auto elapsed = std::chrono::steady_clock::now() - start_time_;
            float secs = std::chrono::duration<float>(elapsed).count();
            if (secs > params_.freeze_after) {
                frozen_.store(true, std::memory_order_relaxed);
                return;
            }
        }
    }
    if (params_.freeze_mode == "on") {
        frozen_.store(true, std::memory_order_relaxed);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_ = std::make_pair(rgb_gray.clone(), ir_gray.clone());
    }
    pending_cv_.notify_one();
}

// ---------------------------------------------------------------------------
// Background ORB worker
// ---------------------------------------------------------------------------

void FusionPipeline::warp_worker() {
    auto orb = cv::ORB::create(params_.orb_features);
    auto bf = cv::BFMatcher::create(cv::NORM_HAMMING, true);

    while (true) {
        std::pair<cv::Mat, cv::Mat> work;
        {
            std::unique_lock<std::mutex> lock(pending_mutex_);
            pending_cv_.wait(lock, [this] { return pending_.has_value() || worker_stop_; });
            if (worker_stop_) return;
            work = std::move(*pending_);
            pending_.reset();
        }

        const auto& [rgb_gray, ir_gray] = work;

        // Detect + compute ORB
        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat des1, des2;
        orb->detectAndCompute(rgb_gray, cv::noArray(), kp1, des1);
        orb->detectAndCompute(ir_gray, cv::noArray(), kp2, des2);

        if (des1.empty() || des2.empty() ||
            static_cast<int>(kp1.size()) < kMinMatches ||
            static_cast<int>(kp2.size()) < kMinMatches) {
            continue;
        }

        // BFMatch
        std::vector<cv::DMatch> matches;
        bf->match(des1, des2, matches);
        if (static_cast<int>(matches.size()) < kMinMatches) continue;

        // Extract matched points
        std::vector<cv::Point2f> pts1(matches.size()), pts2(matches.size());
        for (size_t i = 0; i < matches.size(); ++i) {
            pts1[i] = kp1[matches[i].queryIdx].pt;
            pts2[i] = kp2[matches[i].trainIdx].pt;
        }

        // RANSAC affine estimation
        cv::Mat inliers;
        cv::Mat M = cv::estimateAffinePartial2D(
            pts1, pts2, inliers, cv::RANSAC, 5.0);
        if (M.empty()) continue;

        // EMA smoothing + update
        {
            std::lock_guard<std::mutex> lock(warp_mutex_);
            if (M_.has_value()) {
                M_ = (1.0 - params_.ema_alpha) * (*M_) + params_.ema_alpha * M;
            } else {
                M_ = M.clone();
            }
            M_dirty_ = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Crop computation
// ---------------------------------------------------------------------------

std::optional<FusionPipeline::Roi> FusionPipeline::compute_crop(
    const cv::Mat& M, int W, int H)
{
    // Transform IR frame corners through M to find RGB overlap
    std::vector<cv::Point2f> corners = {
        {0.f, 0.f}, {static_cast<float>(W), 0.f},
        {static_cast<float>(W), static_cast<float>(H)}, {0.f, static_cast<float>(H)}
    };
    std::vector<cv::Point2f> warped;
    cv::transform(corners, warped, M);

    const auto& tl = warped[0];
    const auto& tr = warped[1];
    const auto& br = warped[2];
    const auto& bl = warped[3];

    int x1 = static_cast<int>(std::ceil(std::max({tl.x, bl.x, 0.f})));
    int y1 = static_cast<int>(std::ceil(std::max({tl.y, tr.y, 0.f})));
    int x2 = static_cast<int>(std::floor(std::min({tr.x, br.x, static_cast<float>(W)})));
    int y2 = static_cast<int>(std::floor(std::min({bl.y, br.y, static_cast<float>(H)})));

    if (x2 <= x1 || y2 <= y1) return std::nullopt;
    return Roi{x1, y1, x2, y2};
}

// ---------------------------------------------------------------------------
// Async GPU pipeline
// ---------------------------------------------------------------------------

bool FusionPipeline::launch_async(
    const uint8_t* d_rgb, int rgb_h, int rgb_w,
    const uint8_t* ir_data, int ir_h, int ir_w,
    cudaStream_t stream,
    int& roi_h, int& roi_w)
{
    cv::Mat M_copy;
    bool dirty;
    {
        std::lock_guard<std::mutex> lock(warp_mutex_);
        if (!M_.has_value()) return false;
        M_copy = M_->clone();
        dirty = M_dirty_;
        M_dirty_ = false;
    }

    auto crop_opt = compute_crop(M_copy, ir_w, ir_h);
    if (!crop_opt.has_value()) return false;

    Roi roi = params_.crop ? *crop_opt : Roi{0, 0, ir_w, ir_h};
    roi_h = cur_roi_h_ = roi.y2 - roi.y1;
    roi_w = cur_roi_w_ = roi.x2 - roi.x1;

    ensure_buffers(cur_roi_h_, cur_roi_w_);

    // Recompute inverse affine if warp changed
    if (dirty) {
        // M is 2x3 float64. Build 3x3, invert, take top 2 rows.
        double M3x3[9] = {
            M_copy.at<double>(0, 0), M_copy.at<double>(0, 1), M_copy.at<double>(0, 2),
            M_copy.at<double>(1, 0), M_copy.at<double>(1, 1), M_copy.at<double>(1, 2),
            0.0, 0.0, 1.0
        };
        cv::Mat M_full(3, 3, CV_64F, M3x3);
        cv::Mat M_inv_full = M_full.inv();

        h_M_inv_[0] = static_cast<float>(M_inv_full.at<double>(0, 0));
        h_M_inv_[1] = static_cast<float>(M_inv_full.at<double>(0, 1));
        h_M_inv_[2] = static_cast<float>(M_inv_full.at<double>(0, 2));
        h_M_inv_[3] = static_cast<float>(M_inv_full.at<double>(1, 0));
        h_M_inv_[4] = static_cast<float>(M_inv_full.at<double>(1, 1));
        h_M_inv_[5] = static_cast<float>(M_inv_full.at<double>(1, 2));

        CUDA_CHECK(cudaMemcpyAsync(d_M_inv_, h_M_inv_, 6 * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }

    // Upload IR ROI via 2D memcpy (handles non-contiguous crop from full frame)
    CUDA_CHECK(cudaMemcpy2DAsync(
        d_ir_, cur_roi_w_,                          // dst, dst pitch
        ir_data + roi.y1 * ir_w + roi.x1, ir_w,     // src, src pitch
        cur_roi_w_, cur_roi_h_,                      // width, height in bytes
        cudaMemcpyHostToDevice, stream));

    // Launch fused kernel
    launch_fuse_kernel(
        d_rgb, rgb_h, rgb_w,
        d_ir_, cur_roi_w_,  // ir_stride = roi_w (contiguous after upload)
        d_M_inv_, d_M_color_,
        d_output_,
        cur_roi_h_, cur_roi_w_,
        roi.x1, roi.y1,
        stream);

    // Queue download to pinned host
    size_t out_bytes = static_cast<size_t>(cur_roi_h_) * cur_roi_w_ * 3;
    CUDA_CHECK(cudaMemcpyAsync(h_output_, d_output_, out_bytes,
                               cudaMemcpyDeviceToHost, stream));

    return true;
}

void FusionPipeline::download_sync(uint8_t* out_ptr, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t bytes = static_cast<size_t>(cur_roi_h_) * cur_roi_w_ * 3;
    std::memcpy(out_ptr, h_output_, bytes);
}

void FusionPipeline::gray_to_rgb(
    const uint8_t* ir_data, int h, int w, uint8_t* out_ptr)
{
    for (int i = 0; i < h * w; ++i) {
        uint8_t v = ir_data[i];
        out_ptr[i * 3 + 0] = v;
        out_ptr[i * 3 + 1] = v;
        out_ptr[i * 3 + 2] = v;
    }
}

// ---------------------------------------------------------------------------
// Runtime parameter updates
// ---------------------------------------------------------------------------

void FusionPipeline::set_freeze_mode(const std::string& mode) {
    if (mode == params_.freeze_mode) return;
    params_.freeze_mode = mode;
    if (mode == "off") {
        frozen_.store(false, std::memory_order_relaxed);
        pending_cv_.notify_one();
    } else if (mode == "on") {
        frozen_.store(true, std::memory_order_relaxed);
    } else if (mode == "auto") {
        frozen_.store(false, std::memory_order_relaxed);
        start_time_ = std::chrono::steady_clock::now();
    }
}

void FusionPipeline::set_freeze_after(float seconds) {
    params_.freeze_after = seconds;
    if (params_.freeze_mode == "auto" && frozen_.load(std::memory_order_relaxed)) {
        frozen_.store(false, std::memory_order_relaxed);
        start_time_ = std::chrono::steady_clock::now();
    }
}

void FusionPipeline::set_crop(bool crop) {
    params_.crop = crop;
}
