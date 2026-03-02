#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

/// Per-eye fusion pipeline: GPU buffers, ORB warp worker, async kernel launch.
///
/// Usage from the node:
///   pipeline.submit_warp(rgb_gray, ir_gray);
///   pipeline.launch_async(d_rgb, rgb_h, rgb_w, ir_msg_data, ir_h, ir_w, stream);
///   pipeline.download_sync(out_ptr, stream);
class FusionPipeline {
public:
    struct Params {
        int orb_features = 1000;
        float ema_alpha = 0.2f;
        std::string freeze_mode = "auto";  // "auto", "on", "off"
        float freeze_after = 5.0f;
        bool crop = false;
    };

    explicit FusionPipeline(const Params& params);
    ~FusionPipeline();

    FusionPipeline(const FusionPipeline&) = delete;
    FusionPipeline& operator=(const FusionPipeline&) = delete;

    /// Submit RGB/IR pair for background ORB warp estimation.
    /// rgb_gray and ir_gray are (H,W) uint8 cv::Mat. Copies are made internally.
    void submit_warp(const cv::Mat& rgb_gray, const cv::Mat& ir_gray);

    /// Queue GPU work: upload IR ROI, launch fused kernel, queue download.
    /// d_rgb must already be on GPU. Returns false if no warp exists (fallback needed).
    /// roi_h/roi_w are set to the actual output dimensions.
    bool launch_async(
        const uint8_t* d_rgb, int rgb_h, int rgb_w,
        const uint8_t* ir_data, int ir_h, int ir_w,
        cudaStream_t stream,
        int& roi_h, int& roi_w);

    /// Block until download completes, then memcpy to out_ptr.
    /// out_ptr must have space for roi_h * roi_w * 3 bytes.
    void download_sync(uint8_t* out_ptr, cudaStream_t stream);

    /// Write grayscale IR as RGB into out_ptr (fallback when no warp).
    static void gray_to_rgb(const uint8_t* ir_data, int h, int w, uint8_t* out_ptr);

    // Runtime parameter updates
    void set_freeze_mode(const std::string& mode);
    void set_freeze_after(float seconds);
    void set_crop(bool crop);
    bool get_crop() const { return params_.crop; }

private:
    Params params_;

    // Warp state (protected by warp_mutex_)
    std::mutex warp_mutex_;
    std::optional<cv::Mat> M_;  // 2x3 float64 affine
    bool M_dirty_ = false;

    // Freeze state
    std::atomic<bool> frozen_{false};
    std::chrono::steady_clock::time_point start_time_;

    // Background ORB worker
    std::mutex pending_mutex_;
    std::condition_variable pending_cv_;
    std::optional<std::pair<cv::Mat, cv::Mat>> pending_;  // (rgb_gray, ir_gray)
    bool worker_stop_ = false;
    std::thread worker_thread_;
    void warp_worker();

    // Crop computation
    struct Roi { int x1, y1, x2, y2; };
    std::optional<Roi> compute_crop(const cv::Mat& M, int W, int H);

    // GPU buffers (lazy-allocated, reused across frames)
    int buf_roi_h_ = 0, buf_roi_w_ = 0;
    uint8_t* d_ir_ = nullptr;       // (roi_h, roi_w) uint8
    uint8_t* d_output_ = nullptr;   // (roi_h, roi_w, 3) uint8
    uint8_t* h_output_ = nullptr;   // pinned host, same size as d_output_

    float* d_M_inv_ = nullptr;      // 6 floats
    float* d_M_color_ = nullptr;    // 9 floats

    // Cached inverse affine (recompute only when M changes)
    float h_M_inv_[6] = {};

    // Current frame state (set by launch_async, read by download_sync)
    int cur_roi_h_ = 0, cur_roi_w_ = 0;

    void ensure_buffers(int roi_h, int roi_w);
    void free_buffers();
    void upload_color_matrix();

    // Pre-computed color matrix: M_fuse(3x2) @ M_chroma(2x3) = 3x3
    static constexpr float kMColor[9] = {
         0.701500f, -0.587857f, -0.113643f,
        -0.298864f,  0.413030f, -0.114166f,
        -0.299637f, -0.586863f,  0.886500f,
    };
};
