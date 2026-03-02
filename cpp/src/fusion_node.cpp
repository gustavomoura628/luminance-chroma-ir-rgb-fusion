#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

#include "ir_rgb_fusion_cuda/cuda_check.hpp"
#include "ir_rgb_fusion_cuda/fusion_pipeline.hpp"

using sensor_msgs::msg::Image;

class FusionNode : public rclcpp::Node {
public:
    FusionNode() : Node("fusion_node") {
        declare_params();
        read_params();
        init_cuda();
        init_pipelines();
        init_ros();

        RCLCPP_INFO(get_logger(),
            "Fusion node started — subscribing to %s, %s, %s",
            infra1_topic_.c_str(), infra2_topic_.c_str(), color_topic_.c_str());
    }

    ~FusionNode() override {
        // Free CUDA resources before context teardown
        if (d_rgb_) cudaFree(d_rgb_);
        if (stream_left_) cudaStreamDestroy(stream_left_);
        if (stream_right_) cudaStreamDestroy(stream_right_);
        if (rgb_uploaded_) cudaEventDestroy(rgb_uploaded_);
    }

private:
    // Parameters
    std::string infra1_topic_, infra2_topic_, color_topic_;
    std::string fused_left_topic_, fused_right_topic_;
    double sync_slop_;

    // CUDA resources
    cudaStream_t stream_left_ = nullptr;
    cudaStream_t stream_right_ = nullptr;
    cudaEvent_t rgb_uploaded_ = nullptr;
    uint8_t* d_rgb_ = nullptr;
    int rgb_buf_h_ = 0, rgb_buf_w_ = 0;

    // Pipelines
    std::unique_ptr<FusionPipeline> pipeline_left_;
    std::unique_ptr<FusionPipeline> pipeline_right_;

    // ROS — event-driven 3-topic sync
    // All frame callbacks share one MutuallyExclusive group so they
    // execute sequentially in DDS arrival order.  No mutex needed.
    rclcpp::CallbackGroup::SharedPtr frame_cbg_;
    rclcpp::Subscription<Image>::SharedPtr sub_ir1_;
    rclcpp::Subscription<Image>::SharedPtr sub_ir2_;
    rclcpp::Subscription<Image>::SharedPtr sub_color_;
    rclcpp::TimerBase::SharedPtr stale_timer_;
    rclcpp::Publisher<Image>::SharedPtr pub_left_;
    rclcpp::Publisher<Image>::SharedPtr pub_right_;

    // Frame state (only touched from frame_cbg_ callbacks)
    Image::ConstSharedPtr latest_ir1_, latest_ir2_, latest_rgb_;
    bool ir1_pending_ = false;

    // Pre-allocated output messages
    Image msg_left_, msg_right_;

    // Parameter callback handle (prevent GC)
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;

    // Rolling stats (2s window)
    using SteadyClock = std::chrono::steady_clock;
    using TimePoint = SteadyClock::time_point;
    struct FrameRecord { TimePoint stamp; double total_ms; };
    std::deque<FrameRecord> frame_log_;
    TimePoint last_stats_log_ = SteadyClock::now();
    TimePoint prev_callback_ = {};
    double peak_interval_ms_ = 0;
    double rgb_age_sum_ = 0;
    double rgb_age_peak_ = 0;
    int rgb_age_count_ = 0;
    int rgb_fresh_count_ = 0;

    // ---------------------------------------------------------------
    void declare_params() {
        declare_parameter("infra1_topic", "/camera/infra1/image_rect_raw");
        declare_parameter("infra2_topic", "/camera/infra2/image_rect_raw");
        declare_parameter("color_topic", "/camera/color/image_raw");
        declare_parameter("fused_left_topic", "/camera/fused/left");
        declare_parameter("fused_right_topic", "/camera/fused/right");
        declare_parameter("sync_slop", 0.005);
        declare_parameter("orb_features", 1000);
        declare_parameter("ema_alpha", 0.2);

        rcl_interfaces::msg::ParameterDescriptor freeze_desc;
        freeze_desc.description = "Warp freeze mode: \"auto\", \"on\", or \"off\"";
        declare_parameter("freeze_mode", "auto", freeze_desc);

        rcl_interfaces::msg::ParameterDescriptor freeze_after_desc;
        freeze_after_desc.description = "Seconds of EMA before auto-freezing (auto mode only)";
        declare_parameter("freeze_after", 5.0, freeze_after_desc);

        rcl_interfaces::msg::ParameterDescriptor lock_rotation_desc;
        lock_rotation_desc.description = "Rotation lock mode: \"auto\", \"on\", or \"off\"";
        declare_parameter("lock_rotation", "off", lock_rotation_desc);

        rcl_interfaces::msg::ParameterDescriptor crop_desc;
        crop_desc.description = "Crop to valid overlap region (true) or full IR frame (false)";
        declare_parameter("crop", false, crop_desc);
    }

    void read_params() {
        infra1_topic_ = get_parameter("infra1_topic").as_string();
        infra2_topic_ = get_parameter("infra2_topic").as_string();
        color_topic_ = get_parameter("color_topic").as_string();
        fused_left_topic_ = get_parameter("fused_left_topic").as_string();
        fused_right_topic_ = get_parameter("fused_right_topic").as_string();
        sync_slop_ = get_parameter("sync_slop").as_double();
    }

    void init_cuda() {
        CUDA_CHECK(cudaStreamCreate(&stream_left_));
        CUDA_CHECK(cudaStreamCreate(&stream_right_));
        CUDA_CHECK(cudaEventCreateWithFlags(&rgb_uploaded_, cudaEventDisableTiming));
    }

    void init_pipelines() {
        FusionPipeline::Params p;
        p.orb_features = get_parameter("orb_features").as_int();
        p.ema_alpha = static_cast<float>(get_parameter("ema_alpha").as_double());
        p.freeze_mode = get_parameter("freeze_mode").as_string();
        p.freeze_after = static_cast<float>(get_parameter("freeze_after").as_double());
        p.crop = get_parameter("crop").as_bool();
        p.lock_rotation = get_parameter("lock_rotation").as_string();

        pipeline_left_ = std::make_unique<FusionPipeline>(p);
        pipeline_right_ = std::make_unique<FusionPipeline>(p);
    }

    void init_ros() {
        // All frame callbacks on one MutuallyExclusive group —
        // sequential processing, no mutex needed for frame state.
        frame_cbg_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions opts;
        opts.callback_group = frame_cbg_;

        sub_ir1_ = create_subscription<Image>(
            infra1_topic_, rclcpp::QoS(10),
            [this](Image::ConstSharedPtr msg) { ir1_callback(std::move(msg)); },
            opts);
        sub_ir2_ = create_subscription<Image>(
            infra2_topic_, rclcpp::QoS(10),
            [this](Image::ConstSharedPtr msg) { ir2_callback(std::move(msg)); },
            opts);
        sub_color_ = create_subscription<Image>(
            color_topic_, rclcpp::QoS(10),
            [this](Image::ConstSharedPtr msg) { rgb_callback(std::move(msg)); },
            opts);

        // 5ms one-shot timer for stale-RGB fallback (initially cancelled)
        stale_timer_ = create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(sync_slop_ * 1000)),
            [this]() { stale_timeout(); },
            frame_cbg_);
        stale_timer_->cancel();

        pub_left_ = create_publisher<Image>(fused_left_topic_, 10);
        pub_right_ = create_publisher<Image>(fused_right_topic_, 10);

        // Dynamic parameter callback
        param_cb_ = add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter>& params) {
                return on_param_change(params);
            });
    }

    // ---------------------------------------------------------------
    // Event-driven frame sync
    //
    // Each callback stores its message and checks whether a complete
    // matched frame is ready.  Whichever topic arrives LAST triggers
    // fusion — no blocking waits, no thread coordination.
    // ---------------------------------------------------------------

    void ir1_callback(Image::ConstSharedPtr msg) {
        latest_ir1_ = std::move(msg);
        ir1_pending_ = true;
        if (!try_fuse()) {
            stale_timer_->reset();  // wait up to 5ms for fresh RGB
        }
    }

    void ir2_callback(Image::ConstSharedPtr msg) {
        latest_ir2_ = std::move(msg);
        if (ir1_pending_) try_fuse();
    }

    void rgb_callback(Image::ConstSharedPtr msg) {
        latest_rgb_ = std::move(msg);
        if (ir1_pending_) try_fuse();
    }

    void stale_timeout() {
        stale_timer_->cancel();
        if (!ir1_pending_) return;
        ir1_pending_ = false;
        if (latest_ir1_ && latest_ir2_ && latest_rgb_) {
            do_fusion(false);
        }
    }

    bool try_fuse() {
        if (!latest_ir1_ || !latest_ir2_ || !latest_rgb_) return false;

        auto ir_stamp = rclcpp::Time(latest_ir1_->header.stamp);
        auto rgb_stamp = rclcpp::Time(latest_rgb_->header.stamp);
        double age = std::abs((ir_stamp - rgb_stamp).seconds());

        if (age > sync_slop_) return false;  // RGB not from same frame

        ir1_pending_ = false;
        stale_timer_->cancel();
        do_fusion(true);
        return true;
    }

    // ---------------------------------------------------------------
    rcl_interfaces::msg::SetParametersResult on_param_change(
        const std::vector<rclcpp::Parameter>& params)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        for (const auto& p : params) {
            if (p.get_name() == "freeze_mode") {
                auto v = p.as_string();
                if (v != "auto" && v != "on" && v != "off") {
                    result.successful = false;
                    result.reason = "Invalid freeze_mode: \"" + v + "\"";
                    return result;
                }
                pipeline_left_->set_freeze_mode(v);
                pipeline_right_->set_freeze_mode(v);
                RCLCPP_INFO(get_logger(), "freeze_mode -> %s", v.c_str());
            } else if (p.get_name() == "freeze_after") {
                auto v = static_cast<float>(p.as_double());
                pipeline_left_->set_freeze_after(v);
                pipeline_right_->set_freeze_after(v);
                RCLCPP_INFO(get_logger(), "freeze_after -> %.1f", v);
            } else if (p.get_name() == "lock_rotation") {
                auto v = p.as_string();
                if (v != "auto" && v != "on" && v != "off") {
                    result.successful = false;
                    result.reason = "Invalid lock_rotation: \"" + v + "\"";
                    return result;
                }
                pipeline_left_->set_lock_rotation(v);
                pipeline_right_->set_lock_rotation(v);
                RCLCPP_INFO(get_logger(), "lock_rotation -> %s", v.c_str());
            } else if (p.get_name() == "crop") {
                pipeline_left_->set_crop(p.as_bool());
                pipeline_right_->set_crop(p.as_bool());
                RCLCPP_INFO(get_logger(), "crop -> %s", p.as_bool() ? "true" : "false");
            }
        }
        return result;
    }

    // ---------------------------------------------------------------
    void ensure_rgb_buffer(int h, int w) {
        if (h == rgb_buf_h_ && w == rgb_buf_w_) return;
        if (d_rgb_) cudaFree(d_rgb_);
        size_t bytes = static_cast<size_t>(h) * w * 3;
        CUDA_CHECK(cudaMalloc(&d_rgb_, bytes));
        rgb_buf_h_ = h;
        rgb_buf_w_ = w;
    }

    void prepare_msg(Image& msg, int h, int w,
                     const std_msgs::msg::Header& header) {
        msg.header = header;
        msg.height = h;
        msg.width = w;
        msg.encoding = "rgb8";
        msg.step = w * 3;
        msg.is_bigendian = false;
        size_t bytes = static_cast<size_t>(h) * w * 3;
        if (msg.data.size() != bytes) msg.data.resize(bytes);
    }

    // ---------------------------------------------------------------
    void do_fusion(bool rgb_fresh) {
        auto t0 = SteadyClock::now();

        const auto& infra1_msg = latest_ir1_;
        const auto& infra2_msg = latest_ir2_;
        const auto& color_msg = latest_rgb_;

        // RGB age stats
        {
            auto ir_stamp = rclcpp::Time(infra1_msg->header.stamp);
            double age_ms = (ir_stamp -
                rclcpp::Time(color_msg->header.stamp)).seconds() * 1000.0;
            rgb_age_sum_ += age_ms;
            if (age_ms > rgb_age_peak_) rgb_age_peak_ = age_ms;
            rgb_age_count_++;
            if (rgb_fresh) rgb_fresh_count_++;
        }

        // Decode images — zero-copy cv::Mat wrappers over msg data
        int ir_h = infra1_msg->height, ir_w = infra1_msg->width;
        const uint8_t* ir_left = infra1_msg->data.data();
        const uint8_t* ir_right = infra2_msg->data.data();

        int rgb_h = color_msg->height, rgb_w = color_msg->width;
        cv::Mat rgb_mat(rgb_h, rgb_w, CV_8UC3,
                        const_cast<uint8_t*>(color_msg->data.data()));

        // BGR → RGB if needed
        bool is_bgr = (color_msg->encoding == "bgr8");
        cv::Mat rgb_work;
        if (is_bgr) {
            cv::cvtColor(rgb_mat, rgb_work, cv::COLOR_BGR2RGB);
        } else {
            rgb_work = rgb_mat;
        }

        // Resize RGB to match IR if needed
        if (rgb_h != ir_h || rgb_w != ir_w) {
            cv::Mat resized;
            cv::resize(rgb_work, resized, cv::Size(ir_w, ir_h));
            rgb_work = resized;
            rgb_h = ir_h;
            rgb_w = ir_w;
        }

        // Upload RGB to GPU on stream_left, record event for stream_right
        ensure_rgb_buffer(rgb_h, rgb_w);
        size_t rgb_bytes = static_cast<size_t>(rgb_h) * rgb_w * 3;
        CUDA_CHECK(cudaMemcpyAsync(d_rgb_, rgb_work.data, rgb_bytes,
                                   cudaMemcpyHostToDevice, stream_left_));
        CUDA_CHECK(cudaEventRecord(rgb_uploaded_, stream_left_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_right_, rgb_uploaded_, 0));

        // Submit warp for both eyes (shares one rgb_gray conversion)
        cv::Mat rgb_gray;
        cv::cvtColor(rgb_work, rgb_gray, cv::COLOR_RGB2GRAY);

        cv::Mat ir_left_mat(ir_h, ir_w, CV_8UC1, const_cast<uint8_t*>(ir_left));
        cv::Mat ir_right_mat(ir_h, ir_w, CV_8UC1, const_cast<uint8_t*>(ir_right));

        pipeline_left_->submit_warp(rgb_gray, ir_left_mat);
        pipeline_right_->submit_warp(rgb_gray, ir_right_mat);

        // Launch both eyes async — overlapped on GPU
        int left_h = 0, left_w = 0;
        bool left_ok = pipeline_left_->launch_async(
            d_rgb_, rgb_h, rgb_w, ir_left, ir_h, ir_w,
            stream_left_, left_h, left_w);

        int right_h = 0, right_w = 0;
        bool right_ok = pipeline_right_->launch_async(
            d_rgb_, rgb_h, rgb_w, ir_right, ir_h, ir_w,
            stream_right_, right_h, right_w);

        // Download left + publish
        if (left_ok) {
            prepare_msg(msg_left_, left_h, left_w, infra1_msg->header);
            pipeline_left_->download_sync(msg_left_.data.data(), stream_left_);
        } else {
            prepare_msg(msg_left_, ir_h, ir_w, infra1_msg->header);
            FusionPipeline::gray_to_rgb(ir_left, ir_h, ir_w, msg_left_.data.data());
        }
        pub_left_->publish(msg_left_);

        // Download right + publish
        if (right_ok) {
            prepare_msg(msg_right_, right_h, right_w, infra2_msg->header);
            pipeline_right_->download_sync(msg_right_.data.data(), stream_right_);
        } else {
            prepare_msg(msg_right_, ir_h, ir_w, infra2_msg->header);
            FusionPipeline::gray_to_rgb(ir_right, ir_h, ir_w, msg_right_.data.data());
        }
        pub_right_->publish(msg_right_);

        auto t5 = SteadyClock::now();

        // Rolling stats
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        double total = ms(t0, t5);
        if (prev_callback_ != TimePoint{}) {
            double gap = ms(prev_callback_, t0);
            if (gap > peak_interval_ms_) peak_interval_ms_ = gap;
        }
        prev_callback_ = t0;
        frame_log_.push_back({t0, total});

        if (ms(last_stats_log_, t5) >= 2000.0) {
            // Prune older than 2s
            auto cutoff = t5 - std::chrono::seconds(2);
            while (!frame_log_.empty() && frame_log_.front().stamp < cutoff)
                frame_log_.pop_front();

            size_t n = frame_log_.size();
            if (n >= 2) {
                double span = ms(frame_log_.front().stamp, frame_log_.back().stamp);
                double avg_interval = span / (n - 1);
                double fps = (avg_interval > 0) ? 1000.0 / avg_interval : 0;

                double sum = 0, proc_worst = 0;
                for (auto& f : frame_log_) {
                    sum += f.total_ms;
                    if (f.total_ms > proc_worst) proc_worst = f.total_ms;
                }

                double rgb_avg = (rgb_age_count_ > 0) ? rgb_age_sum_ / rgb_age_count_ : 0;
                int fresh_pct = (rgb_age_count_ > 0) ? rgb_fresh_count_ * 100 / rgb_age_count_ : 0;
                RCLCPP_INFO(get_logger(),
                    "%.0f fps | interval avg %.1fms peak %.1fms | proc avg %.1fms peak %.1fms"
                    " | rgb age avg %.1fms peak %.1fms fresh %d%%",
                    fps, avg_interval, peak_interval_ms_, sum / n, proc_worst,
                    rgb_avg, rgb_age_peak_, fresh_pct);
            }
            peak_interval_ms_ = 0;
            rgb_age_sum_ = rgb_age_peak_ = 0;
            rgb_age_count_ = rgb_fresh_count_ = 0;
            last_stats_log_ = t5;
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FusionNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
