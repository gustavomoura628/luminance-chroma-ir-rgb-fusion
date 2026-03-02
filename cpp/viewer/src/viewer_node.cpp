#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_opengl3.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// ---------------------------------------------------------------------------
// StreamState: one per subscribable image pair (or single image)
// ---------------------------------------------------------------------------
struct ImageSlot {
    GLuint texture = 0;
    int w = 0, h = 0;
    bool has_new_data = false;
    std::vector<uint8_t> buffer; // always RGB8
};

struct StreamStats {
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    static constexpr double WINDOW_SEC = 2.0;

    std::deque<TimePoint> arrivals;
    float avg_fps = 0.0f;
    float avg_ms = 0.0f;
    float peak_ms = 0.0f;

    void record_frame() {
        arrivals.push_back(Clock::now());
    }

    void update() {
        if (arrivals.empty()) return;
        auto now = Clock::now();
        auto cutoff = now - std::chrono::duration<double>(WINDOW_SEC);
        while (arrivals.size() > 1 && arrivals.front() < cutoff)
            arrivals.pop_front();

        size_t n = arrivals.size();
        if (n < 2) { avg_fps = 0; avg_ms = 0; peak_ms = 0; return; }

        double span_ms = std::chrono::duration<double, std::milli>(
            arrivals.back() - arrivals.front()).count();
        avg_ms = static_cast<float>(span_ms / (n - 1));
        avg_fps = (avg_ms > 0) ? 1000.0f / avg_ms : 0.0f;

        double worst = 0;
        for (size_t i = 1; i < n; ++i) {
            double gap = std::chrono::duration<double, std::milli>(
                arrivals[i] - arrivals[i - 1]).count();
            if (gap > worst) worst = gap;
        }
        peak_ms = static_cast<float>(worst);
    }

    void reset() {
        arrivals.clear();
        avg_fps = avg_ms = peak_ms = 0;
    }
};

struct StreamState {
    bool enabled = true;
    bool stereo = false;
    ImageSlot slots[2]; // [0]=left/single, [1]=right
    StreamStats stats;
    std::vector<std::string> topics;
    std::vector<std::string> encodings; // expected encoding per topic
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subs;
};

// ---------------------------------------------------------------------------
// RecorderState: ffmpeg pipe-based video recording
// ---------------------------------------------------------------------------
struct RecordRow {
    int stream_idx;
    bool stereo;
    int y_offset;
    int row_h;
    int row_w;      // total content width for this row
    int left_w;     // frozen left/single image width
    int right_w;    // frozen right image width (stereo only)
};

struct RecorderState {
    bool active = false;
    FILE* pipe = nullptr;
    std::string filename;
    int canvas_w = 0;
    int canvas_h = 0;
    std::vector<RecordRow> rows;
    std::vector<uint8_t> frame_buf;
    std::chrono::steady_clock::time_point last_frame_time;
    std::chrono::steady_clock::time_point start_time;
    int frame_count = 0;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void ensure_texture(ImageSlot& slot, int w, int h) {
    if (slot.texture && slot.w == w && slot.h == h) return;
    if (slot.texture) glDeleteTextures(1, &slot.texture);
    glGenTextures(1, &slot.texture);
    glBindTexture(GL_TEXTURE_2D, slot.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    slot.w = w;
    slot.h = h;
}

static void upload_texture(ImageSlot& slot) {
    if (!slot.has_new_data || !slot.texture) return;
    glBindTexture(GL_TEXTURE_2D, slot.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, slot.w, slot.h, GL_RGB, GL_UNSIGNED_BYTE, slot.buffer.data());
    slot.has_new_data = false;
}

// Copy image data into slot buffer, converting to RGB8
static void copy_to_slot(ImageSlot& slot, const sensor_msgs::msg::Image& msg) {
    int w = static_cast<int>(msg.width);
    int h = static_cast<int>(msg.height);
    size_t npix = static_cast<size_t>(w) * h;
    size_t rgb_size = npix * 3;

    ensure_texture(slot, w, h);
    slot.buffer.resize(rgb_size);

    if (msg.encoding == "rgb8") {
        std::memcpy(slot.buffer.data(), msg.data.data(), rgb_size);
    } else if (msg.encoding == "bgr8") {
        const uint8_t* src = msg.data.data();
        uint8_t* dst = slot.buffer.data();
        for (size_t i = 0; i < npix; ++i) {
            dst[0] = src[2];
            dst[1] = src[1];
            dst[2] = src[0];
            dst += 3;
            src += 3;
        }
    } else if (msg.encoding == "mono8" || msg.encoding == "8UC1") {
        const uint8_t* src = msg.data.data();
        uint8_t* dst = slot.buffer.data();
        for (size_t i = 0; i < npix; ++i) {
            dst[0] = dst[1] = dst[2] = src[i];
            dst += 3;
        }
    } else {
        RCLCPP_WARN_ONCE(rclcpp::get_logger("viewer"),
            "Unknown encoding '%s' on topic, treating as rgb8", msg.encoding.c_str());
        size_t copy_size = std::min(rgb_size, msg.data.size());
        std::memcpy(slot.buffer.data(), msg.data.data(), copy_size);
    }

    slot.has_new_data = true;
}

// ---------------------------------------------------------------------------
// Subscription management
// ---------------------------------------------------------------------------
static void create_subscriptions(StreamState& st, rclcpp::Node* node) {
    st.subs.resize(st.topics.size());
    for (size_t i = 0; i < st.topics.size(); ++i) {
        size_t slot_idx = i;
        st.subs[i] = node->create_subscription<sensor_msgs::msg::Image>(
            st.topics[i], rclcpp::SensorDataQoS(),
            [&st, slot_idx](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                copy_to_slot(st.slots[slot_idx], *msg);
                if (slot_idx == 0) st.stats.record_frame();
            });
    }
}

static void destroy_subscriptions(StreamState& st) {
    for (auto& sub : st.subs) sub.reset();
    st.subs.clear();
}

static void toggle_stream(StreamState& st, rclcpp::Node* node) {
    st.enabled = !st.enabled;
    if (st.enabled) {
        st.stats.reset();
        create_subscriptions(st, node);
    } else {
        destroy_subscriptions(st);
    }
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------
static void start_recording(RecorderState& rec, StreamState streams[3]) {
    if (rec.active) return;

    rec.rows.clear();
    rec.canvas_w = 0;
    rec.canvas_h = 0;

    for (int si = 0; si < 3; ++si) {
        if (!streams[si].enabled) continue;
        auto& s0 = streams[si].slots[0];
        if (s0.w == 0 || s0.h == 0) continue;

        RecordRow row;
        row.stream_idx = si;
        row.stereo = streams[si].stereo;
        row.left_w = s0.w;
        row.right_w = 0;
        row.row_h = s0.h;

        if (row.stereo) {
            auto& s1 = streams[si].slots[1];
            row.right_w = (s1.w > 0 && s1.h > 0) ? s1.w : s0.w;
            row.row_h = std::max(s0.h, (s1.h > 0) ? s1.h : s0.h);
            row.row_w = row.left_w + row.right_w;
        } else {
            row.row_w = row.left_w;
        }

        row.y_offset = rec.canvas_h;
        rec.rows.push_back(row);
        rec.canvas_h += row.row_h;
        if (row.row_w > rec.canvas_w) rec.canvas_w = row.row_w;
    }

    if (rec.rows.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("viewer"),
            "No streams with data — cannot start recording");
        return;
    }

    // yuv420p requires even dimensions
    if (rec.canvas_w % 2 != 0) rec.canvas_w += 1;
    if (rec.canvas_h % 2 != 0) rec.canvas_h += 1;

    // Timestamped filename
    auto now_sys = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now_sys);
    struct tm tm_buf;
    localtime_r(&tt, &tm_buf);
    char fname[64];
    snprintf(fname, sizeof(fname), "viewer_%04d%02d%02d_%02d%02d%02d.mp4",
        tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
        tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
    rec.filename = fname;

    // Allocate zero-initialized frame buffer
    rec.frame_buf.assign(
        static_cast<size_t>(rec.canvas_w) * rec.canvas_h * 3, 0);

    // Open ffmpeg pipe
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d "
        "-framerate 30 -i pipe:0 -c:v libx264 -preset fast -crf 18 "
        "-pix_fmt yuv420p -movflags +faststart '%s' 2>/dev/null",
        rec.canvas_w, rec.canvas_h, rec.filename.c_str());

    rec.pipe = popen(cmd, "w");
    if (!rec.pipe) {
        RCLCPP_ERROR(rclcpp::get_logger("viewer"),
            "Failed to start ffmpeg for recording");
        rec.frame_buf.clear();
        rec.rows.clear();
        return;
    }

    rec.active = true;
    rec.frame_count = 0;
    rec.start_time = std::chrono::steady_clock::now();
    rec.last_frame_time = rec.start_time;
    RCLCPP_INFO(rclcpp::get_logger("viewer"),
        "Recording started: %s (%dx%d)", fname, rec.canvas_w, rec.canvas_h);
}

static void stop_recording(RecorderState& rec) {
    if (!rec.active) return;
    rec.active = false;
    if (rec.pipe) {
        pclose(rec.pipe);
        rec.pipe = nullptr;
    }
    RCLCPP_INFO(rclcpp::get_logger("viewer"),
        "Recording stopped: %s (%d frames)", rec.filename.c_str(), rec.frame_count);
    rec.frame_buf.clear();
    rec.frame_buf.shrink_to_fit();
    rec.rows.clear();
}

static void compose_and_write_frame(RecorderState& rec, StreamState streams[3]) {
    if (!rec.active) return;

    // 30fps pacing
    auto now = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(
        now - rec.last_frame_time).count();
    if (elapsed_ms < 33.0) return;
    rec.last_frame_time = now;

    // Clear to black
    std::memset(rec.frame_buf.data(), 0, rec.frame_buf.size());

    for (auto& row : rec.rows) {
        StreamState& st = streams[row.stream_idx];
        uint8_t* row_ptr = rec.frame_buf.data() +
            static_cast<size_t>(row.y_offset) * rec.canvas_w * 3;

        if (row.stereo) {
            // Left image at x=0
            auto& left = st.slots[0];
            if (left.w > 0 && left.h > 0) {
                int copy_w = std::min(left.w, row.left_w);
                int copy_h = std::min(left.h, row.row_h);
                for (int y = 0; y < copy_h; ++y) {
                    std::memcpy(
                        row_ptr + y * rec.canvas_w * 3,
                        left.buffer.data() + y * left.w * 3,
                        static_cast<size_t>(copy_w) * 3);
                }
            }
            // Right image flush after left (at x=left_w)
            auto& right = st.slots[1];
            if (right.w > 0 && right.h > 0) {
                int copy_w = std::min(right.w, row.right_w);
                int copy_h = std::min(right.h, row.row_h);
                for (int y = 0; y < copy_h; ++y) {
                    std::memcpy(
                        row_ptr + y * rec.canvas_w * 3 + row.left_w * 3,
                        right.buffer.data() + y * right.w * 3,
                        static_cast<size_t>(copy_w) * 3);
                }
            }
        } else {
            // Mono: centered horizontally in canvas
            auto& slot = st.slots[0];
            if (slot.w > 0 && slot.h > 0) {
                int copy_w = std::min(slot.w, row.left_w);
                int copy_h = std::min(slot.h, row.row_h);
                int x_off = (rec.canvas_w - row.left_w) / 2;
                for (int y = 0; y < copy_h; ++y) {
                    std::memcpy(
                        row_ptr + y * rec.canvas_w * 3 + x_off * 3,
                        slot.buffer.data() + y * slot.w * 3,
                        static_cast<size_t>(copy_w) * 3);
                }
            }
        }
    }

    // Write frame to pipe
    size_t frame_size = rec.frame_buf.size();
    size_t written = fwrite(rec.frame_buf.data(), 1, frame_size, rec.pipe);
    if (written != frame_size) {
        RCLCPP_WARN(rclcpp::get_logger("viewer"),
            "Recording broken pipe — stopping");
        rec.active = false;
        pclose(rec.pipe);
        rec.pipe = nullptr;
        rec.frame_buf.clear();
        rec.rows.clear();
        return;
    }
    rec.frame_count++;
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

// Draw a textured quad filling ImGui content region, maintaining aspect ratio
static void draw_image_slot(const ImageSlot& slot, ImVec2 avail) {
    if (!slot.texture || slot.w == 0 || slot.h == 0) {
        ImGui::Dummy(avail);
        return;
    }
    float scale = std::min(avail.x / slot.w, avail.y / slot.h);
    ImVec2 size(slot.w * scale, slot.h * scale);
    // Center horizontally
    float pad = (avail.x - size.x) * 0.5f;
    if (pad > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + pad);
    ImGui::Image(static_cast<ImTextureID>(slot.texture), size);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    signal(SIGPIPE, SIG_IGN);
    auto node = std::make_shared<rclcpp::Node>("stream_viewer");

    // --- SDL2 + OpenGL init ---
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        RCLCPP_FATAL(node->get_logger(), "SDL_Init failed: %s", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    int init_w = 1300, init_h = 900;
    SDL_Window* window = SDL_CreateWindow(
        "Fusion Viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        init_w, init_h, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window) {
        RCLCPP_FATAL(node->get_logger(), "SDL_CreateWindow failed: %s", SDL_GetError());
        return 1;
    }

    SDL_GLContext gl_ctx = SDL_GL_CreateContext(window);
    SDL_GL_SetSwapInterval(1); // vsync

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        RCLCPP_FATAL(node->get_logger(), "glewInit failed");
        return 1;
    }

    // --- ImGui init ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForOpenGL(window, gl_ctx);
    ImGui_ImplOpenGL3_Init("#version 330");

    // --- Stream setup ---
    // Order: IR (index 0), Fused (index 1), RGB (index 2)
    StreamState streams[3];
    const char* stream_names[] = {"IR", "Fused", "RGB"};

    // IR: stereo mono8
    streams[0].stereo = true;
    streams[0].topics = {"/camera/infra1/image_rect_raw", "/camera/infra2/image_rect_raw"};
    streams[0].encodings = {"mono8", "mono8"};

    // Fused: stereo rgb8
    streams[1].stereo = true;
    streams[1].topics = {"/camera/fused/left", "/camera/fused/right"};
    streams[1].encodings = {"rgb8", "rgb8"};

    // RGB: single bgr8/rgb8
    streams[2].stereo = false;
    streams[2].topics = {"/camera/color/image_raw"};
    streams[2].encodings = {"bgr8"};

    // Create initial subscriptions
    for (auto& st : streams) {
        create_subscriptions(st, node.get());
    }

    bool flip_lr = false;
    bool quit = false;
    RecorderState rec;
    auto last_log = std::chrono::steady_clock::now();

    // --- Main loop ---
    while (!quit && rclcpp::ok()) {
        // Log stats every 2 seconds
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_log).count() >= 2.0) {
            last_log = now;
            for (int si = 0; si < 3; ++si) {
                if (!streams[si].enabled) continue;
                auto& s = streams[si].stats;
                RCLCPP_INFO(node->get_logger(), "%-5s  %4.0f fps  %5.1f ms  peak %5.1f ms",
                    stream_names[si], s.avg_fps, s.avg_ms, s.peak_ms);
            }
        }

        // Poll events
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL2_ProcessEvent(&ev);
            if (ev.type == SDL_QUIT) {
                quit = true;
            }
            if (ev.type == SDL_KEYDOWN && !io.WantCaptureKeyboard) {
                switch (ev.key.keysym.sym) {
                    case SDLK_1: toggle_stream(streams[1], node.get()); break; // Fused
                    case SDLK_2: toggle_stream(streams[2], node.get()); break; // RGB
                    case SDLK_3: toggle_stream(streams[0], node.get()); break; // IR
                    case SDLK_f: flip_lr = !flip_lr; break;
                    case SDLK_r:
                        if (rec.active) stop_recording(rec);
                        else start_recording(rec, streams);
                        break;
                    case SDLK_ESCAPE: quit = true; break;
                }
            }
        }

        // Process ROS callbacks (same thread — no mutex needed)
        rclcpp::spin_some(node);

        // Compose recording frame (reads slot buffers before texture upload)
        compose_and_write_frame(rec, streams);

        // Upload new texture data
        for (auto& st : streams) {
            if (!st.enabled) continue;
            for (auto& slot : st.slots) {
                upload_texture(slot);
            }
        }

        // --- Render ---
        int win_w, win_h;
        SDL_GetWindowSize(window, &win_w, &win_h);
        glViewport(0, 0, win_w, win_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // Count enabled streams for layout
        int enabled_count = 0;
        for (auto& st : streams) {
            if (st.enabled) ++enabled_count;
        }

        // Fullscreen image window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(win_w), static_cast<float>(win_h)));
        ImGui::Begin("##viewport", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        if (enabled_count > 0) {
            float row_h = ImGui::GetContentRegionAvail().y / enabled_count;

            for (int si = 0; si < 3; ++si) {
                StreamState& st = streams[si];
                if (!st.enabled) continue;

                if (st.stereo) {
                    int li = flip_lr ? 1 : 0;
                    int ri = flip_lr ? 0 : 1;
                    float half_w = ImGui::GetContentRegionAvail().x * 0.5f;
                    ImVec2 slot_size(half_w, row_h);

                    ImGui::BeginGroup();
                    draw_image_slot(st.slots[li], slot_size);
                    ImGui::EndGroup();
                    ImGui::SameLine();
                    ImGui::BeginGroup();
                    draw_image_slot(st.slots[ri], slot_size);
                    ImGui::EndGroup();
                } else {
                    ImVec2 full_size(ImGui::GetContentRegionAvail().x, row_h);
                    draw_image_slot(st.slots[0], full_size);
                }
            }
        }

        ImGui::End();

        // --- Overlay controls ---
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowBgAlpha(0.6f);
        ImGui::Begin("Controls", nullptr,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNav);

        for (int si = 0; si < 3; ++si) {
            // Map display order to key: IR=3, Fused=1, RGB=2
            int key = (si == 0) ? 3 : (si == 1) ? 1 : 2;
            char label[64];
            snprintf(label, sizeof(label), "[%d] %s", key, stream_names[si]);
            bool val = streams[si].enabled;
            if (ImGui::Checkbox(label, &val)) {
                toggle_stream(streams[si], node.get());
            }
            if (streams[si].enabled) {
                streams[si].stats.update();
                auto& s = streams[si].stats;
                ImGui::SameLine();
                ImGui::TextDisabled("%.0f fps  %.1f ms  peak %.1f ms",
                    s.avg_fps, s.avg_ms, s.peak_ms);
            }
        }
        ImGui::Separator();
        ImGui::Checkbox("[F] Flip L/R", &flip_lr);

        ImGui::Separator();
        if (!rec.active) {
            if (ImGui::Button("[R] Record")) {
                start_recording(rec, streams);
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.1f, 0.1f, 1.0f));
            if (ImGui::Button("[R] Stop")) {
                stop_recording(rec);
            }
            ImGui::PopStyleColor();
            ImGui::SameLine();
            double rec_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - rec.start_time).count();
            bool blink = static_cast<int>(rec_elapsed * 2) % 2 == 0;
            if (blink) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
                ImGui::Text("REC");
                ImGui::PopStyleColor();
            } else {
                ImGui::TextDisabled("REC");
            }
            ImGui::SameLine();
            int secs = static_cast<int>(rec_elapsed);
            ImGui::TextDisabled("%d:%02d  %s", secs / 60, secs % 60, rec.filename.c_str());
        }

        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);
    }

    // --- Cleanup ---
    stop_recording(rec);

    for (auto& st : streams) {
        destroy_subscriptions(st);
        for (auto& slot : st.slots) {
            if (slot.texture) glDeleteTextures(1, &slot.texture);
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();

    rclcpp::shutdown();
    return 0;
}
