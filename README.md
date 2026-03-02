# luminance-chroma-ir-rgb-fusion

ROS2 node that produces **stereo color video** from an Intel RealSense D435i — hardware that only outputs stereo IR (grayscale) + monocular RGB.

## The problem

The D435i has three cameras: a stereo IR pair (left/right) for depth sensing, and a center RGB camera for color. If you're teleoperating a robot through a stereo head-mounted display — like the Unitree H1 — you need stereo color for the operator to see through the robot's eyes. But the D435i doesn't give you that: you get stereo in grayscale, or color from a single viewpoint. Normally you'd bolt on a second RGB camera and deal with extra calibration, weight, and wiring.

This node gets stereo color out of the hardware you already have, no extra cameras needed.

## How it works

Each frame, the node:

1. **Aligns** the RGB image to each IR viewpoint using ORB feature matching + RANSAC affine estimation
2. **Extracts chroma** (color information) from the warped RGB
3. **Replaces luma** (brightness) with the IR image — IR has better contrast and isn't affected by the projector pattern
4. **Outputs** two colorized images: one for the left eye, one for the right

The warp matrix is smoothed with an EMA filter and auto-freezes after convergence, so ORB only runs for the first few seconds. The color math runs on GPU via PyTorch.

The tradeoff with freezing: the warp matrix is a global affine transform, but the true RGB-to-IR mapping is depth-dependent. When ORB runs continuously, it acts as a crude depth adaptation — if a nearby object dominates the frame, the matcher naturally "focuses" on it and adjusts the warp accordingly. Once frozen, you lose this and the alignment is locked to whatever scene geometry was present during convergence. For mostly-static scenes (fixed camera, consistent depth range) freezing is fine. For dynamic scenes you may want `freeze_mode: "off"` — the tradeoff is ~3ms of extra latency per frame (6.9ms vs 3.6ms frozen at 640x480, cause not fully diagnosed).

## Topics

| Direction | Topic | Type | Default |
|-----------|-------|------|---------|
| Subscribe | infra1 | `sensor_msgs/Image` mono8 | `/camera/infra1/image_rect_raw` |
| Subscribe | infra2 | `sensor_msgs/Image` mono8 | `/camera/infra2/image_rect_raw` |
| Subscribe | color | `sensor_msgs/Image` rgb8 | `/camera/color/image_raw` |
| Publish | fused left | `sensor_msgs/Image` rgb8 | `/camera/fused/left` |
| Publish | fused right | `sensor_msgs/Image` rgb8 | `/camera/fused/right` |

All topic names are configurable via ROS params.

## Build and run

There are two implementations: a Python reference node and a C++ CUDA node. The C++ node is the active one (1.4-1.8ms/frame vs 3.7-4.1ms Python). There's also an SDL2/ImGui stream viewer.

### Prerequisites

- ROS 2 Humble
- NVIDIA GPU with CUDA support
- `nvidia-cuda-toolkit` (or CUDA toolkit installed to `/usr/local/cuda`)
- OpenCV 4.x with `features2d` and `calib3d` modules
- SDL2, GLEW, Dear ImGui (for viewer only)

### Build

```bash
# Symlink into your ROS 2 workspace (once)
cd ~/ros2_ws/src
ln -s ~/projects/luminance-chroma-ir-rgb-fusion .

# Build C++ CUDA fusion node
cd ~/ros2_ws
colcon build --packages-select ir_rgb_fusion_cuda

# Build stream viewer (optional)
colcon build --packages-select ir_rgb_fusion_viewer
```

### DDS transport fix (required for image topics)

FastDDS's default shared memory segment (~512KB) is too small for rgb8 images (~900KB at 640x480). Without this fix, subscribers drop ~20-30% of rgb8 frames. Set this env var **before launching any ROS 2 process** (bag play, fusion node, viewer, etc.):

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/cpp/config/fastdds_profile.xml
```

This forces SHM-only transport with a 4MB segment. See `PERF_IMPROVEMENTS.md` for details.

### Run

```bash
# Terminal 1 — play a rosbag (or run a RealSense driver)
ros2 bag play <path-to-bag> --loop

# Terminal 2 — fusion node
ros2 launch ir_rgb_fusion_cuda fusion.launch.py

# Terminal 3 — stream viewer (optional)
ros2 run ir_rgb_fusion_viewer viewer_node
```

Or run the fusion node directly with a custom params file:

```bash
ros2 run ir_rgb_fusion_cuda fusion_node --ros-args --params-file cpp/config/params.yaml
```

### View output

```bash
# Quick check
ros2 topic hz /camera/fused/left

# Image viewers
rqt_image_view /camera/fused/left
ros2 run ir_rgb_fusion_viewer viewer_node  # shows IR, fused, and RGB side by side
```

## Parameters

Configured via `cpp/config/params.yaml` or the launch file's `params_file` argument.

| Parameter | Type | Default | Dynamic | Description |
|-----------|------|---------|---------|-------------|
| `infra1_topic` | string | `/camera/infra1/image_rect_raw` | no | Left IR input topic |
| `infra2_topic` | string | `/camera/infra2/image_rect_raw` | no | Right IR input topic |
| `color_topic` | string | `/camera/color/image_raw` | no | RGB input topic |
| `fused_left_topic` | string | `/camera/fused/left` | no | Left fused output topic |
| `fused_right_topic` | string | `/camera/fused/right` | no | Right fused output topic |
| `sync_slop` | double | `0.005` | no | Max IR-RGB timestamp difference to consider RGB "fresh" (seconds) |
| `orb_features` | int | `1000` | no | Number of ORB features to detect |
| `ema_alpha` | double | `0.2` | no | EMA smoothing factor for warp matrix |
| `freeze_mode` | string | `"auto"` | **yes** | `"auto"`, `"on"`, or `"off"` |
| `freeze_after` | double | `5.0` | **yes** | Seconds before auto-freezing the warp |
| `crop` | bool | `false` | no | Crop output to valid warp region |

```bash
# Change freeze mode at runtime
ros2 param set /fusion_node freeze_mode "off"
ros2 param set /fusion_node freeze_after 10.0
```

## Dependencies

### C++ CUDA node (`ir_rgb_fusion_cuda`)
- ROS 2 Humble: `rclcpp`, `sensor_msgs`, `rcl_interfaces`
- CUDA toolkit (kernel compilation + runtime)
- OpenCV 4.x (`core`, `imgproc`, `features2d`, `calib3d`)

### Stream viewer (`ir_rgb_fusion_viewer`)
- ROS 2 Humble: `rclcpp`, `sensor_msgs`
- SDL2, GLEW, Dear ImGui (vendored or system)

### Python node (`luminance_chroma_ir_rgb_fusion`) — reference implementation
- ROS 2 Humble: `rclpy`, `sensor_msgs`, `message_filters`
- Python: OpenCV, NumPy, PyTorch (CUDA recommended)
