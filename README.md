# luminance-chroma-ir-rgb-fusion

ROS2 node that produces **stereo color video** from an Intel RealSense D435i — hardware that only outputs stereo IR (grayscale) + monocular RGB.

## The problem

The D435i has three cameras: a stereo IR pair (left/right) for depth sensing, and a center RGB camera for color. If you're teleoperating a robot through a stereo head-mounted display — like the Unitree H1 — you need stereo color for the operator to see through the robot's eyes. But the D435i doesn't give you that: you get stereo in grayscale, or color from a single viewpoint. Normally you'd bolt on a second RGB camera and deal with extra calibration, weight, and wiring.

This node gets stereo color out of the hardware you already have, no extra cameras needed.

## How it works

Each frame, the node:

1. **Aligns** the RGB image to each IR viewpoint using ORB feature matching + RANSAC affine estimation
2. **Extracts chroma** (color information) from the warped RGB
3. **Replaces luma** (brightness) with the IR image
4. **Outputs** two colorized images: one for the left eye, one for the right

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

The fusion node is implemented in C++ + CUDA (1.4-1.8ms/frame on 640x480 stereo).

### Prerequisites

- ROS 2 Humble
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit
- OpenCV 4.x with `features2d` and `calib3d` modules

### Build

```bash
# Symlink into your ROS 2 workspace
cd ~/ros2_ws/src
ln -s /path/to/luminance-chroma-ir-rgb-fusion .

# Build C++ CUDA fusion node
cd ~/ros2_ws
colcon build --packages-select ir_rgb_fusion_cuda
```

### DDS transport fix (required for image topics)

FastDDS's default shared memory segment (~512KB) is too small for rgb8 images (~900KB at 640x480). Without this fix, subscribers drop ~20-30% of rgb8 frames. Set this env var **before launching any ROS 2 process** (bag play, fusion node, etc.):

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/cpp/config/fastdds_profile.xml
```

This forces SHM-only transport with a 4MB segment.

### Run (live RealSense camera)

See [`realsense-setup/README.md`](realsense-setup/README.md) for install and
launch instructions. Same DDS environment as bag playback (see below) is required
in every terminal, then:

```bash
# Terminal 1 — camera driver (stereo IR + RGB, emitter off)
ros2 launch realsense-setup/d400_stereo_rgb.launch.py

# Terminal 2 — fusion node
ros2 launch ir_rgb_fusion_cuda fusion.launch.py
```

### Run (local bag playback)

Every terminal needs the same DDS environment. 

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/cpp/config/fastdds_profile.xml
export ROS_DOMAIN_ID=0
unset ROS_DISCOVERY_SERVER
unset ROS_SUPER_CLIENT
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

Then:

```bash
# Terminal 1 — play a rosbag
ros2 bag play <path-to-bag> --loop
```

```bash
# Terminal 2 — fusion node
ros2 launch ir_rgb_fusion_cuda fusion.launch.py
```

To use a different params file:

```bash
ros2 launch ir_rgb_fusion_cuda fusion.launch.py params_file:=/path/to/custom_params.yaml
```

### Run (real robot)

When the fusion node runs on a different machine from the RealSense driver (e.g. a
robot publishing on the network), you need to match the robot's DDS setup:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(pwd)/cpp/config/fastdds_profile.xml
export ROS_DOMAIN_ID=<robot's domain id>
export ROS_DISCOVERY_SERVER="<robot's discovery server address>"
export ROS_SUPER_CLIENT=True
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Verify you can see the camera topics
ros2 topic list | grep camera

# Launch
ros2 launch ir_rgb_fusion_cuda fusion.launch.py
```

All terminals must use the same environment variables.

### View output

In a terminal with the same DDS environment as the run sections above (same `FASTRTPS_DEFAULT_PROFILES_FILE`, `ROS_DOMAIN_ID`, and ROS sourcing — otherwise the subscriber can't discover the fusion node):

```bash
# Quick check
ros2 topic hz /camera/fused/left

# Image viewer (needs ros-humble-rqt-image-view)
ros2 run rqt_image_view rqt_image_view /camera/fused/left
```

## Parameters

The launch file loads `cpp/config/params.yaml` by default. Edit that file directly, or override with `params_file:=/path/to/custom.yaml`.

| Parameter | Type | Default | Dynamic | Description |
|-----------|------|---------|---------|-------------|
| `infra1_topic` | string | `/camera/infra1/image_rect_raw` | no | Left IR input topic |
| `infra2_topic` | string | `/camera/infra2/image_rect_raw` | no | Right IR input topic |
| `color_topic` | string | `/camera/color/image_raw` | no | RGB input topic |
| `fused_left_topic` | string | `/camera/fused/left` | no | Left fused output topic |
| `fused_right_topic` | string | `/camera/fused/right` | no | Right fused output topic |
| `sync_slop` | double | `0.005` | no | Max IR-RGB timestamp difference to consider RGB "fresh" (seconds) |
| `orb_features` | int | `1000` | no | Number of ORB features to detect |
| `ema_alpha` | double | `0.5` | no | EMA smoothing factor for warp matrix |
| `freeze_mode` | string | `"off"` | **yes** | `"auto"`, `"on"`, or `"off"` |
| `lock_rotation` | string | `"off"` | **yes** | Lock warp rotation after convergence: `"auto"`, `"on"`, or `"off"` |
| `freeze_after` | double | `5.0` | **yes** | Seconds before auto-freeze / auto rotation-lock |
| `crop` | bool | `false` | **yes** | Crop output to valid warp region |
| `min_inliers` | int | `20` | **yes** | Min RANSAC inlier count to accept a warp frame |
| `min_inlier_ratio` | double | `0.25` | **yes** | Min RANSAC inlier ratio to accept a warp frame |

```bash
# Change params at runtime
ros2 param set /fusion_node freeze_mode "off"
ros2 param set /fusion_node lock_rotation "auto"
ros2 param set /fusion_node freeze_after 10.0
```

## Dependencies

- ROS 2 Humble: `rclcpp`, `sensor_msgs`, `rcl_interfaces`
- NVIDIA CUDA Toolkit
- OpenCV 4.x (`core`, `imgproc`, `features2d`, `calib3d`)
