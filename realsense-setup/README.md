# RealSense D400 Setup for Fusion Node

How to get a D400-series RealSense (D435i, D435, D455, etc.) publishing the
stereo IR + RGB streams the fusion node needs.

## Install (Ubuntu 22.04 / ROS 2 Humble)

```bash
sudo apt-get install -y \
    ros-humble-librealsense2 \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-camera-msgs \
    ros-humble-realsense2-description
```

### udev rules (one-time)

Required for non-root USB access. Without this the camera won't be detected:

```bash
sudo curl -o /etc/udev/rules.d/99-realsense-libusb.rules \
    https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Unplug and replug the camera after installing the rules.

## Verify the camera

```bash
# Check USB connection
lsusb | grep Intel
# Should show something like: Intel Corp. Intel(R) RealSense(TM) Depth Camera 435i

# Check device nodes exist
ls /dev/video*
```

## Launch for the fusion node

Set this in **every terminal** before running anything. This configures
FastDDS shared memory transport and ensures all processes are on the same
DDS domain:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(realpath cpp/config/fastdds_profile.xml)
export ROS_DOMAIN_ID=0
unset ROS_DISCOVERY_SERVER
unset ROS_SUPER_CLIENT
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

> If your `.bashrc` sets `ROS_DOMAIN_ID` or `ROS_DISCOVERY_SERVER` (e.g. for
> connecting to a robot on the network), you **must** override them as above.
> Mismatched domains = processes can't discover each other and the fusion node
> receives zero frames.

The included launch file enables stereo IR (Y8) + RGB streams and disables
depth, IMU, TF, and the IR projector (projector dots confuse ORB matching):

```bash
ros2 launch realsense-setup/d400_stereo_rgb.launch.py
```

This publishes:
- `/d400/infra1/image_rect_raw` — left IR (mono8, 640x480 @ 30fps)
- `/d400/infra2/image_rect_raw` — right IR (mono8, 640x480 @ 30fps)
- `/d400/color/image_raw` — RGB (rgb8, 640x480 @ 30fps)

Which matches the default topics in `cpp/config/params.yaml`.

The launch file also disables the IR emitter — the projector dot pattern
confuses ORB feature matching. Depth streaming is disabled since we don't
need it.

> **USB 3.0 recommended.** The D435i will work on USB 2.x but warns about
> reduced performance and may drop frames at full resolution/fps.

### Custom resolution

```bash
ros2 launch realsense-setup/d400_stereo_rgb.launch.py \
    rgb_resolution:=1280,720,15 \
    ir_resolution:=1280,720,15
```

## Full pipeline (3 terminals)

Set the DDS environment in **every terminal**:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$(realpath cpp/config/fastdds_profile.xml)
export ROS_DOMAIN_ID=0
unset ROS_DISCOVERY_SERVER
unset ROS_SUPER_CLIENT
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

> **Important:** If your `.bashrc` sets `ROS_DOMAIN_ID` or `ROS_DISCOVERY_SERVER`
> for a robot, you must override them as shown above. Mismatched DDS domains =
> processes can't discover each other.

Then:

```bash
# Terminal 1 — camera driver
ros2 launch realsense-setup/d400_stereo_rgb.launch.py

# Terminal 2 — fusion node
ros2 launch ir_rgb_fusion_cuda fusion.launch.py

# Terminal 3 — viewer (optional)
ros2 run ir_rgb_fusion_viewer viewer_node
```

## Quick reference (copy-paste)

```bash
# -- Run in EVERY terminal --
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/gus/projects/luminance-chroma-ir-rgb-fusion/cpp/config/fastdds_profile.xml ROS_DOMAIN_ID=0 DISPLAY=:0; unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT; source /opt/ros/humble/setup.bash; source ~/ros2_ws/install/setup.bash

# Terminal 1 — camera driver
ros2 launch /home/gus/projects/luminance-chroma-ir-rgb-fusion/realsense-setup/d400_stereo_rgb.launch.py

# Terminal 2 — fusion node (builds first)
cd ~/ros2_ws && colcon build --packages-select ir_rgb_fusion_cuda && source install/setup.bash && ros2 launch ir_rgb_fusion_cuda fusion.launch.py

# Terminal 3 — viewer
ros2 run ir_rgb_fusion_viewer viewer_node
```

## Troubleshooting

### Topics show up under `/d400/d400/...` (doubled namespace)
- This means `camera_namespace` and `camera_name` are both set to `d400`
- The included launch file sets `camera_namespace` to empty and `camera_name`
  to `d400`, giving the correct `/d400/infra1/...` paths

### Camera stuck / "Frames didn't arrived within 5 seconds"
- This happens when the camera gets into a bad state (e.g. after a failed depth
  stream on USB 2.0). Unplugging and replugging often isn't enough
- **Software USB reset** (usually fixes it):
  ```bash
  sudo bash -c 'for dev in /sys/bus/usb/devices/*/product; do
    if grep -qi "RealSense" "$dev" 2>/dev/null; then
      dir=$(dirname "$dev")
      echo "Resetting $(cat $dev)"
      echo 0 > "$dir/authorized"; sleep 2; echo 1 > "$dir/authorized"
    fi
  done'
  ```
- If that doesn't work, physically unplug for 10+ seconds

### Camera not detected
- Check `lsusb | grep Intel` — if empty, try a different USB port (prefer USB 3.0)
- Make sure udev rules are installed (see above)
- Try `initial_reset:=true` in the launch to force a hardware reset

### Streams not starting / timeout
- **Depth requires USB 3.0.** Stereo IR + RGB works on USB 2.0, but adding depth
  exceeds USB 2.0 bandwidth and all streams will timeout
- Check your USB speed: `cat /sys/bus/usb/devices/*/speed` for the RealSense device.
  480 = USB 2.0, 5000 = USB 3.0
- Check `dmesg | tail -20` for USB errors after plugging in

### IR images show projector dots
- The launch file disables the emitter (`depth_module.emitter_enabled: 0`)
- If you still see dots, another RealSense nearby may have its emitter on
- With emitter off, depth stream won't work (that's fine — we don't use it)

### Frame drops / missing topics
- Make sure all terminals use the same `FASTRTPS_DEFAULT_PROFILES_FILE`
- Make sure all terminals use the same `ROS_DOMAIN_ID`
- Try `ros2 topic hz /d400/color/image_raw` to check publish rate
