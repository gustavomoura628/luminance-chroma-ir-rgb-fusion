"""ROS2 node for IR luma + RGB chroma fusion.

Subscribes to D435i infrared and color topics, fuses them via
FusionPipeline, and publishes colorized output.
"""

from __future__ import annotations

import array
import time

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Image

from .fusion_pipeline import FusionPipeline


def _ms(ns: int) -> str:
    return f"{ns / 1e6:.1f}"


def _fmt_pipeline(label: str, total_ns: int, timings: dict[str, int]) -> str:
    parts = [f"{k}={_ms(v)}" for k, v in timings.items()]
    return f"{label}={_ms(total_ns)}({' '.join(parts)})"


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")

        # Declare parameters
        self.declare_parameter("infra1_topic", "/camera/infra1/image_rect_raw")
        self.declare_parameter("infra2_topic", "/camera/infra2/image_rect_raw")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("fused_left_topic", "/camera/fused/left")
        self.declare_parameter("fused_right_topic", "/camera/fused/right")
        self.declare_parameter("sync_slop", 0.02)
        self.declare_parameter("orb_features", 1000)
        self.declare_parameter("ema_alpha", 0.2)
        self.declare_parameter(
            "freeze_mode", "auto",
            ParameterDescriptor(
                description='Warp freeze mode: "auto", "on", or "off"',
                type=ParameterType.PARAMETER_STRING,
            ),
        )
        self.declare_parameter(
            "freeze_after", 5.0,
            ParameterDescriptor(
                description="Seconds of EMA before auto-freezing (auto mode only)",
                type=ParameterType.PARAMETER_DOUBLE,
            ),
        )
        self.declare_parameter(
            "crop", False,
            ParameterDescriptor(
                description="Crop to valid overlap region (true) or full IR frame (false)",
                type=ParameterType.PARAMETER_BOOL,
            ),
        )
        # Read initial values
        infra1_topic = self.get_parameter("infra1_topic").value
        infra2_topic = self.get_parameter("infra2_topic").value
        color_topic = self.get_parameter("color_topic").value
        fused_left_topic = self.get_parameter("fused_left_topic").value
        fused_right_topic = self.get_parameter("fused_right_topic").value
        sync_slop = self.get_parameter("sync_slop").value
        orb_features = self.get_parameter("orb_features").value
        ema_alpha = self.get_parameter("ema_alpha").value
        freeze_mode = self.get_parameter("freeze_mode").value
        freeze_after = self.get_parameter("freeze_after").value
        crop = self.get_parameter("crop").value
        # Pipeline (one per eye — left uses infra1, right uses infra2)
        device = "cuda"
        self._pipeline_left = FusionPipeline(
            device=device,
            orb_features=orb_features,
            ema_alpha=ema_alpha,
            freeze_mode=freeze_mode,
            freeze_after=freeze_after,
            crop=crop,
        )
        self._pipeline_right = FusionPipeline(
            device=device,
            orb_features=orb_features,
            ema_alpha=ema_alpha,
            freeze_mode=freeze_mode,
            freeze_after=freeze_after,
            crop=crop,
        )

        # Dynamic parameter callback
        self.add_on_set_parameters_callback(self._on_param_change)

        # Subscribers via message_filters
        self._sub_infra1 = Subscriber(self, Image, infra1_topic)
        self._sub_infra2 = Subscriber(self, Image, infra2_topic)
        self._sub_color = Subscriber(self, Image, color_topic)

        self._sync = ApproximateTimeSynchronizer(
            [self._sub_infra1, self._sub_infra2, self._sub_color],
            queue_size=10,
            slop=sync_slop,
        )
        self._sync.registerCallback(self._sync_callback)

        # Publishers
        self._pub_left = self.create_publisher(Image, fused_left_topic, 10)
        self._pub_right = self.create_publisher(Image, fused_right_topic, 10)

        self.get_logger().info(
            f"Fusion node started — subscribing to {infra1_topic}, "
            f"{infra2_topic}, {color_topic}"
        )

    def _on_param_change(self, params: list) -> SetParametersResult:
        for param in params:
            if param.name == "freeze_mode":
                if param.value not in ("auto", "on", "off"):
                    return SetParametersResult(
                        successful=False,
                        reason=f'Invalid freeze_mode: "{param.value}"',
                    )
                self._pipeline_left.set_freeze_mode(param.value)
                self._pipeline_right.set_freeze_mode(param.value)
                self.get_logger().info(f"freeze_mode -> {param.value}")
            elif param.name == "freeze_after":
                self._pipeline_left.set_freeze_after(param.value)
                self._pipeline_right.set_freeze_after(param.value)
                self.get_logger().info(f"freeze_after -> {param.value}")
            elif param.name == "crop":
                self._pipeline_left.crop = param.value
                self._pipeline_right.crop = param.value
                self.get_logger().info(f"crop -> {param.value}")
        return SetParametersResult(successful=True)

    def _sync_callback(
        self, infra1_msg: Image, infra2_msg: Image, color_msg: Image,
    ) -> None:
        _t = time.perf_counter_ns
        t0 = _t()

        # Convert ROS images to numpy arrays
        ir_left = np.frombuffer(infra1_msg.data, dtype=np.uint8).reshape(
            infra1_msg.height, infra1_msg.width,
        )
        ir_right = np.frombuffer(infra2_msg.data, dtype=np.uint8).reshape(
            infra2_msg.height, infra2_msg.width,
        )
        rgb = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(
            color_msg.height, color_msg.width, 3,
        )
        if color_msg.encoding == "bgr8":
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        t1 = _t()

        # Resize RGB to match IR if needed
        ir_h, ir_w = ir_left.shape[:2]
        rgb_h, rgb_w = rgb.shape[:2]
        if (rgb_h, rgb_w) != (ir_h, ir_w):
            rgb = cv2.resize(rgb, (ir_w, ir_h))
        t2 = _t()

        # Fuse both eyes
        fused_left = self._pipeline_left.process(ir_left, rgb)
        t3 = _t()
        fused_right = self._pipeline_right.process(ir_right, rgb)
        t4 = _t()

        # Publish
        self._pub_left.publish(self._make_img_msg(fused_left, infra1_msg.header))
        t5 = _t()
        self._pub_right.publish(self._make_img_msg(fused_right, infra2_msg.header))
        t6 = _t()

        self.get_logger().info(
            f"{_ms(t6 - t0)}ms/frame | "
            f"decode={_ms(t1 - t0)} resize={_ms(t2 - t1)} "
            f"{_fmt_pipeline('L', t3 - t2, self._pipeline_left.timings)} "
            f"{_fmt_pipeline('R', t4 - t3, self._pipeline_right.timings)} "
            f"pub_L={_ms(t5 - t4)} pub_R={_ms(t6 - t5)}",
            throttle_duration_sec=1.0,
        )


    @staticmethod
    def _make_img_msg(img: np.ndarray, header) -> Image:
        msg = Image()
        msg.header = header
        msg.height, msg.width = img.shape[:2]
        msg.encoding = "rgb8"
        msg.step = msg.width * 3
        # array.array skips rclpy's per-byte validation (126ms -> 0ms at 640x480)
        msg.data = array.array("B", img.tobytes())
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
