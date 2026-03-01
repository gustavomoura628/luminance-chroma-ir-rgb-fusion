"""ROS2 node for IR luma + RGB chroma fusion.

Subscribes to D435i infrared and color topics, fuses them via
FusionPipeline, and publishes colorized output.
"""

from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Image

from .fusion_pipeline import FusionPipeline


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")
        self._bridge = CvBridge()

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
        # Pipeline (one per eye — left uses infra1, right uses infra2)
        device = "cuda"
        self._pipeline_left = FusionPipeline(
            device=device,
            orb_features=orb_features,
            ema_alpha=ema_alpha,
            freeze_mode=freeze_mode,
            freeze_after=freeze_after,
        )
        self._pipeline_right = FusionPipeline(
            device=device,
            orb_features=orb_features,
            ema_alpha=ema_alpha,
            freeze_mode=freeze_mode,
            freeze_after=freeze_after,
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
        return SetParametersResult(successful=True)

    def _sync_callback(
        self, infra1_msg: Image, infra2_msg: Image, color_msg: Image,
    ) -> None:
        # Convert ROS images to OpenCV
        ir_left = self._bridge.imgmsg_to_cv2(infra1_msg, desired_encoding="mono8")
        ir_right = self._bridge.imgmsg_to_cv2(infra2_msg, desired_encoding="mono8")
        rgb_bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")

        # Resize RGB to match IR if needed
        ir_h, ir_w = ir_left.shape[:2]
        rgb_h, rgb_w = rgb_bgr.shape[:2]
        if (rgb_h, rgb_w) != (ir_h, ir_w):
            rgb_bgr = cv2.resize(rgb_bgr, (ir_w, ir_h))

        # Fuse left eye
        fused_left = self._pipeline_left.process(ir_left, rgb_bgr)
        left_msg = self._bridge.cv2_to_imgmsg(fused_left, encoding="bgr8")
        left_msg.header = infra1_msg.header
        self._pub_left.publish(left_msg)

        # Fuse right eye
        fused_right = self._pipeline_right.process(ir_right, rgb_bgr)
        right_msg = self._bridge.cv2_to_imgmsg(fused_right, encoding="bgr8")
        right_msg.header = infra2_msg.header
        self._pub_right.publish(right_msg)


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
