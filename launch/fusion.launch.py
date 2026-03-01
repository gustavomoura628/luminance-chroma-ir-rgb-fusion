import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("luminance_chroma_ir_rgb_fusion")
    default_params = os.path.join(pkg_dir, "config", "params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to the ROS2 parameters file",
        ),
        Node(
            package="luminance_chroma_ir_rgb_fusion",
            executable="fusion_node",
            name="fusion_node",
            parameters=[LaunchConfiguration("params_file")],
            output="screen",
        ),
    ])
