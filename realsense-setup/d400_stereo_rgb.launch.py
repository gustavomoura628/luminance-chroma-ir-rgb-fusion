"""Launch a D400-series RealSense with stereo IR + RGB for the fusion node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import launch_ros.actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera_name', default_value='d400'),
        DeclareLaunchArgument('camera_namespace', default_value=''),
        DeclareLaunchArgument('fps', default_value='30'),
        DeclareLaunchArgument('rgb_resolution', default_value='640,480,30',
                              description='width,height,fps for color stream'),
        DeclareLaunchArgument('ir_resolution', default_value='640,480,30',
                              description='width,height,fps for infra streams'),
        DeclareLaunchArgument('enable_depth', default_value='false',
                              description='Enable depth stream (requires USB 3.0 at 30fps, or use 15fps on USB 2.0)'),

        launch_ros.actions.Node(
            package='realsense2_camera',
            namespace=LaunchConfiguration('camera_namespace'),
            name=LaunchConfiguration('camera_name'),
            executable='realsense2_camera_node',
            output='screen',
            parameters=[{
                # Enable only the streams we need
                'enable_color': True,
                'enable_infra1': True,
                'enable_infra2': True,
                'enable_depth': LaunchConfiguration('enable_depth'),
                'enable_gyro': False,
                'enable_accel': False,

                # Stream profiles: "width,height,fps"
                'rgb_camera.color_profile': LaunchConfiguration('rgb_resolution'),
                'depth_module.infra_profile': LaunchConfiguration('ir_resolution'),

                # Formats
                'rgb_camera.color_format': 'RGB8',
                'depth_module.infra1_format': 'Y8',
                'depth_module.infra2_format': 'Y8',

                # Don't publish TF (we don't need it)
                'publish_tf': False,

                # Emitter off — IR projector dots confuse ORB matching
                'depth_module.emitter_enabled': 0,
            }],
            emulate_tty=True,
        ),
    ])
