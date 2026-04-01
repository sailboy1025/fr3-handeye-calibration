from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='onr_handeye',
            executable='collect_samples',
            name='handeye_collector',
            output='screen',
            parameters=[{
                'image_topic': '/zed/zed_node/left/color/rect/image',
                'camera_info_topic': '/zed/zed_node/left/color/rect/camera_info',
                'robot_pose_topic': '/right/manip/measured/tool_int_pose',
                'tag_size_m': 0.05,
                'target_tag_id': -1,
                'samples_file': 'handeye_samples.json',
                'publish_debug_image': True,
                'debug_image_topic': '/onr_handeye/debug_image',
                'debug_axis_scale': 0.5,
            }],
        )
    ])
