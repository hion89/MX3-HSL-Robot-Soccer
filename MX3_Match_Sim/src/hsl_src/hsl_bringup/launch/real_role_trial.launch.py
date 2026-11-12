"""Single real robot: existing sensors/world model plus motion and assigned-role behavior."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    hsl = get_package_share_directory('hsl_bringup')
    bringup = get_package_share_directory('humanoid_bringup')
    walking = os.path.join(hsl, 'models', 'walk_v0_6_13500.onnx')
    kicking = os.path.join(hsl, 'models', 'walkkick_v1_5b_22300.onnx')
    for path in (walking, kicking):
        if not os.path.isfile(path):
            raise RuntimeError('Model missing: ' + path)
    return LaunchDescription([
        DeclareLaunchArgument('assigned_role', default_value='attacker'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'control.launch.py')),
            launch_arguments={
                'robot': 'robo04', 'use_teleop': 'false', 'use_kick': 'true',
                'walk_actor_path': walking, 'kick_actor_path': kicking,
                'onnx_provider': 'cpu', 'policy_variant': 'v0_2',
                'gait_mode': 'stop', 'stop_delay_s': '0.0',
                'gravity_source': 'derive', 'fall_manager_imu_topic': '/imu/data_50hz',
            }.items()),
        Node(package='hsl_behavior', executable='behavior', name='hsl_behavior', output='screen',
             parameters=[os.path.join(hsl, 'config', 'hsl.yaml'), {
                 'field_profile': 'practice', 'player_number': 4, 'goalkeeper_number': 1,
                 'assigned_role': LaunchConfiguration('assigned_role'),
                 'default_game_state': 'initial', 'enable_kick': True,
                 'publish_game_stop_lock': True,
                 'vx_max': 0.20, 'vy_max': 0.15, 'wz_max': 0.50,
             }],
             # Local start/stop for this one-robot role trial. No GameState is
             # published to the localizer or the rest of the real robot.
             remappings=[('/game_state', '/hsl/role_trial/game_state')]),
    ])
