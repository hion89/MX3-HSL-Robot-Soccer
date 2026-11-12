"""robo04 hardware, camera, perception and world model in one launch."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    hsl = get_package_share_directory('hsl_bringup')
    perception = get_package_share_directory('humanoid_perception')
    serial = get_package_share_directory('humanoid_serial_bridge')
    world = get_package_share_directory('hsl_world_model')
    config = os.path.join(hsl, 'config', 'hsl.yaml')
    urdf = os.path.join(world, 'urdf', 'XM_robot04.urdf')
    numpy_path = '/home/robo04/sim2real_ws/.policy-python/numpy1'
    inherited_path = os.environ.get('PYTHONPATH', '')
    yolo_path = numpy_path + (os.pathsep + inherited_path if inherited_path else '')

    def include(directory, filename, arguments):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(directory, 'launch', filename)),
            launch_arguments=arguments.items())

    return LaunchDescription([
        GroupAction(actions=[include(serial, 'dynamixel_system.launch.py', {
            'port': '/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_3657367C3036-if00',
            'u2d2_port': '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HKSR-if00-port0',
        })]),
        GroupAction(actions=[include(perception, 'perception_bringup.launch.py', {
            'enable_camera': 'true', 'enable_yolo': 'false', 'enable_ball_map': 'false',
            'rs_camera_profile': '848x480x30',
        })]),
        GroupAction(actions=[
            # Only YOLO needs this NumPy override; keep the native hardware environment.
            SetEnvironmentVariable('PYTHONPATH', yolo_path),
            include(perception, 'yolo_3d.launch.py', {'annotated': 'false'}),
        ]),
        Node(package='hsl_world_model', executable='leg_odometry',
             name='hsl_leg_odometry', output='screen',
             parameters=[config, {'urdf_path': urdf}]),
        Node(package='hsl_world_model', executable='ball_tracker',
             name='hsl_ball_tracker', output='screen', parameters=[config]),
        Node(package='hsl_world_model', executable='self_localizer',
             name='hsl_self_localizer', output='screen',
             parameters=[config, {'field_profile': 'practice', 'player_number': 4}]),
    ])
