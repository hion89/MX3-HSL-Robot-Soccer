"""Observe one local robot. No motion nodes or control publishers."""
import os
import socket
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    here = os.path.dirname(os.path.realpath(__file__))
    profile = os.environ.get('HSL_VIEW_FIELD_PROFILE', 'practice')
    name = os.environ.get('ROBOT_NAME') or socket.gethostname().split('-')[0]
    return LaunchDescription([
        ExecuteProcess(cmd=['/usr/bin/python3', os.path.join(here, 'hsl_view.py'),
                            '--ros-args', '-p', 'field_profile:='+profile,
                            '-p', 'robot_name:='+name], output='screen'),
        Node(package='rviz2', executable='rviz2', name='hsl_rviz',
             arguments=['-d', os.path.join(here, 'hsl_view.rviz')], output='screen'),
    ])
