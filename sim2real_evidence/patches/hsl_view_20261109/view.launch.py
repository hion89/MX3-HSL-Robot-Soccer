"""Add an observer and RViz to the already running robot; no control nodes."""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    here = os.path.dirname(os.path.realpath(__file__))
    return LaunchDescription([
        ExecuteProcess(cmd=['/usr/bin/python3', os.path.join(here, 'hsl_view.py'),
                            '--ros-args', '-p', 'field_profile:=practice'], output='screen'),
        Node(package='rviz2', executable='rviz2', name='hsl_rviz',
             arguments=['-d', os.path.join(here, 'hsl_view.rviz')], output='screen'),
    ])
