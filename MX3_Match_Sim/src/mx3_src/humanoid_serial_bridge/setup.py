from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'humanoid_serial_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools', 'pyserial', 'rclpy', 'sensor_msgs', 'std_msgs'],
    zip_safe=True,
    maintainer='ros103',
    maintainer_email='ros103@todo.todo',
    description='Bridge between ROS 2 JointState msgs and Dynamixel-like serial protocol',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ── MCU/IMU 통신 ──
            'mcu_bridge_node = humanoid_serial_bridge.mcu_bridge_node:main',
            'imu_parser_node = humanoid_serial_bridge.imu_parser_node:main',

            # ── 보드 IO (DIP 등) ──
            'dip_status_node = humanoid_serial_bridge.dip_status_node:main',

            # ── 물리 버튼 입력 (estop, start) ──
            'button_input_node = humanoid_serial_bridge.button_input_node:main',

            # ── 목 제어 ──
            # 2026-08 이관: 목이 U2D2/MX-64 로 넘어가면서
            #   neck_state_parser_node (MCU 프레임 → JointState) 는 삭제되었다.
            #   목 상태는 body_motor_controller 의 SyncRead 가 /joint_states 로
            #   발행하고, deg 환산은 neck_control_node 가 담당한다.
            'neck_control_node = humanoid_serial_bridge.neck_control_node:main',
            'neck_scan_node = humanoid_serial_bridge.neck_scan_node:main',
            'neck_ball_tracker_node = humanoid_serial_bridge.tracking.neck_ball_tracker_node:main',

            # ── MCU 주변장치 제어  ──
            'led_control_node = humanoid_serial_bridge.led_control_node:main',

            # ── 전신 모터 제어 (U2D2) ──
            'body_motor_controller_node = humanoid_serial_bridge.body_motor_controller_node:main',

            # ── 정책/낙상 워크플로 ──
            'robot_init_client_node = humanoid_serial_bridge.robot_init_client_node:main',
            'standup_node = humanoid_serial_bridge.standup_node:main',
            'fall_manager_node = humanoid_serial_bridge.fall_manager_node:main',
            'policy_logger_node = humanoid_serial_bridge.policy_logger_node:main',

            # ── 디버그/모니터링 도구 ──
            'motor_monitor_tui = humanoid_serial_bridge.motor_monitor_tui:main',
            'tilt_monitor_tui = humanoid_serial_bridge.tilt_monitor_tui:main',
            'pose_slider_gui = humanoid_serial_bridge.pose_slider_gui:main',
            # ── 키프레임 시퀀스 편집 GUI (standup/kick phase 단위) ──
            #   pose_slider_gui 가 '자세 하나 → YAML 조각' 이라면,
            #   이건 '시퀀스 전체를 열어 phase 를 오가며 고치고 파일에 저장'.
            'keyframe_editor_gui = humanoid_serial_bridge.keyframe_editor_gui:main',
        ],
    },

    data_files=[
        # 반드시 패키지 인덱스와 package.xml
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
    ],
)
