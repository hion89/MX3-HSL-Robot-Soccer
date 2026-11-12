import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'humanoid_localization'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # 빈 맵(yaml + pgm) — Nav2 map_server 용
        (os.path.join('share', package_name, 'config', 'map'),
            glob('config/map/*')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='하푸스',
    maintainer_email='you@example.com',
    description='KHUBER 위치추정 패키지: 보행 오도메트리, 마커 localization 등',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 실행파일이름 = 패키지.모듈:진입함수
            'walk_odometry_node = humanoid_localization.walk_odometry_node:main',
            'field_lines_node = humanoid_localization.field_lines_node:main',
            'nav_cmd_bridge_node = humanoid_localization.nav_cmd_bridge_node:main',
            # 비전 로컬라이제이션(vision 서브패키지): D405 3D 검출 → 지도 절대 pose 측정
            'field_landmark_localizer_node = '
            'humanoid_localization.vision.field_landmark_localizer_node:main',
            # SW1(좌/우) + 실행버튼 → /initialpose 재진입 시드
            'relocalize_button_node = '
            'humanoid_localization.vision.relocalize_button_node:main',
            # Phase 2: map 공 위치 → Nav2 goal 로 공 근처까지 접근
            'ball_follow_nav_node = humanoid_localization.ball_follow_nav_node:main',
            # Phase 3: 비전 서보 — 로봇·공·골대 일직선 정렬 후 킥
            'visual_servo_kick_node = '
            'humanoid_localization.visual_servo_kick_node:main',
            # 제네릭 go-to-pose 보행 컨트롤러(+회피 합성) — Nav2 드롭인 대체
            'pose_goal_controller_node = '
            'humanoid_localization.pose_goal_controller_node:main',
        ],
    },
)