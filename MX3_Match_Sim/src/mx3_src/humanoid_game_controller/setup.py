import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'humanoid_game_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 폴더를 share 로 설치
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jace',
    maintainer_email='qkrguswo2946@khu.ac.kr',
    description='RoboCup HSL GameController 수신(struct 파싱) + 개인상태 송신(return) '
                '+ game_stop_lock 매핑',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            # GameController 수신 → /game_state
            'game_controller_receiver = '
            'humanoid_game_controller.game_controller_receiver_node:main',
            # 개인 상태(자세/공/넘어짐) → GameController(:3939) 송신
            'game_controller_return = '
            'humanoid_game_controller.game_controller_return_node:main',
            # /game_state → game_stop_lock
            'game_state_to_lock = '
            'humanoid_game_controller.game_state_to_lock_node:main',
            # 디버그/모니터링 — /game_state TUI 대시보드
            'game_state_monitor_tui = '
            'humanoid_game_controller.game_state_monitor_tui:main',
            # 디버그/모니터링 — /game_state 순서도 GUI (현재 상태 점등)
            'game_state_gui = '
            'humanoid_game_controller.game_state_gui:main',
        ],
    },
)
