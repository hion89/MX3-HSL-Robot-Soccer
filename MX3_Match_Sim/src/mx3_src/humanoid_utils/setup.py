from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'humanoid_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools', 'dynamixel-sdk', 'openpyxl'],
    zip_safe=True,
    maintainer='ros103',
    maintainer_email='ros103@todo.todo',
    description='Humanoid 공용 유틸리티 모음 (Dynamixel 일괄 설정 등)',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # ── Dynamixel 일괄 설정 도구 (XM540-W150-R / U2D2) ──
            'khuber_dxl_setup = humanoid_utils.khuber_dxl_setup:main',
            # ── Dynamixel 현재 설정 읽기/표시 ──
            'khuber_dxl_dump = humanoid_utils.khuber_dxl_dump:main',
            # ── IMU 캘리 (자이로 바이어스 + 마운팅 정렬, 브리지 토픽 경유) ──
            'khuber_imu_calib = humanoid_utils.khuber_imu_calib:main',
            # ── 팔 Goal PWM 상한 벤치 테스트 (낙상 시 감속기 보호값 탐색) ──
            'khuber_arm_pwm_test = humanoid_utils.khuber_arm_pwm_test:main',
            # ── 오도메트리 캘리브레이션 실험 로거 ──
            #   /joint_states 를 클럭으로 여러 토픽을 한 CSV 에 정렬 기록.
            #   사용처 : humanoid_bringup/experiment_odom.launch.py
            'experiment_logger_node = humanoid_utils.experiment_logger_node:main',
        ],
    },
    data_files=[
        # ament 패키지 인덱스 + package.xml
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 로봇별 설정 파일 (xlsx/csv) → share/humanoid_utils/config
        (os.path.join('share', package_name, 'config'),
         glob('config/*.xlsx') + glob('config/*.csv')),
    ],
)
