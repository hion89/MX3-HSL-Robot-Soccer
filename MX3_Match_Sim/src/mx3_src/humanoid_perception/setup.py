import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'humanoid_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robo03',
    maintainer_email='robo03@todo.todo',
    description='Vision/perception nodes (YOLOv8 TensorRT detector etc.)',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 신규 3D Cartesian 검출 노드 — CameraInfo 받아 deproject 후 Detection3DArray publish
            'yolo_detector_3d_node = humanoid_perception.yolo_detector_3d_node:main',
            # 공 → map 투영 노드 — 검출(광학프레임)을 TF로 map 좌표 변환해 마커/토픽 발행
            'ball_map_projector_node = humanoid_perception.ball_map_projector_node:main',
            # 단거리 깊이 장애물 검출 — 거리구간별 바닥추정+density, /obstacle/avoid_hint 발행
            'depth_obstacle_detector_node = '
            'humanoid_perception.depth_obstacle_detector_node:main',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        # TensorRT engine 파일도 share 로 install
        # NOTE: .pt 는 install 안 함 (Jetson 추론엔 .engine 만 사용)
        (os.path.join('share', package_name, 'models'),
         glob('models/*.engine')),
        # URDF — robot_tf_bringup.launch.py 가 share/<pkg>/urdf/ 에서 자동으로 찾음
        (os.path.join('share', package_name, 'urdf'),
         glob('urdf/*.urdf')),
        # STL mesh — URDF 의 package://humanoid_perception/meshes/*.STL 가 참조.
        # TF 만 쓸 거면 없어도 되지만, 나중에 RViz 시각화 위해 미리 install.
        (os.path.join('share', package_name, 'meshes'),
         glob('meshes/*.STL') + glob('meshes/*.stl')),
    ],
)
