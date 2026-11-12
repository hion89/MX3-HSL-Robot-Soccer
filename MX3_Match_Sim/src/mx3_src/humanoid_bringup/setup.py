from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'humanoid_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament 패키지 인덱스 마커
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch 파일 전체 설치
        (os.path.join('share', package_name, 'launch'),
         sorted(glob('launch/*.launch.py'))),
        # config(YAML) 전체 설치
        (os.path.join('share', package_name, 'config'),
         sorted(path for path in glob('config/*') if os.path.isfile(path))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robo1',
    maintainer_email='robo1@todo.todo',
    description='KHUBER 전체/부분 시스템 실행 launch 및 config 모음',
    license='TODO: License declaration',
    tests_require=['pytest'],
    # launch 전용 패키지라 실행 노드(console_scripts)는 없다.
    entry_points={
        'console_scripts': [],
    },
)
