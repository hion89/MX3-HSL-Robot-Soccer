# 2026-11-05 로봇 재개 명령

모든 명령은 로봇의 MobaXterm SSH 터미널에서 실행한다.
계속 실행할 탭 7개와 조회용 탭 1개를 사용한다. 기동 완료된 탭은 유지한다.
이미 실행 중인 노드는 중복 실행하지 않는다. 재빌드/소스 덮어쓰기는 필요 없다.

## 모든 새 탭의 공통 환경

각 탭에서 아래 블록을 먼저 실행하고, 해당 탭의 명령을 실행한다.

```bash
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
cd /home/robo04/sim2real_ws
mkdir -p runtime_logs
```

## 탭 8: 최초 실행 목록 확인

```bash
ros2 node list --no-daemon --spin-time 5
ls -l /dev/serial/by-id/
```

재부팅 후 노드가 없는 상태를 기준으로 아래를 기동한다.
이미 목록에 있는 노드는 해당 탭에서 다시 시작하지 않는다.

## 탭 1: 하드웨어

어제 확인한 장치 고유 ID를 사용한다. tty 번호 변경의 영향을 줄인다.

```bash
ros2 launch humanoid_serial_bridge dynamixel_system.launch.py \
  port:=/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_3657367C3036-if00 \
  u2d2_port:=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HKSR-if00-port0
```

다른 창을 먼저 준비하고, 기립용 START는 아래 초기화 순서에서 누른다.

## 탭 2: 카메라와 TF

어제 별도 실행했던 RSP와 D405를 기존 perception_bringup으로 함께 시작한다.
소스의 동일 RealSense leaf launch와 동일 profile/align 설정이다.

```bash
ros2 launch humanoid_perception perception_bringup.launch.py \
  enable_camera:=true \
  enable_yolo:=false \
  enable_ball_map:=false \
  rs_camera_profile:=848x480x30
```

## 탭 3: YOLO

NumPy 1.26.4 경로는 이 탭에서만 적용한다.

```bash
export PYTHONPATH="/home/robo04/sim2real_ws/.policy-python/numpy1:${PYTHONPATH:-}"
ros2 launch humanoid_perception yolo_3d.launch.py annotated:=false
```

## 탭 4: 다리 오도메트리

```bash
ros2 run hsl_world_model leg_odometry --ros-args \
  --params-file /home/robo04/sim2real_ws/install/hsl_bringup/share/hsl_bringup/config/hsl.yaml \
  -p urdf_path:=/home/robo04/sim2real_ws/install/hsl_world_model/share/hsl_world_model/urdf/XM_robot04.urdf
```

## 탭 5: 공 추정 v3

빌드/격리 시험을 다시 수행하는 apply_and_build.sh 대신 설치된 실행 파일을 사용한다.

```bash
ros2 run hsl_world_model ball_tracker --ros-args \
  --params-file /home/robo04/sim2real_ws/install/hsl_bringup/share/hsl_bringup/config/hsl.yaml \
  2>&1 | tee "runtime_logs/ball_tracker_$(date +%Y%m%d_%H%M%S).log"
```

TF_CONFIG listener=parent-node-dedicated 마커가 v3를 식별한다.

## 탭 6: 자기위치 추정

```bash
ros2 run hsl_world_model self_localizer --ros-args \
  --params-file /home/robo04/sim2real_ws/install/hsl_bringup/share/hsl_bringup/config/hsl.yaml \
  -p field_profile:=practice \
  -p player_number:=4 \
  2>&1 | tee "runtime_logs/self_localizer_$(date +%Y%m%d_%H%M%S).log"
```

field 3.9x2.6, player 4를 확인한다.

## 기립 후 탭 8에서 위치 초기화

1~6번 창 준비 후 사용자가 로봇을 지지하고 START로 기립시킨다.
어제와 같은 우리 골라인의 왼쪽 두 T 사이, 상대 골대를 보는 방향에 배치한다.
START는 자기위치 reset도 일으키므로 START 기립을 끝낸 뒤 /initialpose를 입력한다.

```bash
ros2 topic pub --once \
  /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {position: {x: -1.95, y: 0.7583333, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"

timeout 12s ros2 topic echo \
  /hsl/ball hsl_interfaces/msg/BallEstimate \
  --once --no-daemon --qos-profile sensor_data

timeout 12s ros2 topic echo \
  /hsl/robot_pose hsl_interfaces/msg/RobotPoseEstimate \
  --once --no-daemon --qos-profile sensor_data
```

이 중간점은 실측값이 아니라 코드 지도에 따른 임시 초기값이다.

## 탭 7: 행동 판단 시험

공/위치 메시지 확인 후 기동한다. 어제 전달한 스크립트는 source/ROS 설정을 포함한다.

```bash
bash /home/robo04/incoming/start_hsl_behavior_shadow_20261104.sh
```

이 스크립트는 실제 센서 입력을 받고, 모든 행동 출력은 /hsl/shadow/로 보낸다.
킥 실행은 비활성화하고 실제 GameState는 발행하지 않는다.

## 탭 8: 행동 결과 조회

```bash
timeout 12s ros2 topic echo \
  /hsl/shadow/behavior_debug std_msgs/msg/String \
  --once --no-daemon --qos-profile sensor_data

timeout 12s ros2 topic echo \
  /hsl/shadow/cmd_vel_behavior geometry_msgs/msg/Twist \
  --once --no-daemon --qos-profile sensor_data
```

## 복구 후 다음 단계

오늘 첫 단계는 검증된 센서/공/자기위치/행동 시험 상태의 복구다.
보행 모델은 최종 hsl 모델과 실기 launch 기본 사본이 달라 명시적으로 선택해야 한다.
이 복구 명령에는 실제 보행 러너/전체 robot.launch.py 기동을 포함하지 않는다.
다음은 모델 파일/ONNXRuntime 확인 → 시험 출력으로 보행 러너 확인 → 명령 중재/정지 확인 → 실제 저속 동작이다.

모터 과열 우려가 다시 생기면 먼저 로봇을 받친 뒤 STOP으로 토크 해제를 확인하고 종료한다.
전원 OFF를 거쳤더라도 소스/빌드/모델 파일은 보존된다.
