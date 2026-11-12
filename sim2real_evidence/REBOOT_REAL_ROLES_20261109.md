# 재부팅 후 실제 역할 실행 — 2026-11-09

전제: robo04 한 대, Humble, /home/robo04/sim2real_ws, domain42/localhost-only.
오늘 사용자가 재부팅해 기존 터미널이 모두 꺼졌다고 알렸다. 실물 역할 묶음의 이전 적용/빌드 결과는 아직 보고되지 않았다.
이미 real_roles 묶음을 적용했고 REAL_ROLE_INTEGRATION_INSTALLED를 확인했다면 설치 단계는 생략한다.
재부팅만으로 소스/빌드/가중치를 다시 만들 필요는 없다.

## 최초 한 번의 역할 묶음 설치

Windows:

```powershell
cd C:\Users\jes\Desktop\isaac_ws\sim2real_evidence\deploy
scp .\real_roles_20261105.tar.gz .\real_roles_20261105.tar.gz.sha256 robo04@192.168.1.161:/home/robo04/incoming/
```

로봇8번:

```bash
(
set -e
cd /home/robo04/incoming
sha256sum -c real_roles_20261105.tar.gz.sha256
tar -xzf real_roles_20261105.tar.gz
bash real_roles_20261105/apply_and_build.sh
)
```

## 각 새 탭의 공통 환경

```bash
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
cd /home/robo04/sim2real_ws
mkdir -p runtime_logs
```

## 1: 하드웨어

```bash
ros2 launch humanoid_serial_bridge dynamixel_system.launch.py \
  port:=/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_3657367C3036-if00 \
  u2d2_port:=/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HKSR-if00-port0
```

## 2: D405와 카메라 TF

```bash
ros2 launch humanoid_perception perception_bringup.launch.py \
  enable_camera:=true enable_yolo:=false enable_ball_map:=false \
  rs_camera_profile:=848x480x30
```

## 3: YOLO

```bash
export PYTHONPATH="/home/robo04/sim2real_ws/.policy-python/numpy1:${PYTHONPATH:-}"
ros2 launch humanoid_perception yolo_3d.launch.py annotated:=false
```

## 4: 다리 오도메트리

```bash
ros2 run hsl_world_model leg_odometry --ros-args \
  --params-file /home/robo04/sim2real_ws/install/hsl_bringup/share/hsl_bringup/config/hsl.yaml \
  -p urdf_path:=/home/robo04/sim2real_ws/install/hsl_world_model/share/hsl_world_model/urdf/XM_robot04.urdf
```

## 5: 공 추정

```bash
ros2 run hsl_world_model ball_tracker --ros-args \
  --params-file /home/robo04/sim2real_ws/install/hsl_bringup/share/hsl_bringup/config/hsl.yaml \
  2>&1 | tee "runtime_logs/ball_tracker_$(date +%Y%m%d_%H%M%S).log"
```

## 6: 자기위치

```bash
ros2 run hsl_world_model self_localizer --ros-args \
  --params-file /home/robo04/sim2real_ws/install/hsl_bringup/share/hsl_bringup/config/hsl.yaml \
  -p field_profile:=practice -p player_number:=4 \
  2>&1 | tee "runtime_logs/self_localizer_$(date +%Y%m%d_%H%M%S).log"
```

## START 기립 후 8에서 초기 위치

1~6을 먼저 준비한다. 로봇을 지지한 상태에서 START로 기립시킨다.
이전과 같은 우리 골라인 왼쪽 두 T 사이, 상대 골대 방향이라면 다음 초기값을 쓴다.
START 이벤트는 자기위치 reset을 일으키므로 위치 입력은 START 이후에 한다.

```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {position: {x: -1.95, y: 0.7583333, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"
```

## 7: 실제 역할 제어

```bash
bash /home/robo04/incoming/real_roles_20261105/start.sh
```

이 창은 실제 보행·킥·목·낙상 처리와 역할 행동을 시작한다.
초기 상태는 INITIAL/gait stop이다. 정지 정책의 관절 목표는 실제 모터에 연결된다.
이전 start_hsl_behavior_shadow 스크립트와 전체 robot.launch.py는 여기서 추가 실행하지 않는다.

## 8: 실제 역할 명령

처음은 공격10초 후 자동 정지:

```bash
bash /home/robo04/incoming/real_roles_20261105/role.sh attacker --seconds 10
```

이후 원하는 역할을 하나씩 실행한다:

```bash
bash /home/robo04/incoming/real_roles_20261105/role.sh defender --seconds 10
```

```bash
bash /home/robo04/incoming/real_roles_20261105/role.sh goalkeeper --seconds 15
```

정지/상태:

```bash
bash /home/robo04/incoming/real_roles_20261105/role.sh stop
bash /home/robo04/incoming/real_roles_20261105/role.sh status
```

시간 옵션을 생략하면 stop 명령까지 계속 역할을 수행한다.
stop 및 자동 정지는 기립 토크를 유지한다. 모터를 쉬게 할 때는 먼저 로봇을 지지한 뒤 토크OFF/전원 종료한다.

진행 목표는 실제 공격 → 수비/골키퍼 전환 관찰이다. 정책 재검증을 반복하지 않는다.
