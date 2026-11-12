# 재부팅 후 기본 실행 통합 — robo04 한 대

기존 기본 실행 6개를 `base.sh` 한 명령으로 묶었다. 하드웨어, D405와 카메라 TF,
YOLO, 다리 오도메트리, 공 추정, 자기위치를 함께 실행한다.
기존에 확인한 설정과 공 추정 v3 코드는 유지한다. 기본 실행에 보행 정책이나 경기 행동은 포함하지 않는다.
보행·킥·경기 행동은 기립과 초기 위치 입력 후 별도 창에서 실행한다.

## 처음 한 번 설치

Windows PowerShell:

```powershell
cd C:\Users\jes\Desktop\isaac_ws\sim2real_evidence\deploy
scp .\real_robot_20261109.tar.gz .\real_robot_20261109.tar.gz.sha256 robo04@192.168.1.161:/home/robo04/incoming/
```

MobaXterm 로봇 터미널에서, 실행 중인 기존 제어·기본 실행이 없는 상태로:

```bash
(
set -e
cd /home/robo04/incoming
sha256sum -c real_robot_20261109.tar.gz.sha256
tar -xzf real_robot_20261109.tar.gz
bash real_robot_20261109/apply_and_build.sh
)
```

이 묶음은 이전 `real_roles_20261105`의 역할 연결 수정도 포함한다.
이전 역할 묶음을 적용했어도 다시 적용할 수 있다. 원본 또는 이미 적용한 파일의 해시만 허용하며,
수정 전 파일은 `sim2real_ws/patch_backups`에 보관한다.
가중치, 모터 맵·게인, 기존 공 추정 코드는 바꾸지 않는다.
로컬 Python 문법·Bash 문법·배포 해시 검사를 완료했다. 통합 launch의 실제 ROS 실행은 로봇에서 확인한다.

## 터미널 1 — 기본 기능 전체

```bash
bash /home/robo04/incoming/real_robot_20261109/base.sh
```

이 창을 유지한다. 로그는 `sim2real_ws/runtime_logs/base_날짜_시간.log`에 저장한다.
기존 하드웨어·카메라·YOLO·월드모델 개별 실행 명령을 중복해서 켜지 않는다.
base.sh를 두 번 실행하면 두 번째 실행은 잠금에 의해 종료된다.
처음 카메라가 연결되고 TF가 도착하기 전의 일시적인 대기 경고는 잠깐 나타날 수 있다.

로봇을 이전 위치에 두고 지지한 상태에서 물리 START 버튼으로 기립시킨다.
START가 자기위치를 초기화하므로 위치 입력은 반드시 그 이후에 한다.

## 터미널 3 — 초기 위치

이전과 같은 우리 골대 쪽 왼쪽 두 T 사이에서 상대 골대를 바라보는 위치일 때:

```bash
bash /home/robo04/incoming/real_robot_20261109/init_pose.sh
```

초기값은 이전과 같은 근사값 `x=-1.95, y=0.7583333, yaw=0`이다.
로봇을 다른 위치에 세웠다면 이 위치 스크립트를 그대로 사용하지 않는다.

## 터미널 2 — 실제 보행·킥·경기 전략

기립과 초기 위치 입력 후:

```bash
bash /home/robo04/incoming/real_robot_20261109/start.sh
```

이 창도 유지한다. 실제 관절 목표가 모터로 전달된다. 시작은 INITIAL/gait stop이며,
터미널 3에서 역할을 시작하면 경기 행동이 활성화된다.
보행은 전달받은 `walk_v0_6_13500.onnx`, 킥은 `walkkick_v1_5b_22300.onnx`를 사용한다.

## 터미널 3 — 역할 선택

먼저 공격 역할 10초 실행 후 자동 정지:

```bash
bash /home/robo04/incoming/real_robot_20261109/role.sh attacker --seconds 10
```

다음 역할은 원하는 것 하나씩 실행한다:

```bash
bash /home/robo04/incoming/real_robot_20261109/role.sh defender --seconds 10
```

```bash
bash /home/robo04/incoming/real_robot_20261109/role.sh goalkeeper --seconds 15
```

정지 및 상태:

```bash
bash /home/robo04/incoming/real_robot_20261109/role.sh stop
bash /home/robo04/incoming/real_robot_20261109/role.sh status
```

`--seconds`를 생략하면 stop 명령까지 계속 역할을 수행한다.
역할 정지는 기립 토크를 유지한다. 종료할 때는 역할 stop → 로봇 지지 → 물리 STOP으로
토크 OFF → 터미널 2와 1에서 Ctrl+C 순서로 한다. Ctrl+C만으로 토크 OFF를 보장하지 않는다.

남은 확인은 이 통합 실행의 로봇 구동과 실제 공격·수비·골키퍼 행동 관찰이다.
정책·센서·카메라 개별 검증을 처음부터 반복하지 않는다.
