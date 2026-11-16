# MX3 실물 로봇 축구 프로젝트 — AI 모델·채팅 인수인계

작성일: 2026-11-16, 한국 시간. 이 문서는 다른 AI 모델이나 새 채팅에 그대로 전달할 작업 문맥이다. 실제 장비에 새로 접속해 상태를 조회한 문서는 아니다. 장비 정보는 과거 사용자 출력/기록, 소스 정보는 로컬 파일을 기준으로 한다.

## 1. 새 AI에게 먼저 전달할 요청

> 이 문서와 저장소 AGENTS.md를 먼저 읽고 프로젝트를 이어서 진행해 주세요. 한국어로 답하고, 현재 로컬 코드·GitHub·실물 설치 상태를 구분해 주세요. 모르는 접속 정보나 최신 실물 상태를 추측하지 마세요. 변경 전에는 텔레옵·자율 경기·시뮬 중 영향 범위와 예상되는 부작용을 설명하고, 관련 작업과 검증이 끝날 때마다 새 GitHub 저장소에 커밋·푸시해 주세요. 기존에 동작하는 MX3 정책을 기준으로 진행하며, 예전 자체 RL_walk 경로로 임의 전환하지 마세요. 최신 지도/초기 위치의 실물 반영 여부부터 확인하면 됩니다.

이 문서의 오래된 사실은 이후 실제 사용자 출력으로 갱신한다. 새 채팅을 시작했다는 이유로 장비가 재부팅됐거나 모터 토크가 꺼졌다고 가정하지 않는다.

## 2. 프로젝트 목적과 현재 범위

- 최종 목표: 실제 로봇 4대4 모의 축구 경기. 공격수·수비수·골키퍼의 역할 분담, 공 탐색·접근·킥, 위치 추정과 팀 협동을 확인한다.
- 현재 실제 시험 대상: **robo04 한 대**. 역할을 바꿔가며 시험한다. 여러 대의 팀 통신/4대4 실물 성공은 아직 검증되지 않았다.
- 사용자가 제공한 `MX3_Match_Sim`의 `mx3_src + hsl_src`가 실물 적용 기준이다.
- B-Human 방식 상위 기능을 MX3 하드웨어/강화학습 정책 위에 통합했다. B-Human 전체 코드를 그대로 실행하거나 B-Human 보행 정책을 사용하는 구조가 아니다.
- 사용자는 이전 자체 학습 모델의 실물 적용 실패 후 제공된 정책의 실물 운용을 우선했다. 최근에는 잔디 킥 문제 때문에 제공된 학습 코드로 재학습하는 계획도 요청했다. **계획 분석까지만 했고 새 학습은 시작하지 않았다.**

## 3. 장비·접속·서버 정보

### Windows 개발 PC

- 사용자 디렉터리: `C:/Users/jes`.
- 주 작업 공간: `C:/Users/jes/Desktop/isaac_ws`.
- 셸: Windows PowerShell. Ubuntu의 `source`, `export`, Bash 중괄호 문법을 여기에 바로 붙여넣으면 실패한다.
- 로봇 터미널: MobaXterm SSH. RViz는 MobaXterm X server와 X11 forwarding으로 사용한 이력이 있다.
- 이 PC에서는 ROS/Humble 실물 노드나 Isaac 시뮬레이션을 실행해 검증하지 않았다. Python 오프라인 검사와 소스 분석이 중심이다.

### 실물 로봇: robo04

- SSH: `ssh robo04@192.168.1.161`.
- 과거 사용자 프롬프트 호스트명: `robo04-desktop`.
- 보드/GPU 계열: NVIDIA Jetson Orin. 세부 Orin 모델·현재 JetPack·정확한 Ubuntu 버전은 미확인이다.
- OS 계열: Ubuntu/Linux, ROS 2 **Humble**.
- 작업 경로: `/home/robo04/sim2real_ws`.
- 전송 파일: `/home/robo04/incoming`.
- 실행 로그: `/home/robo04/sim2real_ws/runtime_logs`.
- ROS 기본 환경: `ROS_DOMAIN_ID=42`, `ROS_LOCALHOST_ONLY=1`.
- 과거 출력: Python 3.10.12, NumPy 1.26.4, ONNX Runtime 1.23.0. 현재 ONNX 세션은 CPUExecutionProvider로 로딩 검증했다. TensorRT/CUDA provider가 설치돼 있다는 사실만으로 GPU 추론 중이라고 판단하지 않는다.
- 과거 별도 경로의 설치 기록: TensorRT 10.3.0, PyTorch `2.5.0a0+872d972e41.nv24.08`. 이는 현재 정책이 TensorRT engine으로 실행된다는 뜻이 아니다.
- NumPy1 격리 경로: `/home/robo04/sim2real_ws/.policy-python/numpy1`. YOLO/정책 러너에서 필요 시 PYTHONPATH로 사용한다. 전역 NumPy를 임의 업그레이드하지 않는다.
- SSH 포트 변경 기록은 없으며 예시 명령은 기본 SSH 포트를 사용한다. 주소가 DHCP로 바뀌었는지는 재접속 시 확인한다.

### 강화학습 서버

- **과거 접속 기록:** `ssh bami122@163.180.145.122`.
- OS 계열: Ubuntu/Linux 학습 서버. 정확한 OS 버전, GPU 종류·개수·VRAM, 현재 CUDA/드라이버는 미확인이다.
- 작업 경로: `/home/bami122/isaac_ws`.
- 기존 unitree 저장소 경로: `/home/bami122/isaac_ws/unitree_rl_lab`.
- 과거 자체 XM540 프로젝트: `/home/bami122/isaac_ws/robotmodel/XM540_robot`.
- 과거 Conda 환경명: `env_isaaclab`.
- 과거 문서의 IsaacLab 기본 후보 경로: `$HOME/IsaacLab`. 서버에서 실제 존재 여부를 확인해야 한다.
- 사용자는 서버가 이전 Git 작업 상태라고 설명했다. 새 MX3 학습 overlay가 설치됐거나 최신 저장소를 pull했다고 확인한 적은 없다.
- **제공된 킥 패키지 README에 기록된 학습 환경:** Isaac Lab 2.3.2 / Isaac Sim 4.5.0 / rsl_rl 3.1.2 / torch 2.7.0 / Python 3.10. 이는 해당 패키지의 기록이며 위 서버의 현재 설치 상태를 직접 조회한 결과가 아니다. 실제 설치 버전과 호환성은 재현 전에 확인한다.
- Windows가 서버↔로봇 파일 중계점이었다. 서버가 로봇의 사설망 IP에 직접 접근 가능하다고 가정하지 않는다.

비밀번호·토큰·SSH 개인키는 이 문서에 넣지 않는다. 과거 비대화형 SSH 인증이 안 돼 사용자가 비밀번호를 입력해 실행했다. 새 AI는 현재 인증 가능 여부를 확인하되 인증정보를 출력하거나 사용자에게 채팅으로 비밀번호를 보내라고 하지 않는다.

### 실제 연결 장치

- MCU: STM32 Virtual COM, `/dev/ttyACM0` 이력.
- MCU 고정 경로: `/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_3657367C3036-if00`.
- U2D2/모터 통신: FTDI, `/dev/ttyUSB0` 이력.
- 모터 통신 고정 경로: `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HKSR-if00-port0`.
- 카메라: Intel RealSense D405, 사용 프로파일 `848x480x30`.
- 주 모터: Dynamixel XM540 계열. 구체적인 모델·게인·관절 부호/영점은 소스의 servo_map/motor_gains/robot_params와 실물을 대조한다.

## 4. 저장소와 파일 위치 — 서로 혼동하지 말 것

### 현재 기준 GitHub

- URL: https://github.com/hion89/MX3-HSL-Robot-Soccer
- 소유자 `hion89`, 브랜치 `main`. 처음에는 비공개로 생성했으나 **2026-11-16 GitHub 조회에서는 공개(public)** 상태다. 다운로드는 공개 접근 가능하고 push에는 쓰기 인증이 필요하다. 전체 GitHub 구분은 16절을 참고한다.
- 로컬 Git 루트: `C:/Users/jes/Desktop/MX3_Match_Sim`.
- 실제 소스 하위 루트: `C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim`.
- 따라서 실물 소스는 `MX3_Match_Sim/src/mx3_src`, `MX3_Match_Sim/src/hsl_src`이며, 원격 로봇에서는 `/home/robo04/sim2real_ws/src/mx3_src`, `src/hsl_src`다.
- 첫 스냅샷 커밋 `4748fd2`, 다운로드 복원 검증 기록 커밋 `1204a20`. 이후 커밋은 실제 `git log`를 확인한다.
- GitHub에서 별도 clone하여 선택 모델 해시/초기 위치/지도 및 지도·RViz 검사 10개를 통과했다.

### 기존 작업 공간과 기록

- 자료 원본 작업 폴더: `C:/Users/jes/Desktop/isaac_ws/sim2real_evidence`.
- 오래된 별도 Git 루트: `C:/Users/jes/Desktop/isaac_ws/isaac_ws`.
- 오래된 원격: https://github.com/hion89/Isaac-work-space.git . 이 저장소에는 기존 미커밋 변경이 있다. 새 MX3 작업을 무관한 변경과 합쳐 이쪽에 push하지 않는다.
- 전체 진행 기록 원본: `C:/Users/jes/Desktop/isaac_ws/isaac_ws/sim2real_ws/docs/MX3_HSL_DEPLOYMENT_PROGRESS.md`.
- 새 저장소의 동기화 사본: `docs/MX3_HSL_DEPLOYMENT_PROGRESS.md`.
- 최신 상태: `docs/CURRENT_STATUS_KO.md`.
- 이 문서: 새 저장소의 `sim2real_evidence/PROJECT_HANDOFF_KO.md`와 위 자료 원본 폴더에 있다.
- 운영 매뉴얼 Word/PDF: `C:/Users/jes/Desktop/MX3_Match_Sim/Manual`.
- 매뉴얼은 2026-11-09 작성본이다. 이후 변경된 시작 위치·지도·킥 튜닝은 최신 상태 기록이 우선한다.
- 로컬 분석용 Python: `C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/.analysis_env/Scripts/python.exe`.

## 5. 소프트웨어 구조와 제어 흐름

- `mx3_src/humanoid_serial_bridge`: MCU·모터·버튼·토크·관절 피드백.
- `mx3_src/humanoid_perception`: RealSense/YOLO/3D 검출과 관련 TF.
- `mx3_src/humanoid_policy_runner`: 보행/킥 ONNX 러너, 텔레옵 등.
- `mx3_src/humanoid_bringup`: 공통 하위 제어 launch와 모터/러너 구성.
- `hsl_src/hsl_world_model`: 다리 오도메트리, 자기위치 추정, 공 추적(C++).
- `hsl_src/hsl_behavior`: 역할·전략·스킬·목표 기반 이동(Python).
- `hsl_src/hsl_team_comm`: 팀 통신. 현재 단독 시험에서 팀 협동이 검증됐다는 뜻은 아니다.
- `hsl_src/hsl_bringup`: 실물 base/단독 역할 launch, 정책 선택, HSL 설정.
- `src/humanoid_sim`: 기존 MuJoCo 축구 시뮬레이터.

자율 경기: 검출/관절/IMU → 자기위치·공 추정 → 역할/스킬 → 속도 명령 → mux → 보정 → 보행 정책 → 관절 목표 → 모터 제어.

텔레옵도 같은 보행 정책을 사용하지만, 키보드 자체는 지도상 방향/위치 오차를 계속 보정하지 않는다. 자율 목표 이동은 위치 오차를 되먹임하므로 같은 정책이어도 직진성이 다르게 보일 수 있다. robo04의 vy_trim=.024가 들어 있는 이력도 있다. 바닥 조건과 함께 확인해야 하며 곧바로 0으로 바꿀 근거는 아니다.

보행·킥은 제어권을 넘겨 사용하는 구조다. 두 정책의 관절 출력을 합산하지 않는다. 현재 RL 기상 모델은 파일로 제공됐지만 실물 제어 launch의 기상은 키프레임 경로다.

## 6. 현재 선택된 정책과 튜닝

로봇의 모델 디렉터리:
`/home/robo04/sim2real_ws/install/hsl_bringup/share/hsl_bringup/models/`

- 보행: `walk_v0_6_13500.onnx`.
  SHA256 `72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1`.
- 킥: `walkkick_v1_5b_22300.onnx`.
  SHA256 `f71c4db1ecc0db2dd3f40de4a8008c87d88106b1ed89881c30d2722f48c3779f`.
- 보행 입력 240(48항목×5 이력), 킥 입력 265(53항목×5), 출력 12개 다리 관절. 이력은 term-major 규격이다.
- MLP 512→256→128, ELU. nominal policy step .02초/50Hz.
- 관절 목표의 기본 구조: `default_joint_pos + 0.25 * action`.
- 관절 순서는 오른 다리 6개 → 왼 다리 6개이며 각 다리 pelvis pitch/roll/yaw, knee pitch, ankle pitch/roll이다. 패키지 deploy.yaml과 러너 매핑을 대조한다.

로컬 공통 `MX3_Match_Sim/src/mx3_src/humanoid_bringup/launch/control.launch.py`의 킥 수정값:

- `kick_cmd_vx=0.20` m/s.
- `kick_start_hold=0.6`초 → 킥 러너의 `resume_hold_seconds`.
- 킥 params에만 `enable_action_filter=True`, `tau_action_filter=0.02`초.
- `kick_duration=2.0`초. hold가 이 총 시간에 포함돼 유효 동작 시간이 줄어든다.
- 필터는 급격한 목표 변화를 완화하지만 반응도 지연시킨다. 최종 관절 목표 크기를 제한하는 기능은 아니다.
- 같은 킥 러너를 쓰는 **텔레옵 F 킥과 자율 경기 모두** 영향을 받는다.
- 시뮬 launch는 접근 .30/hold .3을 따로 지정하므로 실물 튜닝과 조건이 다르다.
- 일부 파라미터는 노드 시작 때만 읽는다. `ros2 param set` 성공만으로 계산에 반영됐다고 주장하지 않는다. 소스 수정/필요 빌드/러너 재시작을 구분한다.
- 학습의 stiffness=50, damping=2.5를 Dynamixel 레지스터 PID 숫자와 동일하게 취급하지 않는다.

예전 `RL_walk.pt`, `RL_fall_after_walk.pt`, `xm540_dual_policy`, TensorRT engine은 이전 자체 sim2real 경로다. 현재 제공된 walk/kick ONNX와 파일·관측·러너가 다르다.

## 7. 경기장과 초기 위치 — 최신 값

- 현재 목표 프로파일: **practice**, 연습장.
- 필드 라인: 3.9×2.6m. 잔디 전체: 5×3m.
- 센터서클 반지름 .35m.
- 페널티 구역: 깊이 .87m, 폭 2.17m.
- 골 구역: 깊이 .43m, 폭 1.30m.
- 골대 폭 1.13m, 페널티 마크는 골라인에서 .65m.
- 잔디 여유: 골라인 바깥 x방향 .55m, 사이드라인 바깥 y방향 .20m.
- 좌표 원점은 필드 중앙. 우리 골대 x=-1.95, 상대 골대 x=+1.95. 상대를 바라볼 때 왼쪽이 +y, 정면 상대 방향 yaw=0이다.
- **수동 초기 위치: `x=-1.95, y=+0.8675, yaw=0`**.
- 우리 골라인 왼쪽 두 T(y=.65, 1.085) 정중앙에 몸통 중심의 바닥 투영점을 놓고 상대 골대 방향으로 향한다.
- 이전 `y=.7583333`은 대체됐다. 새 위치는 이전보다 왼쪽 약10.9cm다.
- 물리 START가 자기위치를 초기화하므로 START 기립 **후** init_pose.sh를 실행한다. /initialpose는 현재 위치를 알려주는 것이지 그 위치로 이동시키는 명령이 아니다.
- 움직인 로봇에 이 고정 초기값을 다시 입력하지 않는다.

START 직후의 **임시 진입 가정**은 별도다. 라인 밖 .40m에서 .10m로 수정해 잔디 안쪽에 있게 했다. 이것을 위 골라인 수동 초기값과 혼동하지 않는다. C++ 지도 수정/빌드/base 재시작이 필요하다.

사진과 달랐던 현재 HSL C++ 지도/Python 행동 지도/RViz를 함께 수정했다. 레거시 humanoid_localization practice만 원래 사진과 같았고 HSL 지도는 단순 축소값(페널티 폭 약1.7333, 골대 폭 .78, 원 반지름 .325)이었다. 시각화만 고친 작업이 아니다. 지도 변경은 위치 추정·TF·공의 지도 좌표와 골키퍼 구역에도 영향을 준다.

규격 경기장 `s_field` 전환은 자기위치·행동·RViz 프로파일을 함께 맞춰야 한다. 기존 MuJoCo 물리 scene은 이번 지도 수정으로 자동 축소되지 않았다.

## 8. 최신 배포 상태와 즉시 남은 작업

### 완료된 것

- MX3/HSL 13패키지 Humble 실물 빌드 이력.
- 실제 관절/IMU, D405 영상·깊이·CameraInfo, TF, YOLO 및 공 추적 수신.
- 정지 자기위치 20초 시험에서 약30Hz, 위치/각도 변동 작게 유지. 이는 실측 절대 정확도 검증이 아니다.
- 사용자 관찰상 자율 이동·공 탐색/접근과 RViz 표시가 동작했다.
- 정책 ONNX 해시/로딩, shadow 보행 출력 검증. shadow는 실제 모터로 목표를 보내지 않는다.
- 지도/초기 위치/RViz 최신 수정본 제작 및 오프라인 검사, GitHub 업로드·clone 복원 검증.

### 아직 확정하면 안 되는 것

- 2026-11-12 이후 최신 지도와 초기 위치 수정본의 **실물 적용 성공 출력은 아직 받지 못했다**.
- 마지막 화면은 기존의 좁은 페널티 구역처럼 보였다. 화면만으로 C++ 추정 지도 설치 상태까지 단정하지 않는다.
- 마지막 안내는 소스 전송→apply→C++/ROS 빌드→base/start/새 RViz 재시작이었다. 사용자가 그 결과를 보내기 전에는 완료로 기록하지 않는다.
- 현재(11월16일) 로봇 전원·토크·지지·노드 실행 상태, 배터리 상태는 미확인이다.
- 킥 안정성, 실측 위치 정확도, 잔디 보행/킥, 여러 대 팀 통신/실물 4대4는 남은 검증이다.

### 최신 배포 파일

저장소 기준 `sim2real_evidence/deploy/`:

- `hsl_field_20261112.tar.gz` + `.sha256`: 실제 지도·행동 지도·새 RViz 묶음. checksum 파일을 기준으로 검증한다.
- `init_practice_pose_20261112.sh` + `apply_init_pose_20261112.py`: 기존 incoming/real_robot_20261109/init_pose.sh를 백업 후 새 좌표로 교체. 설치 자체로 pose를 발행하지 않는다. 셸 교체만으로는 재빌드가 필요 없다.
- 구버전 hsl_view/real_robot 묶음도 기록용으로 남아 있다. 최신 지도 RViz 시작 경로는 `/home/robo04/incoming/hsl_field_20261112/start.sh`다.

지도 apply_and_build.sh는 source hash 검사/백업, Python 지도 검증, C++ 헤더 컴파일·좌표 검증, `hsl_world_model hsl_behavior` 빌드, 설치 Python 검증을 수행한다. 성공 문자열은 `FIELD_UPDATE_BUILT_AND_VERIFIED`다. 기존 teleop/control/base/viewer 프로세스가 있으면 적용을 거부한다.

## 9. 터미널별 정상 운영 명령

이 명령들은 **관련 배포 묶음이 로봇에 설치된 경우**의 운영 방법이다. 문서를 받았다는 이유로 자동 실행하지 않는다.

로봇의 새 ROS 진단 터미널 공통 환경:

```bash
source /opt/ros/humble/setup.bash
source /home/robo04/sim2real_ws/install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
```

터미널 1: 하드웨어·영상·검출·오도메트리·추적·자기위치 통합 base.

```bash
bash /home/robo04/incoming/real_robot_20261109/base.sh
```

터미널 2: 보행/킥 제어와 행동 노드. 기본 INITIAL로 시작하며 역할 실행과 구분한다.

```bash
bash /home/robo04/incoming/real_robot_20261109/start.sh
```

터미널 3: MobaXterm X11 활성화 상태에서 최신 RViz.

```bash
bash /home/robo04/incoming/hsl_field_20261112/start.sh
```

터미널 4: 실제 시작 위치에 배치하고 물리 START 후 초기값을 알린다.

```bash
bash /home/robo04/incoming/real_robot_20261109/init_pose.sh
```

이후 사용자가 역할 시험을 시작할 때 아래 중 **하나만** 실행한다.

```bash
bash /home/robo04/incoming/real_robot_20261109/role.sh attacker
# 또는 defender / goalkeeper
```

정지/상태 확인:

```bash
bash /home/robo04/incoming/real_robot_20261109/role.sh stop
bash /home/robo04/incoming/real_robot_20261109/role.sh status
```

role stop은 경기 흐름 INITIAL/보행 stop/킥 비활성화를 요청한다. **모터 토크 OFF가 아니다.** 물리 OFF 버튼의 토크 해제와 구분한다. OFF 후 임의 위치에 옮기고 START만 눌렀다고 정확한 위치를 자동 인식하거나 자동 경기 복귀가 구현됐다고 설명하지 않는다. 사용자는 이 부분을 개념적으로 논의했으며 자동 복귀 수정은 요청하지 않았다.

텔레옵은 자율 역할을 먼저 stop한 뒤 보행 gait auto, 킥 motion_enabled를 필요한 시험에 맞게 활성화한다. 키보드 실행 명령은 다음과 같다(공통 ROS 환경 필요).

```bash
ros2 run humanoid_policy_runner keyboard_teleop_node --ros-args \
  -r cmd_vel:=/cmd_vel_teleop \
  -p kick_topic:=/kick_trigger
```

F 킥은 같은 킥 정책을 호출한다. 수동 공 좌표와 motion_enabled 상태를 확인해야 한다. 키를 놓아도 속도 명령이 유지되는 구조이므로 정지는 Space다. 이 명령만으로 base/control 설치·기동이나 모터 토크가 설정되는 것은 아니다.

## 10. 역할 로직의 현재 한계

- attacker: 공이 알려지면 공 뒤 접근·정렬 후 킥 조건을 확인한다. 지정 역할 자체가 고정 대기 좌표를 뜻하지 않는다.
- defender: 현재 강제 defender는 비활성 공 담당자로 수비 위치를 잡는다. 자동 클리어 킥까지 구현된 완성 수비수로 설명하지 않는다.
- goalkeeper: 우리 골대 앞 위치를 유지하고 공의 좌우 위치를 따라 조정한다. 우리 페널티 구역 근처에서 공 담당이 되면 접근/킥 경로를 사용한다. 다이빙 세이브나 공 궤적 예측 방어가 아니다.
- 현재 단독 실행의 역할 선택은 role.sh가 PLAYING으로 전환한다. 공식 GameController와 여러 대 팀 협동은 별도 연결·검증 대상이다.
- ROS localhost 격리를 해제하거나 이름만 바꾼다고 다중 로봇 통합이 되지 않는다. 팀 번호/player 번호/IP/장치 경로/이름/팀 통신/GC와 팀 공·좌표 규격을 함께 검토해야 한다.

## 11. 알려진 실패·문제와 재학습 계획

### 현재 킥 문제

- 잔디에서는 지지 발이 미끄러지거나 몸 안쪽으로 빠르게 들어와 킥 후 넘어지는 사용자 관찰이 있다. 연구실 바닥과 차이가 크다.
- 접근 속도 감소로 약간 나아졌지만 hold/필터 조정만으로 해결된 상태가 아니다.
- 영상 하나만으로 무게중심·지지발 반력·공 접촉 시점의 원인을 단정하지 않는다.
- 별도 킥 bag 분석에서 큰 측면 관절 목표, 기울기 증가, 약1.93초 PROTECT, kick_done 없음이 관찰됐다. 전압 저하도 있었으나 인과는 확정되지 않았다.
- 과거 한 차례 토크 ON 실패는 사용자가 배터리 방전 때문이라고 확인했다. 모든 후속 문제의 원인을 배터리로 일반화하지 않는다.

### 새로 제공된 학습 자료

- `XM_Walking_Policy`, `XM_Kick_Policy`, `XM_Getup_Policy`에는 모델뿐 아니라 학습 체크포인트, env/agent 설정, deploy 규격, unitree overlay, 자산 및 기록이 있다.
- 현재 선택된 walk/kick ONNX와 제공 패키지의 ONNX 해시가 일치한다.
- 세 overlay는 완전한 공통 설치본이 아니며 공유 파일/태스크 등록 내용이 다르다. 같은 디렉터리에 세 개를 무조건 차례로 덮어쓰지 않는다.
- 새 킥 후보는 기존22300 기반 별도 태스크에서 표면 마찰 분포·관측 지연·실제 지지발 미끄럼 평가를 개선하는 방향이다.
- 기존 킥에도 마찰 랜덤화와 support_slip 벌점이 있다. “마찰 랜덤화가 전혀 없어서 실패했다”는 설명은 틀리다.
- 텍스처 무작위화와 접촉 마찰 무작위화는 다르다. 평지에서 표면을 우선 다루고 볼 높이 등이 고정된 상태에서 거친 지형을 무작정 켜지 않는다.
- 학습 조건 hold .3/접근 .3/필터 없음과 현재 실물 hold .6/접근 .2/tau .02를 구분해 기준선을 평가한다.
- 구체적 계획: `sim2real_evidence/policy_package_audit_20261112/RETRAINING_PLAN_KO.md`.
- 서버 전송/새 태스크 구현/새 학습/새 정책 배포는 아직 하지 않았다.

## 12. 진단·빌드에서 반복하지 말아야 할 실수

- ROS node/topic 목록이 처음 비어 보여도 즉시 노드 종료로 단정하지 않는다. daemon 발견 문제가 있었다. `ros2 node list --no-daemon --spin-time 5`를 사용할 수 있다.
- `/hsl/ball`의 정확한 타입은 `hsl_interfaces/msg/BallEstimate`, `/hsl/robot_pose`는 `hsl_interfaces/msg/RobotPoseEstimate`다. 새 터미널은 overlay source가 필요하다.
- 주요 입력/출력: `/joint_states`, `/imu/data`, `/imu/data_50hz`, `/perception/detections_3d`, `/hsl/odom`, `/hsl/ball`, `/hsl/robot_pose`, `/hsl/behavior_debug`.
- 카메라 토픽: `/camera/camera/color/image_raw`, `/camera/camera/color/camera_info`, `/camera/camera/aligned_depth_to_color/image_raw`.
- TF의 주요 프레임: `map`, `odom`, `base_link`, 목/카메라 링크, `camera_color_optical_frame`.
- 과거 tf2_echo에서는 변환이 보이는데 ball_tracker 내부 버퍼가 프레임을 못 찾던 문제가 있었다. 전용 TF 수신 처리 등을 수정한 v3 이력이 있다. TF가 없을 때 identity/최신 TF로 덮어 원점에 공을 그리는 임시 처리를 하지 않는다.
- RViz raw 마커는 관측 클래스/신뢰도이며 위치 추정기가 실제로 채택한 특정 랜드마크 대응이라는 뜻은 아니다.
- `humanoid_bringup/setup.py`의 `glob('launch/*')`가 __pycache__ 디렉터리까지 복사해 빌드 실패한 적이 있다. `*.launch.py`와 일반 config 파일만 설치하도록 수정했다.
- 백업 package.xml이 workspace 아래 있으면 colcon 중복 패키지 오류가 났다. runtime_logs/COLCON_IGNORE를 유지하고 다음처럼 검색 범위를 src로 제한한다.

```bash
colcon build --base-paths src --symlink-install --packages-select hsl_world_model hsl_behavior
```

- 수정한 패키지에 맞게 packages-select를 바꾼다. 이 명령은 실행 중인 노드를 자동 재시작하거나 소스를 먼저 전송해 주지 않는다.
- hsl_world_model CMake가 설치하는 config 디렉터리가 없어 빌드 실패한 적이 있다. 새 Git에는 .gitkeep으로 보존했다.
- Git 줄바꿈 변환으로 배포 manifest SHA가 깨지지 않게 `.gitattributes`는 바이트 보존이다. Linux 셸 스크립트는 LF로 유지한다.
- 백업 없이 build/install/src를 통째로 지우거나 작업 중인 학습 서버 소스를 reset하지 않는다.

## 13. 새 AI의 첫 작업 순서

1. 이 문서, AGENTS.md, docs/CURRENT_STATUS_KO.md, 실제 git status/log를 읽는다. 사용자의 최신 발언이 기록보다 우선한다.
2. 실물 시험을 이어갈 경우 현재 전원·토크·배치·실행 노드 및 최신 지도 적용 로그를 먼저 확인한다. 문서를 만드는 작업만 하는 경우 장비에 접속할 필요는 없다.
3. 지도는 `FIELD_UPDATE_BUILT_AND_VERIFIED`, 실행 중 field_profile와 START margin, 새 RViz 시작 경로를 확인한다. 화면이 옛 지도면 소스/설치/실행 중 프로세스를 나눠 확인한다.
4. 최신 초기 위치는 .8675다. 실제 배치 및 START 순서를 확인하고, 이미 이동했다면 임의 초기화하지 않는다.
5. 다음에는 사용자가 지정한 역할 시험 또는 잔디 킥 재학습 계획을 이어간다. 단독 역할 성공을 4대4 성공으로 확대하지 않는다.

## 14. GitHub 반영과 사용자 선호

- 한국어로 설명한다. 이미 허가된 작업은 구체적 결과까지 진행하고 불필요한 확인 질문으로 끊지 않는다.
- 각 단계에서 완료/다음/남은 작업을 간단히 알린다.
- 어떤 파일의 어떤 값인지, 의미·근거·재빌드·재시작 방법을 알려준다.
- 공통 코드 변경이 텔레옵/자율 경기 모두에 미치거나 시뮬이 다른 값을 덮는 경우 먼저 설명한다.
- 긴 Markdown 표는 쓰지 않는다. 이 요청은 MD 인수인계지만 운영 매뉴얼은 Word/PDF를 선호한다.
- 로봇 시작 위치를 기억하고 변경 시 실제 배치와 좌표를 함께 안내한다.
- 원격 업로드·Git 커밋은 실물 배포/동작과 별개다. 성공한 범위만 보고한다.
- 작업 완료마다 관련 변경과 검증 기록을 **새 저장소 main에 커밋·푸시**한다. 기존 unrelated 변경은 제외한다.

별도 작업 공간의 기록 동기화 예시(PC PowerShell, 새 Git 루트에서):

```powershell
cd C:\Users\jes\Desktop\MX3_Match_Sim
python scripts/sync_working_copy.py --workspace C:/Users/jes/Desktop/isaac_ws
git status --short
```

이후 해당 작업 파일을 선택적으로 add/commit/push한다. sync는 자동 삭제·커밋·원격 배포를 하지 않는다. PC에 Python이 여러 개면 앞서 명시한 분석용 python.exe를 사용할 수 있다.

다운로드 사본의 모델/지도/초기 위치 확인:

```bash
python scripts/verify_checkout.py
```

이 명령은 오프라인 검사다. ROS/모터/Isaac 실행 성공을 의미하지 않는다.

## 15. 후속 상세 자료

- `docs/MX3_HSL_DEPLOYMENT_PROGRESS.md`: 통합 진행 전체 기록.
- `docs/MX3_HSL_REAL_DEPLOYMENT.md`: 실물 배포 기록.
- `sim2real_evidence/manuals/BHUMAN_PORT_AUDIT_KO_20261109.md`: B-Human 방식 이식 범위와 재구현 구분.
- `sim2real_evidence/manuals/BHUMAN_OFFICIAL_REVIEW_20261109.json`: 원본 대조 근거.
- B-Human 원본: https://github.com/bhuman/BHumanCodeRelease . 초반 RoboEireann 링크는 사용자가 정정한 다른 저장소이므로 혼동하지 않는다.
- `sim2real_evidence/policy_package_audit_20261112/`: 학습 자료 inventory/readiness/재학습 계획.
- `sim2real_evidence/kick_balance/`, `kick_video_20261110/`: 킥 분석 결과. 원본 bag/영상은 Git에서 제외돼 다른 PC에는 없을 수 있다.
- `sim2real_evidence/patches/hsl_field_20261112/`: 지도·RViz 적용/검증 소스.
- `Manual/MX3_HSL_로봇운영매뉴얼_20261109.docx` 및 PDF: 운영 매뉴얼. 최신 시작 위치는 이 인수인계 문서를 우선한다.

## 16. 사용하는 GitHub 저장소와 각각의 역할

2026-11-16 기준 로컬 Git remote, 연결된 GitHub 조회, 기존 프로젝트 문서로 확인했다. 이 목록은 **로봇 프로젝트에 관련된 저장소**다. 계정에 있는 의료·기타 프로젝트를 현재 로봇의 의존성으로 섞지 않는다. 공개 상태와 브랜치는 바뀔 수 있으므로 작업 시 다시 확인한다.

### 16-1. 현재 실물 통합과 작업 기록 — hion89/MX3-HSL-Robot-Soccer

- 주소: https://github.com/hion89/MX3-HSL-Robot-Soccer
- 현재 공개(public), 기본 브랜치 main. 11월12일 비공개 생성 기록과 11월16일 현재 상태를 구분한다. 이번 문서 수정으로 공개 설정을 변경한 것은 아니다.
- 앞으로 이 로봇 프로젝트의 변경·검증 기록을 commit/push하는 **기준 저장소**다.
- 실물용 MX3/HSL 소스, 보행·킥·기상 모델과 학습 패키지, 배포 스크립트, RViz/지도 수정, 분석 결과, 매뉴얼, 인수인계가 함께 있다.
- 로컬 루트 `C:/Users/jes/Desktop/MX3_Match_Sim`. 로컬 remote 이름 origin.
- 정책 자료는 포함하지만 Isaac Sim/Isaac Lab 전체 설치본, 원본 bag/영상, 가상환경과 빌드 결과를 모두 포함한 환경 이미지는 아니다.
- 복제: `git clone https://github.com/hion89/MX3-HSL-Robot-Soccer.git`.
- 새 AI의 작업 결과는 이곳에 올린다. push가 로봇 설치/재시작이나 학습 서버 pull을 자동 실행하지는 않는다.

### 16-2. 이전 자체 학습·sim2real 작업 — hion89/Isaac-work-space

- 주소: https://github.com/hion89/Isaac-work-space
- 현재 비공개(private), 기본 브랜치 main. 다운로드에도 계정 권한/인증이 필요하다.
- 로컬 루트 `C:/Users/jes/Desktop/isaac_ws/isaac_ws`. 이곳도 remote 이름이 origin이므로 현재 작업 디렉터리를 확인해야 한다.
- 이전 XM540 자체 강화학습, robotmodel/XM540_robot, unitree_rl_lab 기반 학습 작업과 예전 sim2real 경로의 기록이 있다.
- 서버의 기존 작업 경로 `/home/bami122/isaac_ws`와 관련된 과거 저장소다. 서버가 지금 어떤 remote/커밋을 쓰는지는 직접 조회하지 않았다.
- 이전 RL_walk/RL_fall_after_walk TensorRT 경로와 현재 MX3 ONNX 경로를 비교·복구할 때 참고한다.
- 이 저장소의 코드가 모두 새 실물 통합 저장소로 이동했다고 가정하지 않는다. 로컬에 기존 미커밋 변경이 있으므로 무관한 파일을 함께 commit하거나 reset/clean하지 않는다.
- 새 학습 계획을 실행할 때도 먼저 서버 git remote/status/커밋을 확인하고 별도 작업공간에 필요한 overlay를 적용한다. 새 실물 저장소를 기존 서버 폴더에 무조건 덮어쓰지 않는다.

### 16-3. 별도 비전·위치보정 실험 — hion89/CV-robocup

- 주소: https://github.com/hion89/CV-robocup
- 현재 비공개(private), 기본 브랜치 main.
- 연결된 GitHub에서 README를 확인했다. Gazebo Classic + ROS 2 Humble에서 RGB/Depth, YOLO 검출·세그멘테이션, 랜드마크 위치보정, topological grid/loop 후보, Nav2 목표 이동을 실험하는 저장소다.
- README의 작업 루트는 `CV-robocup/ros2_ws`. 이번 대화에서 대응되는 로컬 clone 경로나 별도 서버 접속 주소를 확인한 것은 아니다.
- **현재 실물 HSL 자기위치/행동 스택과 별도의 실험 저장소**다. 여기의 RTAB-Map/Nav2/TF 구성이나 실행 명령을 현재 MX3 base의 구성으로 그대로 설명하지 않는다.
- 비전 개선을 검토할 때 비교 후보로 참고할 수 있지만, 현재 MX3 소스에 통합됐다는 증거는 없다.

### 16-4. 상위 전략·측위 방식 참고 원본 — bhuman/BHumanCodeRelease

- 주소: https://github.com/bhuman/BHumanCodeRelease
- 사용자 지정 B-Human 원본이다. 역할/전략/스킬, UKF 자기위치 추정, 팀 공·메시지 등의 알고리즘 대응을 검토했다.
- 기존 감사에서 대조한 고정 커밋: `2e3374ba896da8288607f9a1881b876642fe18b8`. 이는 대조 근거이며 원본 전체를 그 커밋 그대로 배포했다는 뜻은 아니다.
- 로컬 구현은 MX3 ROS 2에 맞춘 이식/재구현/통합을 구분해야 한다. 원본 B-Human을 clone한 것만으로 Jetson+XM540 실물이 동작하지 않는다.
- 근거: `sim2real_evidence/manuals/BHUMAN_PORT_AUDIT_KO_20261109.md`, `BHUMAN_OFFICIAL_REVIEW_20261109.json`.
- 이 외부 저장소에 프로젝트 변경을 push하지 않는다. 개선은 현재 본인 실물 통합 저장소에서 관리한다.

### 16-5. 학습 코드 기반 — unitreerobotics/unitree_rl_lab

- 주소: https://github.com/unitreerobotics/unitree_rl_lab
- Unitree 계열 Isaac Lab/RSL-RL 학습 태스크 구조가 기반이다. 기존 XM540 README/출처 고지와 제공된 MX3 정책 패키지에 관련 기록이 있다.
- MX3 사용자 정의 태스크/자산은 원본 기본 태스크와 별개다. 제공 패키지의 unitree_rl_lab_overlay와 rsl_rl_patch를 해당 환경에 맞게 적용해야 한다.
- 기존 로컬 자료의 기본 unitree 태스크만으로 새 MX3 학습이 바로 재현된다고 가정하지 않는다.
- 각 정책 overlay는 공유 파일과 등록이 다르므로 세 패키지를 같은 폴더에 순서대로 덮어써서는 안 된다.
- upstream의 최신 main으로 무조건 갱신하면 학습 계약/호환성이 달라질 수 있다. 기존 커밋·로컬 수정·설치 버전을 먼저 보존한다.

### 16-6. 학습 시뮬레이션 프레임워크 — isaac-sim/IsaacLab

- 주소: https://github.com/isaac-sim/IsaacLab
- 환경·관측·액션·보상·종료·도메인 랜덤화를 구성하는 프레임워크다. 로봇에서 실행하는 ROS ONNX 러너와 역할이 다르다.
- 학습 서버의 실제 설치 경로/버전을 조회한 다음 제공 패키지의 기록과 맞춰야 한다. 최신 release를 설치하는 것이 기존 학습 재현과 같지 않다.
- Isaac Sim, Isaac Lab, unitree_rl_lab, RSL-RL은 서로 구분되는 구성요소다. 제공 패키지의 rsl_rl_patch는 별도 설치 패키지 수정분이며 원본 학습 저장소 전체를 대체하지 않는다.

### 16-7. 구현 확인용 외부 참고

- https://github.com/ros2/geometry2 : tf2/TF 변환과 수신 처리의 원리·구현 확인에 사용한 참고다. 사용 중인 ROS 2 Humble 설치 버전과 원본 최신 main을 동일하게 보지 않는다.
- https://github.com/NVIDIA/TensorRT : 이전 TensorRT 변환/엔진 경로 관련 참고다. 현재 ONNX 러너가 TensorRT provider를 사용한다는 증거는 아니다.
- https://github.com/roboeireann/RoboEireannCodeRelease : 초반 제시된 링크이며 사용자가 이후 B-Human으로 정정했다. 현재 채택 기준이나 실물 통합 원본으로 취급하지 않는다.

### 16-8. 새 AI가 Git 작업 전에 확인할 것

PC의 새 실물 저장소에서:

```powershell
cd C:\Users\jes\Desktop\MX3_Match_Sim
git remote -v
git branch --show-current
git status --short
git log -3 --oneline
```

여기서 origin이 `hion89/MX3-HSL-Robot-Soccer.git`인지 확인한다. 예전 폴더의 origin은 Isaac-work-space이므로 이름만 보고 push하면 안 된다. 서버의 실제 remote/브랜치/수정 여부는 서버에서 따로 확인한다. README의 clone 예시는 자동 동기화 설정이 아니다.

장비 접속 정보와 내부 경로가 포함된 문서다. 현재 실물 저장소는 공개로 조회되므로 추가하는 정보의 공개 범위를 고려하며, 비밀번호/토큰/개인키는 포함하지 않는다.
