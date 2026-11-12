# hsl_src — HSL 축구 상위 계층 (B-Human 이식)

설계·결정·검증 결과는 [DESIGN.md](DESIGN.md). 여기는 빌드·실행 방법입니다.

This product includes software developed by B-Human (http://www.b-human.de).

## 빌드

시뮬 PC (Ubuntu 24.04, ROS 2 Jazzy) — 시뮬 워크스페이스 `~/sim_ws` 에 mx3_src·hsl_src 를 같이 올린다:

```bash
cd ~/sim_ws && colcon build --base-paths ~/Documents/ros2_ws/mx3_src ~/Documents/ros2_ws/hsl_src src --packages-up-to hsl_bringup humanoid_sim
```

conda 가 켜져 있으면 끄고(`conda deactivate`) 빌드한다. 로봇(Humble)에서는 mx3_src 워크스페이스에 `hsl_src` 를 같이 두고
`colcon build --packages-up-to hsl_bringup` (아직 Humble 빌드 미확인).

의존: `ros-<distro>-kdl-parser`, `ros-<distro>-orocos-kdl-vendor`(Jazzy) / `liborocos-kdl-dev`(Humble), `ros-<distro>-tf2-eigen`, `vision_msgs`.

## 시뮬 경기

```bash
~/sim_ws/install/hsl_bringup/lib/hsl_bringup/run_match.sh
```

보행 정책은 최종 v0_6 model_13500 (`hsl_bringup/models/walk_v0_6_13500.onnx`, `sim_robot.launch.py` 의 `walk_actor_path`).
mx3_src 러너 사본 `models/walk_policy.onnx` 는 v0_8 18400 이라 시뮬에서는 쓰지 않는다 (2026-11-01 교체, 이전 경기 로그는 v0_8).
킥 정책은 v1_5b model_22300 (`hsl_bringup/models/walkkick_v1_5b_22300.onnx`, `kick_actor_path`). 경기 시뮬에서 기존 W90B70 18200 보다 킥 중 넘어짐 8.5 → 5.3 %, 공 이동 1.85 → 2.20 m (각 5 판). 실기 mx3_src 러너 사본 `walkkick_v1_policy.onnx` 는 아직 18200 이고, `STACK_ARGS="kick_actor_path:=runner"` 이면 그걸 쓴다. 다른 모델 시험은 `kick_actor_path:=<onnx>` (관측 265 walk_kick v1 계열만), 킥 시간은 `kick_duration:=` (기본 2.0, 실기와 같음).
보행 러너 CSV 로그 (`~/policy_logs`, mx3_src control.launch.py 기본 켬) 는 시뮬에서 끈다 (판마다 8 대가 쓰여 이틀에 7.6 GB). 킥 넘겨받기를 볼 때만 `STACK_ARGS="walk_csv_log:=true"`.

환경 변수:

| 변수 | 기본 | 뜻 |
|---|---|---|
| `SCENE` | `4v4` | `1v0` / `4v0` / `4v4` (`hsl_bringup/models/scene_*.xml/json`) |
| `VIEWER` | `1` | MuJoCo 화면 |
| `DURATION` | `300` | 모니터 시간 [s] (끝나면 전부 종료) |
| `READY_S` | `40` | 첫 READY 시간 [s] |
| `KICKOFF` | `35` | 첫 킥오프 팀 |
| `HALF` | `600` | PLAYING 누적 시간 후 FINISHED |
| `LOG` | `~/sim_ws/logs/match_<장면>_<시각>` | 로그 폴더 |
| `STACK_ARGS` | (없음) | 로봇 스택 launch 인자 덧붙이기 — 예: `team_send_rate_hz:=4.0 team_event_send:=false`, `kick_actor_path:=<onnx>` (시험용) |

예: 화면 없이 4 대만 3 분 — `SCENE=4v0 VIEWER=0 DURATION=180 run_match.sh`

진행: 브리지 → 로봇 스택(로봇마다 ROS_DOMAIN_ID: 35 팀 11~14, 36 팀 21~24) → 20 s 뒤 START →
심판(`match_referee.py`, GC 패킷 STRUCT_VERSION 20 을 실제 GC 처럼 송신): INITIAL 3 s → READY → SET 5 s → PLAYING.
- 점수·킥 팀·세트플레이·남은 시간·로봇별 페널티를 패킷에 싣는다 (로봇 GC 수신 로그에 그대로 찍힘)
- 골 → 센터, 실점 팀 킥오프 / 터치라인 아웃 → 상대 KICK-IN / 골라인 아웃 → CORNER KICK 또는 GOAL KICK
  (브리지가 공에 마지막으로 닿은 로봇을 알려 줌). 세트플레이는 공이 0.3 m 움직이거나 30 s 면 끝
- 반칙 (페널티 30 s, 자기 진영 페널티마크 높이 터치라인에서 복귀): LEAVING THE FIELD(라인 밖 0.8 m),
  PUSHING(상대와 0.6 m 안에서 한쪽만 넘어지고, 넘어지기 직전 서 있는 쪽이 다가가고 있었을 때 → 넘어진 팀 직접 프리킥), ILLEGAL POSITION(SET 순간 상대 진영·센터서클),
  MOTION IN SET(SET 중 0.15 m 이동), INCAPABLE(20 s 넘게 못 일어남). 끄기: `--no_pushing`, `--no_positioning`
- 공이 45 s 정체면 센터 근처로

화면(VIEWER=1): 몸통만 팀 색 — 35 팀 파랑(골키퍼 청록), 36 팀 빨강(골키퍼 주황). 머리 위에 반투명 팀 색 역삼각뿔 +
선수 번호 (7 세그먼트·흰 테두리, 항상 카메라를 봄, 높이 19 cm 고정).
공 담당 로봇(각 로봇이 `/hsl/strategy_state.active` 로 알린 대로) 발밑에 팀 색 고리 — 담당 교대 순간엔 잠깐 두 개일 수 있다.
왼쪽 위 점수(큰 글자), 오른쪽 위 GC 상태(상태·시간·보조 타이머·킥 팀·세트플레이·페널티·마지막 판정). 처음 시점은 경기장 전체.
보조 타이머(TIMER, GC 패킷 secondaryTime 에도 실림): READY·SET 남은 시간 > 세트플레이 남은 시간 > 킥오프 뒤 BALL FREE 10 s.
심판이 뜨기 전(약 30 s)에는 화면 정가운데 PREPARING MATCH (카메라 바로 앞에 3D 로 그림): 로봇 소프트웨어 기동 n/8 → 기립 n/8 → GameController 대기.
좌우 패널은 숨김 — 창에서 Tab / Shift+Tab 으로 켬. W 를 누르면 와이어프레임이 되니 주의.
로봇 충돌 모델은 실제 메시 볼록 껍질 (XM_robot04_convexhull) — `~/unitree_rl_lab/scripts/mujoco_sim/README.md`.

로그 폴더:
- `monitor.txt` — 10 s 마다 로봇별 상태·역할·스킬·측위 오차(정답 대비)·공 오차·킥·넘어짐, 끝에 요약
- `monitor.jsonl` — 1 s 마다 시계열 + 이벤트 + 요약 (분석용)
- `t35p2.log` 등 — 로봇별 스택 로그, `bridge.log`, `referee.log`

모니터 한 줄 읽는 법:
```
t35p3  PLAY suppor A play_ball_align   gt(-0.4,-0.2)   loc  0.07m    2° S ball 0.05 kick 1 fall 0
 이름  상태  역할  공담당  스킬          정답 위치       측위 오차(위치/방향) 품질(P/R/S) 공 오차 킥 수 넘어짐
```
좌표는 각 팀 기준 (+x = 그 팀이 공격하는 방향).

한 대만 띄우기 (디버깅): 브리지 `~/sim_ws/src/humanoid_sim/scripts/run_bridge.sh --scene <장면>` +
`~/sim_ws/install/hsl_bringup/lib/hsl_bringup/run_stack.sh <도메인> <팀> <번호>`.

## 실기 (미검증)

```bash
ros2 launch hsl_bringup robot.launch.py                  # robot = hostname 앞부분 → robot_params.yaml 의 team/player
ros2 launch hsl_bringup robot.launch.py player_number:=3 field_profile:=practice
```

`match.launch.py` 대신 쓴다 (둘을 같이 띄우면 안 됨 — cmd_vel·TF·목 명령이 겹친다).

## 주요 토픽

| 토픽 | 형 | 내는 노드 |
|---|---|---|
| `/hsl/odom`, TF `odom→base_link` | nav_msgs/Odometry | leg_odometry |
| `/hsl/robot_pose`, TF `map→odom` | hsl_interfaces/RobotPoseEstimate | self_localizer |
| `/hsl/ball`, `/hsl/team_ball`, `/hsl/obstacles` | BallEstimate, TeamBall, PoseArray | ball_tracker |
| `/hsl/team_state` | TeamState | team_comm |
| `/hsl/strategy_state`, `/hsl/behavior_debug`(JSON) | StrategyState, String | behavior |
| `cmd_vel_behavior`, `/neck/yaw_deg`, `/neck/pitch_deg`, `/kick_trigger` | | behavior → mx3_src |

`/initialpose` (RViz 2D Pose Estimate) 로 측위를 강제로 맞출 수 있다.

필드 선: self_localizer 가 `/perception/field_lines` (`hsl_interfaces/LineSegmentArray`, 카메라 광학 좌표의 바닥 선분) 를
받으면 B-Human 방식 선 갱신을 한다 (`use_lines`, 기본 켬). 시뮬은 브리지가 정답 선분을 낸다 (`--no_lines` 로 끔).
실기는 선 검출 노드가 아직 없다. 시뮬 시험용 브리지 옵션: `--imu_yaw_drift_dps 0.5` (IMU yaw 드리프트).

## 테스트

```bash
cd ~/Documents/ros2_ws/hsl_src/hsl_behavior && python3 -m pytest -q test/
```
