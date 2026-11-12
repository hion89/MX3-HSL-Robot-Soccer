# humanoid_sim — MuJoCo 로 mx3_src 경기 스택 돌리기

실기 하드웨어(다이나믹셀·IMU·MCU 버튼·RealSense+YOLO)만 MuJoCo 브리지로 바꾸고,
나머지 경기 스택(정책 러너·fall_manager·측위·전략·GameController·팀통신)은 **로봇 코드 그대로** 돌린다.

```
MuJoCo (sim_bridge, 벽시계 1 배속, 물리 200 Hz)
  ├─ /joint_states 50 Hz, /joint_goals → 다음 20 ms 주기에 반영 (실기 SyncWrite/Read 와 같은 1 주기 지연)
  ├─ /imu/data 200 Hz, /imu/data_50hz, /imu/projected_gravity (쿼터니언 body→world, 시작 yaw 기준)
  ├─ /u2d2/* 토픽·서비스 (robot_init, torque, standup*, limits …), /buttons/*
  ├─ /perception/detections_3d 30 Hz — 정답 위치 + 노이즈 (모드 A, YOLO 대체)
  └─ /sim/gt/base_pose, /sim/gt/ball, /sim/event (골·아웃 판정)
로봇마다 ROS_DOMAIN_ID 분리 (스택이 절대 토픽 이름을 써서)
```
\* standup 은 아직 임시: 3 s 뒤 그 자리에 세운다.

## 실행 (터미널 4 개, conda 끈 셸)

```bash
# 1) 시뮬 (화면 보려면 --viewer)
~/sim_ws/src/humanoid_sim/scripts/run_bridge.sh --scene ~/sim_ws/install/humanoid_sim/share/humanoid_sim/models/scene_1v0 --viewer
# 2) 로봇 스택 (robo02, ROS_DOMAIN_ID=2)
~/sim_ws/src/humanoid_sim/scripts/run_robot.sh robo02 2
# 3) GameController (READY 45 s → SET 5 s → PLAYING)
source /opt/ros/jazzy/setup.bash && source ~/sim_ws/install/setup.bash && ros2 run humanoid_sim sim_gc --state initial --sequence initial:3,ready:45,set:5,playing
# 4) START 버튼 + 모니터
source /opt/ros/jazzy/setup.bash && source ~/sim_ws/install/setup.bash && export ROS_DOMAIN_ID=2 && ros2 service call /sim/start std_srvs/srv/Trigger && ros2 run humanoid_sim sim_monitor
```

로봇은 재진입 위치(사이드라인 밖, robot_params 의 x_inset)에 서서 시작한다. START = 실기 START 버튼과 같다
(`/buttons/start_event` + `/u2d2/robot_init`) → relocalize 가 재진입 pose 로 시드한다.

장면 만들기 (USD 를 읽으므로 conda `mujoco_sim` 환경):
```bash
cd ~/unitree_rl_lab/scripts/mujoco_sim && python gen_bridge_scene.py --robot=robo02:-3.1,3.4,-90 --ball=0,0 --out ~/sim_ws/src/humanoid_sim/models/scene_1v0
```
새 장면을 넣었으면 `colcon build --base-paths ~/Documents/ros2_ws/mx3_src src --packages-select humanoid_sim`.

## 실기와 다르게 한 것 (sim 런치에서만, 로봇 소스는 그대로)

| 무엇 | 왜 |
|---|---|
| twist_mux 를 sim 런치에서 직접 띄움 (`use_stamped: false`) | Jazzy twist_mux 기본값이 TwistStamped 라 스택의 Twist 와 끊긴다. 전역 `SetParameter` 로 주면 안 된다 — yaml 의 노드별 값이 런치 dict 를 이겨 측위가 practice 필드로 뜬다 |
| 정책 러너 onnxruntime 1 스레드 (`pylib_sim/sitecustomize.py`, `SIM_ORT_THREADS`) | 러너가 SessionOptions 없이 세션을 만들어 코어 수만큼 스레드가 spinning → 러너 1 개 CPU 209 %, `/joint_goals` 43~46 Hz 로 주기 누락. 1 스레드로 18 %, 50.0 Hz |
| `robot` 인자로 robot_params·재진입 inset 선택 | 실기는 hostname 으로 고른다 |
| 카메라·YOLO·depth_obstacle 제외 | YOLO 는 Jetson TensorRT 엔진뿐. 검출은 브리지가 정답 위치로 낸다 (D405 FOV 87×58°, 깊이 0.07~3.0 m, 공 0.30 m 미만 제외 — yolo_3d.launch 와 같은 게이트) |

## 파이썬 환경
ROS Jazzy 의 rclpy 는 시스템 Python 3.12 용이다. mujoco·onnxruntime 은 워크스페이스의 `pylib` (이 PC: conda 의 pip 로
`--target --python-version 3.12`) 또는 `.venv` (`python3 -m venv --system-site-packages`, 공유 패키지의 setup.sh) 에 두고,
`scripts/sim_env.sh` 가 PYTHONPATH 에 넣는다 (ROS 노드는 /usr/bin/python3 로 돌아 가상환경을 켜는 것만으로는 안 잡힌다).
실행 스크립트는 워크스페이스를 자기 위치에서 찾는다 (`SIM_WS` 로 덮을 수 있음). 시스템 Python 은 건드리지 않았다.

## 첫 통합 시험 결과 (2026-10-31, robo02 1 대, GC READY→SET→PLAYING)

- START → `NORMAL`, 재진입 위치 시드 → READY 에서 포메이션 목표 (−1.04, 0.92) 로 걸어가 정지 (측위 오차 1 cm·1°).
- PLAYING: 킥오프 공 접근 → 정렬 → 킥. 노이즈 0 이면 PLAYING 8 s 만에 킥. 한 판(it4)에선 2 번 차서 골대 안으로 넣었다.

**드러난 문제 (실기에도 해당할 수 있는 것)**
1. 정렬 진동: visual servo 가 최소 속도 ±0.20 으로 반올림해서, 검출 노이즈 1~2 cm 에서 공 앞 0.23~0.40 m 를 64 s 왕복
   (노이즈 0 이면 바로 수렴). 실기 D405+YOLO 의 정지 공 검출 노이즈를 재서 맞춰 봐야 한다.
2. 오도메트리 = 명령 적분: 로봇이 제자리에서 왕복하는 동안 측위가 +x 로 4.7 m 흘렀다 (공을 내려다봐 랜드마크도 안 보임).
   그 상태로 골 방향 킥 → 공이 터치라인 밖.
3. 틀린 시드에서 회복 못 함: 시드가 (−4.5, 3.2, −90°) 로 잘못 들어가자 70 s 동안 잔차 0.5 m 로 계속 거부.
4. 러너 onnxruntime 스레드 미설정 → CPU 209 %, 주기 누락 (위 표). 실기 Jetson 도 같은 코드다.

**sim 이 실기와 다른 것 (다음 보정 대상)** — `odom_calib.yaml` 실측과 비교
| 명령 | 실기 | MuJoCo |
|---|---|---|
| vx −0.2 / −0.3 | 0.0 / −0.02 (거의 못 감) | 명령대로 |
| vy | vy_scale 0 (안 움직이는 것으로 처리) | 명령대로 |
| wz | 0.7 × 명령 | ≈ 1.0 × 명령 |
sim 로봇이 실기보다 민첩해서, 실기 기준 오도메트리와 어긋나고 정렬 동작도 실기와 다르게 나온다.

## 킥오프 정렬 튜닝 비교 (2026-10-31, `run_trials.sh`, 배치 8 × 설정 3, 한 판 45 s)
로봇을 센터 공 뒤 0.8~1.5 m 무작위 위치·방향에 세우고 START → 정답 시드 → SET → PLAYING. 결과 `~/sim_ws/logs/kick_trials_ab1/`.

| 설정 | 킥 | 킥까지 중앙 (최대) | 공 방향 오차 중앙 | 킥 순간 측위 오차 중앙 |
|---|---|---|---|---|
| base (현재 yaml) | 4/8 | 20.4 s (40.1) | 10° | 0.42 m |
| lowmin (vx·후진·vy 0.08, wz 0.15) | 4/8 | 11.4 s (33.3) | 9° | 0.27 m |
| lowmin_tol (+ kick_pos_tol_x 0.10) | 3/8 | 12.3 s (44.5) | 15° | 0.32 m |

최소 속도를 낮추면 찰 때는 빨라지지만(중앙 20 → 11 s) **성공률은 그대로 절반**. 실패 12 판의 원인:
- 가까운 공 분실 (ALIGN → LOST → 전략이 포지션으로 복귀) 5 판 — 공이 발 옆 0.3 m 안쪽으로 오면 시야/깊이 게이트에서 사라진다
- 핸드오프 전 접근 단계(ball_follow/pose_goal)에서 막힘 4 판 — 공 0.14~0.65 m 앞에서 SETTLE/TRAVEL 반복
- 반대편에서 ALIGN 진입 (골대 방위 ±140°) 2 판 — 게이트를 만족할 수 없는 자리
- GETBEHIND 최소속도 진동 1 판 (base) — lowmin 에서는 같은 배치가 10 s 에 찼다

## 목 yaw 고정 비교 (2026-10-31, 배치 12 × 2, 둘 다 lowmin 서보 설정)
`neck_ball_tracker.yaml` 의 `yaw_lock_pitch_deg: -40` (목을 40° 넘게 숙이면 yaw 를 정면에 고정) 을 −71(사실상 끔)로 바꿔 비교.
sim 런치 인자 `neck_yaw_lock_deg:=-71` (원본 yaml 은 그대로). 결과 `~/sim_ws/logs/kick_trials_neck/`.

| 설정 | 킥 | 킥까지 중앙 | 공 분실(LOST) | 가까운 공 시야 밖 평균 | 킥 순간 측위 | 공 방향 오차 중앙 |
|---|---|---|---|---|---|---|
| yaw 고정 있음 (−40) | 8/12 | 11.8 s | 3 회 | 2.4 s | 0.23 m | 16° |
| yaw 고정 해제 (−71) | **12/12** | 11.6 s | **0 회** | **0.0 s** | 0.18 m | 17° |

가까운 공 분실의 원인 = 0.30 m 최소 깊이 게이트가 아니다 (진단에서 0 회. 카메라가 높아 발밑 공도 0.36 m 이상).
공이 발 옆(전방 0~0.07, 측면 0.33 m)에 왔을 때 목이 정면 고정 + pitch 한계 60° 라 시야(87×58°) 밖으로 나간다.
같은 배치·같은 설정도 판마다 결과가 다르다 (lowmin: 첫 비교 4/8, 이번 8/12) — 검출 노이즈 때문. 판 수를 넉넉히.
남은 문제: 공 방향 오차 중앙 16~17° — 킥 순간 측위 방향 오차(평균 9~11°, 최대 29~42°) + 킥 정책 자체 오차.

## visual servo GETBEHIND 교착 수정 비교 (2026-10-31, 공 바로 뒤 배치 4 × 2, 공 3 m 기준)
원인: perpendicular 모드에서 GETBEHIND(direct)는 로봇을 공 일직선 뒤(delta 0)로 몰지만, ALIGN 복귀 조건은 delta ≈ ±80°(±45)라
공 뒤에서 접근하면 영원히 못 나온다 (최소속도 반올림으로 공 뒤에서 앞뒤 진동). 수정: `_delta_target` 후보에 0 추가 (SIM_SERVO_FIX=1, sitecustomize).

| | 킥 | 킥까지 (판별) | 공 방향 오차 |
|---|---|---|---|
| 수정 전 | 3/4 | 10.9 / 실패 / 13.7 / 9.7 s | 15.7 / – / 1.9 / 7.9° |
| 수정 후 | **4/4** | 12.4 / **6.8** / 9.5 / 6.5 s | 15.0 / 4.5 / 4.1 / 2.0° |

## 브리지 변경 (2026-10-31, hsl_src 다중 로봇 경기용)

- **로봇별 executor 를 스레드로** — 예전엔 메인 루프가 남는 시간 내내 `spin_once` 를 돌렸다. rclpy 는 호출마다 wait set 을
  새로 만들어 8 대에서 CPU 의 절반 이상을 먹었다 (cProfile). 지금은 로봇마다 `exe.spin()` 스레드가 메시지 올 때만 깨어나고
  메인 루프는 물리 스텝 후 남는 시간을 잔다. 브리지 노드는 파라미터 서비스·QoS 이벤트 핸들러를 만들지 않는다.
  콜백이 물리 상태를 바꾸는 것(공 재배치, START 위치)은 요청만 남기고 메인 루프가 스텝 사이에 적용한다.
- `/sim/place_ball` (PointStamped, 첫 로봇 도메인, 월드 x, y) — 심판 스크립트(hsl_bringup/match_referee.py)가 골·아웃 뒤 공을 놓는다.
- **START 때 출발 pose 로 세움** — 사람이 사이드라인에 세워 놓고 누르는 것과 같게. START 전 대기 중 기울거나 넘어진 로봇이 있었다.
- **임시 기상 방향 수정** (`heading_of`) — 뒤로 누운 로봇은 몸 x 축이 수직이라 `yaw_of` 가 180° 뒤집힌 방향으로 세웠다.
- 참고: 보행 러너 로그의 `[rate] actual=..Hz` 는 노드 시작부터의 누적 평균(START 전·킥 양보 포함)이라 실시간 지표가 아니다.
  실시간 여부는 브리지 로그의 '실시간보다 … 뒤처짐' 경고로 본다.
- 팀 색: 로봇 이름 t<팀>p<번호> 로 첫 팀 파랑·둘째 팀 빨강, 1 번(골키퍼)은 청록·주황. 뷰어 좌우 패널 기본 숨김(`--viewer_ui` 로 표시).
- 심판 연동 토픽 (첫 로봇 도메인): `/sim/ball_touch`(마지막 공 접촉 로봇), `/sim/robots`(5 Hz, 이름 → [x, y, yaw°, 서 있음]),
  `/sim/place_robot`("이름 x y yaw°"), `/sim/scoreboard`(JSON → 뷰어 왼쪽 위 점수 / 오른쪽 위 GC 상자). 아웃 이벤트에 `last=<로봇>`.
