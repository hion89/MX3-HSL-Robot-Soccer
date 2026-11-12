# hsl_src 설계 문서

RoboCup HSL 용 상위 계층 스택입니다. B-Human 2026 code release(SPL 우승팀)를 분석해 구조를 가져오고,
핵심 알고리즘을 ROS 2 노드로 이식했습니다. 하위 계층(하드웨어, 보행·킥 정책, 목, GC 수신)은
`mx3_src` 것을 **수정 없이** 그대로 씁니다.

> 작성: 2026-10-31 새벽 자율 작업 (사용자 지시: "알아서 분석하고 알아서 결정해서 만들어줘").
> 결정마다 이유를 남겼습니다. 바꾸고 싶은 결정이 있으면 해당 절을 보면 됩니다.

---

## 1. 한눈에 보기

```
┌──────────────────────── hsl_src (새로 만든 것) ────────────────────────┐
│ hsl_behavior (Py)   전략: 역할 배정·공 담당 선정 → 스킬: 걷기/공 차기/탐색 → 목 제어 │
│        ▲ /hsl/team_state           │ cmd_vel_behavior, /kick_trigger, /neck/*   │
│ hsl_team_comm (Py)  UDP 10000+팀번호 │                                            │
│        ▲ /hsl/robot_pose, /hsl/ball│                                            │
│ hsl_world_model (C++)                                                          │
│   leg_odometry   → TF odom→base_link (관절각 FK + IMU heading)                   │
│   self_localizer → TF map→odom, /hsl/robot_pose  (B-Human SelfLocator 이식)       │
│   ball_tracker   → /hsl/ball, /hsl/team_ball                                     │
└────────────────────────────────────────────────────────────────────────┘
┌──────────────────────── mx3_src (그대로) ──────────────────────────────┐
│ twist_mux → lateral_correction → 보행 러너 ⇄ 킥 러너 (/kick_trigger, /kick_done) │
│ neck_control (/neck/yaw_deg, /neck/pitch_deg) · fall_manager · RSP(TF)          │
│ RealSense + YOLO → /perception/detections_3d  (시뮬: 브리지가 정답+노이즈로 대신) │
│ game_controller_receiver → /game_state, game_state_to_lock → twist_mux 락       │
└────────────────────────────────────────────────────────────────────────┘
```

빠진 mx3_src 노드 (hsl 이 대체): vision/EKF 측위, walk_odometry, pose_goal, neck_ball_tracker,
ball_follow, visual_servo, strategy, team_comm, depth_obstacle.

### 패키지

| 패키지 | 언어 | 내용 |
|---|---|---|
| `hsl_interfaces` | msg | RobotPoseEstimate, BallEstimate, TeamBall, Teammate, TeamState, StrategyState, SkillRequest |
| `hsl_world_model` | C++ | leg_odometry, self_localizer, ball_tracker, 필드 모델(S 필드 9×6 / practice 3.9×2.6) |
| `hsl_team_comm` | Python | 팀 메시지 51 B 송수신 → `/hsl/team_state` |
| `hsl_behavior` | Python | 전략(strategy.py) + 스킬(skills.py) + 목(head.py) + 노드(behavior_node.py), 단위 테스트 |
| `hsl_bringup` | launch | `robot.launch.py`(실기) · `sim_robot.launch.py`(시뮬) · `hsl_stack.launch.py`(공통), 경기 스크립트·장면 |

---

## 2. B-Human 대응표

| B-Human (Src/…) | 우리 | 이식 수준 / 차이 |
|---|---|---|
| Tools/Modeling/UKFPose2D | `ukf_pose2d.hpp` | **코드 이식** (수식 그대로, mm→m, float→double) |
| Modules/Modeling/SelfLocator (+UKFRobotPoseHypothesis) | `self_localizer_node.cpp` | **코드 이식**: 가설 12 개, 모션 노이즈 모델, validity 이동평균(60 프레임), 가중 재표본화, sensor resetting, 대칭 짝 처리, 진입/페널티 복귀 pose. 차이: 선(line) 관측 없음 → 아래 '프레임 절대 pose' 로 대체 |
| Modules/Modeling/PerceptRegistrationProvider | 같은 파일 `registerPercept` | 가설 pose 기준 최근접 같은 종류 랜드마크 (게이트 교차점 0.5 m, 골대 0.6 m) |
| Modules/Modeling/AlternativeRobotPoseProvider | 같은 파일 `frameCandidates` + `sensorResetting` | 같은 프레임 랜드마크 쌍의 기하 정합으로 절대 pose 후보 → 3 s 버퍼(오도메트리로 현재 시각에 옮김) → 6 회 이상 일치하면 최악 가설 교체 |
| Modules/Modeling/SideInformationProvider | `mustBeInOwnHalf`, `checkMirror` | 우리 진영 확정 = 진입·SET·페널티 복귀만. 팀원 공으로 좌우 반전 감지 |
| Modules/Modeling/OdometerProvider + 보행엔진 odometryOffset | `leg_odometry_node.cpp` | 직접 구현: 관절각 FK(KDL) 디딤발 고정 가정, heading 은 IMU |
| BallStateEstimator / BallLocator | `ball_tracker_node.cpp` | 단순화: 다중 가설 대신 단일 칼만 [x,y,vx,vy] (odom 좌표) + 3 연속 이상치면 재초기화, 등감속 마찰 |
| TeamBallModelProvider | 같은 파일 `publishTeamBall` | 팀원 공을 현재 시각으로 예측 → 0.6 m 군집 → 최대 군집 가중 평균. 자기 공 1 s 안이면 우선 |
| BallLostProvider | `time_since_disappeared` | 카메라 시야·깊이 안인데 안 보인 누적 시간 |
| ObstacleModelProvider | `ball_tracker` → `/hsl/obstacles` | YOLO `robot` 검출을 odom 좌표로 1.5 s 유지, 몸통 좌표로 10 Hz 발행 |
| TeamMessageHandler / TeamData | `hsl_team_comm` | 구조체 직렬화, 포트 10000+팀번호, 2 Hz + 즉시 송신 (공 담당·넘어짐·페널티 플래그 변화, 공 도달 시간 1 s 넘게 변화 → 바로 보내고 주기 시계 재시작, 토큰 버킷 0.25 Hz). HSL 2026 메시지 예산(팀당 게임당 12,000 개) → 실측 팀당 20 분 약 9,500 개. 처음엔 4 Hz 였다 (예산 규칙이 없다고 잘못 알았음). 고정 2 Hz 는 넘어짐이 늘어 즉시 송신을 넣었다 (5-3e) |
| BehaviorControl/StrategyBehaviorControl + Config/Behavior/Tactics | `strategy.py` | **개념 이식**: 위치(position) + 우선순위, 비용 행렬 배정, 공 담당 = 도달시간 최소 + 현 담당 보너스, 킥오프 배치. 차이: Voronoi 대신 공 위치의 규칙식 |
| Skills (GoToBallAndKick, WalkToPoint, SearchForBall), InWalkKick | `skills.py` | 개념 이식. 킥은 우리 RL 킥 정책(walkkick_v1)에 학습 분포 안에서 넘김 |
| HeadControl / LookActive | `head.py` | 공 추적(TF 닫힌 루프) + 주기적 좌우 훑기(측위용) + 탐색 스윕 |

이식하지 않은 것: CABSL 행동 언어, 자체 프레임워크(MODULE/STREAMABLE), SimRobot, 선 인식,
whistle, 세트플레이 세부(프리킥·코너 위치), 패스 대상 선택, 상대 추적(GlobalOpponentsTracker).

---

## 3. 결정 기록

| # | 결정 | 이유 |
|---|---|---|
| D1 | B-Human 프레임워크는 안 가져오고 ROS 2 노드로 이식 (A+B 혼합) | 사용자 결정. 하위 계층(mx3_src)이 ROS 2 이고 실기 검증돼 있음 |
| D2 | 월드모델은 C++, 행동·팀통신은 Python | 측위는 30 Hz × 12 가설 UKF 라 C++, 행동은 자주 바꾸므로 Python. 둘 다 Humble/Jazzy 공통 API 만 사용 |
| D3 | 인식은 기존 YOLO(`/perception/detections_3d`) 그대로 | 실기에서 돌고 있음. 클래스: ball, goalpost, robot, L/T/X-Intersection |
| D4 | 오도메트리를 '명령 적분' 에서 '다리 기구학' 으로 | 기존 walk_odometry 는 /cmd_vel 을 적분 → 정렬 진동·킥·밀림에서 5 분에 11 m 흘렀음. 관절각+IMU 로 바꾸니 시뮬에서 보행 이동량 비율 1.00, 1 초당 오차 1 cm |
| D5b | 선 측정 경로를 B-Human 그대로 이식해 두고 입력은 `/perception/field_lines` 로 분리 | 시뮬(정답 선분)로 효과를 먼저 확인 (5-3c). 실기는 선 검출기만 만들면 연결된다 |
| D5 | 선 대신 '프레임 절대 pose' 측정 | YOLO 가 선을 안 낸다. 같은 프레임 랜드마크 2 개 이상의 거리 정합으로 pose 후보를 만들고, **가설 근처에 후보가 하나뿐일 때만** 절대 pose 갱신. 센터서클 X 3 개가 30° 간격이라 위치만 보고 고르면 30° 별칭을 잡았다 (시험 1v0 에서 확인 → 수정) |
| D6 | 킥은 '정밀 정렬' 대신 '학습 분포 안이면 넘김' | 기존 visual_servo 의 정밀 정렬이 공 앞 앞뒤 진동의 원인. 킥 정책은 공 0.2~0.5 m · 방위 ±90° · 목표 ±90° 에서 걸어가 차도록 학습됨 → 거리 0.25~0.45, 방위 ±55°, 목표 ±35° 면 즉시 넘김 (6 s 넘게 못 맞추면 목표 ±60° 로 완화). B-Human 의 InWalkKick 과 같은 발상 |
| D7 | 골키퍼 = 1 번 고정, 나머지는 공 위치의 함수로 위치 생성 + 거리 비용 배정 | 4 인 팀이면 Voronoi/전술 파일까지는 과함. 우선순위 defender > supporter > midfielder (B-Human t112 와 같은 순서) |
| D8 | 공 담당은 PLAYING 에서만, 현 담당에 1.5 s 보너스 | 담당 진동 방지 (B-Human 과 같은 방식) |
| D9 | 우리 진영 강제는 진입(INITIAL→첫 READY)·SET·페널티 복귀 5 s 뿐 | 골 뒤 READY 까지 걸었더니 상대 진영에서 돌아오던 로봇이 거울 pose 로 뒤집혀 경기 끝까지 못 돌아옴 (4v4_b 의 t36p4) |
| D10 | 시뮬 로봇의 robot_params 섹션은 'sim' (common 만) | 실기 기체별 vy_trim·IMU 장착각은 시뮬 모델에 맞지 않음 |
| D11 | 시뮬은 로봇마다 ROS_DOMAIN_ID 분리, 팀통신·GC 는 실제처럼 UDP 브로드캐스트 | 스택이 절대 토픽 이름을 써서. UDP 는 SO_REUSEADDR 로 한 PC 의 여러 노드가 같이 받음 |
| D12 | 대회 조건(라이선스) 은 문서에만 | 사용자 지시. 아래 7 절 |
| D13 | 시뮬 로봇 충돌 모델 = 실제 메시 볼록 껍질 (XM_robot04_convexhull) | 사용자 요청 (2026-10-31). Isaac 충돌 제외 쌍을 MuJoCo exclude 로 옮기고 자가충돌 검증 (`scripts/mujoco_sim/README`) |

---

## 4. 노드별 요점

### leg_odometry
- `/joint_states` → KDL FK(XM_robot04.urdf) → 양 발바닥 중심(발목 롤 링크 + (0.057, 0, −0.04)).
- IMU 자세에서 yaw 를 뺀 기울기로 수평 정렬 → 더 낮은 발 = 디딤발(히스테리시스 4 mm).
- 몸통 이동 = −(직전 프레임 디딤발의 위치 변화). heading = IMU yaw(시작 0).
- `robot_state` ≠ NORMAL(넘어짐·기상) 이면 적분 중지.

### self_localizer (주요 파라미터는 `hsl_bringup/config/hsl.yaml`)
- 30 Hz. 오도메트리 증분 = TF odom→base_link 차분.
- 검출 시각의 몸통→카메라 TF 가 올 때까지 기다렸다 처리(최대 0.25 s).
- 회전 노이즈는 B-Human 값보다 작게 (heading 이 IMU 라 정확): odometry_deviation θ 0.05, rotation_deviation 0.2 rad/m.
- 출력 quality: SUPERB(validity ≥ 0.8, σ < 10 cm/10°) / OKAY / POOR(가설이 2 m·84° 넘게 갈라짐).
- `/initialpose` 로 강제 지정 가능. START 버튼·INITIAL 이면 사이드라인 진입 pose (x_inset = 0.7 × 선수번호, 기존 reentry 규칙).

### ball_tracker
- odom 좌표 칼만(로봇이 돌거나 걸어도 공 추정이 안 흔들림). 측정 노이즈 σ_r = 0.03 + 0.03 d².
- `/hsl/ball`: rel(몸통), field(지도), cam(카메라 광학 — 목 제어용), 속도, 본 지 경과, '보여야 하는데 안 보인' 시간.
- `/hsl/team_ball`: 자기 공 + 팀원 공.
- `/hsl/obstacles` (PoseArray, base_link): YOLO `robot` 검출. 행동 노드가 TF 를 안 듣도록 여기서 좌표 변환까지 한다.

### behavior
- 20 Hz. 상태: INITIAL·SET·FINISHED·페널티 = 정지 / READY = 킥오프 위치 / PLAYING = 전략.
- PLAYING: 팀 공 없음 → 탐색(제자리 한 바퀴 → 흩어진 탐색 자리) / 상대 킥오프 10 s·상대 세트플레이 → 공에서 거리 유지 /
  공 담당 → GoToBallAndKick / 그 외 → 역할 위치로 (팀원 pose·`/hsl/obstacles`·공 1 m 회피, 정면 0.45 m 로봇이면 전진 금지).
- TF 를 구독하지 않는다 (Python TF 리스너가 로봇당 CPU 15~30 % 였다).
- 보행 명령: 최소 노름 0.12 (러너 standing_threshold 0.1 아래면 제자리 정지), 가속 제한 0.6 m/s², 2.5 rad/s².
- 킥: 킥 러너에 SetParameters(ball_pos_b_x/y, kick_target_yaw_offset_rad) → `/kick_trigger` → `/kick_done` 대기(최대 6 s) → 1 s 쿨다운.

### team_comm
- 패킷 `<4sBBBBB4f5ffH` 51 B: 'HSLT', ver, team, player, flags(penalized/fallen/ball_valid/playing_ball), role,
  pose(x, y, θ, validity), ball(x, y, vx, vy, age), time_to_ball, seq.

---

## 5. 시뮬 검증 결과

환경: MuJoCo 브리지(`humanoid_sim`) + 로봇마다 스택 1 개, 벽시계 1 배속, 검출 = 정답 + 노이즈(σ = 0.01 + 0.01 d²) + 5 % 누락.
측위 오차는 정답을 측위 출력 시각으로 보간해 비교 (5-4 참고).

### 5-1. 경기 결과

| 시험 | 경기 | 결과 | 비고 |
|---|---|---|---|
| hsl_1v0_a | 1 대, 150 s | 2 골 (킥오프 후 14 s 만에 첫 골) | 킥 4 회, 넘어짐 0 |
| hsl_4v0_a | 4 대, 220 s | 3 골, 넘어짐 0 | 3 번이 차고 4 번(서포터)이 이어받아 득점하는 장면 반복 |
| hsl_4v4_a | 4v4, 300 s | 2 : 0 | 넘어짐 14 (상대 회피 없음) |
| hsl_4v4_b | 4v4, 300 s | 2 : 1 | 로봇 회피 추가 → 넘어짐 9. t36p4 좌우 반전 (D9 로 수정) |
| hsl_4v4_c | 4v4, 420 s | 1 : 2 | 측위 수정 전부 반영. 넘어짐 18 (14 회가 다른 로봇 0.6 m 안 충돌) |
| hsl_4v4_d | 4v4, 420 s | 2 : 0 | 회피 강화(정면 0.45 m 전진 금지, 비담당은 공 1 m 밖) → 넘어짐 10 |
| hsl_4v4_g | 4v4, 420 s | 1 : 2 | 브리지 스레드화·행동 노드 TF 제거 후. t35p4 가 START 전에 넘어져 180° 돌려 세워짐 → 브리지 수정 |
| hsl_4v4_h | 4v4, 480 s | 1 : 2 | 최종 (5-3) |

### 5-2. 시뮬로 찾아 고친 문제

| 문제 | 증상 | 원인 | 수정 |
|---|---|---|---|
| 30° 방향 튐 | 킥오프 위치에서 방향이 ±30° 로 튀었다 돌아옴 | 센터서클 X 3 개(30° 간격) 별칭 후보를 위치 거리로만 골라 절대 pose 갱신 | 가설 근처(1 m, 20°) 후보가 하나뿐일 때만 사용 |
| 리셋이 늦은 방향 주입 | 회전 중 30~50° 틀린 가설로 리셋 | 대안 pose 버퍼(3 s)를 오도메트리 보정 없이 평균 | 버퍼 각 관측을 오도메트리로 현재 시각에 옮김 (B-Human 과 동일) |
| 좌우 반전 | 골 뒤 READY 에서 거울 pose 로 뒤집혀 복구 안 됨 | READY 전체에 '우리 진영' 강제 | D9 + 팀원 공 기반 반전 감지 |
| 오도메트리 과소 | 보행 8 %, 킥 15 % 적게 | 디딤발 바뀌는 프레임 이동을 버림 | 직전 디딤발 기준 적분 → 보행 1.00 |
| 충돌 넘어짐 | 4v4 에서 14 회 | 상대 로봇 회피 없음 | YOLO robot 검출로 회피 (공 0.7 m 안 다툼은 허용) |
| 심판 오판 | PLAYING 시작 순간 공을 옮김 | 공 정체 타이머가 READY 부터 돌았음 | 상태 바뀔 때 타이머 리셋 |
| 모니터 착시 | 방향 오차 p90 13° 로 보임 | 10 Hz 정답과 최신값 비교 (회전 중) | 시각 보간 비교 → 실제 평균 0.9°, 빠른 회전 중 2.5° |
| 공 다툼 넘어짐 | 비담당 선수가 공 담당 옆으로 붙어 충돌 | 공 회피 반경 0.6 m | 공 담당이 아니면 공 1 m 밖, 정면 0.45 m 로봇이면 전진 금지 |
| CPU 포화 | 8 대에서 부하 18.7 / 20 스레드 | 브리지 busy-spin, Python 행동 노드의 TF 리스너(로봇당 15~30 %) | 브리지 executor 스레드화. 카메라 기준 공·로봇 장애물을 C++ ball_tracker 가 계산(`BallEstimate.cam_*`, `/hsl/obstacles`) → 부하 15 |
| 180° 돌려 세움 | START 전 넘어진 로봇이 반대 방향으로 세워짐 | 브리지 임시 기상이 누운 몸의 x 축으로 방향 계산 | 브리지 수정 (humanoid_sim README) + START 때 출발 pose 로 세움 |

### 5-3. 최종 4v4 (hsl_4v4_h, 480 s 모니터 ≈ 경기 시간 6 분)

| 로봇 | 측위 평균 / p90 [m] | 방향 평균 / p90 [°] | 공 추정 평균 [m] | 킥 | 넘어짐 |
|---|---|---|---|---|---|
| t35p1 (GK) | 0.034 / 0.062 | 0.6 / 1.3 | 0.26 | 4 | 1 |
| t35p2 | 0.099 / 0.233 | 2.6 / 4.8 | 0.15 | 11 | 6 |
| t35p3 | 0.064 / 0.141 | 1.3 / 2.7 | 0.14 | 12 | 5 |
| t35p4 | 0.066 / 0.121 | 1.1 / 2.2 | 0.16 | 10 | 1 |
| t36p1 (GK) | 0.034 / 0.075 | 0.6 / 1.5 | 0.22 | 7 | 0 |
| t36p2 | 0.072 / 0.148 | 1.5 / 2.9 | 0.18 | 10 | 5 |
| t36p3 | 0.066 / 0.128 | 1.4 / 2.8 | 0.15 | 10 | 2 |
| t36p4 | 0.065 / 0.124 | 1.4 / 2.6 | 0.16 | 12 | 4 |

- 점수 1 : 2, 킥 76 회, 좌우 반전 0, 0.5 m 넘는 측위 오차 시간 ≤ 1.2 %, 브리지 실시간 유지(지연 경고 1 회).
- **넘어짐 24 회 분석**: 22 회가 다른 로봇 0.6 m 안 → **전부 상대 팀**(같은 팀 충돌 0 — 팀원 회피는 동작).
  그중 14 회는 '킥 중' — 양 팀 공 담당이 같은 공을 동시에 차며 부딪힘. 판마다 편차가 큼(10~24 회 / 7~8 분).
  대응 후보(사용자 판단 필요): 상대가 공 0.5 m 안이면 킥 대신 몸으로 막고 기다리기 / 옆으로 비켜 차기(킥 목표각 이용) /
  RL 킥 정책을 '상대 있는 상황' 으로 추가 학습. 실기 규칙(pushing 반칙)과도 관련.

### 5-3b. 이전 스택 대비 측위 A/B (2026-10-31 오후, 1 대, 같은 시뮬·같은 심판·약 290 s)

측정: 정답 pose(10 Hz) 마다 같은 시각의 TF `map→base_link`(스택이 믿는 위치) 와 비교. READY·SET·PLAYING 만 집계.
이전 스택 = mx3_src 비전 측위 + EKF + walk_odometry, 밤 시험의 개선 설정(목 yaw 고정 해제·서보 최소속도·교착 수정) 적용.

| | 위치 평균 | 위치 p90 | 0.5 m 넘은 시간 | 방향 평균 | 방향 p90 | 골 |
|---|---|---|---|---|---|---|
| 이전 1 판 | 3.22 m | 4.26 m | 83 % | 63° | 92° | 0 (PLAYING 10 s 뒤 위치 상실, 끝까지 회복 못 함) |
| 이전 2 판 | 0.13 m | 0.30 m | 2.8 % | 11.0° | 27.7° | 있음 |
| 이전 3 판 | 0.24 m | 0.50 m | 10.2 % | 12.8° | 38.4° | 4 |
| **새 스택** | **0.051 m** | **0.11 m** | **0 %** | **0.8°** | **1.9°** | 4 |
| 새 스택 4v4 (view4, 페널티로 들려 나간 구간 제외) | 0.072 m | 0.125 m | 0.5 % | 1.6° | 2.4° | |

로그: `~/sim_ws/logs/loc_ab_old{,2,3}`, `loc_ab_new`. 행동 계층도 바뀌었으므로(정렬 진동 없음) '시스템 전체' 의 측위 정확도 비교다.

### 5-3c. 필드 선 사용 효과 (2026-10-31 오후, 1 대, 판당 약 290 s, 측정은 5-3b 와 같음)

선 측정 = B-Human `PerceptRegistrationProvider` 선 등록 + `UKFRobotPoseHypothesis::updateByLine` + `UKFPose2D::lineSensorUpdate`
이식 (`self_localizer`, `use_lines`). 입력 `/perception/field_lines` (`hsl_interfaces/LineSegmentArray`, 카메라 좌표 선분).
시뮬은 브리지가 정답 선분을 시야(87×58°)·깊이(0.2~3 m)로 잘라 노이즈(검출과 같은 σ)·누락 10 % 로 낸다.
센터서클·오검출·로봇에 가림은 없음. 'IMU 드리프트' = 브리지 `--imu_yaw_drift_dps 0.5` (yaw 가 0.5 °/s 로 흐름, 6 축 IMU 흉내).

| 조건 | 선 | 위치 평균 / p90 / 최대 [m] | 0.5 m 넘음 | 방향 평균 / p90 / 최대 [°] |
|---|---|---|---|---|
| IMU 정상 | 없음 | 0.051 / 0.11 / 0.48 | 0 % | 0.81 / 1.90 / 9.0 |
| IMU 정상 | 있음 | 0.060 / 0.12 / 0.42 | 0 % | **0.63 / 1.34 / 4.0** |
| 드리프트 0.5 °/s | 없음 | 0.070 / 0.13 / 0.65 | 1.7 % | 1.58 / 3.08 / **160** |
| 드리프트 0.5 °/s | 있음 | **0.050 / 0.10 / 0.37** | **0 %** | **0.99 / 2.14 / 5.4** |

- IMU 가 정확하면 선은 방향만 20~30 % 좋게 한다 (위치는 판 사이 편차 안).
- IMU 가 흐르면 선 효과가 커진다: 방향 평균 −37 %, 위치 평균 −29 %, 그리고 선 없을 때 한 번 생긴 큰 이탈
  (약 197 s, 0.65 m·방향 순간 160°) 이 선 있을 때는 없었다.
- 판마다 1 회씩이라 편차가 있다. 실기 효과는 선 검출기 품질(오검출·가림·센터서클)에 달렸다 — 검출기는 아직 없음.

### 5-3d. 실기 같은 검출 오차 + 가림 (2026-10-31 오후)

브리지 기본값을 실기 경험치로 바꿨다 (사용자: D405+YOLO, 작은 로봇이라 먼 물체 픽셀이 적음):
깊이 방향 σ = 2.5 m 까지 선형으로 10 cm, 그 뒤 급증해 4 m 에서 50 cm (`--depth_sigma 0:0.005,2.5:0.10,4.0:0.50`),
방위 σ 0.5°, 최대 검출 4.5 m, 로봇·공 가림(MuJoCo 광선), 오검출 없음 (실기에서도 공만 가끔이라 생략).
측위·공 추적의 측정 공분산도 같은 곡선 × 1.3 (`percept_range_sigma_knots`, `range_sigma_knots`).

| 시험 (선 없음) | 위치 평균 / 중앙값 / p90 / 최대 | 0.5 m 넘음 | 방향 평균 |
|---|---|---|---|
| 이전 스택 1 대 | 0.15 / 0.12 / 0.30 / 0.85 m | 2.9 % | 9.0° |
| 새 스택 1 대 (수정 전) | 0.16 / 0.027 / 0.12 / **7.1** m | **7.1 %** | 13.8° (180° 뒤집힘) |
| **새 스택 1 대 (수정 후) ×2** | **0.042~0.045 / 0.026 / 0.09~0.10 / 0.35** m | **0 %** | **0.7~0.8°** |
| 새 스택 4v4 (수정 전, 페널티 구간 제외) | 0.062 / 0.036 / 0.12 m | 0.8 % | 1.2° |
| 새 스택 4v4 (①② 후) | 평균 0.20 — 7 대는 0.025~0.07, **t35p2 만 1.35** | 4 % | 4.7° |
| **새 스택 4v4 (①②③ 후)** | **0.051 / 0.033 / 0.096 m** (로봇별 최대 0.23~0.96) | **0.3 %** | **0.95°** |

수정 (오차가 커지자 드러난 문제): ① 리셋용 절대 pose 후보를 2.5 m 이내 랜드마크로만 계산
(`candidate_max_percept_distance`) — 먼 랜드마크로 만든 후보가 센터서클 근처에서 180° 돌아간 위치로 리셋시켰다.
② 출력 가설 히스테리시스 (`best_switch_validity_margin` 0.05) — 맞는 가설과 대칭 가설의 점수가 비슷하면 출력이 0.1 s 마다 뒤집혔다.
③ **정지 상태에서 0 명령을 teleop 슬롯으로** (`stop_via_teleop_topic`) — t35p2 가 회전 중 페널티를 받고 30 s 내내
제자리 회전 → 복귀 때 '필드를 보고 서 있다' 는 가정과 방향이 달라 측위가 틀린 곳으로 수렴. 원인은 **twist_mux 는
game_stop_lock 이 걸리면 0 을 내지 않고 출력을 끊는다** (Jazzy 에서 직접 확인) → 보행 러너가 마지막 명령을 계속 실행.
실기 mx3_src 스택에도 같은 구조라 READY 가 끝날 때 걷던 로봇은 SET 중에도 걸을 수 있다 (Motion in Set 위험).
hsl 행동 노드는 INITIAL·SET·FINISHED·페널티 동안 0 을 `cmd_vel_teleop`(우선순위 100 > 잠금 50) 으로도 보내 막는다.
근본 수정(잠금 시 0 발행)은 mx3_src 쪽 일이라 미적용. 심판은 페널티가 끝날 때 로봇을 터치라인에 필드를 보게 다시 세운다.


### 5-3e 팀 통신 주기와 넘어짐 (2026-10-31, 4v4 화면 없음, 판당 PLAYING 약 5 분 × 3 판씩)

| 설정 | 송신 (로봇당) | 넘어짐 /분 (판별) | 킥 중 넘어짐 | 상대 0.6 m 안 킥 |
|---|---|---|---|---|
| 4 Hz (예산 초과) | 4.0 Hz | 0.58 (0.23 / 0.45 / 1.01) | 5.4 % | 32 % |
| 2 Hz 고정 | 2.0 Hz | 1.61 (1.81 / 0.91 / 2.27) | 13.2 % | 40 % |
| 2 Hz + 즉시 송신 (채택) | 2.0 Hz | 0.81 (1.01 / 0.45 / 0.91) | 6.2 % | 26 % |

- 2 Hz 고정의 넘어짐 증가는 팀원 충돌·팀원 위치 오차(p50 8 cm)·공 담당 공백과는 무관했고, 상대와 공에 동시에 도착하는 킥이 늘었다
  → 공이 굴러간 뒤 공 도달 시간·담당 변화가 팀원에게 최대 0.5 s 늦게 가는 탓으로 보고 바뀌는 순간만 즉시 보냄.
- 즉시 송신은 로봇당 분당 약 10 번. 즉시 보낸 뒤 주기 시계를 다시 세므로 총 송신은 2.0 Hz 그대로.
- 재현: `STACK_ARGS="team_send_rate_hz:=2.0 team_event_send:=false" SCENE=4v4 VIEWER=0 DURATION=380 run_match.sh --no_lines`.

### 5-4. 측위 정밀 측정 (1v0, 타임스탬프 정렬)

| | 정지·느린 회전 | 0.2~0.6 rad/s | 빠른 회전 > 0.6 rad/s |
|---|---|---|---|
| 다리 오도메트리 heading 오차 | 0.05° | 0.20° | 0.45° |
| 측위 출력 방향 오차 평균 / p90 | 0.9° / 2.1° | 1.4° / 2.9° | 2.6° / 4.0° (약 40 ms 지연) |

랜드마크 측정 편향(TF 체인 검증): 목 yaw −90~+90° 전 구간 방위 편향 |평균| < 0.5°, 표준편차 ~1.8°.

---

## 6. 한계와 다음 할 일

1. **실기 미검증.** `robot.launch.py` 는 Humble 에서 빌드·실행해 본 적 없음. 먼저 빌드 → 정지 상태에서 측위만 → 보행 → 경기 순서로.
2. **실기 다리 오도메트리 상수**: 발바닥 오프셋(0.057, 0, −0.04)과 URDF(XM_robot04) 는 시뮬 모델 기준. 실기 미끄러짐이 크면 비율 보정 필요 (D4 의 1.00 은 시뮬 값).
3. **인식이 이상적**: 시뮬 검출은 정답+작은 노이즈, 오검출·가림 없음. 실기 YOLO 는 오검출이 있으므로 self_localizer 의 게이트·validity 가 실제로 걸러내는지 확인 필요.
4. **세트플레이 미구현**: GC set_play 는 '상대면 거리 유지' 만. 프리킥·코너·골킥 배치, 페널티킥, 페널티 처리 시험 안 함.
5. **킥 목표 선택이 단순**: 항상 골대 안쪽 고정점. 상대 위치·패스 미고려.
6. **시뮬 기상 = 순간이동** (브리지가 3 s 뒤 세움). 실제 기상 시간(수 초)·실패가 반영 안 됨.
7. **브리지 CPU**: 8 대에서 단일 스레드 ~97 % (실시간 유지, 가끔 50 ms 지연 경고). 더 늘리려면 브리지 최적화 필요.
8. 공 마찰(ball_friction 0.5 m/s²)은 시뮬 공(평균 3 m 굴림) 기준 대략값.
9. ~~**공 다툼 충돌**~~ → `duel.py` 로 대응 (2026-10-31 오후): 상대가 공 0.6 m 안이면 상대가 더 가까울 때 막기(공과 우리 골대 사이
   0.6 m), 아니면 상대 쪽에서 50° 이상 비켜 차기, 몸 바로 앞 0.4 m 에 로봇이 있으면 킥 시작 안 함. 넘어짐 1.6~1.9 회/분 →
   0.17 회/분 (캡슐 모델 5.8 분) → 0 회 (메시 볼록 껍질 모델 3.9 분). 심판 밀기 판정도 '넘어지기 직전 누가 다가갔나' 로 바꿈
   (처음엔 서 있는 쪽을 가해자로 봐서 발에 차인 로봇이 퇴장당했다). 남은 것: 막기↔차기 판단이 자주 바뀜(로그 '공 다툼:').
10. **mx3_src twist_mux 잠금 = 출력 끊김** (5-3d ③): 잠금 동안 마지막 명령이 유지된다. hsl 은 teleop 슬롯 0 으로 우회했지만
    mx3_src 의 기존 스택(match.launch)은 그대로다. 근본 수정 후보: game_state_to_lock 이 잠금 때 0 Twist 를 보내거나,
    lateral_drift_correction 이 잠금이면 0 을 내기. (사용자 승인 필요)

---

## 7. 라이선스 (대회 전 할 일)

B-Human code release 2026 의 `License.md`(사본: `LICENSE_BHUMAN.md`) 조건:

1. 이식한 소스(`ukf_pose2d.hpp`, `self_localizer_node.cpp`)에 저작권 고지 유지 — **넣어 둠**.
2. 문서에 "This product includes software developed by B-Human (http://www.b-human.de)." — **이 문서에 명시**.
3. **대회에서 쓰려면**: 첫 대회 **1 개월 전까지** HSL 메일링 리스트(robocup-humanoid@lists.robocup.org)에
   사용 사실과 **사용한 부분**을 공지해야 함. 우리가 코드로 이식한 부분은 self-localization(UKFPose2D, SelfLocator,
   AlternativeRobotPose 방식) — '단일 컴포넌트' 에 해당한다고 판단. 전략·스킬은 개념만 참고(코드 이식 아님).
4. 만약 B-Human 시스템의 '주요 부분' 을 쓰는 것으로 보면(derived software): 공지에 자체 기여 설명 추가 +
   그 해 마지막 대회 버전 소스를 연말까지 공개·공지해야 함.

This product includes software developed by B-Human (http://www.b-human.de).
