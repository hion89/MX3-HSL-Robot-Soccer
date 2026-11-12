# B-Human에서 가져온 부분과 현재 MX3 코드의 대응

검토 기준일: 2026-11-09.
대상 로컬 소스: `C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim/src/hsl_src`와 `src/mx3_src`.
공식 대조 기준: [BHumanCodeRelease의 고정 커밋 2e3374ba896da8288607f9a1881b876642fe18b8](https://github.com/bhuman/BHumanCodeRelease/tree/2e3374ba896da8288607f9a1881b876642fe18b8).

## 1. 지금 로봇에서 사용하는 방식

사용자가 기준으로 지정한 MX3_Match_Sim 폴더에는 이미 B-Human 대응 구현인 `hsl_src`가 들어 있었다.
이번 실물 작업은 그 소스와 MX3 하위 계층을 배포하고 실제 센서·정책·역할 제어에 연결한 것이다.
공식 GitHub 전체를 새로 받아 원본 실행 파일을 그대로 실행하는 방식이 아니다.

공식 소스 일부는 대조 목적으로 별도 `sim2real_evidence/bhuman_official_2026`에 보관돼 있다.
이 폴더의 PROVENANCE.json에는 기준 커밋과 선택한 원본 파일의 Git blob SHA가 기록돼 있다.
해당 커밋은 이번 검토의 대조 기준이다. 로컬 이식의 최초 작성자가 정확히 같은 커밋만 사용했는지까지
이 기록으로 소급해 증명되는 것은 아니다.

실행 흐름은 다음과 같다.

```text
실제 관절·IMU·카메라/YOLO
          ↓
hsl_world_model : 오도메트리 · 자기위치 · 공/팀 공
          ↓                    ↕ hsl_team_comm
hsl_behavior   : 역할/공 담당 결정 → 접근·정렬·탐색·목 동작
          ↓ cmd_vel / kick_trigger
mx3_src        : 기존 보행·킥 ONNX 정책 → 모터 제어
```

## 2. 수식·코드 구조의 이식이 직접 확인되는 부분

### UKF 위치 추정 핵심

| 항목 | 공식 원본 | 현재 로컬 구현 | 확인한 대응 |
|---|---|---|---|
| UKF 운동 예측 | `Src/Tools/Modeling/UKFPose2D.cpp` | `hsl_world_model/include/hsl_world_model/ukf_pose2d.hpp`의 motionUpdate | 7개 시그마점, 방향을 반영한 오도메트리 적용, 평균/공분산 계산, 과정·오도메트리 노이즈 추가 |
| 시그마점 생성 | 같은 원본의 generateSigmaPoints | 같은 로컬 파일의 generateSigmaPoints | Cholesky 분해와 양/음 시그마점 구성 |
| 랜드마크 갱신 | 같은 원본의 landmarkSensorUpdate | 같은 이름의 로컬 함수 | 상대 랜드마크 예측과 측정 잔차로 위치 보정 |
| 선 갱신 | 같은 원본의 lineSensorUpdate와 UKFRobotPoseHypothesis의 선 처리 | ukf_pose2d.hpp 및 self_localizer_node.cpp | 선 방향에 따른 관측 좌표/각도 선택과 갱신 경로 |

로컬·원본 함수를 직접 대조해 계산 순서와 수식 대응을 확인했다.
로컬은 Eigen 기반 double과 m 단위, 로컬 Pose2D 및 ROS 노드를 쓰므로 파일 바이트가 동일한 복사본은 아니다.
[공식 UKFPose2D](https://github.com/bhuman/BHumanCodeRelease/blob/2e3374ba896da8288607f9a1881b876642fe18b8/Src/Tools/Modeling/UKFPose2D.cpp),
[공식 UKFRobotPoseHypothesis](https://github.com/bhuman/BHumanCodeRelease/blob/2e3374ba896da8288607f9a1881b876642fe18b8/Src/Modules/Modeling/SelfLocator/UKFRobotPoseHypothesis.cpp).

### SelfLocator 구조

현재 `hsl_world_model/src/self_localizer_node.cpp`는 다중 위치 가설을 유지하고,
관절/IMU 기반 오도메트리로 예측한 뒤 관측으로 갱신하며 유효도와 재표본화로 가설을 관리한다.
현재 설정은 12개 가설이다. 최선 가설을 `/hsl/robot_pose`와 TF `map→odom`으로 내보낸다.
이 구조는 원본 SelfLocator의 가설 집합과 운동·관측 갱신 구조에 대응한다.
[공식 SelfLocator](https://github.com/bhuman/BHumanCodeRelease/blob/2e3374ba896da8288607f9a1881b876642fe18b8/Src/Modules/Modeling/SelfLocator/SelfLocator.cpp).

로컬에서는 기존 YOLO의 L/T/X·골포스트 검출과 RealSense 깊이를 사용한다.
랜드마크 대응, 같은 프레임의 랜드마크 쌍으로 만드는 위치 후보, 대칭 가설과 팀 공을 활용하는 보정이 들어 있다.
원본의 인식 입력·프레임워크까지 그대로 가져온 것은 아니다.
추정 오차를 실측으로 검증해야 한다는 점은 이식 여부와 별개다.

선 측정 입력 `/perception/field_lines` 경로도 코드에 있지만, 현재 실물 카메라에서 해당 선분을 공급하는
검출 노드의 동작은 확인되지 않았다. YOLO의 line 클래스 검출만으로 이 선분 입력이 활성화됐다고 보지 않는다.

## 3. 개념을 참고해 단순화하거나 다시 구현한 부분

다음은 로컬 설계 문서·코드와 원본 기능의 대응을 확인한 항목이다.
원본 전체 알고리즘과 같은 동작을 보장하거나 파일을 그대로 복사했다는 의미는 아니다.

| 기능 | B-Human 원본 대응 | 현재 로컬 구현 | 실제로 하는 일 / 차이 |
|---|---|---|---|
| 팀 전략·역할 | StrategyBehaviorControl, Behavior, Tactics | `hsl_behavior/hsl_behavior/strategy.py` | 공 담당 선정, 위치 역할과 우선순위, 거리 비용 기반 배정. 원본 Voronoi 전술 구조 대신 규칙식 위치 생성 |
| 공 담당 유지 | 원본의 active role 구조 | `choose_active`, `time_to_ball` | 공 도달 예상 시간과 현재 담당 보너스로 담당을 선정 |
| 공 접근과 킥 | GoToBallAndKick | `hsl_behavior/hsl_behavior/skills.py`의 GoToBallAndKick | 멀리 접근 → 공 주변 회전 → 공 뒤 정렬 → 학습 범위 안이면 우리 킥 정책 호출 |
| 탐색·목 동작 | BallSearch/Head 계열 설계 | 같은 패키지의 `skills.py`, `head.py` | 회전/이동 탐색, 공 추적, 주기적 랜드마크 훑기. 원본 탐색 파티클/행동 언어 전체 이식은 아님 |
| 공 추적 | BallStateEstimator | `ball_tracker_node.cpp` | 단일 4차원 칼만 상태와 마찰 예측, 이상치 게이트/재초기화. 원본의 전체 가설/인식 처리는 아님 |
| 팀 공 | TeamBallModelProvider | 같은 파일의 팀 공 계산 | 자기 관측과 팀원 공 정보를 이용한 군집/가중 융합. 원본 전체 구현과 같지는 않음 |
| 팀 메시지 | TeamMessageHandler/TeamData | `hsl_team_comm/hsl_team_comm/team_comm_node.py` | 위치·공·담당·넘어짐/페널티 등을 자체 UDP 패킷으로 공유. 주기+이벤트 송신 |
| 다리 오도메트리 | Odometer/보행 오도메트리의 역할 | `leg_odometry_node.cpp` | 우리 URDF/KDL과 관절·IMU로 직접 구현. 원본 보행엔진을 가져온 것이 아님 |
| 공 다툼·회피 | 원본 공 처리·킥 선택/상대 고려라는 설계 대응 | `duel.py`, behavior_node.py | 상대가 가깝다면 가로막기/비껴 차기/대기. 원본 Zweikampf 전체 이식이라고 할 근거는 없음 |

원본 전략은 [Behavior.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/2e3374ba896da8288607f9a1881b876642fe18b8/Src/Modules/BehaviorControl/StrategyBehaviorControl/Behavior.cpp)에서
위치 배정과 역할 배정을 처리한다. 로컬 코드가 언급한 수비 우선·전방·미드필더 우선순위는
[t112.cfg](https://github.com/bhuman/BHumanCodeRelease/blob/2e3374ba896da8288607f9a1881b876642fe18b8/Config/Behavior/Tactics/t112.cfg)의 priorityGroups와 대응한다.

공 접근/킥의 원본 비교 대상은
[GoToBallAndKick.cpp](https://github.com/bhuman/BHumanCodeRelease/blob/2e3374ba896da8288607f9a1881b876642fe18b8/Src/Modules/BehaviorControl/SkillBehaviorControl/Skills/Ball/GoToBallAndKick.cpp)다.
현재 로컬은 그 스킬 이름과 접근→킥 연결 개념을 사용하면서 실제 킥 동작은 제공된 ONNX 정책으로 넘긴다.

팀 공 및 통신 원본 비교 대상은
[TeamBallModelProvider](https://github.com/bhuman/BHumanCodeRelease/blob/2e3374ba896da8288607f9a1881b876642fe18b8/Src/Modules/Modeling/TeamBallModelProvider/TeamBallModelProvider.cpp)와
[TeamMessageHandler](https://github.com/bhuman/BHumanCodeRelease/blob/2e3374ba896da8288607f9a1881b876642fe18b8/Src/Modules/Communication/TeamMessageHandler/TeamMessageHandler.cpp)다.
현재 통신은 `HSLT` 헤더의 자체 51바이트 구조체이며 원본 B-Human 패킷을 그대로 사용하지 않는다.

## 4. B-Human에서 가져온 것으로 분류하지 않는 것

| 항목 | 현재 출처/상태 |
|---|---|
| 보행 ONNX 가중치 | 제공된 MX3/HSL 폴더의 walk_v0_6_13500.onnx |
| 킥 ONNX 가중치 | 제공된 MX3/HSL 폴더의 walkkick_v1_5b_22300.onnx |
| 관측 조립·관절 순서·모터 출력 러너 | mx3_src의 humanoid_policy_runner |
| 모터 통신·버튼·토크/기립 | mx3_src의 humanoid_serial_bridge |
| RealSense·YOLO | mx3_src의 humanoid_perception |
| 로봇별 IMU·보행 trim/모터 맵 | MX3 로봇 설정 |
| 경기장 치수 | 프로젝트의 FieldModel/Field 설정. 실제 바닥과의 일치는 따로 확인 |
| RViz 관찰 노드 | 이번 작업에서 추가한 hsl_view_20261109 |

CABSL/MODULE/STREAMABLE 프레임워크, SimRobot, 원본 인식 파이프라인·보행/킥 엔진,
휘슬 검출, 패스 대상 선정과 상세 세트플레이, 전체 상대 추적을 실물에 이식했다고 말할 근거는 없다.
골키퍼 위치 선정이 있다는 사실만으로 원본의 모든 골키퍼 동작/다이빙이 들어 있다고 보지 않는다.

## 5. 이번 실물 연결 작업에서 추가·수정한 것

- 제공된 mx3_src+hsl_src를 로봇에 배포하고 빈 디렉터리 누락과 의존성/빌드 문제를 해결했다.
- ball_tracker의 실제 TF 수신 문제를 수정해 현재 v3 경로를 연결했다.
- 고정 공격/수비/골키퍼 선택과 실행 중 역할 변경 인터페이스를 추가했다.
- 전달받은 최종 보행/킥 가중치를 실제 제어 launch에서 명시적으로 선택했다.
- 정지·명령 유효시간과 킥 활성화 제어를 보완하고 실제 역할 제어 스크립트를 만들었다.
- 기본 기능 통합 실행과 위치 관찰용 RViz 구성을 추가했다.

이 항목들은 이번 세션의 실물 통합 수정이며, B-Human 공식 GitHub에서 그대로 내려받은 기능으로 설명하지 않는다.
정답 폴더의 기존 이식과 이번 통합 수정을 구분해서 이력을 남긴다.

## 6. 근거와 검증 범위

| 근거 | 무엇을 확인하는가 |
|---|---|
| `hsl_src/DESIGN.md` | 작성자가 설명한 이식 대응표와 단순화 결정. 선언만으로 완전한 동일성을 증명하지 않음 |
| 로컬 소스의 실제 함수와 토픽 | 지금 들어 있는 동작과 연결 방식 |
| 고정 커밋의 공식 함수 대조 | UKF 계산 및 원본 기능/경로의 대응 |
| `bhuman_official_2026/PROVENANCE.json` | 선택 보관한 원본 파일의 출처와 blob SHA |
| 사용자 실물 관찰 | 실제 이동·공 탐색 접근·RViz 표시의 성공 보고 |

자기위치 수치 정확도, 실물 킥/골키퍼 성능, 다중 로봇 통신과 경기 완주 여부는 별도 결과가 필요하다.
시뮬 설계 문서에 있는 성능 수치를 이번 실물의 성능으로 옮겨 적지 않는다.

원본 고지 파일은 `hsl_src/LICENSE_BHUMAN.md`에 있고 일부 이식 C++ 파일에도 출처 고지가 있다.
문서/배포에서도 해당 출처를 유지한다. 이 문서는 코드 출처 확인 기록이다.
