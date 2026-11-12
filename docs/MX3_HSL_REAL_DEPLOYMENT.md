# MX3 원본 + 공식 B-Human 기반 상위 계층의 실기 적용

## 확정된 사용자 지시 — 2026-11-03

- 기존 355/345 관측 WALK/RECOVERY sim2real 작업을 중단한다.
- 원본 저장소는 `https://github.com/bhuman/BHumanCodeRelease`다. RoboEireann 저장소로 진행하지 않는다.
- 배포 기준은 `C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim/src/` 아래
  `mx3_src`와 `hsl_src`다. 사용자가 이 조합을 명시적으로 선택했다.
- `humanoid_sim`과 경기 시뮬레이터의 설치/실행 스크립트는 로봇 배포 대상에서 제외한다.
- 각 사용자 실행 결과를 받고 완료/다음/남은 작업을 안내한다.

## 원본 및 확인 근거

공식 저장소의 확인한 HEAD는 2026 릴리스 커밋
[`2e3374ba896da8288607f9a1881b876642fe18b8`](https://github.com/bhuman/BHumanCodeRelease/commit/2e3374ba896da8288607f9a1881b876642fe18b8)다.
[공식 초기 설정](https://docs.b-human.de/coderelease2026/getting-started/initial-setup/)의
실기 지원 플랫폼은 Booster K1/T1이다. 이 프로젝트의 XM540 모터 통신은 로컬 ROS 2 하위 계층으로 연결해야 한다.

공식 원본의 다음 파일을 실제로 읽고 로컬 이식 코드와 대조했다.

- `Src/Tools/Modeling/UKFPose2D.cpp`, `.h`: 로컬 `hsl_world_model/include/hsl_world_model/ukf_pose2d.hpp`와 대응.
  오도메트리 예측, 7개 시그마점, 랜드마크/선/절대 pose 갱신을 원문 수준에서 확인했다.
  로컬 코드는 단위 m와 double 자료형을 쓴다. 원본 바이너리와의 수치 동등성 시험은 아직 하지 않았다.
- `Src/Modules/Modeling/SelfLocator/UKFRobotPoseHypothesis.cpp`: 로컬 가설 초기화/validity/가중치 구현을 검토할 기준.
- `License.md`: 원문을 확보하고 출처와 함께 보존한다.

확보 파일은 `C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/bhuman_official_2026/`에 있다.
`PROVENANCE.json`에 저장소 URL, 커밋, 파일별 GitHub blob SHA와 원문 링크를 기록했다.
저장 파일의 Git blob SHA를 직접 다시 계산하여 4개 파일 모두 일치함을 확인했다.
이는 선택한 원본 파일의 확보이며 전체 공식 저장소를 복제한 상태는 아니다.

## 배포 구성

```text
/home/robo04/sim2real_ws/
  src/
    mx3_src/     # 사용자 지정 하위 계층 8개 패키지
    hsl_src/     # 사용자 지정 상위 계층 5개 패키지
  MX3_HSL_SOURCE_MANIFEST.json
  BHUMAN_PORT_NOTES.md
  reference/BHumanCodeRelease_2026/  # 선택 원문 + 라이선스 + 출처
```

하위 계층은 `humanoid_behavior_interfaces`, `humanoid_bringup`, `humanoid_game_controller`,
`humanoid_localization`, `humanoid_perception`, `humanoid_policy_runner`,
`humanoid_serial_bridge`, `humanoid_utils`다.
상위 계층은 `hsl_interfaces`, `hsl_world_model`, `hsl_team_comm`, `hsl_behavior`, `hsl_bringup`다.
패키지 이름은 총 13개이며 중복이 없어야 한다.

새 전송 묶음은 사용자 지정 소스의 바이트를 보존한다. 이 단계에서 임의의 알고리즘/모델/게인을 변경하지 않는다.
제외 항목은 Python bytecode와 캐시, 시뮬레이터 패키지, 기존 빌드 산출물이다.
하위 계층과 hsl_bringup 안에 있는 원래 모델은 포함한다.

## 현재 완료된 검사

- `mx3_src + hsl_src`의 Python 147개 파일이 Python 3.10 문법 검사를 통과했다.
- 13개 package.xml을 파싱하고 이름의 중복이 없음을 확인했다.
- 기존 `hsl_behavior/test/test_strategy.py`의 인자 없는 8개 테스트 함수를 직접 실행해 모두 통과했다.
  역할 배정, 공 담당, 킥 인계, 궤도 접근, 명령 shaping, 도착 latch, duel 판단 검사다.
  ROS 없이 수행한 테스트이며 노드 실행이나 실제 동작 시험은 아니다.
- 새 기준 하위 계층의 `servo_map.yaml`, `motor_gains.yaml`, `robot_params.yaml`은 앞서 배포한 소스와 바이트가 같다.
  로봇의 백업 대비 동일성은 앞선 사용자 출력에서 확인됐다. 실제 EEPROM/레지스터 값의 검증은 별도다.
- 최초 배포 묶음을 만들 때 로컬 소스 원본은 수정하지 않았다.
  이후 2026-11-04 실제 TF 수신 문제를 조사하며 ball_tracker의 부모 노드 TF 구독/진단 수정본을 준비했다.
  변경 파일, 해시와 검증 상태는 [진행 기록](MX3_HSL_DEPLOYMENT_PROGRESS.md)을 따른다.

## 실기 연결에서 확인할 항목

1. **Humble 빌드:** 로컬 문서는 Jazzy 시뮬 검증만 기록한다. `hsl_world_model`은 KDL/Eigen/tf2_eigen과 C++ 빌드를 추가한다.
   package.xml의 `orocos_kdl_vendor`와 실제 Humble에서 제공되는 KDL 패키지 구성을 대조한 뒤 필요한 변경만 적용한다.
2. **카메라:** 기준 소스의 `perception_bringup.launch.py`는 RealSense D405를 선택한다.
   앞선 워크스페이스에는 ZED Mini 테스트 기록도 있다. 로봇에 실제 연결된 장치를 확인하고 카메라 입력을 연결한다.
   장치가 다르면 해당 어댑터를 수정하며 카메라 모델을 임의로 가정하지 않는다.
3. **센서/TF:** `/joint_states`, `/imu/data`, `/imu/data_50hz`, `/robot_state`,
   `/perception/detections_3d`, `odom→base_link`, `map→odom`, `base_link→camera`의 시각/좌표/발행 주체를 확인한다.
4. **모델 선택:** 하위 `control.launch.py`의 기본 ONNX와 `hsl_bringup/models`의 시뮬 선택 모델은 다르다.
   기존 TensorRT 355/345 모델을 이 스택에 임의로 넣지 않는다. 실제 실행 경로와 ONNX 관측 규격을 먼저 확인한다.
5. **실행 순서:** `robot.launch.py`는 하드웨어·보행/킥·인지·게임컨트롤러·행동을 함께 시작한다.
   먼저 월드모델의 입력과 출력만 확인하고, 행동 명령과 구동을 단계적으로 연결한다.
   이 문서에 원래 launch 이름을 기록한 것이 즉시 전체 실행을 권고하는 것은 아니다.
6. **실제 입력:** 시뮬의 정답 검출/필드선/자동 기립을 실물 성능으로 세지 않는다.
   실물에는 실제 검출, 측위 품질, 게임 상태와 보행/킥 완료 신호가 필요하다.

## 사용자와 진행할 순서

1. 새 13개 패키지 묶음을 전송하고 SHA-256을 확인한다.
2. 실행 중인 노드를 확인하고, 현재 로봇의 소스/빌드를 별도 백업한 뒤 기준 소스로 교체한다.
3. Humble 의존성을 확인하고 빌드 오류를 해결한다. 전체 빌드 성공을 사용자 출력으로 확인한다.
4. 새 설치 경로/모델/실제 센서 장치와 토픽/TF를 확인한다.
5. 월드모델 → 행동 명령의 관찰 → 하위 제어 연결 → 실제 단일 로봇 동작 순서로 검증한다.

앞선 백업 `/home/robo04/sim2real_backup_20261103_VagdmK`와 모델/실패 bag/카메라·정책 환경은 보존한다.
이 작업의 빌드 성공과 실물 성공은 아직 확인되지 않았다.

This product includes software developed by B-Human (http://www.b-human.de).
