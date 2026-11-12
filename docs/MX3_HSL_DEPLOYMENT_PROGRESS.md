# MX3 + HSL 실기 배포 진행 기록

기준일: 2026-11-03. 현재 목표와 구성은 [실기 적용 기록](MX3_HSL_REAL_DEPLOYMENT.md)을 따른다.

## 사용자 출력으로 확인한 완료 항목

- 기준 소스: `C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim/src/mx3_src`와 `hsl_src`.
- 공식 원본: `https://github.com/bhuman/BHumanCodeRelease`, 확인 커밋 `2e3374ba896da8288607f9a1881b876642fe18b8`.
- 새 묶음: `mx3_hsl_robot_20261103.tar.gz`, 총 13개 ROS 패키지, 원본 소스 280개 파일.
- 묶음 전체 파일 해시 목록은 286개이며 별도 manifest 파일 자체는 이 개수에 포함하지 않는다.
- 압축파일 SHA-256: `5914b1502594633f9b2c1f467b3534dab2aa737f6a66a8afb33c758815144a00`.
- 사용자가 로봇에서 `mx3_hsl_robot_20261103.tar.gz: OK`를 확인했다.
- 이어 사용자 출력으로 임시 경로의 `SOURCE VERIFIED: 286 files`, `PACKAGES VERIFIED: 13`,
  `MX3_HSL_SOURCE_REPLACED`를 확인했다. 로봇 소스 교체가 완료됐다.
- 직전 9개 패키지 소스/빌드의 새 백업은 `/home/robo04/mx3_hsl_backup_20261103_FjGF0d`다.
- 사용자가 `rclcpp`, `tf2_eigen`, `kdl_parser`, `vision_msgs`, `libeigen3-dev`, `liborocos-kdl-dev` 설치를 확인했다.
- 첫 13개 패키지 빌드는 6개 완료 후 `hsl_world_model` 설치 단계에서 실패했다.
  3개 C++ 실행 파일 설치 항목까지 진행했으나 소스의 `config` 디렉터리가 없어 중단됐다.
  실패 요약: `6 packages finished`, `1 package failed`, `6 packages not processed`.
- 로컬 원본의 `hsl_world_model/config`는 빈 디렉터리이고, 전송 묶음에 해당 디렉터리가 없음을 확인했다.
  묶음 생성 시 파일만 수집하여 빈 폴더를 누락한 것이 직접 원인이다. 실제 공통 설정은
  `hsl_bringup/config/hsl.yaml`에 있으며 이번 문제는 설정값이나 C++ 컴파일 오류가 아니다.
- 원본의 빈 디렉터리 3개 복원 명령을 안내한 뒤 사용자가 재빌드 결과
  `Summary: 13 packages finished [31.9s]`를 제공했다. 전체 Humble 빌드 완료를 확인했다.
- 사용자 출력으로 `hsl_world_model`과 `hsl_bringup`이 현재 workspace install 경로에 있음을 확인했다.
  `leg_odometry`, `self_localizer`, `ball_tracker`, `behavior`, `team_comm` 실행 파일이 등록됐다.
- ROS_DOMAIN_ID=42 / ROS_LOCALHOST_ONLY=1 조회의 node list는 비어 있었다.
- USB 장치: RealSense D405 (`8086:0b5b`), STM32 CDC (`0483:5740`), FTDI (`0403:6014`).
  STM32 by-id `usb-STMicroelectronics_STM32_Virtual_ComPort_3657367C3036-if00`는 `/dev/ttyACM0`,
  FTDI by-id `usb-FTDI_USB__-__Serial_Converter_FTB8HKSR-if00-port0`는 `/dev/ttyUSB0`를 가리킨다.
  두 tty 장치는 root:dialout, 권한 0660이다. 사용자 dialout 그룹/실제 통신은 아직 미확인이다.
- 2026-11-04 사용자가 하드웨어 전용 launch와 버튼 기립 안내 후 "자세 유지중이고 로그 잘나와"라고 보고했다.
  사용자 관찰상 자세 유지는 확인됐다. 정확한 도착/진단 로그와 센서 수신 수치는 아직 받지 않았다.
- 이어 사용자 측정으로 `/joint_states` 약 48.03~48.09Hz, `/imu/data` 약 200Hz,
  `/imu/data_50hz` 약 50Hz 수신을 확인했다. 관절 메시지 payload 자체는 최초 echo에서 받지 못했다.
  IMU 한 샘플의 quaternion norm은 약 1, roll=-0.086°, pitch=2.171°, yaw=-32.701°, tilt=2.173°다.
  이는 해당 시각의 IMU 계산값이며 센서 장착각 교정이나 전체 균형 검증을 뜻하지 않는다.
  behavior_status는 `behavior=none`, `active=false`였다. timeout 종료도 동일 상태가 되므로 도착 성공 증거는 아니다.
- 최초 node list에는 `/imu_parser`, `/neck_scan`만 보였지만 이후 관절과 behavior 데이터가 수신됐다.
  실제 발행 노드 부재로 단정하지 않고 CLI graph/초기 discovery 문제 가능성을 먼저 확인한다.
  기존 하드웨어 launch를 중단하거나 재시작하지 않는다.
- 이후 `--no-daemon --spin-time 5` 조회에서 하드웨어 8개 노드가 모두 확인됐다.
  타입과 sensor_data QoS를 명시한 직접 구독으로 관절 payload도 수신했다.
  고유한 관절 20개는 `hsl_world_model/urdf/XM_robot04.urdf`의 가동 관절 이름 집합과 정확히 같다.
  각도는 모두 유한값, 속도는 해당 샘플에서 모두 0, effort는 비어 있다.
  소스의 초기자세 대비 최대 각도 차이는 왼 무릎 0.02258rad(약 1.294°)였다.
  이 한 샘플은 초기자세 근처임을 보이지만 장시간 안정성/부하 성능 검증은 아니다.
- leg_odometry 단독 실행 후 사용자가 `/hsl/odom` 약 47.78~48.12Hz와 두 pose 샘플을 제공했다.
  첫 x/y/z는 0.0002288/-0.0001203/0.3959255m, 끝은 0.0008476/-0.0002652/0.3959415m다.
  두 샘플 사이 수평 변위는 0.6356mm, z 변화는 0.0160mm다. 중간 최대값이나 실제 이동 거리 오차는 알 수 없다.
  yaw는 -1.7300° → -3.9761°로 -2.2461° 변했다. 실제 몸통 회전 여부는 추가 확인 대상이며,
  정지 상태였다면 IMU 기반 방향 누적 오차 가능성을 추적해야 한다. 헤더 시각이 없어 초당 드리프트는 계산하지 않는다.
  현재 leg_odometry의 heading은 IMU yaw를 시작 yaw에서 뺀 값이다. IMU 파서는 gyro+accel Madgwick을 사용한다.
  z=약 0.396m는 FK 추정 높이이며 실측 높이와 대조한 결과는 아니다. TF 수신도 아직 별도로 확인하지 않았다.
- 사용자는 작은 각도 변화는 오차로 고려해 진행하라고 지시했다. 해당 변화를 현 단계 차단 항목에서 제외한다.
  별도 yaw 튜닝이나 보정값 변경 없이 카메라/TF/실제 검출 연결을 계속한다.
- D405 CameraInfo 수신: 848×480, `camera_color_optical_frame`, plumb_bob,
  fx=433.99249, fy=433.00204, cx=422.15857, cy=240.70897.
  컬러는 약 29.8~30.0Hz, 정렬 depth는 약 29.0~29.75Hz 수신됐다.
  주기 조회의 초기 토픽 경고 후 실제 메시지가 수신됐으므로 경고만으로 통신 실패로 보지 않는다.
- 사용자가 TF 조회 결과를 첨부했다. `base_link→camera_color_optical_frame`와
  `odom→camera_color_optical_frame` 모두 초기 discovery 대기 후 연속 변환을 수신했다.
  몸통 기준 카메라 위치는 약 [0.084, 0.001, 0.154]m로 반복 샘플에서 일정했다.
  odom 기준 카메라 높이는 약 0.550m다. 이는 연결/출력 확인이며 외부 실측으로 보정한 결과는 아니다.
  optical 축 변환이 포함된 RPY를 기체의 roll/pitch로 해석하지 않는다.
  마지막 SIGINT/SIGTERM은 timeout 명령으로 조회 프로세스가 끝난 것이다.
  원본 로그는 `sim2real_evidence/observations_20261104/camera_tf.txt`에 보존했다.
- 원본 `humanoid_perception/models/best.engine`의 JSON 메타데이터를 직접 읽었다.
  detect, batch=1, imgsz=640×640, FP16, Ultralytics 기록 버전 8.4.53이며 클래스 9개는
  노드/launch의 매핑(ball, goalpost, robot, field_edge, line, X/T/L-Intersection, obstacle)과 정확히 같다.
  engine SHA-256은 `7554a441d022ee5dfe1a22f8f3c6f0bd0fefa2847df945b77a1edaf4829368d6`다.
  이는 모델 파일 메타데이터 확인이며 Jetson에서 실제 검출 추론은 아직 실행 결과가 없다.

- YOLO 실제 검출 결과를 사용자 첨부로 확인했다. frame은 `camera_color_optical_frame`,
  공 score=0.8933094, 광학 좌표 [0.0014902, -0.0042714, 1.128]m다.
  공 bbox의 영상 역투영 폭/높이는 약 0.1376/0.1500m이며 실제 규격/거리 대조는 아직 하지 않았다.
  같은 메시지에 T/L/X 교차점도 들어 있었고, 전체 검출 발행은 약 29.0~30.1Hz였다.
  원본 로그는 `sim2real_evidence/observations_20261104/detections_3d.txt`에 보존했다.
- 사용자가 실제 경기장을 3.9×2.6m 연습장으로 확인했다. 자기위치/행동의 field_profile은 `practice`를 사용한다.
- 다음 공 추정 단독 시험에서는 `valid`, `rel_x/rel_y`, `cam_valid`, `time_since_seen`을 먼저 확인한다.
  코드상 map→odom이 없으면 field_x/y에 odom 좌표를 대신 넣으므로, 자기위치 연결 전 필드 좌표는 검증 완료로 보지 않는다.

- 공 추정 실행 후 사용자가 `camera_color_optical_frame ... frame does not exist` 경고 한 건을 제공했다.
  시작 직후 일시적 탐색인지 지속적인 TF 수신 실패인지는 아직 알 수 없다.
  이전에는 전체 TF 연결이 실제 수신됐다. 이 경고만으로 연결 단절이나 노드 종료를 확정하지 않는다.
  다음은 valid 필터를 제거한 `/hsl/ball` 출력과 현재 `odom→camera_color_optical_frame` TF를 확인하는 것이다.
  하드웨어/카메라/TF 창을 유지하며 근거 없이 static TF를 추가하거나 파라미터를 바꾸지 않는다.

- 후속 조회에서 `/hsl/ball` 두 echo 모두 `The passed message type is invalid`가 발생했다.
  같은 구간의 외부 `tf2_echo odom camera_color_optical_frame`는 초기 discovery 후 연속 변환을 수신했다.
  사용자는 공 추정 쪽 `camera_color_optical_frame` 경고가 계속 반복된다고 확인했다.
  초기 경고로 종결하지 않는다. 조회 터미널의 hsl_interfaces Python 메시지 로딩과
  ball_tracker 자체의 TF 수신 문제를 구분해 진단한다. 아직 valid 공 추정값을 받은 적은 없다.
  다음은 조회 환경 재-source, 메시지 prefix/import/type-support 확인, 관련 프로세스의 ROS 환경 비교,
  필터 없는 BallEstimate 구독이다. 근거 없이 CMake/TF 수치/모터 설정을 변경하지 않는다.
- 사용자가 반복 경고 스크린샷도 제공했다. 지속 경고로 취급한다.
  다음 조치는 공 추정 실행창만 종료/재-source하고 BallEstimate import/type support를 확인한 뒤 단독 재실행하는 것이다.
  조회창도 현재 install을 다시 source하고 `/hsl/ball` 및 `/tf`, `/tf_static`의 발행/구독/QoS를 확인한다.
  다른 하드웨어/기립/카메라/오도메트리/robot_state_publisher 창은 유지한다.
  Humble의 기본 TransformListener는 `transform_listener_impl_*` 자체 노드를 만들어 구독하므로
  TF 구독자의 노드 이름이 hsl_ball_tracker와 다를 수 있다. listener 생성 패턴만으로 코드 오류를 단정하지 않는다.

## 2026-11-04 TF 수신 수정본 준비

- 사용자가 코드 검토의 정확성을 요구했다. 프레임 정의를 모르는 문제가 아니라, 외부 조회에서
  수신되는 TF가 BallTracker의 내부 버퍼에는 없어 보이는 현상임을 분명히 한다.
- 정확히 확인한 경로: leg_odometry의 `odom→base_link`, RSP의
  `base_link→neck_yaw_link→neck_pitch_link→camera_link`, RealSense 내부 TF를 통한
  `camera_color_optical_frame` 연결이다. 두 끝점 변환은 이미 실제 수신됐다.
- 기존 `tf_listener_(tf_buf_)`도 Humble 지원 API다. 이것만으로 원래 코드가 잘못됐거나
  실제 DDS/환경 원인이 확정됐다고 주장하지 않는다.
- 로컬 기준 소스의 ball_tracker_node.cpp를 `tf_listener_(tf_buf_, this, false)`로 변경했다.
  TF 구독을 부모 노드와 executor에 명시적으로 연결한다. 모든 TF 호출은 non-blocking이다.
  원본/변경 후 SHA-256은 각각 `0487c09b44dd8e32b150808640ec32c4a83555f891c581d476738f2714999e1e`,
  `0349808f7c3789dc634dbf9c6512c86d5569119c4a581dd7e0cf1f881680ea55`다.
- 프레임 정의와 모터/게인/모델은 변경하지 않았다. 검출 시각의 변환을 계속 요구한다.
  최신 변환은 진단에만 사용하며 대체 입력으로 쓰지 않는다.
- `TF_CONFIG`, 최초 `TF_READY`, `BALL_INITIALIZED`를 추가했다. 실패 시 3초마다
  실제 버퍼의 프레임 목록, TF 발행자 수, 검출 경과 시간, 최신/검출시각 변환 가능 여부와 오류를 출력한다.
- 패치는 `sim2real_evidence/deploy/ball_tracker_tf_fix_20261104_v2.tar.gz`로 준비했다.
  SHA-256은 `95ab3a51e35f340cba8dad0d3c703c6d1f056f6bcbbb863a534b91bf0cb4730b`다.
- 적용 도구의 원본 해시/백업/중복 적용/실행 중 차단/실패 복원 6개 테스트를 통과했다.
  Python 3.10 문법, Bash 문법, 압축 파일 readback도 통과했다.
  Windows 호스트에는 실행 가능한 ROS/WSL 빌드 환경이 없어 C++ 대상 빌드와 ROS 통합 시험은 미실행이다.
- 자동 실행 스크립트는 소스 백업/적용, hsl_world_model 단독 빌드, ROS 도메인 77 격리 시험,
  성공 시 실제 도메인 42의 공 추정 시작 순서다. 시험은 정적 TF late-join과 미래 시각 검출 거부,
  정상 시각 검출의 몸통 좌표 (1.1,0) 출력을 확인한다. 모터 명령이나 액추에이터 서비스는 사용하지 않는다.
- 현재 로봇에는 이 새 패치를 아직 적용하지 않았다. 프레임 수신 문제가 해결됐다고 보고하지 않는다.
  다음은 ball_tracker만 종료하고 수정 묶음을 적용/부분 빌드하여 위 시험을 실행하는 것이다.

## 2026-11-04 v2 실제 시험 실패 및 v3 수정

- 사용자가 v2 적용 결과를 제공했다. 소스 해시 검증과 `hsl_world_model` C++ 빌드는 성공했다
  (`Finished ... [55.6s]`, `Summary: 1 package finished [56.3s]`).
  로봇의 v2 적용 전 백업은
  `/home/robo04/sim2real_ws/patch_backups/ball_tracker_tf_20261104T081157Z_a663177k`이다.
- 뒤이은 격리 도메인 77 시험은 실패했다. `spin_thread=false`인 v2 수정 때문에
  Humble Buffer의 dedicated-thread 검사가 변환 조회를 거부했다. 이는 작성한 패치의 오류다.
  앞서 기록한 non-blocking 호출이면 전용 스레드가 필요 없다는 설명을 정정한다.
  최초 `PASS`는 미래 시각 입력이 무효라는 조건만 확인했으므로 TF 수신 성공의 증거가 아니다.
- 근거: Humble `tf2_ros/src/buffer.cpp`의 `Buffer::canTransform`은 ROS-time overload에서
  timeout=0이어도 dedicated-thread 여부를 검사한다. `transform_listener.hpp`의
  spin_thread=true 경로는 실제 전용 executor/thread를 만들고 buffer 플래그도 설정한다.
  공식 소스: https://github.com/ros2/geometry2/blob/humble/tf2_ros/src/buffer.cpp 및
  https://github.com/ros2/geometry2/blob/humble/tf2_ros/include/tf2_ros/transform_listener.hpp
- v3는 `tf_listener_(tf_buf_, this, true)`를 사용한다. 실제 전용 스레드를 실행하며,
  최신 TF로 검출 시각의 TF를 대체하지 않는다. 프레임 정의와 모터 설정은 그대로다.
  v2가 추가한 오류 원인은 확인했으나, 수정 전 원래의 지속적인 missing-frame 원인은 아직 확정하지 않는다.
- v3 격리 시험은 미래 시각 거부뿐 아니라 같은 C++ 버퍼에서
  `latest_odom=yes latest_base=yes`가 실제 확인되어야 첫 조건을 통과한다.
  이후 정상 시각 검출의 유효한 공 좌표 (1.1,0), v3 실행 마커, dedicated-thread 오류 부재를 확인한다.
  시험에 실패하면 실제 도메인 42의 공 추정은 시작하지 않는다.
- 로컬에서 적용 도구 7개 시험, Python 3.10 문법, Bash 문법 및 압축 파일 바이트 대조를 통과했다.
  v3의 Jetson C++ 빌드와 ROS 통합 시험은 아직 실행 전이다.
  현재 로봇 소스는 사용자가 적용한 v2이며, v3 적용 도구는 해당 해시를 인식하여 백업 후 갱신한다.
- v3 소스 SHA-256: `24f90fcc09275786e982a64daf998c813f802df1e030ae2d2943b59e1a1ba545`.
  배포 파일: `sim2real_evidence/deploy/ball_tracker_tf_fix_20261104_v3.tar.gz`.
  압축 SHA-256: `db0c2055521b4b02f7b6c55d71b73249917614b9ae3b070c892186b4818533c9`.
  추출 최상위 폴더는 `ball_tracker_tf_20261104_v3`이다.
- 사용자 실패 로그는 `sim2real_evidence/observations_20261104/ball_tracker_tf_v2_test_failure.txt`에 보존했다.
  원문 SHA-256: `b157e3f5b243a2f8e2180f0434f11a97606935720988caf179c9bf0e5f05229f`.
- 다음은 v3 전달/적용, world_model 부분 빌드, 격리 시험, 실제 `/hsl/ball` 확인이다.
  기존 하드웨어/카메라/오도메트리/RSP/YOLO 실행 창은 유지한다.

## 2026-11-04 v3 로봇 빌드·격리 시험 및 실제 초기화 성공

- 사용자 로그에서 v3 해시 검증, v2에서 v3로 소스 교체, C++ 빌드를 확인했다.
  `Finished ... [57.4s]`, `Summary: 1 package finished [58.0s]`이다.
  로봇 백업: `/home/robo04/sim2real_ws/patch_backups/ball_tracker_tf_20261104T082253Z_x7f2a5n0`.
- 격리 도메인 77 시험은 모두 통과했다. 최신 TF가 실제 존재하는 상태의 미래 시각 거부,
  노드 시작 전 발행한 정적 TF 수신, 몸통 공 좌표 (1.1000000238,0), v3 실행 마커가 확인됐다.
  마지막 출력은 `ISOLATED_TF_TEST_PASSED`이다.
- 실제 도메인 42에서도 약 2.8초 후 `TF_READY`와 `BALL_INITIALIZED`가 확인됐다.
  실제 초기 관측: base=(1.080,-0.011,-0.307)m, optical=(0.002,-0.001,1.098)m, score=0.905.
  이는 실제 검출 시각의 TF 수신과 초기화 성공이며, 지속 갱신과 실측 정확도는 별도 확인 대상이다.
- 실제 캐시가 입증한 경로는
  `odom -> base_link -> neck_yaw_link -> neck_pitch_link -> camera_link -> camera_color_frame -> camera_color_optical_frame`이다.
  이전에 RealSense 내부 중간 프레임을 생략해 기록한 경로를 이번 캐시로 구체화한다.
- 시작 직후의 프레임 누락은 이후 해소됐다. 반복 경고는 `latest_odom=yes latest_base=yes`인
  미래 extrapolation이며, 로그상 검출 시각과 최신 TF 간 차이는 약 7.735~16.243ms이다.
  processPending은 실패한 검출을 보관하여 다음 주기에 재시도한다. 약 0.25초 경과 시 폐기,
  대기열 5개 초과 시 오래된 항목 제거 조건은 그대로다. 경고만으로 최종 관측 폐기를 단정하지 않는다.
  `valid`는 최대 6초 이전 관측으로도 유지될 수 있어 `time_since_seen`의 연속 측정이 필요하다.
- 다음 한 단계는 공을 정지시킨 채 `/hsl/ball`의 수신 주기, 유효 비율,
  `time_since_seen` 및 몸통 상대 좌표를 측정하고 실제 수평 거리와 대조하는 것이다.
  이 결과 전에는 경고를 숨기거나 타임스탬프를 임의 보정하지 않는다.
- 로그 보존: `sim2real_evidence/observations_20261104/ball_tracker_tf_v3_runtime.txt`.
  SHA-256: `733a3e3cfe4fa581e55a30ff853c9b61e66f55a8366317aa0bbf2fe828da2962`.

## 2026-11-04 후속 구독 시험: received=0

- 사용자가 도메인 42 및 localhost-only 환경에서 BallEstimate 12초 직접 구독을 실행했으나
  `received: 0`이었다. 메시지 import와 구독 생성은 성공했으므로 이전의 generic type 오류와 다르다.
- 소스상 cycle은 공의 검출 여부와 무관하게 `/hsl/ball`을 발행한다. 따라서 이 결과를
  공 미검출이나 좌표 오차로 해석하지 않는다. 앞선 실제 초기화 성공은 유지하되,
  현재 실행 지속 여부와 구독 전달은 확인 전이다.
- apply_and_build.sh의 마지막 명령은 foreground `exec ros2 run ... ball_tracker`이다.
  해당 창의 종료/Ctrl+C 여부는 아직 확인하지 않았다. 노드 종료로 단정하지 않는다.
- 다음 한 단계는 ball_tracker OS 프로세스, no-daemon ROS 노드 목록,
  `/hsl/ball` 발행자와 QoS 조회다. 실행 중인 노드가 확인되기 전 중복 기동하거나 소스를 다시 수정하지 않는다.

## 2026-11-04 received=0의 현재 원인 확인

- 사용자 조회에서 `pgrep -af '[b]all_tracker'` 출력이 없고, no-daemon 노드 목록에
  `/hsl_ball_tracker`도 없으며 `/hsl/ball`은 Unknown topic이다.
  현재 공 추정 프로세스 부재가 확인됐다. 종료 이유는 로그가 없어 확정하지 않는다.
- 하드웨어 8개 노드, 카메라, hsl_leg_odometry, robot_state_publisher,
  yolo_detector_3d는 같은 조회에서 확인됐다. 이 노드들은 유지한다.
- 다음은 설치된 v3 ball_tracker만 foreground로 재실행하고 stdout/stderr를
  `$HOME/sim2real_ws/runtime_logs/ball_tracker_<timestamp>.log`에 보존하는 것이다.
  빌드와 격리 시험을 반복할 필요는 없다. 실행 창을 유지한 상태에서 별도 터미널로
  `/hsl/ball`을 조회하고, 종료 시 로그와 실제 프로세스 종료 코드를 확인한다.

## 2026-11-04 재실행 후 실제 BallEstimate 수신 성공

- 사용자 최신 echo(1793781102.8774)에서 BallEstimate를 정상 수신했다.
  앞부분 1793778125~1793778133의 타입 오류/TF 조회는 과거 출력이므로 현재 실패로 재분류하지 않는다.
  환경 source 및 도메인 설정 뒤 실행한 최신 조회는 성공했다.
- 실제 출력: valid=true, cam_valid=true, rel_x=1.0864718m, rel_y=-0.0110548m,
  time_since_seen=0.0523476s, time_since_disappeared=0,
  camera=(0.00155046,0.0175689,1.1134373)m.
  이 표본은 최근 관측 갱신을 입증하며, 연속 수신 주기는 아직 측정 전이다.
- 사용자 스크린샷의 실행 로그 경로는
  `/home/robo04/sim2real_ws/runtime_logs/ball_tracker_20261104_173126.log`이다.
  v3 marker, TF_READY, 실제 BALL_INITIALIZED(base 약 1.089,-0.011,-0.311m)가 보인다.
- field_x=-0.63709m, field_y=-0.87703m은 소스의 map->odom 부재 시 odom 대체 동작으로도
  나올 수 있으므로 경기장 절대 위치 검증으로 취급하지 않는다. 자기위치 추정 단계에서 확인한다.
- 다음은 짧은 수신 주기 측정과 실제 수평 거리 대조다. 실측 위치는 사용자에게 요청했으며 답변 전이다.
  자기위치 추정은 field_profile=practice를 사용할 예정이다. 전체 hsl_stack은 오도메트리와
  공 추정 중복 및 행동 노드 자동 시작을 수반하므로 단독 self_localizer 실행을 준비한다.

## 2026-11-04 공 위치 사용자 대략 측정

- 사용자는 현재 공이 전방 대략 120cm라고 답했다. 정확한 줄자 측정값인지와 좌우 위치는 미확인이다.
- 직전 rel_x=1.0864718m와 단순 대조하면 약 0.1135282m(11.35cm), 기준 1.20m 대비 약 9.46% 짧다.
  서로 같은 순간의 정밀 측정이라는 근거는 없으므로 확정 센서 편향이나 보정 계수로 해석하지 않는다.
- 현재 다음 요청은 `/hsl/ball` 10초 수신 주기 측정 결과다. 해당 결과는 아직 받지 않았다.
  이후 정확한 거리 대조와 practice 자기위치 추정을 진행하며 임의의 깊이 배율/카메라 외부 파라미터 변경은 하지 않는다.

## 2026-11-04 사용자 상대 거리 수용 및 자기위치 단계 준비

- 사용자는 공의 상대 거리가 '얼추 맞는 것 같다'고 확인했다. 현재 시험의 상대 거리 확인을
  사용자 수용으로 마무리하고 자기위치 단계로 진행한다. 정밀 깊이/외부 파라미터 보정 성공을 뜻하지 않는다.
  `/hsl/ball` 연속 주기 결과는 아직 받지 않았으며 측정 완료로 기록하지 않는다.
- 소스를 재확인했다: self_localizer는 `/hsl/robot_pose`와 `map -> odom`을 발행하며
  모터/보행 명령 발행자는 없다. 기본 TF listener는 실제 전용 스레드가 있는 유효한 Humble API다.
  이전 ball_tracker v2의 false 설정을 이 노드에 적용하지 않는다.
- practice 지도는 3.9 x 2.6m이며 원점은 중앙, +x는 상대 골대 방향, +y는 그 방향을 볼 때 왼쪽이다.
  내부 골대/센터서클/영역 치수도 9x6m 지도에서 같은 비율로 축소된다.
  경기장 외곽 크기만으로 내부 랜드마크 배치까지 실제와 같다고 확정하지 않는다.
- 기본 초기값은 `walkInPose()`의 사이드라인 밖 진입 위치다. player 번호는 이 노드 단독 실행에서
  inset을 자동 계산하지 않는다. 실제 로봇 위치와 다를 수 있어 정상 위치로 간주하지 않는다.
  `/initialpose`는 실제 위치·방향을 알려준 뒤 초기화할 수 있다. 이 메시지의 validity=0.8은
  초기 설정값이므로 이후 관측 갱신과 실측 비교 없이 위치추정 성공이라고 판단하지 않는다.
- 사용자에게 실제 연습장 내 현재 위치와 바라보는 골대 방향을 요청했다. 경기장 밖인지도 확인 전이다.
  답변에 따라 practice 단독 self_localizer 기동 및 실제 위치 기반 초기화 명령을 정한다.
  출력에서 num_landmarks_used, validity, quality, x/y/theta와 실제 위치를 확인할 예정이다.
  실제 위치를 모르는 상태에서 중앙이나 특정 선수 진입 좌표를 정답으로 임의 지정하지 않는다.
- 기존 하드웨어/카메라/RSP/YOLO/leg_odometry/ball_tracker는 유지한다.
  전체 hsl_stack은 행동 노드 시작과 기존 노드 중복을 수반하므로 이번에는 단독 self_localizer만 사용한다.

## 2026-11-04 자기위치 초기 위치 설명 수신

- 사용자는 '경기장 골대쪽에서 상대 골대를 바라보는 방향으로 왼쪽에 L마커 다음 T마커 두개 사이'라고 답했다.
  상대 골대를 보는 방향은 지도 convention상 theta 약 0이다. 실제 정렬 오차는 미측정이다.
- L-T 구간은 우리 진영 왼쪽 긴 터치라인의 코너 L~중앙선 T 구간일 수도 있고,
  우리 골대 쪽 짧은 골라인의 코너 L~다음 T 구간일 수도 있다. 초기 x/y 차이가 커
  둘 중 어느 구간인지 한 가지 선택 질문을 보냈다. 아직 답변 전이다.
  마커 사이의 정확한 위치/선에서 안쪽 거리도 제공되지 않아 중간점을 실측 정답으로 기록하지 않는다.
- 로봇 config에서 robo04의 player_number=4를 확인했다. 단독 노드에 넘겨야 하는 값은 확인됐고,
  적용할 초기 x/y는 구간 설명이 해소된 뒤 정한다. 소스의 /initialpose 핸들러는 입력 covariance를
  사용하지 않고 고정 0.1m/0.1rad 편차로 초기화하므로, 대략 위치를 정밀 위치로 과신하지 않는다.

## 2026-11-04 우리 골라인의 두 T 사이로 위치 구체화

- 사용자는 상대 골대를 보는 시선으로 우리 골대 쪽 맨 바깥 선의 왼쪽부터
  `L T 로봇 T` 순서라고 구체화했다. 이전의 L~T 사이라는 가설을 대체하여,
  우리 골라인 왼쪽의 두 T 사이로 해석한다. 방향은 theta 약 0이다.
- field_model.hpp practice 치수: x_goal=-3.9/2=-1.95m,
  T_penalty_y=(4.0*(3.9/9.0))/2=0.8666667m,
  T_goal_area_y=(3.0*(3.9/9.0))/2=0.65m.
  구간 중간 y=0.7583333m를 임시 초기값으로 사용한다. 중간점은 사용자 실측값이 아니며,
  실제 마커 배치가 코드 지도와 같다는 가정에 따른 초기 가설이다.
- 다음 실행 명령은 설치된 `hsl_world_model self_localizer` 단독 기동에
  hsl.yaml, field_profile=practice, player_number=4를 넘기는 것이다.
  새 노드의 기본 사이드라인 pose를 `/initialpose`의 (-1.95,0.7583333,theta=0)로 한 번 초기화하고,
  `/hsl/robot_pose` 출력 및 실행 로그를 확인한다. 아직 로봇에서 실행한 결과는 받지 않았다.
- theta=0의 quaternion은 (0,0,0,1)이다. 노드는 초기 covariance를 무시하고
  validity=0.8로 시작하므로 높은 초기 validity 또는 입력과 같은 좌표 자체를
  인식 기반 자기위치 성공으로 보지 않는다. 후속 관측의 num_landmarks_used와 위치 유지/보정을 확인한다.
- 관측 업데이트 확인 전까지 기존 센서/공 추정은 유지하고 행동/보행 노드는 시작하지 않는다.

## 2026-11-04 자기위치 실제 관측 갱신 첫 확인

- 사용자가 /initialpose 발행과 /hsl/robot_pose 출력을 제공했다.
  명령 표시 일부가 깨져 있지만 실제 publishing #1 메시지의 quaternion은 (0,0,0,1)이고,
  초기 좌표는 (-1.95,0.7583333)로 출력됐다. 손상된 표시만 보고 잘못된 초기화로 단정하지 않는다.
- 1793781707.545155137의 추정은 x=-1.7290766m, y=0.6117641m,
  theta=-0.18787086rad(약 -10.764도), validity=0.942476, quality=2,
  num_hypotheses=12, num_landmarks_used=8이다. last_jump=1793781662.076976289.
- 입력한 임시 중간점 대비 x +22.09cm, y -14.66cm, 수평 약 26.51cm 변경됐다.
  임시 초기값이 실측 정답은 아니므로 이를 실제 오차나 실제 로봇 이동량으로 해석하지 않는다.
- 센서 갱신 코드상 num_landmarks_used는 등록된 랜드마크와 선 관측 개수의 합이다.
  현재는 YOLO 교차점/골대가 주 입력이며, 이번 주기에 관측 8개가 사용됐다는 증거다.
  quality=2는 내부 SUPERB 판정으로 실제 절대 정확도를 독립 검증한 값이 아니다.
- 다음 한 단계는 로봇 정지 상태에서 20초간 /hsl/robot_pose를 구독하여 수신 주기,
  관측 사용 주기 수, 좌표 범위, 방향 범위와 점프를 확인하는 것이다.
  일정한 방향 편차만으로 IMU bias나 카메라 보정을 변경하지 않는다.
- 사용자 한국어 응답 요구를 확인했다. 이후 안내는 한국어로 유지한다.

## 2026-11-04 자기위치 정지 안정성 확인 및 행동 출력 시험 준비

- 사용자 20초 측정: 601 messages, 30.0Hz, 관측 사용 548/601회(약 91.18%).
  x=-173.32~-172.64cm(폭 0.68cm), y=60.29~61.79cm(폭 1.50cm),
  theta=-11.31~-10.09도(폭 1.22도). 마지막 값은 (-172.96cm,60.64cm,-10.25도).
  validity=0.98720485~0.99744117이며 quality=2가 601회 모두 유지됐다.
- 정지 상태의 연속 관측 갱신과 추정 안정성은 확인됐다. 실제 절대 위치/방향의 정밀 정확도나
  이동 중 정확도는 별도이며, 사용자 초기 방향은 대략값이므로 -10도 편차를 센서 고장으로 단정하지 않는다.
- 행동 노드 소스를 확인했다. 출력은 cmd_vel_behavior/teleop, neck yaw/pitch/speed,
  kick_trigger, strategy_state, behavior_debug의 8개 publisher와 킥 파라미터 service client다.
  행동 노드를 원래 출력으로 바로 띄우면 목에도 명령이 전달되므로 출력 시험을 먼저 진행한다.
- 실행 스크립트 `sim2real_evidence/deploy/start_hsl_behavior_shadow_20261104.sh`를 준비했다.
  실제 도메인 42의 pose/ball/team_ball/obstacles/목 상태를 입력으로 사용하고,
  8개 출력을 전부 `/hsl/shadow/`로 remap한다. GameState 입력도 시험 토픽으로 바꾸고
  이 노드에만 default_game_state=playing을 적용하므로 실제 경기 상태나 자기위치 초기화를 변경하지 않는다.
  enable_kick=false, neck_speed_rpm=0, stop_via_teleop_topic=false,
  kick_node=hsl_shadow_kick_target을 사용한다. 킥 서비스 호출 경로까지 시험 대상으로 분리한다.
- AST로 소스의 8개 publisher 전부가 remap 목록에 포함됨을 대조했고 Bash 문법을 확인했다.
  아직 로봇에서 해당 행동 시험을 실행한 결과는 없다.
  다음은 스크립트 전달/실행 후 `/hsl/shadow/behavior_debug`와
  `/hsl/shadow/cmd_vel_behavior`를 조회하여 실제 공을 따라가려는 판단과 명령을 확인하는 것이다.
- 행동 판단 후에는 twist_mux와 하위 보행 러너/모델 로딩을 확인한다.
  control.launch.py의 기본은 package-share/models/walk_policy.onnx, policy_variant=v0_2이며
  해당 모델의 로봇 파일 존재·정책 provenance·런타임은 아직 확인 전이다.
  과거 RL_walk TensorRT 검증을 이 ONNX 러너의 검증으로 대체하지 않는다.

## 2026-11-04 행동 출력 확인 후 사용자 요청으로 오늘 실험 중단

- 사용자 행동 시험 결과: PLAYING, player=4, active=4, skill=play_ball_align,
  kick=null, kicks=0. 시험 속도는 vx=0.2908923m/s, vy=-0.02147965m/s,
  wz=0.07878143rad/s(약 +4.514도/s)이다. 실제 보행/킥/목 제어로 연결한 시험은 아니다.
  행동 로그: `/home/robo04/sim2real_ws/runtime_logs/hsl_behavior_shadow_20261104_174932.log`.
- 사용자가 모터 과열 우려로 오늘 실험을 마치고 내일 같은 위치에 다시 세워 재개하겠다고 요청했다.
  로봇 실험 진행을 중단한다. 추가 기동/보행/추론 시험 명령을 요청하지 않는다.
  물리적 토크 해제/전원 종료를 실행했다는 답변은 아직 없으므로 완료로 기록하지 않는다.
- 종료 안내: 로봇을 먼저 안정적으로 지지한 뒤 STOP으로 토크 해제를 확인하고 모터 전원을 끈다.
  button_input_node.py의 기본 auto_torque_off=true 경로는 STOP에서 torque_enable(false)를 호출한다.
  기존 ROS 실행 창은 토크 해제 확인 후 종료한다.
- 재개 위치: 실제 연습장 3.9 x 2.6m, 상대 골대를 보는 방향으로 우리 골라인 왼쪽의
  `L T 로봇 T` 배치 중 두 T 사이. 임시 초기값 (-1.95,0.7583333,theta=0).
  마지막 정지 추정은 약 (-1.7296,0.6064,-10.25도)이었지만 이를 다음 날 실측 정답으로 쓰지 않는다.
- 내일(2026-11-05) 재개 순서: 냉각 후 하드웨어/센서/카메라/RSP/YOLO/leg_odometry,
  패치된 ball_tracker를 중복 없이 기동하고, self_localizer practice를 실제 위치로 초기화한다.
  ROS_DOMAIN_ID=42, ROS_LOCALHOST_ONLY=1을 모든 터미널에 유지한다.
  기존 pose/관측 히스토리는 재시작 시 사라지므로 실제 공·자기위치 수신을 짧게 재확인한다.
- 다음 미완료 단계: 보행 모델 정체/로봇 파일/ONNXRuntime 확인 →
  보행 러너 시험 출력 및 명령 중재/정지 확인 → 하위 제어 연결 → 실제 저속 동작 시험.
  기존 shadow 행동은 출력 분리된 시험이며 정상 실제 행동 연결로 대체하지 않는다.
- 재개 전 기억할 모델 차이: source README와 sim_robot.launch.py의 최종 보행 모델은
  hsl_bringup/models/walk_v0_6_13500.onnx이나, 실기 robot.launch.py가 포함하는
  control.launch.py 기본은 humanoid_policy_runner/models/walk_policy.onnx이다.
  README는 후자를 v0_8 18400 사본으로 설명한다. 로컬 바이너리와 전달 manifest의 해시도 다르다.
  최종 보행 SHA=72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1,
  러너 기본 SHA=9f6efa34d19cf90382a056f01ad9add5e71596d528cafe4855e04d1f3ba11531.
  킥도 hsl의 v1_5b 22300과 러너 사본 18200이 서로 다르다.
  다음 날 전체 launch 기본값으로 모터에 연결하기 전에 모델 선택/전달 경로를 맞춰야 한다.
  로봇 ONNXRuntime 로딩과 설치 모델 해시는 아직 확인 전이다.
- 별도 재개 메모: `sim2real_evidence/NEXT_SESSION_20261105.md`.

## 2026-11-05 재개 요청 및 터미널별 명령 정리

- 사용자가 다시 진행할 때 필요한 터미널별 명령을 요청했다. 전날의 중단 요청은 이번 재개 요청으로 해제됐다.
  아직 오늘 실제 로봇에서 기동한 결과는 받지 않았다.
- 재개 안내를 `sim2real_evidence/RESTART_TERMINALS_20261105.md`에 저장했다.
  1 하드웨어, 2 카메라+RSP, 3 YOLO, 4 leg_odometry, 5 ball_tracker v3,
  6 self_localizer practice, 7 행동 shadow 시험, 8 초기화/조회 탭으로 구성한다.
- 카메라+RSP는 소스의 기존 perception_bringup.launch.py에서 enable_camera=true,
  enable_yolo=false, enable_ball_map=false, rs_camera_profile=848x480x30으로 함께 시작한다.
  어제 사용한 RealSense leaf launch와 동일 profile/align 설정을 소스에서 확인했다.
- 하드웨어 포트는 어제 확인한 STM32/FTDI 고유 by-id 경로를 사용한다.
  실제 기동 전에 노드 목록과 /dev/serial/by-id를 확인하여 기존 노드 중복을 피한다.
- 모터 기립 유지 시간을 줄이기 위해 1~6번 창을 준비한 뒤 START 기립,
  그 후 /initialpose 입력과 실제 공/자기위치 조회, 마지막으로 행동 shadow 시험을 진행한다.
  START 이후 /initialpose를 입력해야 START 이벤트의 자기위치 reset으로 덮어써지지 않는다.
- 재빌드/패치 적용 반복은 필요 없다. 실제 보행 모델과 명령 중재 확인은 복구 이후 미완료 단계로 남는다.

## 2026-11-05 행동 시험 출력 복구 확인, 보행 모델 로딩 확인 준비

- 사용자 재개 조회에서 최초 노드 목록은 비어 있었고 STM32/FTDI by-id 두 장치는 어제와 같았다.
  이어 실제 행동 시험 출력이 수신됐다: player=4, PLAYING, active=4,
  skill=play_ball_align, kick=null, kicks=0.
  시험 Twist는 vx=0.36615082m/s, vy=-0.03559922m/s, wz=0.03555022rad/s이다.
  debug 문자열은 CLI에서 잘려 전체 pose/ball 필드는 이번 출력으로 확인할 수 없다.
- 행동 판단과 시험 출력 전달은 재개 확인됐다. 오늘 실제 공/자기위치 표본이나
  정지 안정성의 새 수치는 아직 제공되지 않았으므로 어제 수치를 오늘 측정으로 기록하지 않는다.
- 보행 러너는 아직 기동하지 않는다. 다음 한 단계는 선택 모델의 로봇 파일 해시,
  NumPy/ONNXRuntime import와 CPUExecutionProvider의 모델 로딩 및 입력/출력 shape 확인이다.
  이는 실제 보행/추론 출력 성능 검증을 의미하지 않는다.
- 기준 README+sim_robot.launch.py의 최종 모델은 hsl_bringup/models/walk_v0_6_13500.onnx다.
  로컬 파일 및 전달 manifest SHA는 72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1.
  실기 control.launch 기본 walk_policy.onnx와 다르므로 그 기본값을 그대로 사용하지 않는다.
- 읽기 전용 검사 파일을 sim2real_evidence/deploy/check_walk_model_20261105.py에 준비했다.
  ament_index로 설치 경로를 확인하고, 선택 파일의 정확한 해시를 확인한 뒤
  CPU session을 생성하여 입력240/출력12를 확인한다. ROS 노드/토픽/서비스는 만들지 않는다.
  NumPy1.26.4 환경을 별도 subshell에 적용하여 실행할 예정이다.
  로컬 Python3.10 문법을 확인했으나 Jetson import/ORT 로딩은 아직 실행 전이다.
- 모델의 차원만 같다고 v0_2/v1s 또는 서로 다른 가중치가 호환된다고 확정하지 않는다.
  현재 정책 선택 근거는 기준 소스의 명시적 모델 경로와 해시다.

## 2026-11-05 선택 모델 로딩 통과, 속도 0 러너 시험 준비

- 사용자 검사 결과: 선택한 walk_v0_6_13500.onnx가 설치 경로에 있으며
  SHA=72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1로 정확히 일치했다.
  launch 기본 walk_policy.onnx도 존재하지만 SHA=9f6efa34d19cf90382a056f01ad9add5e71596d528cafe4855e04d1f3ba11531로 다르다.
- 로봇 NumPy1.26.4(.policy-python/numpy1), ONNXRuntime1.23.0 확인.
  available providers=Tensorrt/CUDA/CPU, 실제 검사 session은 CPU를 사용했다.
  입력 obs=[1,240] float, 출력 actions=[1,12] float이며 WALK_MODEL_LOAD_CHECK_PASSED다.
  GPU/TensorRT 실행이나 실제 센서 정책 추론까지 검증한 것은 아니다.
- 사용자에게 현재 자세를 물었으며 '기립 자세를 유지 중이며 토크 ON'이라고 확인받았다.
  다음 출력의 목표-현재 관절 차이는 이 기립 상태를 맥락으로 해석한다.
- 검사 파일 sim2real_evidence/deploy/check_walk_shadow_20261105.py를 준비했다.
  설치된 보행 러너 소스 hash(477dcff62aa2572b12d7cad58dc71c4a1628871d0c8695a869450ffc5bdf52aa)와
  5개 publisher의 remap 포함 여부, 모델 hash를 확인한 뒤 실제 설치 entrypoint를 실행한다.
  실관절/IMU/robot_state를 읽고, 속도 0을 시험 입력에만 발행한다.
- 실행 파라미터는 기준 소스 경로의 v0_6 13500, policy_variant=v0_2,
  CPU provider, policy_rate=50, action_scale=.25, gait_frequency=2,
  gait_mode=auto, gravity_source=derive, calibrate_on_startup=false다.
  킥 양보는 비활성화하고 cmd_vel/gait_mode/gait_frequency/kick 입력은 시험 토픽으로 분리한다.
  5개 출력(joint_goals/kick_done/gait 3개)은 모두 /hsl/shadow/walk로 remap한다.
- 약 20초 후 본인이 시작한 보행 러너 자식 프로세스만 종료하며 하드웨어 노드는 종료하지 않는다.
  CSV/runner.log/launch.json/summary.json을 runtime_logs/walk_shadow_<timestamp>_<suffix>/에 저장한다.
  실제 추론 RUN CSV가 있어야 결과를 통과 처리하며 HOLD 출력만으로 통과시키지 않는다.
  유한값, 속도0, 관절12개 매칭, 센서/출력 수신, 실제 ROS 출력 remap을 확인한다.
  출력 주기/간격, 목표-현재 최대 차이와 관절명, 추론 median/p99/max도 보고한다.
- 로컬에서는 Python3.10 문법, 현재 소스의 5개 publisher 대조를 확인했다.
  hold-only, raw NaN, 목표 inf, 비영속도 거부 및 정상 CSV 집계의 5개 검사를 통과했다.
  스크립트 SHA=96836157bff7ce4fdaf67d482addff785bbe3db98850e446f18cdcc039b47ef6.
  실제 Jetson 실행은 아직 전이다. 이 시험은 물리적 균형/명령 추종 시험이 아니다.
- 원본 러너의 출력 제한 기본값(max_target_delta_from_current_rad=10.35,
  max_target_abs_from_default_rad=10.80)은 이번 시험에서 변경하지 않았다.
  물리 연결 전 실제 관절 한계와 출력 범위를 확인해야 하며 유한값 통과만으로 연결하지 않는다.

## 2026-11-05 속도 0 보행 러너 실제 센서 시험 통과 및 목표 차이 분석 대기

- 사용자 결과: RUNTIME_CHECK_PASSED, 실관절 960개, 실IMU 3952개,
  시험 관절목표 906개, 수신 주기 47.999486Hz, 수신 간격 p99=45.011327ms/max=48.522444ms.
  실제 ROS 그래프에서 5개 정책 출력 모두 /hsl/shadow/walk 아래임을 확인했다.
  ROS 기본 /parameter_events와 /rosout 외에 실제 관절목표 출력은 없다.
- 실제 추론 CSV 859행, inference median=.290ms, p99=2.194ms, max=6.429ms.
  유한값/속도0/관절12개/센서 수신/추론 행 검사를 통과했다.
  이 통과는 모터 추종/자세 안정성/물리적 보행 성공을 뜻하지 않는다.
- 목표-현재 최대 차이는 l_knee_pitch_joint의 0.4286175935rad(약 24.56도)였다.
  어떤 방향인지, startup 이후 지속되는 값인지, 좌우 대칭/다른 관절 차이인지는 CSV 분석 전이다.
  이 숫자를 로봇이 실제로 움직인 양으로 해석하지 않는다.
- 사용자가 육안상 현재 기본 자세 그대로 멈춰 있다고 추가로 확인했다.
  실제 관절목표는 전달되지 않은 시험이므로 예상한 상태다. 현재 기립·토크ON 맥락은 유지한다.
- runner.log에는 STOPPED, phase0/f0, hip action a0 약 +1.02,
  qt0 약 +0.876이 반복됐다. 속도0에서도 관절 목표는 반드시 현재 자세와 같지 않다.
  이를 곧바로 오류나 정상 균형 보정이라고 확정하지 않고 방향/지속/관절 범위를 확인한다.
- 원본 rate 로그의 actual32.6→45.3Hz는 start_wall_time 기준 누적 평균으로 startup hold 등을 포함한다.
  관찰된 실제 목표 출력 약48Hz와 구분한다. 정책은 관절 수신 이벤트마다 실행하며
  내부 policy_dt는 20ms 고정이다. 수신 p99 간격만으로 느린 추론을 원인으로 단정하지 않는다.
- CSV closed:859 rows 이후의 rosout context-invalid 오류는 종료 시 로그 발행에서 나타났다.
  파일 종료와 실제 추론 행 수는 확인됐다. 이를 주행 중 추론 실패로 재분류하지 않는다.
- 로봇 로그 폴더: /home/robo04/sim2real_ws/runtime_logs/walk_shadow_20261105_162155_qeh2uxxq
  CSV: v0_2_log_20261105_162156.csv. runner.log, launch.json, summary.json도 같은 폴더다.
- 다음 한 단계는 새 로봇 구동 없이 위 폴더를 PC의
  sim2real_evidence/observations_20261105/로 복사해 실제 CSV를 분석하는 것이다.
  관절별 q/q_default/q_target/차이, raw action, 입력 중력/IMU,
  실제 추론 시각 간격을 대조한다. 아직 CSV 파일 원문은 로컬에서 받지 않았다.

## 2026-11-05 CSV 원문 분석 및 정지 위상 재생 완료

- 사용자가 복사를 완료했고 로컬에서 runner.log/launch.json/summary.json/CSV를 직접 읽었다.
  CSV SHA=3ea70087338429b85b40dfd931631c1754f4e716380f901cdd1c6201318d81cf.
  CSV 전체는 906행(47 HOLD_START + 859 RUN)이며 로그의 CSV closed 859는 실제 추론 행 수다.
- 속도 명령은 전 구간0. 원본 auto+stop_delay_s=1.0 로직 때문에
  hold 종료 후 1.0121~1.9896초에 48프레임 WALKING/f=2Hz가 발생했다.
  2.0104초 STOPPING 1프레임 뒤 2.0297초부터 STOPPED810프레임이다.
  이는 실물 보행이 아니라 시험 출력의 내부 위상이다.
- 왼쪽 무릎 최대 차이는 1.3503초 WALKING에서 current=-1.217981,
  target=-1.646598rad, -24.558도였다. 오른쪽도 1.6104초 +23.800도.
  STOPPED 구간 왼쪽 무릎 최대 차이는 4.969도다. 초기 최대가 계속 남은 것으로 해석하지 않는다.
- 기록 마지막5초 중앙 차이: 오른골반+14.067도, 왼골반-12.279도,
  오른무릎+9.198도, 왼무릎-4.374도, 오른발목roll-6.057도/왼-7.795도.
  실제 위치는 초기 자세를 유지하고 있었으며 이들은 미적용 정책 목표다.
- 별도 로컬 .analysis_env(NumPy1.26.4, ONNXRuntime1.23.0 CPU, matplotlib3.9.4)를 만들었다.
  전역 Python/로봇 패키지 환경은 바꾸지 않았다. 기록 과거5프레임으로 실제 모델 출력854행을 재구성했고
  최대 raw action 오차1.87755e-6으로 원본 관측 구성과 출력 재현을 확인했다.
- 이어 기록된 같은 관절/IMU를 고정 피드백으로 쓰되 위상만 처음부터stop으로 재생하고
  이전 액션 히스토리는 새 출력으로 갱신했다. 첫 hold 상세 센서가 없어 첫 RUN 측정 복제/이전액션0으로 초기화했다.
  stop 재생 무릎 최대 차이는 왼4.650도/오른9.319도로 줄었고 정지 후 골반 차이는 남았다.
  새 목표에 대한 물리 피드백이 없으므로 균형/실물 수렴 검증으로 취급하지 않는다.
- 기록 목표의 URDF 관절 한계 초과는0개. 실제 추론 간격은 median20.0ms/p9945.2ms/max47.1ms,
  30ms 초과36/858간격이다. CPU 추론만 느려서 생긴 지연이라고 단정하지 않는다.
- 첫 물리 인계는 기존 파라미터 gait_mode=stop으로 시작하고 이동 명령을 줄 때 auto로 전환할 계획이다.
  모델/러너 소스/보정값은 이번 분석에서 변경하지 않았다.
- 산출물: sim2real_evidence/analysis_walk_shadow_20261105/ANALYSIS.md,
  analysis.json, joint_summary.csv, replayed_stop_targets.csv, zero_command_analysis.png.
  재현 코드: sim2real_evidence/analyze_walk_shadow.py.
- 다음 분리 시험 파일 check_mux_stop_20261105.py를 준비했다. 실제 설치된 twist_mux만 도메인77에서
  기동해 전진입력/입력중단/경기정지락/teleop0 전달을 확인한다. 모터/보행 정책 프로세스는 시작하지 않는다.
  실제 cb_cmd_vel 소스를 추출해 단순 캐시로 재생하여 silence와0명령의 차이를 대조한다.
  config와 source hash를 기록하고 실기 정책의 실제 주행 성공으로 표현하지 않는다.
  Python3.10 문법을 확인했으며 Jetson mux 실행 결과는 아직 전이다.

## 2026-11-05 최우선 목표를 실물 역할 수행으로 명시, 단독 로봇 통합 준비

- 사용자 명시 지시: 제공된 정책은 이미 검증된 것이므로 추가 정책 재검증에 시간을 쓰지 말고,
  함께 들어 있는 B-Human 경기 전략을 실제 로봇에 적용해 공격/수비/골키퍼 역할 수행을 보는 것을 최우선으로 한다.
  실제 가동 로봇은 robo04 한 대라고 확인받았다. 이후 분리 검증 반복은 재개 조건으로 요구하지 않는다.
- 직전 mux 시험 로그는 원문 observations_20261105/mux_stop_test_result.txt에 보존했다.
  입력 단절/경기 정지 락 모두 mux 출력0개이며 소스 콜백 캐시 vx=.2가 남았고,
  명시적 teleop0 전달은17회 확인됐다. 단절 보호만 실제 연결 수정에 포함한다.
- 원본 자동 역할 배정은 필드 로봇 한 대를 계속 공 담당으로 고르므로 단독 역할 전환이 불가능했다.
  assigned_role=auto/attacker/defender/goalkeeper를 추가해 기존 전략/스킬을 사용하도록 연결했다.
  attacker=공 접근·정렬·킥, defender=기존 수비 위치 이동,
  goalkeeper=골문 앞 위치 조정·우리 페널티 구역 근처 공 처리다. jersey 번호는4 그대로다.
- 실물 실행 묶음 real_roles_20261105.tar.gz를 준비했다.
  SHA256=5f876eb86737a27a4179955f2d799ec1ef48703b0c05b6b3bbf37ff518462235.
  archive18파일82462bytes, readback검증 완료. 추출폴더 real_roles_20261105.
  적용 전5개 원본 파일 해시는 기존 로봇 전달 manifest와 전부 일치한다.
  적용 도구는 원본 해시/실행 중 노드/백업/원자 교체/실패 복원/중복 적용을 처리한다.
- 배포 수정6개: walking runner(수신후0.5초 command expiry), kick runner(motion_enabled 즉시 킥 종료),
  control.launch(최종 킥 경로/CPU provider/stop_delay 전달), strategy.py(지정 역할/none fallback/작은필드 수비거리),
  behavior_node.py(지정역할과 경기상태 동적변경, 로컬 trial의 game_stop_lock),
  새 real_role_trial.launch.py. 가중치/action scale/관절순서/게인/IMU 보정은 바꾸지 않았다.
- 수비 거리의9m 필드 기준1.5/1.2m 상수를 practice 길이 비율로 줄였다.
  기존 작은 필드에서 수비 목표가 공보다 상대 골대 쪽으로 놓일 수 있는 경우를 회귀 검사했다.
  일반9m 지도 동작은 유지된다. 공격/GK는 원본 기존 스킬과 pose함수를 사용한다.
- real_role_trial은 기존 하드웨어/센서/월드모델을 유지하고
  실제 mux→보정→보행·킥, 기존 fall_manager, 역할 행동만 추가한다.
  최종 보행v0_6_13500과 킥v1_5b_22300, CPU provider를 명시한다.
  최초 상태는 INITIAL/gait stop이다. 정지 정책의 관절 목표는 실제 모터에 연결된다.
  최초 시험 속도 상한은 vx=.20,vy=.15,wz=.50이다.
- role.sh attacker/defender/goalkeeper로 경기 상태PLAYING과 해당 역할을 설정한다.
  role.sh stop은 행동INITIAL, 보행gait stop, 킥motion_enabled=false를 요청한다.
  역할 변경은 먼저 정지하며, enable을 올린 후 PLAYING으로 들어간다.
  --seconds 10 옵션은 실제 역할10초 후 자동 정지/토크 유지다. status는 전체 debug JSON조회다.
- 로컬 역할/시작·정지10검사(기존 전략8개 하위검사 포함), 적용/복원5검사가 통과했다.
  앞서 실제 변경된 walking 메서드의 freshness/control_tick7검사도 통과했다.
  Python3.10/Bash 문법을 확인했다. 실제 ROS launch와 물리 역할 수행은 아직 배포 전이다.
- 사용자 실행 순서: 기존7번 shadow행동만 Ctrl+C → archive전달 →8번 apply_and_build.sh(관련4패키지)
  →기립/위치 초기화 상태에서7번 start.sh →8번 role.sh attacker --seconds 10.
  기존1~6은 유지한다. 그다음 수비/골키퍼 전환 결과를 실제로 확인한다.
- 위 작업은 정책 재학습/추가 모델 비교가 아니라 원본 전략과 실물 제어의 통합이다.
  실제 역할 수행에 성공했다는 판정은 사용자 관찰/실행 결과가 있어야 한다.

## 2026-11-09 재부팅 후 전체 기동 안내

- 사용자가 현재 재부팅되어 기존 터미널이 모두 꺼져 있다고 확인했다.
  기존1~6을 유지한다는 직전 안내의 전제를 철회하고 전체 기동 명령을 다시 정리한다.
- real_roles 묶음 적용/빌드 및 실제 역할 수행 결과는 아직 보고되지 않았다.
  설치 완료를 이미 확인한 경우 재부팅만으로 재적용/재빌드할 필요는 없다.
  미설치라면 먼저 기존 real_roles_20261105.tar.gz를 전송·적용한다.
- 새 기동 순서: 1 하드웨어, 2 카메라+TF, 3 YOLO, 4 leg_odometry,
  5 ball_tracker, 6 self_localizer practice → START 기립 →8 초기위치 입력
  →7 실물 역할제어 start.sh →8 공격10초 실행이다.
- 이전과 같은 우리 골라인 왼쪽 두 T 사이/상대골대 방향이면 임시 seed(-1.95,.7583333,0)를 쓴다.
  실제 배치가 달라졌다면 해당 좌표를 정답으로 사용하지 않는다.
- shadow행동·모델재검증·전체 robot.launch 중복 기동을 추가 선행 조건으로 요구하지 않는다.
  현재 목표는 실제 공격/수비/GK 역할 관찰이다.
- 재부팅 전 파일과 새 역할 스크립트를 로컬에서 다시 확인했으며 묶음 SHA는
  5f876eb86737a27a4179955f2d799ec1ef48703b0c05b6b3bbf37ff518462235로 유지된다.
  재부팅 후 로봇 기동 결과는 아직 전이다.
- 재개 안내: sim2real_evidence/REBOOT_REAL_ROLES_20261109.md.

## 현재 및 다음

현재 로봇의 소스는 사용자 지정 `mx3_src + hsl_src` 13개 패키지 배포본이다.
전체 13개 패키지의 Humble 빌드, 실행 파일 등록과 실제 센서/포트 조회가 완료됐다.
D405 컬러·정렬 depth·CameraInfo 수신이 확인됐다. 현재 카메라 드라이버는 유지한다.
사용자는 처음 기체 상태를 "지지 준비 전이거나 토크 상태를 모름"으로 답했고, 이어
"내가 토크 온 버튼을 누르면 일어나니 그 상태로 테스트"하는 진행 방식을 요청했다.
이에 사람이 기존 버튼으로 기립시키고 자세를 유지한 상태에서 센서/월드모델부터 확인하는 순서로 전환한다.
이후 사용자 보고로 자세 유지가 확인됐다. 모터 레지스터의 실제 토크 상태와 구체적인 도착 로그는 아직 미확인이다.
소스 확인: 기체 START 버튼은 `/u2d2/robot_init`을 호출해 토크 ON과 초기자세 이동을 시작한다.
토크 서비스만 호출하는 경로는 초기자세 완료를 보장하지 않는다. 서비스의 `success/started` 응답은 시작 확인이며,
실제 완료는 `robot_init arrived` 로그 및 실제 자세/관절 피드백으로 확인한다. timeout은 완료로 보지 않는다.
leg_odometry의 실제 센서 입력 기반 출력과 수신 주기가 확인됐다. 작은 각도 변화는 사용자 지시에 따라 오차로 감안한다.
robot_state_publisher 추가 후 몸통/odom에서 카메라 optical frame까지 TF 연결이 확인됐다.
YOLO 3D의 실제 공 검출/좌표와 출력 주기가 확인됐다.
`hsl_world_model/ball_tracker` v3는 로봇에서 빌드 및 격리 시험을 통과했고,
실제 도메인 42에서도 검출 시각의 TF와 첫 공 초기화가 확인됐다.
실제 공의 최근 관측 수신과 사용자에 의한 대략 거리 수용이 확인됐다.
practice 자기위치 노드도 실제 관측을 사용하며 20초간 30Hz 정지 추정 안정성이 확인됐다.
실제 센서 입력을 쓰는 행동 노드가 시험 토픽에서 공 접근/정렬 명령을 생성한 것도 확인됐다.
11월 4일 중단 후 11월 5일 사용자가 재개 명령을 요청했다. 현재는 검증된 기동 상태를 복구하는 단계다.
현재는 사용자 우선순위에 따라 실물 역할 수행 통합 묶음을 준비한 상태다.
이후 단계는 배포 → 실제 공격 → 수비/골키퍼 역할 전환 관찰이다.
정책 파일 재검증이나 추가 분리 시험을 선행 조건으로 반복하지 않는다.
이 노드는 NVIDIA PyTorch/Ultralytics와 cv_bridge를 사용하므로 검증된 NumPy 1.26.4 경로를
새 검출 터미널에만 적용한다. 기존 실행 중인 노드의 환경을 변경하지 않는다.
카메라 관측 없이 방향까지 검증 완료됐다고 판단하거나 임의의 yaw 보정값을 넣지 않는다.
단독 시험 노드를 실행한 후에는 전체 hsl_stack을 시작할 때 leg_odometry가 중복되지 않도록 해야 한다.
이후 카메라/TF/검출을 연결한다.
센서/월드모델 확인이 끝날 때까지 보행·킥·B-Human 행동·전체 robot.launch.py는 시작하지 않는다.
원본의 빈 디렉터리는 `hsl_world_model/config`, `hsl_behavior/config`, `hsl_team_comm/launch` 3개다.
이후 배포 묶음을 만들 때는 원본 빈 디렉터리도 포함해야 한다. 기존 전송 묶음과 SHA-256은 변경하지 않았다.
카메라 검사도 이어서 진행한다. 카메라 검사 자체에는 모터 명령이 필요하지 않다.

그 후 Humble 의존성 및 13개 패키지 빌드, 설치 경로, 실제 센서/TF, 월드모델 출력,
행동 명령과 하위 제어의 연결, 실제 단일 로봇 시험으로 진행한다.
기존 `models`, `debug_bags`, `.policy-python`과 과거 백업은 보존한다.
## 2026-11-09 기본 실행 통합과 최신 재개 기준

- 사용자 요청: 재부팅 후 기본 기능을 한 명령으로 켠다.
- real_base.launch.py는 하드웨어, 카메라+RSP, YOLO, leg_odometry,
  ball_tracker, self_localizer를 통합한다. NumPy1 경로는 YOLO 그룹에만 적용한다.
- practice 3.9x2.6m, player4, 실제 포트 by-id와 기존 카메라 프로파일을 유지한다.
- 보행/킥/경기 행동은 START 기립과 /initialpose 이후 별도 start.sh로 켠다.
- base.sh는 로그를 모으고 중복 base.sh 실행은 flock으로 막는다.
- real_robot_20261109.tar.gz에 기존 real_roles 수정도 포함했다. 이전 묶음은 선행 설치할 필요 없다.
  파일 해시 확인 및 백업은 유지한다. 가중치는 바꾸지 않는다.
- SHA256: c735bb7a0bc0768f0f99b15b5eaa8457966974182cdda9ed35a8450a66ca3ba1.
- 7개 소스 동일성, Python3.10 AST, Bash 구문, LF 체크섬, 아카이브 재검증 통과.
  새 manifest 기준 설치기 기존 5개 테스트(백업/멱등성/알 수 없는 소스 거부/실행 중 거부/롤백) 통과.
- 실제 ROS 통합 실행과 역할 동작은 로봇에서 확인 전이다.
- sim2real_evidence/RESTART_3_TERMINALS_20261109.md가 이전 8터미널 안내를 대체한다.
- 현재 목표는 robo04 한 대에서 기존 정책으로 실제 공격/수비/골키퍼 행동을 관찰하는 것이다.
  이미 확인한 정책·센서 검증이나 sim2real 재학습을 새 선행 조건으로 반복하지 않는다.
- 순서: 새 묶음 설치 → 터미널1 base.sh → 물리 START 기립 → 터미널3 init_pose.sh
  → 터미널2 start.sh → 터미널3 attacker --seconds 10 → 수비/골키퍼 전환 관찰.
- 역할 stop은 기립 토크를 유지한다. 종료 시 지지 후 물리 STOP으로 토크 OFF하고 launch를 종료한다.
  models/debug_bags/.policy-python/과거 백업은 보존한다.

### 사용자 요청: 시작 위치를 기억하고 변경이 필요하면 먼저 안내

- 우리 골대에서 상대 골대를 볼 때 우리 골라인 왼쪽 두 T마커 사이에 세운다.
- 몸통 중심 바닥 투영점은 골라인 위에서 중앙보다 왼쪽 약 76cm, 정면은 상대 골대 중앙 방향이다.
- 근사 초기값은 `x=-1.95, y=+0.7583333, yaw=0`이다. 실측 정답으로 간주하지 않는다.
- START 기립 후 초기 위치를 입력한다. 다른 실제 위치에 기존 좌표를 그대로 입력하지 않는다.
- 시작 위치나 방향 변경이 필요한 시험에서는 새 배치와 변경 이유를 먼저 안내하고,
  초기 위치 스크립트의 좌표도 실제 배치에 맞춰 함께 변경한다.
- 역할 전환만으로 시작 위치를 임의 변경하지 않는다.

### 통합 실행 이후 사용자 관찰

- 사용자가 "잘 움직인다"고 보고했다. 실물 움직임은 사용자 관찰로 확인됐으며,
  공 접근·정렬·킥 완료나 수비/골키퍼의 수행 결과까지 확인한 것으로 확대 해석하지 않는다.
- 이어 현재 공격 역할의 동작과 지도 기준을 질문했다.
  real_base.launch.py의 자기위치와 real_role_trial.launch.py의 행동 모두 field_profile=practice다.
  지도는 3.9x2.6m, 중앙 원점, +x 상대 골대, +y 왼쪽이다.
- 단일 로봇 attacker는 공 담당을 robo04에 지정한다. 공 탐색 → 공 뒤 접근/정렬 →
  유효 관측·공 속도·킥 범위 조건 충족 시 기존 킥 정책 호출로 동작한다.

### 실제 이동 보고 후 자기위치 시각화 요청

- 사용자는 시작 위치 유지 후 공을 찾아가는 동작도 되는 것 같다고 보고했다.
  자기위치 정확도를 RViz 등으로 보고 싶으며 MobaXterm tunneling 가능 여부를 질문했다.
- 기존 프로세스와 초기 위치를 유지한 채 관찰용 새 SSH X11 세션에서 RViz를 실행하는 묶음
  `sim2real_evidence/patches/hsl_view_20261109`를 준비했다. 전송본은 deploy/hsl_view_20261109.tar.gz다.
- /hsl/robot_pose, /hsl/ball만 구독하고 /hsl/view/markers만 발행한다.
  지도, 추정 위치·방향, 관찰 중 궤적, 공, quality/validity/관측 수/수신 경과를 표시한다.
  제어·초기 위치·TF를 발행하지 않으며 원격 소스 덮어쓰기와 colcon 빌드는 필요 없다.
- launch 전 실제 /hsl_self_localizer의 field_profile이 practice인지 읽어서 확인한다.
- 중요한 발견: 현재 HSL 추정 지도와 기존 humanoid_localization practice 시각화의 내부 치수가 다르다.
  외곽은 양쪽 모두 3.9x2.6m지만 HSL의 센터서클 반지름/페널티박스 폭/골대 폭은
  0.325/약1.733/0.78m, 기존 시각화는 0.35/2.17/1.13m다.
  새 관찰 노드는 설치된 hsl_behavior.geometry.Field로 현재 HSL 지도를 그대로 표시한다.
  실제 필드가 어느 치수인지 실측하지 않았으므로 추정 지도를 임의 수정하지 않는다.
- 높은 validity나 화면의 안정성은 실측 위치 정확도를 보장하지 않는다.
  다음에는 정지 위치의 실제 라인/마커 거리와 화면 좌표·방향을 비교한다.
  이미 이동한 로봇에 기존 시작 좌표를 다시 입력하지 않는다. 초기 배치 변경은 먼저 안내한다.
- 로컬 문법/구문/YAML/지도 좌표/압축파일 검사는 완료했다. 실제 로봇의 RViz 및 X11/OpenGL
  동작은 사용자 실행 전이다. Qt/GLX 환경에 따라 별도 그래픽 설정 확인이 필요할 수 있다.

### 사용자 RViz 성공 보고 및 운영 매뉴얼 요청

- 사용자가 RViz 안내 후 "디게 잘되네"라고 보고했다. 현재 로봇 이동/공 탐색 접근과
  RViz 표시는 사용자 관찰상 성공이다. 실측 위치 오차나 실물 킥/4대4 경기 성공으로 확대하지 않는다.
- 요청: 각 로봇 시작과 역할 부여, RViz 실행, practice/s_field 전환, 로봇 증가 시 4대4 확장,
  B-Human에서 가져온 기능을 매뉴얼로 정리한다. 현재 성공한 실행 소스는 변경하지 않는다.
- 작성 파일: sim2real_evidence/manuals/ROBOT_OPERATION_MANUAL_KO_20261109.md,
  BHUMAN_PORT_AUDIT_KO_20261109.md, ROBOT_FLEET_TEMPLATE_20261109.csv.
- 현재 래퍼는 robo04/home/USB/player4/practice 전용이다. start.sh는 launch 인자를 전달하지 않고,
  role.sh는 auto를 지원하지 않는다. 행동 노드의 assigned_role=auto는 지원한다.
- 지도 변경은 자기위치/행동 launch 및 RViz launch/프로파일 검사와 실제 초기 위치를 함께 맞춘다.
  라이브 field_profile 파라미터만 바꿔 내부 지도가 재생성됐다고 보지 않는다.
- 4대4 준비 시 단독 game_state remapping 제거, GC lock 발행기 중복 방지, default initial 유지가 필요하다.
  기존 GC return은 구형 PoseWithCovarianceStamped/PointStamped 토픽을 읽으므로 HSL 타입 변환이 남아 있다.
  또한 GC INITIAL 동안 자기위치가 매 주기 walkInPose로 고정되므로 현재 단독 골라인 초기화 방식을
  그대로 적용할 수 없다. 실제 진입 배치와 walk_in 파라미터/초기 상태 처리를 함께 정하고 시작 위치 변경을 먼저 안내한다.
- 팀 통신은 HSLT 자체 UDP 10000+팀번호다. localhost ROS 격리를 유지하면서 통신할 수 있다.
  실물 두 대의 팀 통신/협동부터 확인한다. 원본 시뮬 경기 스크립트를 실물 실행 명령으로 안내하지 않는다.
- 공식 B-Human 고정 커밋의 전략/스킬/SelfLocator/팀 공/팀 메시지와 로컬 소스를 추가 대조했다.
  UKF 수식·구조 이식, 상위 전략/통신의 개념 재구현, 제공된 MX3 정책/하드웨어 및 이번 통합 수정을 구분했다.
  공식 대조 기록은 manuals/BHUMAN_OFFICIAL_REVIEW_20261109.json이다.

### 사용자 요청: 매뉴얼을 Word/PDF로 제공

- 사용자는 Markdown 표가 깨져 보인다며 `C:/Users/jes/Desktop/MX3_Match_Sim/Manual`에
  Word 또는 PDF로 정리해 달라고 요청했다. 이후 매뉴얼 배포는 이 형식/폴더를 우선한다.
- 운영 절차, B-Human 이식 내역, 로봇 8대 설정 기록란을 통합한
  `MX3_HSL_로봇운영매뉴얼_20261109.docx`와 `.pdf`를 해당 폴더에 저장했다.
- 긴 표는 항목별 설명으로 바꾸고 명령어 박스, 빠른 실행 안내, 링크 목차를 넣었다.
  PDF는 19쪽이며 한글 표시, 원본 명령어 보존, 빈 페이지/페이지 밖 텍스트 여부를 검사하고 화면을 확인했다.
- Word COM의 PDF 내보내기가 종료되지 않아 해당 작업 소유의 내보내기 프로세스를 정리했다.
  최종 PDF는 같은 DOCX 문단에서 만든 인쇄용 HTML을 headless Chromium으로 렌더링했다.
  기존 사용자 문서나 로봇 실행 코드는 변경하지 않았다.

### 2026-11-12: 정책 학습 자료 및 RViz 표시 확장

- 사용자는 수정 전 적용 범위(텔레옵/자율 경기/시뮬), 부작용, 재시작/원복 방법을 명확히 알릴 것을 요구했다.
- 현재 로컬 공통 control.launch의 킥 실험값: 접근 0.20, hold 0.6초, 출력 필터 켬/tau 0.02, 총 2초. 실물 반영은 사용자 적용 결과로 확인해야 한다. 시뮬 launch는 접근 0.30/hold 0.3을 별도 지정한다.
- 추가 폴더 XM_Walking_Policy, XM_Kick_Policy, XM_Getup_Policy를 분석했다. 보행/킥은 현재 선택된 ONNX와 해시가 같다. 원본 학습 코드, 체크포인트, 배포 규격, USD, 평가 자료가 있다. 기상은 파일은 같아도 현재 실물 제어 launch에서 RL 기상으로 실행하지 않는다.
- 재학습 계획은 `sim2real_evidence/policy_package_audit_20261112/RETRAINING_PLAN_KO.md`에 기록했다. 기존 킥에도 마찰 DR/미끄럼 벌점이 있으며, 새 태스크에서 표면 배정·관측 지연·지지 접촉 평가를 개선할 계획이다. 서버 전송/학습은 아직 안 했다.
- 사용자 요청에 따라 canonical `hsl_behavior/skills.py`, `behavior_node.py`에 표시 전용 target_local/target 필드를 추가했다. 목표 계산·속도 명령·역할 로직은 바꾸지 않았다.
- 새 RViz 묶음: `sim2real_evidence/patches/hsl_view_20261112`, 배포 `deploy/hsl_view_20261112.tar.gz`.
  SHA256 a3f26b71fe75a4a478a0eee5a9edcf78a4383fcc0699243dc314ded56c3db936.
- 표시: hostname 기반 robo04/robo03 이름, 역할/경기 상태/스킬, 실제 WalkToPose 목표(map)/연결선, 추정 공 위치, detection-time TF로 변환한 원시 관측 클래스/신뢰도. TF가 없으면 원점/최신 TF로 대체하지 않는다.
- 새 start.sh는 localizer의 practice/s_field를 읽어 표시한다. 한 로봇 로컬 관측 구성이며 이름 설정만으로 다중 로봇 중앙 관측이 되는 것은 아니다.
- 로컬 검증: 스킬 3000입력 원본 대비 동일 출력, TF/표시/만료 검사, 소스 hash guard/백업/반복 적용/실행중 거부, Python3.10/Bash/RViz 설정/압축 readback 통과. 실제 ROS/RViz 실행은 로컬에서 불가.
- 사용자 적용 절차: 기존 teleop/start.sh/RViz 종료(base 유지), tar+sha 전송, checksum 후 압축 해제, apply_behavior.py, `colcon build --base-paths src --symlink-install --packages-select hsl_behavior`, start.sh 재시작, 새 `/home/robo04/incoming/hsl_view_20261112/start.sh` 실행. 표시 소스 설치 시 역할 시험은 중단되므로 필요하면 role.sh로 다시 시작한다. 움직인 로봇에 초기 위치를 임의 재입력하지 않는다.

### 2026-11-12: 사용자 사진 기준 연습장 지도 수정

- 사용자는 RViz뿐 아니라 실제 위치 추정 지도를 사진의 치수와 일치시키라고 요청했다. 사진은 라인 3.9×2.6, 잔디 5×3, 센터서클 r=.35, 페널티 구역 .87×2.17, 골 구역 .43×1.30, 골대 폭 1.13, 페널티 마크 거리 .65m이다.
- legacy humanoid_localization practice는 이미 사진과 동일했으나 현재 HSL C++ FieldModel/Python Field는 S-Field 균일 축소로 페널티 폭 1.7333, 골대 폭 .78, 센터 r=.325를 사용했다. 두 HSL 정의를 사진의 명시값으로 수정했다.
- C++ 후보 유효 잔디 영역은 기존 4.9×3.6에서 5×3m로 수정했다. border_x=.55/border_y=.20으로 분리했다. 기본 START 시드가 잔디 밖으로 나가지 않도록 practice walk_in_outside 기본 .4→.1m; s_field는 .4 유지. self_localizer 시작 로그에 전체 치수 진단 추가.
- Python Field에 carpet_length/width를 추가하고 RViz에 실제 잔디 외곽/치수를 표시했다. 역할·이름·관측·목표·공 표시도 포함한다.
- 기존 약속 초기 좌표(-1.95,+.7583333,0)는 유지했다. 새 도면의 왼쪽 두 T y=.65/1.085 사이 정중앙은 .8675라 기존 seed보다 약10.9cm 왼쪽이다. 물리적 시작 위치와 seed를 혼동하지 않도록 사용자에게 알렸다.
- 영향: 위치 추정 랜드마크/TF/공 지도좌표와 골키퍼 구역이 바뀐다. 일반 GK 좌우 범위 ±.24→±.415m. 보행/킥 모델·튜닝과 s_field 치수는 유지. 기존 MuJoCo s_field 물리 scene XML은 변경하지 않았다.
- 배포: `sim2real_evidence/deploy/hsl_field_20261112.tar.gz`, SHA256 f843487d571d1df914aae989f68940c684f377b09ecc558c696bb0c87f32191f. 기존 telemetry 설치 전/후 양쪽 소스를 hash guard로 지원하며 미리보기는 `sim2real_evidence/practice_field_20261112/practice_field_comparison.png`.
- 로컬 검증: 사진/Python/C++ 상수 대조, legacy 31 랜드마크, s_field 불변, 설치 백업/반복/실행중/충돌 가드, 뷰어 회귀, 기존 전략 테스트, Bash/Python 문법/압축 무결성 통과. C++ 실제 컴파일과 전체 ROS 빌드는 로봇에서 `apply_and_build.sh`가 수행한다. 실제 로봇에는 아직 전송/실행하지 않았다.
- 적용 시 이전 teleop/start/base/RViz 모두 종료 필요. 새 묶음의 apply_and_build.sh는 source 검사→백업→Python 검증→C++ 헤더 컴파일/랜드마크 체크→hsl_world_model/hsl_behavior 빌드→설치 Python 검증 순서다. 이후 base/start와 새 `/home/robo04/incoming/hsl_field_20261112/start.sh`를 사용한다. 자기위치 노드 재시작으로 추정 상태가 초기화되므로 실제 위치에 맞는 seed가 필요하다.

### 2026-11-12: 사용자 승인으로 정확한 T 중간점 초기 위치 적용

- 위의 기존 위치 유지 기록은 이 요청으로 대체한다. 사용자 요청에 따라 수동 초기 위치를 `x=-1.95, y=+0.8675, yaw=0`으로 수정했다. y=(.65+1.085)/2이며 기존보다 왼쪽 약10.9cm다.
- 로컬 `patches/real_robot_20261109/init_pose.sh`와 배포용 `deploy/init_practice_pose_20261112.sh`를 수정했다. `deploy/apply_init_pose_20261112.py`는 로봇의 기존 incoming/real_robot_20261109/init_pose.sh만 백업 후 교체한다. 기존 압축 배포본은 보존했다.
- 설치만으로 위치를 발행하거나 모터를 동작시키지 않는다. 실제 새 시작 위치에 놓고 물리 START 후 기존 init_pose.sh 명령으로 입력한다. 이 셸 스크립트만 교체할 때 재빌드는 불필요하다.
- START 직후 임시 진입 가정은 라인 밖 .10m로, 잔디 경계 .20m 안쪽이다. 수동 골라인 초기 위치와 별개이며 선행 hsl_field_20261112 지도 수정/C++ 빌드/base 재시작이 필요하다. 실물 배포 성공은 아직 확인되지 않았다.
- 로컬 Bash/Python3.10 구문, 정확한 좌표, 격리 설치/백업/반복 적용/사용자 커스텀 값 거부를 확인했다. 보행·킥 정책은 변경하지 않았다.

### 2026-11-12: 새 GitHub 저장소로 작업 보관 및 이후 커밋 원칙

- 사용자 요청으로 `https://github.com/hion89/MX3-HSL-Robot-Soccer` 비공개 저장소를 생성했다. 로컬 루트는 `C:/Users/jes/Desktop/MX3_Match_Sim`이며, 기존에 커밋/원격이 없던 Git 초기화 상태를 사용했다. 기존 Isaac-work-space 원격과 변경 파일은 그대로 두었다.
- 실물/시뮬 소스, 보행·킥·기상 정책/체크포인트/학습 overlay, 배포 묶음, 분석 결과, Word/PDF 매뉴얼 및 진행 기록을 보관한다. 가상환경·빌드·브라우저 프로필·비밀정보·원본 bag/영상은 제외했다.
- 정책 해시/사진 지도/초기 위치 확인 도구, 자료 동기화 도구와 최신 상태 문서를 추가했다. 새 clone에서도 필요한 hsl_world_model/config 빈 디렉터리가 유지되도록 .gitkeep을 추가했다. 지도/RViz 최신 검사와 묶음 생성 도구는 저장소 상대 경로를 우선하도록 했다.
- 배포 manifest의 해시 보존을 위해 Git 줄바꿈 자동 변환을 끄고 파일 원래 바이트를 보존했다. 셸 파일은 LF다. 첫 업로드 검토: 756파일/약189MiB, 비밀정보 패턴 및 제외 대상 검사, 스테이징 바이트 일치, 체크아웃 검증, 지도 4개/RViz 6개 로컬 검사 통과. ROS 실물 검증으로 확대하지 않는다.
- 사용자 요청을 루트 AGENTS.md와 기존 작업 공간 AGENTS.md에 기록했다. 이후 관련 변경·검증이 완료되면 새 저장소에 선택적으로 커밋·푸시하고 결과를 보고한다. 인증/네트워크 실패 시 미완료를 명시한다. GitHub 보관은 로봇 배포/동작과 별개다.
