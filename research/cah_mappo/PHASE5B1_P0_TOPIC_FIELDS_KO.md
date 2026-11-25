# P0 토픽·필드·시각·QoS 대응표

이 문서는 소스 정적 분석과 과거 bag에 근거한다. 현재 로봇의 설치본·프로세스·QoS를 확인한 결과가 아니다. 센서 및 모든 실물 ROS 실행은 별도 승인 대상이다. 과거 bag 토픽의 원본 type/serialization/offered_qos_profiles는 `phase5b1_p0_outputs/study_v1/bag_catalogs.json`에 보존했다.

## 공통 기록 규약

원본 bag은 CDR payload와 bag 수신 시각을 보존한다. measurement-v1은 `record_id`, experiment/robot/trial, source path/sha256/locator/topic/message_type, decoded payload, `timestamp`, `header_timestamp`, 명시적 결측 측정 필드를 갖는다. 원본 record 순서를 유지하고 정렬해서 역전을 숨기지 않는다.

bag 입력의 timestamp는 recorder clock이다. 향후 passive subscriber는 콜백에서 읽은 ROS clock을 receive_ros로 기록한다. 둘은 의미가 다르다. header 없는 메시지에 가짜 header를 생성하지 않는다. 외부 GT 시각은 external_gt로 분리하고 검증된 clock 변환 없이는 서로 빼서 지연을 만들지 않는다.

기존 JSONL 구독기는 대부분 sensor_data(BEST_EFFORT/volatile), /tf_static은 RELIABLE/transient_local/depth=1이다. BEST_EFFORT 구독은 일반적으로 reliable 또는 best-effort 센서 publisher의 데이터를 받을 수 있지만 원본 전량 보존을 보장하지 않는다. transient-local 구독은 publisher의 durability와 맞아야 한다. 깊이 1은 여러 TF batch의 버스트를 충분히 보존하지 못할 수 있으므로 필요한 parent-child 연결의 실제 수신 여부까지 현장에서 확인한다. 과거 SQLite QoS는 live 협상 증거가 아니다.

## 비구동 센서 기록 후보

`/imu/data`, `/imu/data_50hz`: sensor_msgs/msg/Imu.

- decoded.header.stamp/frame_id, orientation.xyzw, angular_velocity.xyz(rad/s), linear_acceleration.xyz(m/s²), covariance를 보존한다. IMU parser는 host 처리 중 now로 stamp를 만든다. 50Hz 출력은 decimation 경로이며 원본 stamp를 재사용한다.
- 센서 하드웨어 샘플링 순간이나 MCU 송신 순간을 header로 확정할 수 없다. 시각 신선도는 입력→파싱→기록 단계를 구분해야 한다.
- 소스 publisher는 depth 10 reliable, 수집기는 sensor_data다. 원천 /mcu/raw_rx 확보는 MCU open/IMU enable/PING 전송을 수반하므로 물리 격리·firmware 확인이 선행되어야 한다.

`/mcu/raw_rx`: std_msgs/msg/ByteMultiArray. 원시 바이트 data와 수신 시각만 있다. 기존 기본 25토픽에는 포함되지 않으므로 별도로 승인된 원본 recorder 또는 수집 manifest 구독 항목이 필요하다. timestamp 없는 바이트에 실제 센서 획득 시각을 만들어 붙이지 않는다. /mcu/raw_tx는 기록 대상 송신 관측과 명령 발행을 구분하며 P0에서 발행하지 않는다.

`/camera/camera/color/camera_info`: sensor_msgs/msg/CameraInfo.

- header, width/height, distortion_model, d/k/r/p, ROI를 decoded로 보존한다. frame_id를 실제 이름으로 manifest에 등록한다. 기존 기본 frames에는 camera optical 이름이 없으므로 템플릿 그대로 사용하면 부족하다.
- header의 물리적 의미는 외부 RealSense driver와 장치 timestamp domain 확인이 필요하다. CameraInfo 수신은 이미지·깊이가 실제 기록됐다는 증거가 아니다.

`/camera/camera/color/image_raw`, `/camera/camera/aligned_depth_to_color/image_raw`: sensor_msgs/msg/Image.

- 원본 recorder에 header/width/height/encoding/step/is_bigendian/data를 보존한다. 기존 JSONL 기본 25토픽에는 이미지가 없으며, JSONL이 원본 image/bag 대체품이 아니다.
- code의 848×480×30 설정과 실시간 encoding/실제 도착률을 구분한다. raw→aligned 처리 지연과 색상/깊이 시각 차이도 구분한다. RealSense driver의 실제 timestamp/QoS는 현장 검증 대상이다.

## 세계모델·추정 기록: 생산자 의존성 충족 시에만

`/perception/detections_3d`: vision_msgs/msg/Detection3DArray. array/개별 detection header, bbox/results/hypothesis/pose를 decoded에 보존한다. YOLO는 color header를 복사하고 최신 depth cache를 이용한다. color/depth stamp가 존재해도 동기화 및 추론 신선도가 자동 보장되지 않는다. 카메라 입력은 BEST_EFFORT, detection 출력은 depth10이다. detection 좌표는 camera optical(m), 공 추정 정확도 검증에는 독립 GT가 필요하다.

`/hsl/odom`: nav_msgs/msg/Odometry. pose.pose.position/orientation, twist.twist.linear/angular와 header/child_frame을 보존한다. normalized estimated_position=[x,y,yaw]는 header frame, estimated_velocity=[vx,vy,wz]는 child frame이다. 단위 m/rad/m/s/rad/s. joint_states 시각을 이어받는 leg odometry이며 독립 GT가 아니다. 실제 joint feedback이 없으면 신뢰 가능한 실물 odom을 만들 수 없다.

`/hsl/robot_pose`: hsl_interfaces/msg/RobotPoseEstimate. x/y/theta, covariance, validity, quality, num_landmarks_used, last_jump를 decoded에 보존하고 estimated_position으로 변환한다. 출력 header는 localizer의 now다. 반복 발행률이나 작은 공분산만으로 실제 위치 정확도 또는 새 관측률을 인증하지 않는다.

`/hsl/ball`: hsl_interfaces/msg/BallEstimate. valid/cam_valid, rel_x/y(base_link,m), field_x/y(map,m), vel_x/y(m/s), covariance, time_since_seen/disappeared(s), cam_xyz(optical,m)를 보존한다. 현재 공통 변환기는 이 토픽을 decoded로 보존하며 normalized estimated_position으로 자동 올리지 않는다. header=map이라고 rel_x/y와 cam_xyz까지 map 좌표로 해석하지 않는다. 주기적 예측 발행과 새 공 관측을 구분한다.

`/tf`, `/tf_static`: tf2_msgs/msg/TFMessage. 최상위 header가 없고 transforms 각각에 header.stamp/frame_id/child_frame_id와 transform이 있다. 기존 generic 변환의 header_timestamp=null은 맞는 표현이며 새 P0 검사기는 nested header를 별도 검사한다. /tf_static의 0 시각은 동적 센서의 0 시각과 구분한다. neck의 동적 TF는 실제 목 관절 정보에 의존한다.

## 제어 상태 기록: P0를 위해 생산자를 켜지 않음

`/joint_states`, `/joint_goals`: sensor_msgs/msg/JointState. name/position(rad)/velocity(rad/s)/effort를 보존하고 **이름으로** 결합한다. 배열 인덱스만으로 조인트를 결합하지 않는다. effort가 빈 배열이면 전류·토크 측정으로 대체하지 않는다. joint_goals는 명령이며 위치 GT가 아니다.

body controller는 read 시작 시각을 feedback stamp로 쓰는 기본 설정이 있다. walkkick runner는 최신 joint feedback stamp를 goal에 복사하고 없으면 now를 쓴다. neck adapter는 now를 쓴다. 따라서 같은 goal 토픽이라도 header 의미가 다르다. source 근거는 source_review.json과 runner의 header.stamp 할당 구문이다.

`/cmd_vel`, `/cmd_vel_raw`, `/cmd_vel_behavior`, `/cmd_vel_teleop`: geometry_msgs/msg/Twist. linear.x/y(m/s), angular.z(rad/s)를 command_velocity로 정규화하며 frame은 기존 계약상 base_link다. 메시지에 header가 없으므로 수신 시각을 기록한다. 명령 변화는 실제 지면 이동 시작이 아니다. 킥 러너가 고정 내부 cmd를 쓰는 경우 0 Twist도 움직임 부재를 뜻하지 않는다.

`/robot_state`, `/u2d2/behavior_status`, `/hsl/behavior_debug`, `/motor_diagnostics`: std_msgs/msg/String. data 원문과 가능한 JSON 필드를 보존한다. header가 없으므로 수신 시각이며, 상태 전이 구간은 기록상의 dwell이다. 진단 V/온도/전류/hw_error는 코드의 변환 단위와 모터 ID를 함께 남긴다. 정상 범위나 안전 인증으로 쓰지 않는다.

`/u2d2/torque_enabled`, `/buttons/stop_pressed`, `/kick_trigger`, `/kick_done`: std_msgs/msg/Bool. data와 수신 시각을 기록한다. 이벤트가 0회인 것은 missing periodic stream과 다르다. torque flag는 별도의 하드웨어 상태 검증을 대신하지 않는다. P0에는 이벤트를 발생시키기 위한 버튼 조작이나 kick trigger 발행이 없다.

## 팀 상태·역할 정보: 통신 지연 측정과 구분

`/hsl/team_state`: hsl_interfaces/msg/TeamState. header/my_player_number/teammates를 보존한다. team_comm 노드의 header는 host now다. 타인의 상태 유효기간이나 sender timestamp가 있어도 실제 packet 송신·수신 trace 및 양단 clock 검증 없이는 단방향 지연을 계산하지 않는다.

`/hsl/strategy_state`: hsl_interfaces/msg/StrategyState. header, game_state/tactic/set_play/position_role/active/active_player/my_time_to_ball/skill을 보존한다. 정책 선택 내용이지 물리적 대응 가능성의 GT가 아니다. 팀·역할 노드 구동은 이번 P0 준비에서 수행하지 않았고 자동 실행 목록에 포함하지 않는다.

## 결측과 현장 확인

기본 JSONL 스키마에 source timestamp가 남아도 command_start/motion_start/target_reached/communication_send/receive를 의미 있게 채우는 데 필요한 이벤트가 없으면 null이다. target_position도 실제 목표 trace가 없으면 null이다. 모든 토픽이 도착했다는 이유로 독립 GT, 실제 네트워크 지연, 목표 도달 성공이 측정됐다고 하지 않는다.

예시 프로파일의 min_messages=1 또는 2는 0수신/짧은 기록을 탐지하는 기능 검사 기준이다. 정상 주파수·안전 기준이나 충분한 실험량이 아니다. 실제 기록시간과 expected stream 목록은 현장 승인 구성에 맞춰 고정하고, 근거 없는 필수 토픽을 추가해 제어 노드를 켜도록 유도하지 않는다.
