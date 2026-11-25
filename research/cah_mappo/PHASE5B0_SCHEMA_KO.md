# Phase 5B-0 공통 기록 규격 · measurement-v1

이 규격은 로봇을 구동하지 않는 수집·분석 도구의 계약이다. 기존 ROS 메시지/팀 packet/연구 관측 규격을 바꾸지 않는다. 시간·좌표·측정 출처를 모르는 값을0이나 다른 센서값으로 채우지 않는다.

## 저장 단위와 출처

trial 폴더는 `manifest.json`, `records.jsonl` 또는 종료된 추출물 `records.jsonl.gz`, `seal.json`으로 구성된다. 원본 사본을 붙이면 `raw_sources/`에 hash가 포함된 파일명으로 보관한다. `data_origin`은 real_robot/real_robot_shadow/synthetic 중 하나다. 합성 테스트는 synthetic을 유지하고 실측 결과에 합산하지 않는다.

manifest에는 experiment_id/robot_id/trial_id, repeat_index, trial_condition, clocks, synchronization, frames, sources, mode, safety가 있다. 조건은 바닥·배터리·온도·모델 hash·러너/실효 명령 제한·초기 자세·GT 교정과 누락 이유를 기록할 수 있는 객체다. 개별 행은 `trial_condition_ref=manifest.trial_condition`으로 이를 참조한다. source.path/SHA256/locator가 원본 message ID/CSV 줄 번호를 가리킨다. bag 행은 원래 CDR payload의 SHA256도 남긴다. 과거 자료의 반복 번호는 미확인이므로 repeat_index=null이다. 새 trial은 실제 반복 번호를 요구하며 SET_ 식별자를 채우지 않은 template로는 session을 만들지 못한다.

## 각 행의 필드

- schema_version, experiment_id, robot_id, trial_id, record_id, trial_condition_ref.
- timestamp, header_timestamp. 전자는 recorder/callback/외부 측정 시각 등 행의 기준이며 후자는 원문 header를 그대로 분리한 값이다. header0을 recorder 시각으로 자동 대체하지 않는다.
- command_velocity, estimated_velocity, ground_truth_position, estimated_position, target_position. 없는 항목도 키를 유지하고 JSON null을 쓴다.
- command_start_time, motion_start_time, target_reached_time, communication_send_time, communication_receive_time. 관측·연결한 event만 채우며 파일명이나 명령값으로 물리 시작/도착을 추정하지 않는다.
- event_type, success, failure_reason. 성공은 true/false/null이고 소프트웨어 신호만 있는 과거 bag에서 물리 임무 성공을 자동 작성하지 않는다.
- source와 decoded. decoded는 원문 메시지/CSV 값을 보존하는 영역이다. 단위가 다른 원문 필드를 임의로 body velocity나 GT로 승격하지 않는다.

## 시간과 좌표

시간 객체는 `value` 정수, `unit=ns`, clock_id, source, semantic, uncertainty_ns를 가진다. epoch nanosecond를 float로 저장하면 정밀도를 잃으므로 거부한다. uncertainty_ns=null은 미확인이며0오차를 뜻하지 않는다. clock 선언에 basis와 host를 기록한다. 서로 다른 clock ID는 측정된 offset/error와 유효 구간이 없으면 비교 지연을 계산하지 않는다.

동기 offset은 target_clock−source_clock이다. `end−start−offset`으로 정렬하며 source 시각이 유효 구간 안에 있는지 검사한다. 오차 상한은 선언된 세 항의 합이며 미측정 항이 있으면 상한도 null이다. 음의 정렬 지연은0으로 보정하지 않고 오류 사유를 남긴다. 이 차이는 선언한 event 사이의 시간이며 자동으로 물리 지연·순수 무선 지연이 되는 것은 아니다.

평면 pose는 components[x,y,yaw]/units[m,m,rad], velocity는[vx,vy,wz]/[m/s,m/s,rad/s]다. frame_id는 manifest.frames에 선언돼야 한다. map은 필드 좌표, odom은 로컬 추정, base_link는 로봇 전방/왼쪽 기준이며 서로 자동 정렬하지 않는다. GT pose에는 independent=true, device_id, calibration_id, method가 필요하다. method=odometry는 거부하며 synthetic_truth는 synthetic 데이터에서만 허용한다. 운영자가 잘못된 장치 출처를 선언하는 것을 소프트웨어만으로 인증할 수는 없다.

raw decoded의 단위는 원래 ROS 타입 계약과 source frame을 유지한다. Imu.orientation은 무차원 quaternion, angular_velocity는rad/s, linear_acceleration은m/s², header는sec/nanosec+frame_id다. 이 로봇의 회전 관절 JointState.position/velocity는rad/rad·s⁻¹이며 body m/s가 아니다. Twist의 linear/angular는m/s와rad/s다. `/motor_diagnostics`의 current_ma/voltage_v/temp_c는mA/V/°C, stamp는별도 호스트 wall 초다. covariance도 원문 타입에 종속된다. 카메라 광학 x오른쪽/y아래/z전방과 지도 x/y를 혼동하지 않는다. 새로운 센서/기구에서는 이 계약을 확인한 뒤 frame/단위 mapping을 선언한다.

## 통신·GT 결측과 분류

통신 분석에는 두 시각뿐 아니라 source.packet_match의 packet_id/sender/receiver/tx_locator/rx_locator가 필요하다. 송신 로그 하나 또는 matched packet 출처 없는 두 값은 단방향 지연 결과로 쓰지 않는다. clock 정렬이 없으면 지연은 null이다. RTT/2로 대체하지 않는다.

결과 등급은 A=원문에서 직접 측정 가능한 기록/명령/간격, B=독립 GT가 없는 추정·상대 event 시간, C=코드 설정, D=미측정이다. A의 recorder 간격도 실제 센서 취득 주기·누락 확정이 아니다. 위치 정확도/절대 지면 속도 정확도/성공 기상은 기존 자료로 계산하지 않는다. future GT 행을 저장할 수 있다는 사실과 실제 GT 오차 계산기가 검증됐다는 사실은 다르다.

## 무결성과 검증

session/출력 파일은 exclusive 생성이다. append는 기존 ID와 충돌하면 거부하고 batch를 먼저 검사한다. 파일 잠금은 동시에 쓰는 collector를 막고, live writer는 ID를 한 번 읽어 재개 시 충돌을 검사한다. 전원 장애로 마지막 줄이 잘릴 수 있으며 validator가 이를 검출한다. 이미 기록한 원문을 고쳐 재개하는 자동 복구 기능은 없다.

seal 후 기록/manifest/첨부 원본 사본의 hash와 행 수를 검사한다. sealed journal append나 원본 추가를 거부한다. 원본 외부 파일은 읽기 전용 입력으로 hash를 비교하고, bag SQLite에는 mode=ro/query_only를 쓴다. source 원본을 복사한 경우에도 원본 파일을 수정하지 않는다.

validator는 필수 필드/명시적 null/finite 수치/단위·프레임/GT 출처/중복 ID/잘린 JSONL/시각 역전/동기 결측/봉인 변경을 보고한다. 시각 검사는 원래 record 순서에서 topic+clock별로 수행한다. header 역전은 mixed stream 또는 늦은 source 표본일 수도 있어 물리 clock reset의 증명으로 해석하지 않는다. 분석에서 시간순 정렬로 지우지 않는다.

preflight_blockers는 미확인 안전 수치·담당자·지지/정지 점검·GT/시각 조건을 표시한다. `motion_readiness=NOT_AUTHORIZED_BY_THIS_TOOL`은 항상 유지한다. 도구가 실제 로봇 가동을 승인하거나 자동으로 명령을 내리지 않는다.
