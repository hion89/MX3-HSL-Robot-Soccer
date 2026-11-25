# Phase 5B-1 P0 비구동 사전검증 준비 결과

작성일: 2026-11-23.

## 판정

**P0 오프라인 준비 완료 / 현장 P0 미실행 / P1 이동 실험 진입 불가.**

기존 5B0 결과·README·schema, 5B1 preflight, 5A 측정 프로토콜과 실제 추출·수집·분석·검증 코드를 검토했다. 실제 저장소 루트는 `C:/Users/jes/Desktop/MX3_Match_Sim`이며 신규 코드·문서·결과는 `research/cah_mappo/` 안에만 생성했다. 기존 코드와 결과를 고치지 않았다. 현장 센서 연결, 실물 ROS 실행, 토크·보행·회전, 원격 접속, 새 패키지 설치, Git 커밋·푸시, 재학습 및 보호 Holdout 열람은 수행하지 않았다.

## 1. P0 준비 상태와 미충족 조건

로컬 기준 source는 MX3/HSL 프로젝트이며 `real_base`는 XM_robot04 URDF와 MCU/FTDI 장치 설정, 카메라 및 세계모델을 사용한다. 역할 실행 경로에는 walk_v0_6_13500.onnx와 walkkick_v1_5b_22300.onnx가 선택된다. 이는 **현재 로컬 소스 설정**이다. 실제 로봇의 현재 모델·OS·설치 경로·hash·전원·실행 프로세스·가용 공간은 이번에 확인하지 않았다. 과거 Orin/ROS2 Humble 운용 기록을 현재 설치 검증으로 대체하지 않았다.

확보된 것은 원본 bag/CSV의 읽기 전용 재분석, 기록 규격·합성 검사, 실제 소스의 정적 검토와 현장 절차다. 미충족 항목은 다음과 같다.

- 실명 장비 담당자·안전 담당자와 확인 일시.
- 실제 모터·배터리 기종에 근거한 온도/전류/전압 중단 기준.
- 전원 계통 및 모터 구동 격리, 비상정지·기구적 보호 검증.
- 실제 설치 정책/runner/실효 제한과 기존 프로세스의 확인.
- 독립 GT 장비·좌표 교정·독립 검증 오차와 clock offset/drift/잔여 오차.
- 실물 DDS 수신·QoS·기록 부하·종료 및 오류 처리의 현장 P0 검증.
- 실제 저장 공간과 쓰기 속도, 원본 봉인 및 현장 실험 승인.

`readiness.json`은 BLOCKED이며 `onsite_evidence_UNFILLED.json`은 빈 양식이다. 체크리스트는 제출된 증거의 진위나 장비 안전성을 자동 인증하지 않는다. 모든 칸이 채워져도 사람이 검토하고 별도 실행 승인을 받아야 한다.

## 2. 분리 가능한 기능과 구동 위험 경로

실제 source 17개를 import 없이 text/AST로 검토하고 path/sha256/줄 번호/호출 식을 `source_review.json`에 저장했다. 외부 드라이버·펌웨어·설치본의 전체 안전 인증은 아니다.

`real_base.launch.py` → `dynamixel_system.launch.py`는 body controller, 버튼 입력, 목 제어·스캔, MCU 등을 포함한다. 따라서 기존 base/start 실행은 P0 센서 전용 실행으로 인정할 수 없다. `real_role_trial` → `control`에는 보행·킥 등 제어 경로가 있다. gait 정지나 cmd_vel=0만으로 다른 제어 경로가 차단되지 않는다.

body_motor_controller는 초기화 중 hardware.open()과 startup gains 적용을 수행한다. U2D2Hardware.open() 자체는 SDK 객체 준비/PWM limit 읽기 등을 하며, 상위 노드가 설정 write를 한다. 시작 시 자동 torque ON을 하지 않는다는 주석과 내부 `_torque_enabled=False`는 실제 전체 모터 토크 readback 또는 물리 전원 차단 증거가 아니다. P0에서 진단값을 얻으려고 이 노드를 켜지 않는다.

button_input은 START의 robot_init/torque 및 STOP의 torque 제어를 호출할 수 있다. neck_control은 관절 목표·profile velocity·torque 서비스를 사용한다. neck_scan은 enable 경로에서 neck torque true와 목표 각도를 발행한다. 이들은 기록 프로세스가 아니다.

MCU bridge는 serial open 후 IMU enable, PING, watchdog 재전송을 수행한다. 순수 바이트 수신기가 아니므로 firmware의 명령 의미·USB reset·전원 계통이 확인돼야 한다. IMU parser 자체는 입력 바이트를 해석해 IMU/중력 값을 발행하는 분리 후보지만 실시간 입력 원천은 이 MCU 경로에 의존한다.

카메라/YOLO 및 TF/세계모델은 별도 검토 후보지만 RealSense 외부 driver 초기화는 repository 소스만으로 인증하지 못했다. perception launch에는 URDF 및 선택적 joint_state_publisher_gui도 있다. GUI/replay 값은 실물 관절 측정이 아니고, 실제 제어 그래프와 섞이면 입력을 오염시킬 수 있다. 실제 joint feedback을 제외하면 동적 neck TF와 leg odometry 등은 필요한 실물 입력이 없을 수 있다. 결측으로 표시한다.

기존 passive recorder는 이미 승인된 토픽에 구독만 만들고 publisher/service/모터 명령을 만들지 않는다. 그러나 생산자 안전성이나 저장 성능을 보장하지 않는다. 실제 하드웨어 없이 검증 가능한 것은 schema/변환/파일 잠금·봉인/오프라인 검사까지다.

## 3. 실제 기록 대상·데이터 계약

전체 대응은 PHASE5B1_P0_TOPIC_FIELDS_KO.md에 정리했다. 센서 후보는 IMU와 camera_info 및 별도 원본 color/depth다. 실제 관절·진단은 현재 controller 의존성 때문에 P0 필수 생산자로 자동 켜지 않는다. odom/pose/ball/TF는 해당 실제 입력이 준비된 경우의 추정 자료다.

기존 collector는 25개 상태·명령·TF 토픽을 지원하지만 raw image/MCU raw byte는 기본 목록에 없다. 토픽별 type과 raw decoded 필드는 보존하되 normalized velocity/pose로 자동 변환되는 것은 Twist, Odometry, RobotPoseEstimate다. BallEstimate 등은 raw decoded로 보존한다. 목표 도달/실제 이동 시작/packet 송수신 시각은 별도 근거 없으면 null이다.

IMU stamp는 host parser 처리 시각, odom은 joint feedback을 따른 시각, localizer와 TeamState는 host now, 검출은 color header 복사, ball 추정 발행과 last_seen은 별도다. TF는 각 transform 내부 header를 검사해야 한다. /joint_goals에는 header 의미가 다른 스트림이 섞인다.

구독 QoS는 기본 sensor_data이고 /tf_static은 reliable/transient_local/depth1이다. source의 reliable10 출력 또는 카메라 best-effort 입력과 구분했다. 과거 bag의 offered QoS 문자열을 원형 보존했지만 live DDS 연결/지터/누락률 검증은 아니다. static TF 전체 연결 수신과 burst 처리도 현장 확인 대상이다.

기록 오류는 schema/type/frame/시각/필수 스트림을 별도로 보고한다. 코드상의 flush/fsync/예외 정리는 확인했지만 실제 disk-full, 강제종료 중 부분 write, callback 적체나 처리량은 실물/기록 PC에서 미검증이다. 오류 trial은 불완전 원본으로 보존하고 자동 재개하지 않는다.

## 4. header 역전 168회 재검증

원본 kick bag은 19.334396895초에 4,253개 메시지다. 그중 /joint_goals는 1,692개이며 다음 결과를 원래 record 순서에서 재현했다.

- 합쳐진 goal header: 역전 168회.
- 서로 다른 관절 이름 서명 사이의 역전: 168회.
- 다리 12개 서명 910개, 몸통 18개 서명 60개, neck_yaw 361개, neck_pitch 361개: **각각 내부 역전 0회**.
- goal bag 수신 시각 역전: 0회.

현재 코드의 walkkick runner는 최신 joint feedback stamp를 goal에 복사하며 없으면 now를 쓴다. neck_control은 now를 쓴다. 오래된 feedback 시각을 갖는 goal과 현재 발행 시각을 갖는 goal이 섞이면 aggregate header가 뒤로 보일 수 있다. 이 메커니즘은 관측 결과와 일치한다. 다만 당시 설치 코드/각 메시지 publisher GID가 없으므로 모든 과거 메시지의 생산자와 clock을 확정한 것은 아니다. names 서명은 publisher ID가 아니다.

따라서 집계 역전을 곧바로 시스템 시계가 168회 뒤로 갔다고 해석하지 않는다. 반대로 서명별 monotonic이라는 사실을 동기화 증거로 사용하지도 않는다. recorder/header/ROS time/외부 GT의 offset·잔여 오차가 측정되지 않아 **단방향 지연은 계산 불가**다.

기존 5B0 검증기는 같은 topic header 집계를 이유로 kick 결과를 FAIL/168오류로 유지한다. 새 서명별 감사의 오류 0은 이 알려진 혼합을 설명하는 추가 관점이지 기존 오류 삭제나 동기화 PASS가 아니다. 개별 역전 record ID와 원본 messages.id locator는 kick_timestamp_audit.json에 있다.

## 5. 실제 오프라인 Dry-run

기존 분석 환경 Python 3.12.14 / rosbags 0.10.11, 승인된 연구 .venv를 사용했다. 새 패키지를 설치하지 않았다. 실행 결과는 `phase5b1_p0_outputs/study_v1/`에 보존했다.

원본 walk bag 120,364행, kick bag 4,253행, shadow CSV 906행을 다시 추출했다. 합계 125,523행이며 각 dataset의 manifest/정규화 gzip JSONL/seal이 기존 것과 SHA256으로 일치했다. 기존 분석도 각각 두 번 실행하여 analysis/validation/command_segments 파일이 바이트 단위로 일치했다. P0 stream 감사 역시 동일 입력 반복 결과가 일치했다.

신규 31개 합성 테스트와 기존 285개, 총 **316개 테스트를 실제 실행하여 통과**했다(전체 unittest 18.635초). 단위/형식 오류, 필요한 센서 frame 결측·미등록, 필수 스트림 0/부족, receive/header/nested TF 역전, static TF의 0 stamp 허용, GT 결측 보존, 긴 간격과 확정 손실의 구분, signature 혼합, 현장 증거 부재, 시각 유효 범위·불확실성, 파일 충돌·재현성을 포함한다. 합성 입력은 실측 결과가 아니다.

현존 kick bag에 없는 /imu/data를 요구한 진단에서는 의도대로 required_stream_missing_or_short가 발생했다. /kick_done처럼 발생하지 않을 수 있는 이벤트는 min_messages=0으로 구분했다. 이는 원래 kick 기록을 현장 P0 성공으로 인정한 시험이 아니다.

실제 CLI에 기존 study_v1 경로를 다시 전달한 검사도 FileExistsError/종료코드1로 거부됐고 completion.json hash가 그대로였다. 원본 영상 자체는 확보하지 못했고 기존 메타데이터·추출 자료만 보존했다. 독립 GT가 없는 모든 행은 그대로 결측이며 위치 정확도·실제 지면 속도·네트워크 단방향 지연을 생성하지 않았다.

기존 코드/문서 및 보호 체크포인트를 포함한 사전 hash 대상 **397개 파일 변경 0**, 기존 원본 자료 **8개 hash 일치**를 확인했다. 이전 출력 및 checkpoint를 덮어쓰지 않았다. 자세한 범위는 protected_hashes_before.json, protection_check.json, replay/verification.json에 남겼다.

## 6. GT·저장 공간 준비

외부 고정 카메라, 검증된 실제 거리 기준점, 강체 표식/몸통 기준점 변환, 렌즈·원근·표식 높이를 반영한 공간 교정, 교정에 사용하지 않은 점에서의 오차 측정, 프레임 PTS와 로그의 공통 시각 단서가 필요하다. 기준 장비 정확도가 검증되지 않으면 절대 위치 GT로 인정하지 않는다. 구체적 오차·가림·동기 요구는 PHASE5B1_P0_GT_REQUIREMENTS_KO.md에 분리했다.

색상/깊이 848×480×30, BGR8+16UC1 한 번 기록을 가정하면 raw payload만 61.056MB/s, 30초 1.83168GB, 90초 5.49504GB다(10진 단위). 이는 실측 throughput이 아닌 조건부 계산이다. 원본 bag의 작은 용량은 raw image 녹화 예산으로 사용할 수 없다. 실제 로봇의 여유 공간·지속 쓰기는 미측정이다. 로컬 PC disk snapshot은 별도로 저장했고 장비 사용 합격 판정으로 쓰지 않았다.

## 7. 현장 절차와 다음 승인 범위

현장 8단계 문서는 PHASE5B1_P0_OPERATOR_PROCEDURE_KO.md다. 담당자 확인 → 전원/구동 격리 → 기록 장치/새 경로 → 시각/좌표 계약 → 승인된 센서 기록 가능성 → GT 교정/동기 → 제한 기록과 무결성 → 봉인·재분석 순서다.

우선 준비할 것은 다음과 같다.

1. 장비 담당자·안전 담당자, 정확한 모터/배터리 자료와 중단 기준·물리 격리 방법.
2. 독립 GT 카메라/표식/거리 기준 및 교정·시각 공통 단서.
3. 실제 기록 장치의 저장 공간·성능과 승인 대상 센서 구성을 한정한 실행 계획.
4. 현장 비구동 P0에 대한 별도 승인 및 그 결과 검토.

다음 승인 요청 범위는 **승인된 센서·기록 경로의 현장 비구동 P0 검증**이다. 기존 base/start 일괄 실행이나 모터 버스 제어 시작을 자동 포함하지 않는다. P1 이동은 현장 P0 성공과 나머지 진입 조건 확인 후 별도로 판단한다. 이번 단계에서 실제 실험을 시작하지 않았다.
