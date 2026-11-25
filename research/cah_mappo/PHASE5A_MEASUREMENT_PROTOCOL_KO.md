# Phase 5A 실물 성능 측정 프로토콜

설계일: 2026-11-23. **미실행 계획서**다. 이 문서의 명령 크기·반복 수·장소 배치는 예비안이며 로봇을 구동하라는 지시가 아니다. Phase5B에서 장비·안전·계측 변경 범위를 별도로 승인받은 뒤 시행한다. 실제 측정값은 [자료 목록](PHASE5A_DATA_INVENTORY_KO.md)과 구분한다.

## 1. 단계와 공통 착수 조건

처음에는 robo04 한 대로 시각/기록/정지 인식→직선·정지→회전→목표 이동을 확인한다. 이후 정지 상태 통신, 두 대의 역할 반응, 마지막으로 승인된 복구 시험을 분리한다.4대·4대4 경기나 실제 공격/충돌을 첫 측정에 넣지 않는다. 한 대 측정으로 로봇 간 성능 분포를 얻었다고 주장하지 않는다.

각 가동일에는 조작자1명과 안전 담당자1명을 둔다. 카메라/기록 확인과 로봇 받침을 혼자 동시에 수행하지 않는다. 사람·케이블·장애물 없는 구획, 미끄럼/단차 점검, 안전 줄 또는 기립 보조대와 완충 공간을 준비한다. 줄은 보행 측정 중 하중을 받지 않아야 하고 접촉/장력 발생 시각을 기록한다. 장력으로 지지한 구간은 자유 보행 보정에 섞지 않는다.

다음 현장 기록란이 채워지기 전에는 torque ON 측정을 시작하지 않는다.

- 로봇 ID/정확한 모터·배터리 모델, 제조사·연구실의 허용 온도/전압/전류, 시험 중단값 `T_stop`, `V_stop`, 이상 상태 대응 절차. 기존 로그의45°C/11.6V는 안전 기준이 아니다. 이번 단계에서 임의 수치를 정하거나 장치 제한을 변경하지 않는다.
- 충전/커넥터/관절·발바닥·잔디 상태, 전원 분리와 물리 OFF 버튼 담당자. 토크 OFF는 지지력을 제거하므로 받침 없이 정지 방법으로 기대하지 않는다.
- 센서·GT·기록·시각 동기 상태, 시간 오차 상한, disk 여유, 로그 저장 성공, 역할/teleop/mux 제어권. 처음에는 자율 공격/자동 킥과 중복 명령원을 비활성화할 계획을 별도 승인받는다.
- 실제 설치 모델 hash, source/launch/runner variant, gait mode, drift 보정, 필터, kick/fall gate, 실효 cmd 제한. 로컬 소스 기본값이 실물에 적용됐다고 가정하지 않는다.
- 비상정지는 안전 담당자가 받침·정지/전원 절차를 관리한다. 목표/속도 갱신 중단과 zero 명령만으로 토크·낙상 FSM까지 정지했다고 간주하지 않는다.

모든 움직임 시험은 다음 경우 즉시 중단한다: 균형 상실·예상 밖 동작/가속·명령원 간섭·안전 줄 하중·사람 진입·경계 접근·hw_error·확정한 온도/전압 범위 이탈·시각 역전/동기 실패·GT/필수 피드백 상실·기록 누락. 멈추는 데 필요한 여유 거리는 예비 정지 실험 전에는 알 수 없으므로 안전 담당자가 넓은 구역을 확보한다. 실패 후 자동 재시도로 가열·배터리 저하를 누적하지 않는다.

## 2. 좌표와 독립 기준 장비

지도 map은 필드 중앙 원점, +x 상대 골대/+y 왼쪽, m/rad다. base_link는 로봇 전방/+왼쪽, 카메라 optical은 +x 오른쪽/+y 아래/+z 전방이다. map↔odom↔base_link↔목↔camera의 TF를 보존한다. GT의 몸체 기준점은 base_link에 고정된 기준점의 바닥 투영으로 정의하고 tag→base 외부 변환을 측정한다.

권장 장비는 독립 고정 카메라/모션캡처, 몸통의 강체 표식, 측량한 바닥 기준점·거리 기준자, 공 중심 추적용 영상, 시각 동기 표식, 별도 기록 PC다. 장비 보유 여부는 미확인이다. 카메라 방식이면 intrinsics/렌즈왜곡/촬영 자세/시야 내 기준점의 실제 좌표와 불확실성을 먼저 측정한다. 몸통 표식은 바닥 위에 있으므로 단순 바닥 homography를 그대로 적용하면 시차가 생긴다. 표식의3D 자세와 몸체 외부 변환을 사용하거나 높이/투영 오차를 명시한다. 공도 반지름·중심 높이를 실제로 측정한다.

차선책은 줄자로 등록한 시작/종료 위치와 동기 촬영이다. 이 경우 끝점·총 시간의 제한된 측정만 가능하며 정밀 순간 가속·지연·전체 경로를 얻었다고 하지 않는다. 로봇의 자체 pose/odom/바퀴 없는 다리 추정/IMU 적분은 독립 GT가 아니다. GT의 calibration check points와 본 trial points는 분리한다.

**교정용 시작 위치는 경기 운영의 고정 초기값을 바꾸지 않는다.** 이동 예비 시험은 연습장 안쪽 예: map(-0.75,0,0), 반대 방향은(+0.75,0,π)를 후보로 한다. 실제 로봇을 그곳에 놓은 후에만 해당 실측 초기값을 알려야 한다. 기존 경기 시작 x=-1.95/y=+0.8675/yaw0을 다른 장소의 로봇에 그대로 발행하지 않는다. START 임시 시드와도 구분한다.

## 3. 시각·동기·event 정의

한 호스트에서 처리 구간은 monotonic 시계로 재고 ROS/wall 시계와의 대응을 함께 기록한다. 다른 프로세스·PC의 monotonic 절댓값을 직접 빼지 않는다. `use_sim_time`, 시계 점프, boot/session ID를 기록한다. message header가 노출·센서 취득·수신·발행 중 무엇인지 topic별로 확인한다.

다른 PC/로봇은 가능한 장비에서 NTP/PTP 등으로 동기화하고 사용 방법·전후 offset·drift·오차 상한을 기록한다. 설치나 실제 오차 확인은 이번 단계에서 하지 않는다. “동기 기능을 켰다”는 사실만으로 ms 정확도를 가정하지 않는다. 독립 영상은 양쪽 기록에서 관측되는 LED/펄스 등을 연결하고 영상 frame timestamp·rolling shutter·표식 발생 회로 지연을 불확실성에 포함한다. 수동 박수·버튼/로그 수신 시각을 정밀 공통 시각으로 간주하지 않는다.

통신에서 offset δ를 receiver clock−sender clock으로 정의하면 추정 지연은 `d = t_receive − t_send − δ`다. 최악 오차는 동기·TX timestamp·RX timestamp 오차의 합으로 상한을 제시한다. 독립성 근거가 있을 때만 제곱합을 이용한다. 음의 지연이 나오면0으로 숨기지 않고 동기/시각 의미 오류로 표시한다. RTT는 동일 호스트 시계로 계산할 수 있지만 RTT/2를 실제 단방향 지연이라고 단정하지 않는다.

표준 event는 `trial_start`, command 생성/발행/수신/use, GT motion onset/still, goal 생성/arrived, 관측 취득/수신/소비, packet send/receive/strategy use, 물리 넘어짐 기준/PROTECT/getup start/upright/NORMAL/service resumed, abort/end다. 각 event는 trial_id/robot_id/source_clock/time/uncertainty/source evidence를 가진다. 기존 토픽에 없는 필드는 향후 sidecar trace로 추가하는 **설계**다. 현재 기록되고 있다고 가정하지 않는다.

## 4. 공통 수집 목록과 데이터 관리

기존 토픽의 후보는 다음과 같다. 미래 실행 전 실제 node/topic/type/QoS/remap과 센서 표본 시각을 확인하며, 지금 이 목록을 실행하지 않는다.

- 이동/정책: `/cmd_vel_behavior`, `/cmd_vel_teleop`, `/cmd_vel_raw`, `/cmd_vel`의 linear.x/y, angular.z. Twist에는 header가 없으므로 callback 수신 시각과 향후 source publish trace를 구분한다.
- 몸체/관절: `/joint_states`의 header/name/position/velocity, `/joint_goals`의 header/name/position, `/imu/data`, `/imu/data_50hz`의 header/orientation/angular_velocity/linear_acceleration. goal header는 현재 최신 joint feedback stamp를 재사용할 수 있다.
- 안전/상태: `/motor_diagnostics` JSON의 stamp/id/name/pwm/current_ma/voltage_v/temp_c/hw_error, `/robot_state`, `/u2d2/torque_enabled`, `/u2d2/behavior_status`, `/buttons/stop_pressed`, `/kick_trigger`, `/kick_done`.
- 위치/비전: `/hsl/odom`의 header/pose/twist, `/hsl/robot_pose`의 x/y/theta/covariance/validity/quality/num_landmarks_used/last_jump, `/hsl/ball`의 rel/field/cam/vel/covariance/valid/time_since_seen, `/perception/detections_3d`의 header/results/class_id/score/pose.
- 영상·프레임: `/camera/camera/color/image_raw`, aligned depth, CameraInfo의 header/encoding/size/K/D/깊이 단위, `/tf`와 `/tf_static`, 목 관절 상태. 원시 영상은 필요한 세션에서만 기록하고 카메라 부하·저장 손실을 평가한다. `/tf_static`의 transient-local QoS/늦은 구독도 사전 확인한다.
- 상위/팀: `/hsl/strategy_state`의 header/position_role/active_player/skill, `/hsl/behavior_debug`의 target/target_kind/pose/cmd, `/hsl/team_state`의 header/teammates.received/player_number/pose_validity/ball_age/fallen/role. marker 그림을 원래 관측 또는 명령 trace로 대체하지 않는다.
- 런너 CSV는 tick/time_sec/cmd/action/q/qd/q_target/gait_state/inference_ms/run_state/publishing/robot_state와 metadata를 원형 보존한다. raw source stamp와 소비시각은 현재 CSV에 완전하게 있지 않아 추가 trace가 필요하다.

`/measurement/trial_event`, GT pose CSV, packet/decision sidecar 등의 이름은 **향후 계측 산출물 제안**이다. 현재 존재하는 ROS 패키지/토픽이라고 부르지 않는다. 계측은 수동 관측 전용이고 `/joint_goals` 등 제어 토픽의 새 publisher를 만들지 않는 방향으로 설계한다. driver·패킷·제어 코드를 바꾸는 경우에는 별도 승인/검증이 필요하다.

저장은 session→robot→trial 단위로 하고 bag/영상/GT/sidecar/launch·모델·지도 hash/실험 조건/시각 보정/중단 이유를 한 manifest로 연결한다. frame 누락·bag drop·source별 seq gap을 기록한다. host별 bag은 로컬에 기록하고 나중에 오프라인 결합한다. 여러 로봇에 같은 ROS topic 이름이 있으므로 공통 domain에 그대로 연결해 pose/명령을 섞지 않는다. 기존 ROS_LOCALHOST_ONLY=1과 UDP 팀 통신은 별개이며 다중 호스트 ROS 구독을 위해 임의 해제하지 않는다. 네임스페이스/도메인/프레임/팀·선수 번호 설계는 후속 승인 대상이다.

## 5. P0: 계측·시각·부하 예비 시험

목적: GT 독립성과 각 timestamp의 의미, 기록 부하를 먼저 검증한다. 로봇1대와 기록 PC/GT 카메라를 사용하고, 초기는 안전 받침·torque OFF/정지 센서 상태다. 외부 기준점을 촬영하고 동기 event를 세션 시작/끝에 넣는다. 무기록/상태만/영상 포함 조건을 구분해 각30초×3블록을 후보로 한다. 가동 불필요 항목은 모터를 켜지 않는다.

위 공통 토픽 중 실제 발생하는 센서·상태·clock/QoS와 영상의 연속 source stamp를 기록한다. GT checkpoint 재투영 오차, 길이/각도 오차, 동기 offset/drift 범위, topic별 중복/누락/간격 분포와 recorder 부하를 계산한다. 과도한 저장 부하·시계 역전·미확인 시각 의미·GT 가림이면 중단한다. 이 시험을 통과하지 못하면 이동 지연/정확도 수치를 만들지 않는다.

## 6. P1: 선속도·가감속·정지 분포

목적: 명령 대비 실제 이동, 출발/정지 지연, drift와 반복 변동을 분리한다. 로봇1대/외부 몸체 GT/모터 상태 기록, 사람 없는 중앙 통로를 사용한다. 기본 기립·동일 발 배치·목 자세로 시작한다. 바닥재(연구실/잔디), 배터리 전압·부하, 모터 온도, 보정·gait mode를 조건으로 기록한다.

예비 행동은 전진 명령0→0.15m/s를4초 유지→0→GT 정지1초 관측이다. 안전성이 확인되면0.20m/s를 같은 방식으로 추가한다. 시작/종료 여유와 실효 cap을 먼저 확인하고 GT 이동1m/구획 경계/균형 이상에서 시간보다 먼저 중단한다. 반대 방향은 몸체 yawπ로 바꾸어 같은 전진 명령을 주어 바닥 방향 효과를 본다. 초기 예비안은 속도2개×진행 방향2개×독립5회다. 실제 최고 속도를 찾기 위해 한계를 올리는 시험은 제외한다.

횡이동/후진은 직선 시험 후 별도 승인한다. 예비 횡명령±0.12m/s는 작은 명령 보정과 제한의 영향을 확인한 뒤 후보로 사용한다.0.10을 요청해도 standing threshold/min_norm 때문에 다른 동작이 될 수 있으므로 requested/shaped/muxed/consumed cmd를 모두 기록한다.4초는 주행 거리 보증이 아니며 안전 구획에 맞춰 예비 조정 후 고정한다.

관측은 cmd pipeline·관절/IMU·정책 CSV·GT 위치/yaw·모터 건강이다. GT를 고정한 smoothing/미분 방법으로 처리하여 body-frame v_x/v_y, yaw rate를 계산한다. 예비 정지 잡음에서 motion onset 기준(예:정지 분포3σ 이상이 일정시간 지속)과 정지 기준을 정하고 본 측정 전에 고정한다. onset 정의·filter 지연·시간창을 함께 보고한다. 연속 표본을 독립 반복으로 세지 않는다.

가속은 smoothing 후 기울기/10~90% 전이시간, 감속은 zero command 이후 실제 속도 감소·정지까지 시간/거리로 평가한다. 평탄 구간이 없으면 steady speed를 억지로 추정하지 않는다. 각 trial의 bias/평균·중앙·분위수·표준편차, 동기/미분 불확실성을 제시한다. 하네스 하중·넘어짐·안전 중단은 숨기지 않는다. 휴식·재충전 기준은 P0의 실제 건강값으로 관리하고 순서를 블록 내 균형화한다.

## 7. P2: 회전과 방향 변경

로봇1대, 중앙 고정 위치, slack 안전 줄, GT yaw와 위치를 사용한다. 선속도0에서 회전±0.25rad/s를2초→zero 후 정지 관측, 안전 확인 후±0.50을 추가한다. 각 부호/크기별 예비5회이며 실제 cap 이내에서만 한다. 이후±90도 목표 회전 각5회는 별도 조건으로 측정한다. 목표 yaw와 실제 고정시간 명령 시험을 섞지 않는다.

기록은 P1 공통 목록과 yaw GT/목표/step event다. 실제 회전율·onset/stop 지연·overshoot·몸체 중심 drift·회전 완료시간을 산출한다. yaw는 unwrap 후 미분하고 최종 오차는 wrap한다. 선회와 제자리 회전은 별개 분포로 둔다. 발 미끄러짐·하네스 장력·균형 상실·구획 이탈이면 중단한다. 방향 변경 시 몸체가 이미 움직였는지와 이전 보행 위상을 남긴다.

## 8. P3: WalkToPose 도달과 상위 실행 지연

로봇1대, 독립 GT와 기존 목표 이동 경로를 사용한다. 시작 예(-0.75,0,0)→목표(+0.25,0,0), 동일 위치에 최종 yaw90도를 요구하는 조건, 반대 방향을 분리한다. 각 조건 예비5회, 한 trial에 초기화/목표1개만 준다. 목표 시각부터15초를 예비 관측 상한으로 두되 공간·건강 조건이 먼저 중단시킨다. 이 상한은 성공을 맞추기 위한 튜닝이 아니며 예비 이후 본 측정 전에 고정한다.

기록은 goal 요청/할당/결정/속도 publish·receive·policy use의 trial ID trace, `/hsl/strategy_state`, debug target, cmd 각 단계, GT pose/속도/관절/상태다. 현재 목표/arrived 전용 이벤트가 없으므로 정확한 내부 지연은 향후 계측 hook 없이는 측정할 수 없다.5Hz debug/string 수신을 소급한 정밀 결정 시각으로 쓰지 않는다.

GT 도착은 예비안으로 위치≤0.15m, yaw≤15도, 정지 기준을1초 유지한 첫 시점으로 정의한다. 이는 코드 래치와 별도 운영 정의이며 연구 과제의 허용 오차와 GT 분해능을 확인해 사전 고정한다. 코드 arrived 시간, GT 최초 진입, GT 안정 도착, 명령 정지, 물리 정지를 각각 보고한다. 추정값만 먼저 도착했는지와 도착 후 drift/해제도 평가한다. GT transform은 독립 측량으로 고정하고 각 궤적에 맞춰 다시 정렬해 오차를 지우지 않는다.

지연은 결정→속도 발행, mux/correction, policy 소비, 관절 목표 발행, 실제 운동으로 분해한다. 완전한 trace가 없으면 확인 가능한 두 끝점의 복합 지연만 그 이름으로 보고한다. 디버그 목표가 이전 skill의 잔존 값일 수 있어 goal ID와 실제 선택된 skill의 대응도 확인한다.

## 9. P4: 위치추정·공 검출·신선도

로봇1대, 외부 GT, 줄자로 등록한 기준점/공 중심, D405의 CameraInfo·원영상·depth·TF를 사용한다. 초기 정지 pose 예비안은 중앙4개 점(±0.6,±0.5m)×heading0/90도×독립 재배치3회, 각10초다. 사람이 지지한 자료와 자립 자료는 조건을 분리한다. 초기 GT pose를 알려주는 조건과 알려주지 않는 재위치 탐색은 별개이며 매 frame GT를 주입하지 않는다.

같은 독립 좌표계/시각에서 `p_est−p_GT`, wrapped yaw error의 bias/RMSE/median/p95, validity/quality와 실제 오차의 관계, jump/invalid 비율을 구한다. 다음으로 P1/P3 주행 궤적에서 이동 중 오차를 계산한다. 정지 반복 precision을 절대 accuracy라고 부르지 않는다. covariance는 GT 오차 대비 예측 일치도도 별도로 검토한다.

공 정지 예비안은 거리0.5/1.0/1.5m×방위−30/0/+30도×재배치3회, 각10초다. 실측 FOV/깊이/잔디/조명과 실제 공 크기를 확인해 가능한 조합만 사전 정의한다. 움직이는 공은 로봇 정지 상태에서 표시된 통로를 양방향으로 굴려 각 예비3회부터 시작한다. 공을 로봇에 맞히거나 킥시키지 않는다. 외부 영상으로 속도/위치를 측정하며 손으로 준 명목 세기나 공 tracker 속도를 GT로 쓰지 않는다.

기록은 비전·공·pose·TF·목 상태와 색/깊이의 각각 source stamp, detection 생성/소비·추적 update trace다. 배열 Hz, 공 positive 관측 Hz, 동일 source frame 중복, 추적 발행 Hz, 유효 결과 비율을 나누어 계산한다. 물리 GT로 가시 범위를 판단한 시간만 missed detection의 분모로 삼고, 화면 밖/가림을 오검출 실패와 구분한다. 안전한 가림판으로 짧은 occlusion을 주는 경우 별도 조건/예비3회로 원래 관측 age·invalid 전환·재획득 시간을 잰다.

color/depth skew, 카메라 노출/호스트 처리 시각 차이, TF 대기, 추정→전략/정책 소비 age를 분리한다. BallEstimate header는 주기 발행 시각이고 새 관측은 time_since_seen 등과 연결해야 한다. 정확도와 latency를 동시에 식별할 수 없는 수동 영상만 있으면 그 항목을 미측정으로 남긴다. GT/TF 유실·마커 가림·프레임 과다 drop·기립 과열이면 중단한다.

## 10. P5: 팀 통신·상태 발견

먼저 모터를 구동하지 않는2개 endpoint(로봇1대+PC 또는 두 PC)에서 기록 체계를 검증한다. 이것은 실제 두 로봇의 무선 부하·움직임 시험과 다른 조건이다. 이후 로봇2대의 정지/안전한 간단 이동, 장비가 준비되면4대로 확장한다. 시작 위치·팀/선수 번호·IP/AP/채널·거리·가림·네트워크 부하를 기록한다. 두 PC의 결과로4대 통신을 검증했다고 하지 않는다.

기존2Hz+사건 송신을 유지한 조건부터5분×독립3세션을 예비안으로 한다. 명시 상태 변화(역할/가용성)를10회/세션의 별도 고정 일정으로 만들되 모터 고장을 유발하지 않는다. 후속 지연/손실 주입은 독립 승인·격리 네트워크에서만 계획하며 외부망 부하를 만들지 않는다.

TX/RX capture 또는 passive sidecar에 session/robot/team/player/seq/packet bytes, timestamp 의미, send성공/실패, kernel/app 수신, ROS 발행, 전략 최초 use를 기록한다.16비트 wrap, 재부팅, 재순서, 중복, 세션 말미 미도착의 timeout을 구분한다. `/hsl/team_state` 반복 발행 수를 새 packet 수로 세지 않는다. source pose/ball 취득 stamp를 송신 seq와 연결하는 trace는 추가 승인 대상이고 기존 packet payload를 바꾸지 않는 계측을 우선 검토한다.

송신 간격과 수신 간격의 median/p95/지터, 성공 send 대비 unique matched receive 손실률, burst loss 길이, app TX→app RX 지연, RX→ROS/strategy use, 물리/명시 상태 변화→타 로봇 최초 인식 시간을 구한다. app 지연에는 OS queue와20ms polling이 포함되며 순수 무선 전파 지연이라고 부르지 않는다. sender sample age까지 얻었을 때만 전체 AoI를 계산한다. clock 오차가 효과 크기와 비슷하면 지연 순위를 정하지 않고 구간/RTT/간격만 보고한다.

미승인 공통 ROS 연결·팀/선수 번호 충돌·명령 토픽 혼선·시각 동기 실패·capture drop이면 중단한다. 실제 운영에서는 received TTL2초가 원본 정보 freshness를 보장하지 않는지도 관찰하되 이번 단계에서 timeout/패킷/전략 코드는 수정하지 않는다.

## 11. P6: 감지·기상·작업 재개

필수 선행조건은 P0~P3 완료, 장비 담당자의 기상 절차 확인, 넓은 완충 바닥·지지 장비·조작자/안전 담당자2명, 정확한 자동 복구/정지 gate 확인이다. 이 시험은 **별도의 고위험 장비 작업 승인**을 받아야 한다. 보행 중 일부러 밀어 넘어뜨리거나 모터를 고장 내지 않는다.

단계를 분리한다. 첫째, torque OFF/안전 지지 상태의 센서 방향 변화는 “IMU/검출 경로”만 검사한다. 이때 STOPPED gate 때문에 PROTECT가 안 나올 수 있으며 무반응을 검출 실패로 세지 않는다. 둘째, 승인된 지지/완충 조건에서 감지 시각을 확인한다. 셋째, 정지·torque OFF로 안전하게 눕혀 놓은 시작 상태에서 승인된 기상 시도만 평가한다. 눕힌 자세에서 시작한 성공률은 자연 낙상 후 복구 성공률과 별도다.

처음에는 허용된 앞/뒤 시작 자세별 예비3회 이내와 충분한 휴식을 제안한다. 자동3회 재시도를 독립 표본3개로 세지 않는다. session/attempt별 실패·수동 개입·중단·모터 건강을 남긴다. 최초 시도 실패나 이상 전류/온도, 관절 걸림, 갑작스러운 동작, 기록 상실이면 자동 반복을 멈추고 재승인을 검토한다.

GT 영상의 tilt/접촉/움직임, IMU, PROTECT, 실제 관절 운동 onset, upright 안정, NORMAL, 새 목표 요청/실제 서비스 재개를 동기화한다. 물리 낙상 시작 정의(예:사전 고정한 GT tilt 교차)와 IMU45도 threshold를 구분한다. `t_detect−t_GT`, `t_getup_motion−t_detect`, `t_upright−t_getup_motion`, `t_service−t_upright`, 전체 downtime을 각각 계산한다. 추정 tilt가 GT가 아니다.

성공은 GT로 안정 기립 후 승인된 짧은 목표/서비스를 다시 수행한 경우와 기립까지만 성공한 경우를 나눈다. 현재10초 timeout·NORMAL·kick_done을 성공의 독립 증거로 쓰지 않는다. 관측 끝 미복구는 우측 검열, 고장·실패는 성공률 분모에 유지한다. 현재는 성공 반복 자료가 없어 시간/성공률의 수치를 제안하지 않는다.

## 12. P7: 실제 행동 점유와 성공 판정

P1~P4 이후에만 로봇1대/공/독립 영상으로 접근→정렬→킥 전 hold→정책 동작→안정화→다음 명령 수용 시간을 분리하는 예비안을 검토한다. 현재 킥은 잔디에서 넘어짐 이력이 있으므로 이 계획서가 킥 가동을 승인하지 않는다. 처음에는 이미 안정성이 확인된 바닥/정책/실효 설정의 단일 조건에서 예비3회부터 검토하고 실패 반복을 강행하지 않는다.

기록은 trigger/done/state/goal/IMU/관절/GT 공 궤적/건강과 모든 gate다. 소프트웨어 done, 발 접촉, 공의 지정 구역 이동, 기립 유지, 다음 서비스 가능을 구분한다. 공 성공 기준은 실제 도면의 사전 목표 영역과 평가창을 별도 고정하고 결과를 본 뒤 좁히거나 넓히지 않는다. 킥에 쓰인 보행/킥 모델과 전략 MAPPO 모델을 혼동하지 않는다. 추상 서비스2초/12초를 곧바로 이 측정으로 대체할 수 있는지 검토하고 대응이 없으면 별도 과제로 둔다.

## 13. 반복 수·불확실성·누락 처리

각5회/3회와5분 세션은 **실행 가능성을 확인하는 예비 수**이지 통계적 충분성 근거가 아니다. 장비 상태를 안정화한 뒤 독립 trial/session의 분산·autocorrelation·일간 차이를 추정한다. 본 측정의 허용 CI 반폭 ε는 연구 과제의 마감 여유/목표 오차와 계측 분해능으로 사전 결정한다. 평균의 예비 근사 `n ≈ (t × s / ε)^2`로 비용을 점검하되 작은 예비 표본의 s 불안정성·세션 군집·비정규 지연 꼬리를 반영해 재검토한다. p95나 실패율에는 이 평균 공식만 적용하지 않는다.

trial 내부 수천 frame은 반복 수가 아니다. CI는 trial/session을 재표집 단위로 하고 성공률은 분모와 이항 불확실성을 병기한다. 실패0건만으로 실패확률0이나 안전성을 주장하지 않는다. 조건 순서는 블록 내 균형화하고 floor/day/battery/온도/모델을 기록해 상관과 순서 효과를 점검한다.

위치 오차에는 GT calibration/표식 외부변환/GT 재현성/시간 정렬 오차를, 지연에는 시계/표본 간격/검출 정의를 포함한다. GT 속도 v일 때 시간 오차 u_t가 위치 비교에 대략 v×u_t로 영향을 주는 것도 기록한다. 미분/smoothing 오차와 정지 기준에 따른 민감도는 별도다.

안전 중단은 정책 실패와 계측 실패의 이유를 구분해 원시 로그에 남긴다. 사전 규칙에 따라 전체 시도 수·유효 측정 수·중단/재시도 수를 모두 보고한다. 위험 동작을 성공률에서 삭제하지 않는다. GT 누락 구간을 공백0/성공으로 채우지 않고 미판정 시간·상하한을 제시한다. 본 측정/최종 평가는 예비·보정 trial과 분리한다.

## 14. Phase 5B 승인 묶음

1. 장비 목록·수치 안전 한계·GT/동기 확인 및 P0의 수동 기록.
2. P1~P3의 한 대 제한 명령/배치/목표 이동, 기록·중단 담당자와 관측 범위.
3. P4의 정지 인식/위치·제한 주행 관측. raw 영상 저장·용량·데이터 관리.
4. P5의 별도 endpoint/계측 hook/capture 권한/네트워크 설정, 이후 두 로봇 참여 범위.
5. P6/P7의 기상/킥은 장비 담당 확인 후 별도 승인. 기존 제어·패킷·가상환경 변경이 필요하면 구체적 변경을 먼저 검토한다.

현재 Phase5A는 이 계획서 작성에서 끝난다. 로봇 START/토크/launch/원격·모터 명령은 실행하지 않는다.
