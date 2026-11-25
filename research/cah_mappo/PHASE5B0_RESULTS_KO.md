# Phase 5B-0 기존 기록 분석 및 수집 도구 검증 결과

2026-11-23. 이번 단계는 오프라인 분석·실측 기록 준비다. 새 실물 측정, ROS2 launch/구독기 실행, 모터 명령, 원격 접속, 설치, Git commit/push, 재학습을 하지 않았다. Phase5A 문서5개를 기준으로 했으며 기존 문서·코드를 수정하지 않았다. CAH-MAPPO는 구현하지 않았다.

**판단: 오프라인 도구와 규격은 준비됐으나 Phase5B-1의 로봇 이동 측정 착수 조건은 아직 충족되지 않았다.** 정확한 장비 안전 수치, 독립 GT/시각 동기와 실제 ROS 환경의 구독·기록 부하 검증이 필요하다. 도구의 검증 PASS는 가동 승인이 아니다.

## 1. 실제 사용 환경과 자료

기존 Python3.12.14 환경을 사용했다. bag decoder는 W의 `.analysis_env`에 이미 설치된 rosbags0.10.11, 표준 라이브러리 수집·분석과 기존 연구 회귀는 Q의 `.venv`다. 새 패키지를 설치하거나 전역/ROS/Isaac 환경을 변경하지 않았다.

Q는 `C:/Users/jes/Desktop/MX3_Match_Sim/research/cah_mappo`, W는 `C:/Users/jes/Desktop/isaac_ws/sim2real_evidence`다. 원본 bag2개는 SQLite mode=ro/query_only로 열어 CDR을 해독했다. 기존 분석 스크립트는 원본 analysis 디렉터리에 쓰는 경로가 있어 실행하지 않았다. 새 출력만 Q 아래에 만들었다.

최종 자료는 `phase5b0_outputs/study_v2/`, 재현·감사는 `verification_v2/`다. 개발 v1과 중간 테스트도 보존했다. 과거 반복 번호를0으로 부여하던 중간 metadata는 원실험 번호로 오해되지 않도록 v2에서 null로 고쳤다. raw 메시지·원본 시각·명령값을 바꾼 것은 아니다.

## 2. 실제 추출 결과

### 10월7일 bag

원본: `W/walk_20261007_161217/walk_20261007_161217_0.db3`.

- 메시지120,364개, 관측 구간422.145327413초.
- recorder 시작/종료 ns:1791357137876041620 / 1791357560021369033. Unix epoch로 표시하면07:12:17.876041620~07:19:20.021369033 UTC, 한국 시각16:12:17~16:19:20이다. 표시 변환이며 외부 시계 정확도 검증은 아니다.
- `/joint_states` JointState20,257개, 약47.9841Hz. recorder 간격 중앙19.9965ms/p99 41.8440ms/최대348.7435ms, 긴 간격 후보3개.
- `/imu/data` Imu84,367개, 약199.8506Hz, 긴 간격 후보2개.
- `/joint_goals`와 preview JointState 각각126개, 약50.0291Hz. 목표 recorder 간격 중앙19.9765ms/p99 20.5220ms/최대20.7727ms.
- `/cmd_vel` Twist4,127개, 약10.0003Hz. 기록된 vx/vy/wz는 모두0이다. 동일 값 표본 창412.5889초를 확인했으나 마지막 명령 종료 시점은 관측되지 않아 duration은 null/우측 검열로 남겼다.
- `/xm540_dual_policy/state` String2,071개, 약5Hz. RECOVERY→FAULT의 기록 상태 체류시간은0.398450961초다. 성공 기상 시간이 아니다.
- `/motor_diagnostics` String845개, 약2Hz. 모터 표본16,900개에서 전압11.6~12.0V, 온도33~46°C, 기록 hw_error 최대0, 전류 범위−1814.4~2362.1mA다. 전체 bag 범위이며 과거 정책 구간만 계산한35~45°C와 분석 창이 다르다. 부호가 다른 전류 평균을 소비전력/안전 전류로 쓰지 않는다.
- `/u2d2/behavior_status` String8,442개, `/u2d2/torque_enabled` Bool2개, `/buttons/stop_pressed` Bool1개도 원문과 시각을 보존했다. event topic의 평균 빈도는 설정 주기가 아니다.

독립 GT·odom/자기위치·공 추적·팀 통신 토픽은 없다. bag 이름이 walk여도 실제 전진 속도 실험이라고 부르지 않는다. 이번 수치는 전체 bag의 **recorder 시각** 통계이며 이전 제한 창/header 통계와 섞지 않는다.

### 11월10일 킥 bag

원본: `W/kick_balance/kick_balance_20261110_151804/kick_balance_20261110_151804_0.db3`.

- 메시지4,253개,19.334396895초. recorder 시작/종료 ns:1794291485202669082 / 1794291504537065977. epoch 표시로 한국 시각15:18:05.202669082~15:18:24.537065977이다.
- `/joint_states`876개/약49.4389Hz, `/imu/data_50hz`901개/약50.0160Hz, `/motor_diagnostics`35개/약2Hz, `/robot_state`193개/약10Hz.
- `/cmd_vel`555개는 모두0이며 관측 표본 창19.2125초다. 킥 러너에는 토픽 대신 고정 내부 명령을 쓰는 경로가 있으므로 zero Twist를 로봇 정지·유효 명령 부재의 증거로 해석하지 않는다.
- `/kick_trigger`1개, `/kick_done` topic은 있으나 메시지0개다. NORMAL→PROTECT 이후 관측 종료까지 성공 복구는 확인되지 않았다.
- trigger→첫 바뀐 robot_state(PROTECT) 수신까지1.931258497초, IMU tilt45도 최초 교차→PROTECT 수신까지0.080676536초다. 두 값 모두 recorder event 차이의 **상대 측정 B**이며 실제 기계 반응/물리 낙상 시작 지연이 아니다. 근거 message ID는 분석 JSON에 있다.

`/joint_goals`1,692개 전체는 약87.4648Hz로 나타난다. 이를 보행/킥 정책 주기라고 해석하면 안 된다. 관절 이름 구성으로 나누면12개 다리910행, neck_yaw361행, neck_pitch361행,18개 몸체60행이다. 이들은 publisher ID 증명이 아니라 메시지 구성별 구분이다.

이 topic을 합쳐 header를 보면168회 역전이 있지만 **각 관절 이름 구성 안에서는 역전0회**다. recorder 시각의 역전도0회다. 따라서 물리 시계가168번 뒤로 갔다고 단정하지 않는다. 서로 다른 생성/취득 시각을 가진 mixed stream일 가능성을 분리해야 한다. 원래 순서를 정렬해 지우지 않았고 validator는 이 무구분 header 시계열 사용에 FAIL을 보고한다. decode 오류는0개이며 추출 실패와 다르다.

### shadow·인식 콘솔·영상

- shadow CSV906행/약47.9997Hz, infer 값이 있는859행. 추론 중앙0.29ms/최대6.429ms. 이번 p99는 선형 보간 `(n−1)p` 방식으로2.25238ms다. 과거 summary의2.194ms는 정렬 정수 index 방식으로, 실행 성능이 바뀐 의미가 아니다. CSV time_sec는 상대 시각으로 보존하고 날짜를 붙여 ROS epoch로 합치지 않았다.
- 실물 센서 기반 shadow이지만 출력 remap 시험이다. 기록 명령은 모두0이고 실제 몸체 이동 GT는 없다. q/qd/action/목표/inference/state 등 원문 열을 보존했다.
- `observations_20261104/detections_3d.txt`의 printed rate29.033~30.052Hz를 줄 번호와 함께 추출했다. TF_WAIT/TF_READY와 printed age도 별도 보존했다. console 평균에서 존재하지 않는 개별 메시지 간격이나 공 positive 주기를 복원하지 않았다.
- `kick_video_20261110/metadata.json`과 기존 JPEG 추출물 경로를 확인했다. metadata에는 약30.012fps/291프레임/9.696초다. 이전 Desktop MP4 경로에는 현재 파일이 없으며 원영상이 다른 곳에 있는지는 미확인이다. GT/ROS 동기/좌표 교정이 없어 절대 속도·위치 오차를 계산하지 않았다. 영상을 편집하거나 다시 생성하지 않았다.

## 3. 분류와 계산하지 않은 값

**A 직접 활용:** 실제 recorder/header 값의 기록, 간격·지터·긴 간격 후보, 명령 표본·표본 창, 센서/모터 진단 원문, CSV에 기록된 추론 값. 이것이 자동으로 물리 정확도나 센서 취득 주기를 뜻하지는 않는다.

**B 상대 측정/추정:** IMU orientation/상태, header 의미가 확인되지 않은 시각 관계, trigger/IMU 임계→상태 수신 차이, 소프트웨어 recovery 체류, 콘솔·영상 metadata. 독립 GT 정확도와 구분한다.

**C 코드 설정:** 역할 시험의 vx0.20/vy0.15m/s/wz0.50rad/s, fall manager 임계·timeout 등은 소스 줄/hash로 남겼다. 실행 당시 적용된 실물 파라미터나 안전 범위로 승격하지 않는다.

**D 미측정:** 실제 지면 속도 정확도·절대 위치 오차, 원본 목표→물리 이동 onset/GT 도착, 공의 실제 위치 오차·유효 새 관측 간격, 팀 통신 단방향 지연/지터/손실·상태 발견, 성공 기상/작업 재개 시간은 충분한 기록이 없다. odometry를 GT로 쓰거나 gyro를 지면 속도로 변환하지 않았다. 통신 송신 자료 자체도 없어 전송 지연을 산출하지 않았다.

긴 간격 후보는2.5×해당 topic의 양의 간격 중앙값 기준이다. walk의 joint_states 최대348.7ms, kick의 IMU 최대341.9ms 같은 recorder 공백은 실제 센서 손실·전송 지연·기록 queue의 원인을 단독 확정하지 못한다. source sequence/취득 시각·clock 의미가 없으면 유실 건수를 만들어 내지 않는다.

## 4. 도구와 기록 규격

별도 `src/cah_mappo/measurement/`에 schema/storage/importers/analysis/validation/collection/workflow를 만들었다. 기존 시뮬레이터·ROS2·정책·통신 노드는 수정하지 않았다. CLI와 파일 역할/재현 명령은 [README](PHASE5B0_README_KO.md), 필드·단위·clock·결측은 [기록 규격](PHASE5B0_SCHEMA_KO.md)에 있다.

JSONL은 실험/로봇/trial/조건 참조·반복 번호, 원문 출처, 정수 ns/clock/시각 의미·오차, 명령/추정/GT/목표, 주요 event 시각·성공/실패를 저장한다. 기존 두 bag에는 없는 필드는 null이다. GT CSV와 manual event를 같은 journal에 연결할 수 있으나 독립 GT 교정/clock 정렬 자체를 자동 수행한 것은 아니다.

exclusive 출력/record ID 충돌 거부, 잠금, batch 사전 검증, seal hash/원본 첨부 보존, 잘린 행·중복·시각 역전·동기 결측·단위/frame/GT 출처 검사를 구현했다. 예기치 않은 종료의 부분 행을 숨겨 복구하거나 sealed 데이터를 고쳐 쓰지 않는다. source locator와 hash로 raw부터 결과를 재생성한다.

Passive ROS adapter는 source/센서/상태·카메라/검출 metadata/TF를 구독만 하도록 준비했다. opt-in 전에는 ROS import조차 하지 않는다. custom interface/QoS/부하/실물 종료 동작은 **실제 ROS 환경에서 미실행·미검증**이다. raw image/전체 CDR/packet 수집은 별도 원본 recorder가 필요하고 내부 source publish/use hook·하드웨어 시각 동기 기능을 자동 추가하지 않았다.

## 5. 실제 검증과 보존

두 bag124,617메시지와 shadow906행, 총125,523행을 정규화했다. 원본에서 다시 추출한 manifest/normalized gzip/seal이 동일했고, 같은 normalized 원시 자료로 다시 생성한 분석/검증/명령 구간 결과도 바이트 동일성을 확인했다. gzip의 생성 시각은0으로 고정했다. 임의 표본 몇 개만 비교한 것이 아니다.

기록 validator 결과는 walk PASS, shadow PASS, kick은 header 역전168회 때문에 FAIL이다. 이를 PASS로 바꾸지 않았다. 단위/schema decode 오류는 없고 raw record 전체가 보존됐다. 실제의 이 결함을 탐지했다는 점과 합성 검증의 성공을 구분한다.

최종 실제 테스트는 기존243개+신규42개, **총285개가18.672초에 통과**했다. 실패/오류/skip0이며 `verification_v2/verification.json`과 `unittest.txt`에 실제 출력이 있다. 합성 테스트는 결측/NaN·단위/frame/GT 위장, clock offset/오차/유효구간/음의 지연, sender-only/매칭 없는 통신, EOF 검열·시각 역전, 충돌·봉인·원본 사본 변경·부분 JSONL, GT CSV/수동 callback/QoS 설정·결정적 gzip를 검사한다. 실제 ROS callback을 실행한 시험은 아니다.

원본 bag·CSV·관련 로그/metadata8개와 운영 Word 매뉴얼은 hash를 대조했다. 코드 작성 전의 기존 연구/실물 소스·체크포인트382개도 SHA256이 동일함을 확인했다. 최종 hash·파일 목록·검사 결과는 completion_v2에 기록했다. 보호 Holdout 내용/ID/seed는 열람하지 않았고, 새 패키지·Git 쓰기·원격·실물·학습 실행은 없다.

## 6. Phase5B-1에 필요한 준비

준비 완료 범위는 오프라인 추출/분석·기록 계약·검증·재현 도구다. **실제 로봇 한 대의 이동 측정은 아직 착수 준비 미완료**다.

우선순위는 장비 담당자가 확인한 실제 온도/전압/전류·균형 중단 기준과 안전 담당자→외부 GT/표식/좌표 교정→시각 동기/잔여 오차→현재 설치 모델·실효 명령/gate→비구동 ROS/원본 기록·부하 확인이다. 이 P0를 먼저 별도 승인받은 뒤 직선/정지→회전→목표 도달의 작은 예비 trial로 진행한다.

기존 운영 매뉴얼에는 지지·STOP·토크 해제 순서가 있지만 정확한 모터/배터리 안전 수치의 근거는 확보하지 못했다. 관측 온도·전압·전류를 중단 임계값으로 채우지 않았다. 실제 추가 로봇/GT 장비 가용성도 확인해야 한다. 통신은 정지 endpoint부터, 기상/킥·자연 낙상·4대 경기는 별도 승인이다. 체크 항목은 [Phase5B-1 준비표](PHASE5B1_PREFLIGHT_KO.md)에 있다.

본 보고 뒤 자동으로 실물 측정을 시작하지 않는다.
