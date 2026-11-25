# Phase 5A 기존 자료와 측정 가능성 목록

조사일: 2026-11-23, 한국 시간. 이번 단계는 로컬 자료의 읽기 전용 조사와 연구 문서 작성이다. 새 실물 측정·시뮬레이션·학습·패키지 설치·원격 접속·Git 쓰기를 하지 않았다. 보호 Holdout의 내용·ID·시드는 읽지 않았다. Phase 4C 결과나 이전 계획서를 새 결론에 맞춰 고치지 않는다.

## 1. 자료 위치와 판정 규칙

- 저장소 R: `C:/Users/jes/Desktop/MX3_Match_Sim`. 이번 조사 시작 HEAD는 `575bfeb6cdf45732ff8580a48ee5ca155652effa`다. GitHub/실물의 현재 상태를 조회한 값은 아니다.
- 연구 Q: `R/research/cah_mappo/`. 로봇 소스 S: `R/MX3_Match_Sim/src/`.
- 원본 증거 W: `C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/`. R의 `sim2real_evidence/`에는 동기화 사본과 별도 시뮬레이션 결과가 있다. 경로가 다르다고 독립 실험으로 중복 집계하지 않는다.
- Isaac 로컬 사본: `C:/Users/jes/Desktop/isaac_ws/isaac_ws/unitree_rl_lab_kick16/`, R의 `XM_Walking_Policy/`, `XM_Kick_Policy/`. 학습 서버에 접속해 현재 설치·장비·파일 상태를 확인하지 않았다.

모든 항목은 다음 네 종류로 구분한다. **[1] 실제 로봇 기록의 측정값**, **[2] 코드에 설정된 제한·주기·임계값**, **[3] 시뮬레이션 가정 또는 시뮬레이션에서 얻은 값**, **[4] 자료 없음/미측정**. [1]도 독립 Ground Truth(GT)가 없는 수신 시각·센서 추정·콘솔 보고인지 구분한다. 설정 주기, 모델 신뢰도, 성공 신호는 물리 성능의 GT가 아니다.

최신 장비 가용성은 미확인이다. 과거 실물 기록으로 확인된 대상은 robo04 한 대이며 여러 대의 실물 협력 성공 자료는 확보하지 못했다. 과거 robo04는 Jetson Orin/ROS2 Humble, 작업 경로 `/home/robo04/sim2real_ws`였다. 학습 서버의 현재 GPU·CUDA·OS는 이번 조사 범위에서 미확인이다.

## 2. 확보한 실물 자료

**R1 [1] 2026-10-07 정지 명령 중 무너짐 bag.** W의 `walk_20261007_161217/walk_20261007_161217_0.db3`, 57,982,976 bytes. SQLite를 `mode=ro`로 열어 topic/type/count만 확인했다. 메시지120,364개이며 `/joint_states`20,257, `/imu/data`84,367, `/joint_goals`126, `/cmd_vel`4,127, `/motor_diagnostics`845개 등이 있다. 독립 궤적 GT·팀 통신·자기위치·공 추적 토픽은 없다. 이름 walk가 실제 전진 주행을 입증하지 않는다.

기존 `analysis_walk_20261007_161217/analysis.json`의 정책 구간 속도 명령 최대 절댓값은 [0,0,0]이다. 목표 간격 중앙19.976ms/p99 20.357ms, 관절 피드백 중앙20.000ms/p99 41.426ms/최대57.455ms다. 당시 옛 xm540_dual_policy 경로이며 현재 MX3/HSL 이동 모델의 측정 실험이 아니다.

기존 분석의 목표 시점 기준 직전 관절 피드백 나이 중앙5.459ms/p99 25.573ms, IMU 나이 중앙3.370ms/p99 4.121ms는 기록 시각을 이용한 freshness 대용량이다. 물리 센서 취득부터 정책 사용까지 지연 전부가 아니다. header→recorder 차이도 센서·제어 종단 지연으로 쓰지 않는다.

동일 구간 모터 진단 전압11.6~12.0V, 온도35~45°C, 최대 절대 전류2,362.1mA, 기록된 비영 hw_error 없음이 기존 분석에 있다. 이 범위는 당시 표본이며 허용 전압·안전 온도·토크 한계가 아니다.2Hz 진단으로 순간 과부하를 배제하지 못한다.

**R2 [1] 2026-11-10 킥 실패 bag.** W의 `kick_balance/kick_balance_20261110_151804/kick_balance_20261110_151804_0.db3`, 2,043,904 bytes. `/kick_trigger`1, `/kick_done`0, `/robot_state`193, `/imu/data_50hz`901, `/joint_states`876, `/motor_diagnostics`35개다. 분석은 하위 `analysis/summary.json`에 있다. 킥 이후45도 IMU 임계 최초 교차의 recorder 시각1.850581961초, PROTECT 최초 수신1.931258497초, 차이는 약80.68ms다. **이는 두 기록 이벤트의 차이이며 실제 넘어짐 시작→감지 지연의 GT가 아니다.** `/robot_state`는 header가 없는 String이다. 관측 끝까지 성공 복구·작업 재개는 확인되지 않는다.

**R3 [1] 2026-11-05 실제 센서 기반 shadow 추론.** W의 `observations_20261105/walk_shadow_20261105_162155_qeh2uxxq/{summary.json,runner.log,v0_2_log_20261105_162156.csv}`. 실제 관절960/IMU3,952메시지, 목표906개, 약47.9995Hz, 추론 중앙0.29ms/p99 2.194ms/최대6.429ms다. 출력은 시험 토픽으로 remap됐으므로 이 결과는 실제 이동 속도·안정성 측정이 아니다. CSV에는 명령·관측·action·관절 목표·inference_ms·run_state 등이 있지만 독립 몸체 궤적은 없다.

**R4 [1, 콘솔 보고] 2026-11-04 인식·TF 시험.** W의 `observations_20261104/detections_3d.txt`에 Detection3DArray 수신 평균29.033~30.052Hz, 일부 최대 간격 약0.067초가 있다. ball 포함 단일 메시지도 확인했다. 이 수치는 배열 발행/수신 주기이며 **매번 공을 새로 검출한 주기·tracker 유효 갱신률·정확도가 아니다**. `ball_tracker_tf_v3_runtime.txt`에는 source stamp 시점의 TF future extrapolation과 TF_WAIT, detection age 약0.052~0.053초 기록도 있다. 현재 수정본의 실패 빈도나 지연 분포로 일반화하지 않는다.

**R5 [1, 사용자 콘솔] 정지 pose 안정성.** 이전20초 출력601메시지/30Hz, x 범위6.8mm/y15mm/yaw1.22도다. Phase4C `evidence/user_reported_stationary_console.json`은 원시 GT가 없는 콘솔 출처임을 명시한다. 절대 위치 오차·표준편차가 아니다. 공 약1.2m라는 눈대중과 rel_x 약1.09m의 비교도 정확도 측정으로 사용하지 않는다.

**R6 [1, 시각 자료] 킥 영상 추출물.** W의 `kick_video_20261110/metadata.json`에는 원영상 약30.012fps/291프레임/9.696초 메타데이터와 추출 이미지가 있다. 독립 지도 좌표 보정·ROS 시간 동기·배터리/바닥 조건과의 정밀 연결이 없어 속도·검출 지연·균형 원인의 정량 GT로 사용하지 않는다. 프레임을 새로 처리하거나 원영상을 수정하지 않았다.

이 조사에서 로컬 원본 실물 bag은 위2개를 확인했다. 대화에 언급됐던 다른10월7일 bag 전부가 이 PC에 존재한다고 가정하지 않는다. 성공 반복·직선 왕복·GT 동기 촬영·다중 로봇 packet capture 자료는 검토 범위에서 찾지 못했다.

## 3. 항목별 확보 상태

### A. 이동

- A1 명령 선속도/회전 제한 [2]: `S/hsl_src/hsl_bringup/launch/real_role_trial.launch.py:37`의 vx0.20/vy0.15m/s/wz0.50rad/s. 일반 `config/hsl.yaml`은0.40/0.25/1.0이며 경로별 override가 다르다. 실제 로봇 런타임 값은 이번에 조회하지 않았다.
- A2 실제 선속도·반복 편차·횡방향 drift [4]: R1은 정지 명령, R3은 shadow다. `/cmd_vel`·관절 회전속도·다리 오도메트리를 몸체 GT 속도로 간주하지 않는다.
- A3 가속·감속 [2/4를 분리]: `skills.py`의 accel_lin0.6/accel_ang2.5는 명령 shaping 설정이다. 정지 입력에서는 즉시 zero를 반환하고 작은 출발 명령에는 min_norm0.12 보정을 한다. 실제 가감속 분포 [4].
- A4 회전 속도·회전/정지 지연 [4]: 제한값A1은 확보했지만 실제 yaw 궤적 GT가 없다.
- A5 목표 도달시간·도달 오차·반복 경로 [4]: 사용자 관찰상 이동 성공은 정량 궤적 실험을 대체하지 않는다. WalkToPose 판정 설정은 E3이다.

### B. 인식·위치추정

- B1 검출 배열 수신 주기 [1]: R4의 약29~30Hz 짧은 콘솔. 공 positive 새 관측·유효 추적 갱신 분포 [4].
- B2 tracker 주기/신선도 판정 [2]: `hsl_world_model/src/ball_tracker_node.cpp` 기본50Hz, ball_timeout6초. 킥 핸드오프의 max_ball_age0.4초는 `skills.py` 설정.50Hz 반복 예측 발행을50Hz 새 관측으로 세지 않는다.
- B3 공 위치 오차·동적 tracking 오차 [4]: 독립 공 중심 GT와 동기 궤적이 없다. BallEstimate covariance/valid/cam_valid는 [2] 알고리즘 출력 의미이며 정확도 보증이 아니다.
- B4 자기위치 변동 [1]: R5. 절대 위치·방향 오차, 주행 중 오차·재초기화 편차 [4]. localizer30Hz/노이즈 설정은 [2].
- B5 관측 지연 [1, 부분]/[4]: R4 TF_WAIT의 시각 차이는 과거 특정 pipeline 진단이다. 노출→검출→TF→tracker→Actor 소비의 전체 지연은 [4].
- B6 추정 실패·신뢰도 저하 [1, 사례]/[4]: 과거 TF 실패는 존재한다. GT 가시성 대비 미검출률, validity의 보정된 정확도, 점프/복구 확률 분포는 [4].
- B7 입력 freshness [1, 대용]/[4]: R1 직전 표본 나이는 확보했다. 현재 보행/상위 정책이 실제 소비한 image/depth/pose/IMU/관절 각각의 source/receive/use 시각을 연결한 기록은 [4]. YOLO는 컬러 header를 검출에 복사하며 latest depth를 캐시한다. 그 depth stamp 캐시는 있으나 동기 오차를 제한하는 실행 계약으로 확인되지 않아 두 입력 시각을 따로 기록해야 한다.

### C. 팀 통신

- C1 실제 송신·수신 간격/지터/손실 [4]. R1/R2에는 팀 통신 원시 기록이 없다. `hsl_team_comm/team_comm_node.py` 기본 송신2Hz/사건 token0.25Hz/burst3, 수신 polling20ms/발행100ms/사건 최소 간격100ms는 [2].
- C2 단방향 지연·왕복 지연·상태 발견 지연 [4]. 명목 송신 간격0.5초와 전송 지연을 혼동하지 않는다.
- C3 유효기간 [2]: teammate_timeout2초. 이는 마지막 **수신** 시각 기준이고 원래 pose 취득 시각 기준이 아니다. `behavior_node.py::_mates()`는 현재 팀 리스트를 전략 Agent로 변환하며 각 관측의 원래 시각을 별도 입력으로 넘기지 않는다.
- C4 packet 식별성 [2]: 형식 `<4sBBBBB4f5ffH`에16비트 seq는 있으나 sender timestamp 없음. 수신 `_seq`를 Teammate.msg에 저장하지 않는다. 현재 `/hsl/team_state`만으로 유실·중복·순서 역전을 재구성할 수 없다. capture/sidecar 계측이 필요하다.
- C5 상태 나이 [2/4 분리]: Teammate.received와 ball_age 필드는 존재하지만 pose source age·전송 중 age를 분리한 실측은 [4]. 수신 루프의 now는 한 번 잡혀 여러 queued packet에 쓰이므로 실제 개별 도착 시각과 같다고 가정하지 않는다.

### D. 고장·복구

- D1 IMU 임계→PROTECT 기록 차이 [1, 부분]: R2 약80.68ms. 실제 물리 낙상 시작→감지 지연 [4].
- D2 복구 행동 시작 지연 [4]: 옛 R1 recovery 진입→FAULT 약0.398초는 실패 상태 전이이며 현재 FSM의 정상 기상 시작 성능이 아니다.
- D3 성공 기상 시간·성공률 [4]: 성공 반복 자료가 없다. fall_manager의45도 감지/20도 회복/settle1초/PROTECT 최소0.4초/standup timeout10초/재시도3/사후 안정2초/50Hz는 [2].
- D4 복구 후 실제 작업 재개 지연 [4]: NORMAL 신호나 torque ON만으로 실제 서비스 가능성을 보증하지 않는다.
- D5 건강 상태 [1]: R1/R2 전류·온도·전압·hw_error 표본. 허용 운전 범위·안전 중단 수치는 별도 장비 확인이 필요하며 [4]로 남긴다. 전류를 지면 마찰계수·지지력·몸체 속도로 직접 역산하지 않는다.

### E. 실행 제어

- E1 역할 결정→목표/속도 전달 지연 [4]. `/hsl/strategy_state`는 header/role/skill/active_player를 가지지만 목표 ID와 결정 시작·소비 시각이 없다. `/hsl/behavior_debug`는 약0.2초 간격/좌표 반올림/String이라 밀리초 계측이나 명시적 arrived 이벤트를 대체하지 못한다.
- E2 명령→실제 이동 시작 지연 [4]. 현재 흐름은 cmd_vel_behavior/teleop→mux→cmd_vel_raw→선택적 drift correction→cmd_vel→보행 러너→joint_goals→모터다. mux·킥/낙상 gate·통신·기계 반응을 한 값에 합치지 않는다.
- E3 목표 도달 판정 [2]: `skills.py::WalkToPose`의 거리<0.15m AND 방향 오차<15도에서 arrived 래치. 이후0.35m/35도를 넘으면 해제. `GoToBallAndKick` 내부 walker는0.04m/8도의 별도 값이다. 이를 실측 위치 정확도로 쓰지 않는다.
- E4 실제 성공/실패 [4]: `/kick_done`·timeout·FSM은 소프트웨어 상태다. 공이 목표로 이동했는지, GT로 목표에 도달했는지, 복구 후 걸을 수 있는지와 독립적으로 검사해야 한다.
- E5 정책·관절 시각 [2]: walking runner의 `publish_joint_goals()`는 최신 joint_states.header.stamp를 목표에 복사한다. IMU parser header도 호스트 처리 시각이다. message header를 그대로 “생성/센서 취득 시각”으로 해석하지 않는다. 별도 provenance가 필요하다.

## 4. 시뮬레이션·학습 자산 [3]

Phase2의 Config/RobotSpec/JobSpec/EventSpec, capacity1 중재, Phase3A 관측 캐시·지연·slot/mask·Actor/Critic 분리, Phase4B의 제어/물리 시계 분리와 위협 도달 GT 평가를 재사용할 수 있다. `phase2/{engine,model,policies,metrics}.py`, `phase3a/{perception,contracts}.py`, `phase4b/{engine,model,metrics,planning,scenarios}.py`가 연결점이다. 지금은 아무것도 수정하지 않는다.

Fixed는 초기 guard/지정 worker의 FIFO, Greedy는 중요도→ETA, Exact는 ETA+미할당 비용+전환 비용의 전수 유효 할당, RuleBasedHandover는 준비/마감/위험 gate다. Exact는 Hungarian이나 완전 시간 최적 계획기가 아니다. 현재 센서146/10 로컬 Fixed/ETA만 있으며 동일 조건의 로컬 Exact/인계는 아직 별도 어댑터가 필요하다. 중앙 버전이 이미 로컬로 존재한다고 주장하지 않는다.

기존 guard1m/s, worker0.45~0.55m/s, 회전πrad/s, 반경0.25m, 서비스2/12초, 응답 예산2초, 사건32초, 제어1초는 [3]. Phase4B 기본 물리 dt0.25초, Phase4C 제안 dt0.125초도 실물 주기 측정값이 아니다. Phase3A 클래스 기본 센서5m/잡음0.02/통신2초/cache4초와 Phase3B-2B 실행의100m/잡음0/통신0 설정을 구분한다.

선택된 Standard MAPPO는 Actor146→64→64→10/Critic141→64→64→1/Tanh/공유 Actor의2D 할당 모델이다. 실물 보행·킥 PPO ONNX와 다른 모델이다. 실물 상위 HSL에 MAPPO가 배포됐다는 증거는 없다. 선택 근거/전체 hash는 `phase4b_outputs/study_v1/design/freeze.json`, 학습·보상 설정은 `phase3b2b_outputs/main_v1/frozen_plan.json`이다.

- seed101: `phase3b2b_outputs/main_v1/seed_101/checkpoints/step073728.pt` / SHA256 시작 `bbd847ac5c57e965`.
- seed211: `phase3b2b_outputs/main_v1/seed_211/checkpoints/step090112.pt` / `2ed4dfd9e9bc97f6`.
- seed307: `phase3b2b_outputs/main_v1/seed_307/checkpoints/step100352.pt` / `75fd474d6da0627e`.
- seed401: `phase3b2b_outputs/main_v1/seed_401/checkpoints/step100352.pt` / `8b2f342b59291d37`.
- seed503: `phase3b2b_outputs/main_v1/seed_503/checkpoints/step100352.pt` / `3d7226550e9dec1e`.

실물 보행 패키지 deploy의vx[-0.5,0.7]/vy[-0.5,0.5]/wz[-1.5,1.5]와 Isaac dt0.005×decimation4, 마찰 설정은 [3] 학습 계약이다. 현재 로봇 능력이 아니다. Isaac MX3 walk_kick_v1_5/v1_6 코드, MuJoCo scene 기반의 `kick_robust_20261116/results_*` 및 새 v16h gate summary도 [3]이다. 파일명·전압/마찰 sweep·fall 비율을 실물 검증으로 분류하지 않는다. 과거 배포 문서는 오래됐고 현재 로봇의 ONNX/launch override/러너 variant는 Phase5B 전에 확인해야 한다.

## 5. 평가·실패 원인·Holdout 자산

기존 JSONL은 원시 GT 구간/사건·정책 입력·요청/실행 행동·일부 Actor 확률을 보존한다. 기존 작업 성공/응답/완료/전환/중복과 legacy CGR, 소유권/응답 가능성/위협 가중/작업 후 공백을 재사용하되 물리 정의를 다시 대응시켜야 한다. 가중/무가중 분자·분모, 무응답·검열·위협 없음(null)을 유지한다.

Phase4C의2,880 정책-에피소드 중 임무 실패2,308, A 물리 하한384/C 동일 정보 선택 반례752/전체 원인 미확정1,172였다. 부분 원인1,608을 여기에 합산하지 않는다. 정보 이력108쌍과 Actor120쌍, Native reset seed 보완, 원인 미확정7세계는 `PHASE4C_RESULTS_KO.md`와 `analysis_final/`에 보존돼 있다.97/120 세계의 온라인 성공 증인과16개 불가능성 근거도 독립 반복의 실물 성능이 아니다.

기존 보호 Holdout400개는 봉인 상태를 유지한다. 관리 규칙만 기존 보고서에서 읽었으며 내부 내용/ID/seed/상태는 읽지 않았다. 실측 분포가 달라지면 대표성이 자동 보장되지 않는다. 기존400개를 폐기·이름 변경·새 최종 데이터로 재포장하지 않는다. 새 final set은 보정·정책·평가 규칙을 고정한 뒤 별도로 준비한다.

Phase4C의243개 테스트/5,022 raw 감사는 **이전 단계의 실행 결과**다. Phase5A에서는 시뮬레이터·학습기·실물 노드·회귀 테스트를 실행하지 않았다. 읽기 전용 자료 분석과 문서 검사를 수행했으며 테스트 통과를 새로 주장하지 않는다. 작성 전후 기존 연구/실물 소스·5개 checkpoint를 포함한377개 파일의 SHA256이 동일함을 확인했다. 별도로5개 checkpoint의 기존 freeze hash, Phase4C 산출물21개의 기존 hash와 원본 실물 bag2개의 기존 hash도 일치했다. 새 문서5개의 UTF-8/상대 링크9개/표 미사용을 검사했다. 이 범위를 전체 저장소나 모든 원시 로그의 재검증으로 과장하지 않는다.

## 6. 아직 필요한 자료

우선순위는 독립 GT와 공통 시각→한 대의 실제 직선/회전/목표 도달→인식/자기위치 오차→두 endpoint의 송수신·상태 발견→안전한 복구·작업 재개다. 반복당 바닥·배터리·모터 온도·모델 hash·러너/launch/mux·시각 오차·중단 이유를 같이 남겨야 보정 자료로 사용할 수 있다. 구체적인 절차와 필요한 향후 승인 범위는 [측정 프로토콜](PHASE5A_MEASUREMENT_PROTOCOL_KO.md)에 있다.
