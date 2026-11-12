# 2026-11-05 재개 메모

> 문서 형식 선호(2026-11-09): 사용자는 Markdown 표가 깨져 보인다고 지적했다.
> 이후 운영 매뉴얼은 `C:/Users/jes/Desktop/MX3_Match_Sim/Manual`의 Word/PDF로 제공한다.
> 통합본은 `MX3_HSL_로봇운영매뉴얼_20261109.docx`와 같은 이름의 PDF다.
> 긴 표 대신 항목별 설명과 명령어 박스를 사용한다.

> 2026-11-09 최신 재개 안내는 `RESTART_3_TERMINALS_20261109.md`를 따른다.
> 사용자 요청: 시작 위치를 기억하고 변경이 필요하면 먼저 알려준다.
> 기준은 우리 골대에서 상대 골대를 볼 때 우리 골라인 왼쪽 두 T 사이,
> 2026-11-12 사용자 승인으로 사진의 정확한 두 T 중간점으로 변경했다.
> 몸통 중심 바닥 투영점은 골라인 중앙보다 왼쪽 86.75cm, 방향은 상대 골대 방향이다.
> 초기값은 `x=-1.95, y=+0.8675, yaw=0`이며 START 기립 후 입력한다.
> START의 임시 진입 시드는 별개로, 라인 밖 10cm(잔디 안)에 해당한다.
> 이 임시 시드는 hsl_field_20261112 지도 수정/빌드 후 base 재시작이 필요하다.
> 배치 변경 시 새 위치·방향과 이유를 안내하고 초기 위치 스크립트도 함께 맞춘다.

2026-11-04 사용자가 모터 과열 우려로 실험 중단을 요청했다.
로봇에 추가 시험을 요청하지 않는다. 토크 해제/전원 종료 완료 여부는 아직 보고받지 않았다.
안내한 종료 순서는 로봇 지지 → STOP으로 토크 해제 확인 → 실행 노드 종료 및 모터 전원 차단이다.

## 계속 유지할 목표

- 자체 RL sim2real 개선 작업은 중단한 상태다.
- 기준은 MX3_Match_Sim의 mx3_src+hsl_src, B-Human 이식 계층을 실제 로봇에 적용하는 작업이다.
- 한국어로, 한 단계씩 진행하고 완료/다음/남음을 안내한다.
- 로봇은 robo04@192.168.1.161, workspace=/home/robo04/sim2real_ws, Humble/Orin.
- 모든 로봇 터미널은 ROS_DOMAIN_ID=42, ROS_LOCALHOST_ONLY=1.
- 직접 원격 실행은 불가하며 사용자가 MobaXterm에서 실행한다. 비밀번호를 요청하지 않는다.

## 완료된 검증

- mx3_src+hsl_src 13패키지 Humble 빌드, 실제 관절/IMU, D405 컬러/깊이/카메라 정보, 실제 TF, YOLO.
- ball_tracker v3의 TF 전용 스레드 수정 및 Jetson 격리 통합 시험 통과.
  실제 공 위치 rel_x 약 1.086m, rel_y 약 -0.011m, 최근 관측 약 52ms.
  사용자는 전방 약 1.20m라고 대략 설명했고 현재 시험에 충분히 맞는 것으로 수용했다.
- practice 자기위치: 정지 20초 동안 601개, 30Hz, 관측 사용 548/601회.
  x 변동 폭 0.68cm, y 1.50cm, 방향 1.22도, quality=2 전 구간 유지.
  이는 정지 안정성 검증이며 절대 정밀 정확도/보행 중 검증은 아니다.
- 시험 출력으로 분리한 행동 노드: player4/PLAYING/active4/play_ball_align,
  vx=0.2908923, vy=-0.02147965, wz=0.07878143, kick=null, kicks=0.
  모든 publisher는 /hsl/shadow/ 아래이며 실제 모터 출력 연결은 아직 안 했다.

## 내일 배치와 기동

- 실제 연습장 3.9x2.6m, 우리 골라인에서 상대 골대를 보고 왼쪽부터 L T 로봇 T 배치.
  로봇은 두 T 사이에 있다. 현재 승인된 정확한 중간점 초기값=(-1.95,+0.8675,theta=0).
- 마지막 추정 약 (-1.7296,+0.6064,-10.25도)을 내일 실측 정답으로 복사하지 않는다.
- 냉각 후 기존 검증한 하드웨어 기동 → 센서 → leg_odometry → 카메라/RSP/YOLO →
  ball_tracker v3 → self_localizer practice 초기화 순서로, 실제 실행 목록을 보고 중복 없이 시작한다.
- START는 토크 ON+robot_init 및 자기위치 start_event 초기화를 유발한다.
  수동 기립은 /initialpose 입력보다 먼저 끝내고, 그 뒤 실제 위치 초기값을 넣는다.
- YOLO의 NVIDIA PyTorch/cv_bridge에는 /home/robo04/sim2real_ws/.policy-python/numpy1 경로의
  NumPy1.26.4를 사용했다. 전역 NumPy 환경은 바꾸지 않는다.
- 새 세션에서 실제 공/자기위치 수신을 짧게 재확인한 뒤 미완료 단계로 넘어간다.

## 다음 미완료 단계와 확인된 모델 차이

1. 보행 모델 정체/실제 로봇 파일/ONNXRuntime 로딩 확인.
2. 보행 러너의 출력을 시험 토픽에서 확인하고 명령 중재/정지 확인.
3. 하위 제어 연결 후 실제 저속 동작 시험. 킥 통합은 보행 확인 후 진행.

중요: sim_robot.launch.py와 README의 최종 보행은
hsl_bringup/models/walk_v0_6_13500.onnx이다.
실기 robot.launch.py → control.launch.py의 기본 walk_policy.onnx는 README상 v0_8 18400 사본으로 다르다.
로컬 파일과 전달 manifest에서 두 해시가 다름을 확인했다. 실제 기동 전에 선택/전달을 맞춘다.

- v0_6 13500 SHA256: 72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1
- 기본 walk_policy SHA256: 9f6efa34d19cf90382a056f01ad9add5e71596d528cafe4855e04d1f3ba11531
- 최종 킥 v1_5b 22300 SHA256: f71c4db1ecc0db2dd3f40de4a8008c87d88106b1ed89881c30d2722f48c3779f
- 러너 킥 사본 18200 SHA256: 65a768430798004c0f08412f5341eaa6b1e6cb1e5dacb45786d34695674013a5

기본 policy_variant=v0_2는 관측240이다. 같은 차원만으로 가중치 의미까지 확인된 것은 아니다.
과거 RL_walk TensorRT 시험을 이 ONNX 모델의 검증으로 대체하지 않는다.
로봇의 ONNXRuntime 설치/버전/모델 load는 아직 확인 전이다.
오늘 종료 요청 후에는 모델 선택 코드를 수정하거나 새 시험 스크립트를 실행하지 않았다.

## 로그 및 상세 기록

- /home/robo04/sim2real_ws/runtime_logs/ball_tracker_20261104_173126.log
- /home/robo04/sim2real_ws/runtime_logs/hsl_behavior_shadow_20261104_174932.log
- self_localizer 로그는 runtime_logs/self_localizer_<timestamp>.log 형식.
- 상세: isaac_ws/sim2real_ws/docs/MX3_HSL_DEPLOYMENT_PROGRESS.md
- 행동 출력 시험 스크립트: sim2real_evidence/deploy/start_hsl_behavior_shadow_20261104.sh

기존 models/debug_bags/.policy-python 및 모든 백업은 보존한다.
