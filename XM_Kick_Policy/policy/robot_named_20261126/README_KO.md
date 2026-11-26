# 로봇에 이름 붙여 설치한 킥 정책 (2026-11-26)

2026-11-25 묶음 (`../robot_named_20261125`) 에 `kick_push_31800.onnx` 를 더했다. robo04 `hsl_bringup/models` 에 같은 파일명으로 설치하고, `real_role_trial.launch.py` 의 `kicking` 줄 하나로 고른다.

- `kick_power_22300.onnx`: V1-5B 22300, 원래 정책이다.
- `kick_stable_27500.onnx`: V1-6H long model_27500 이다. MuJoCo 일반 게이트 넘어짐이 가장 적다.
- `kick_accurate_30899.onnx`: V1-6I2 long model_final (30899) 이다. 방향 오차가 22300 수준이다.
- `kick_push_31800.onnx`: V1-6L3 model_31800 이다. K5 31599 에서 디딤발 쪽 외력 0~8 N·몸 roll 벌점 −10 으로 200 iteration 더 학습했다.
  - MuJoCo 일반 게이트 real_like 넘어짐 3 % (30899 는 0.3~1.7 %) 다.
  - 디딤발 쪽 외력 8 N 넘어짐은 26 % (30899 는 54 %) 다.
  - 차는 다리 골반 roll 벌림은 44° (30899 는 61°) 다.
  - 원본 체크포인트는 `isaac_ws/sim2real_evidence/kick_robust_20261116/candidate_v16l3_31800/` 에 있다.

네 파일 모두 입력 265, 출력 12 다. 해시는 `SHA256SUMS` 에 있다.

모델 바꾸기: start.sh 를 끄고 로봇에서 `bash install_named_kicks.sh kick_accurate_30899` 처럼 실행한다. 이름을 주지 않으면 kick_push_31800 이다. 스크립트는 다음을 순서대로 한다.
- 실행 중 러너가 있는지 확인한다.
- 해시를 검증하고, 기존 세 파일이 로봇 설치본과 같은지 확인한다.
- launch 파일과 기존 모델을 백업한다.
- `kicking` 줄을 바꾼다.
- hsl_bringup 을 빌드한다.
