# 로봇에 이름 붙여 설치한 킥 정책 (2026-11-25)

robo04 `hsl_bringup/models`에 같은 파일명으로 설치했다. 선택은 `real_role_trial.launch.py`의 `kicking` 줄 하나로 한다.

- `kick_power_22300.onnx`: V1-5B 22300, 원래 정책이다. `../policy.onnx`와 같다.
- `kick_stable_27500.onnx`: V1-6H long model_27500이다. MuJoCo 넘어짐이 가장 적다.
- `kick_accurate_30899.onnx`: V1-6I2 long model_final(30899)이다. 방향 오차가 22300 수준이다.

세 파일 모두 입력 265, 출력 12다. 해시는 `SHA256SUMS`에 있다.

모델 바꾸기: start.sh를 끄고 로봇에서 `bash install_named_kicks.sh kick_stable_27500`처럼 실행한다. 스크립트는 다음을 순서대로 한다.
- 실행 중 러너가 있는지 확인한다.
- 해시를 검증한다.
- launch 파일을 백업한다.
- `kicking` 줄을 바꾼다.
- hsl_bringup을 빌드한다.

실물 시험 결과(2026-11-25): 세 모델 모두 인조잔디에서 찬 뒤 디딤발이 바깥 모서리로 들려 넘어지는 일이 많았다. `../../KICK_V1_6_KO.md` 실물 진단 절을 본다.
