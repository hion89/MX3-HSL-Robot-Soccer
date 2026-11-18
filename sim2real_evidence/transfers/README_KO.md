# 22300 정책을 학습 서버 로그 경로로 복사 · 2026-11-18

요청한 서버 경로는 `/home/bami122/isaac_ws/unitree_rl_lab_kick16/logs/rsl_rl/unitree_mx3_walkkick_v1_6c`다. 기존 로그와 충돌하지 않도록 그 아래 `import_v1_5b_22300_20261118`에 보관한다.

현재 상태: 로컬 원본과 전송 묶음/설치 도구를 준비하고 검증했다. 서버 `bami122@163.180.145.122`의 비대화형 SSH 인증이 거부되어 **실제 전송은 아직 하지 못했다**. 사용자가 다음 명령을 PC PowerShell에서 실행해 비밀번호를 직접 입력해야 한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\jes\Desktop\isaac_ws\sim2real_evidence\transfers\send_kick22300_20261118.ps1"
```

다른 PC에서는 이 저장소의 `sim2real_evidence/transfers/send_kick22300_20261118.ps1` 경로를 사용한다. 스크립트는 자신의 위치를 기준으로 payload를 찾는다.

복사되는 파일:

- `model_22300.pt`: 학습 actor/critic 체크포인트.
- `walkkick_v1_5b_22300.onnx`: 로봇에서 선택된 ONNX와 같은 해시의 모델.
- `policy.pt`: 제공된 TorchScript 내보내기.
- `deploy.yaml`: 원본 배포 계약.
- `params/env.yaml`, `params/agent.yaml`: 원본 학습 설정 기록. v1_6c 설정으로 바꾼 것이 아니다.
- `manifest.json`: 파일별 SHA256과 출처.

서버의 성공 출력은 `IMPORTED_AND_HASH_VERIFIED`이며, 재실행 시 내용이 동일하면 `ALREADY_IMPORTED`다. 이미 같은 이름의 다른 파일이 있으면 덮어쓰지 않고 중단한다. 원본 로컬 파일은 보존한다.

학습을 시작하거나 v1_6c 학습 가중치로 변환하는 작업은 아니다. 체크포인트를 새 태스크에서 재개하는 방법과 설정 호환성은 학습 전에 따로 확인해야 한다. 로봇 정책/토크/러너 설정은 바꾸지 않는다.
