# MX3 · HSL 실물 로봇 축구 작업 저장소

MX3 로봇의 제공된 강화학습 보행·킥 정책과 HSL 상위 기능을 연결해 실제 로봇에서 역할별 축구 동작을 확인하는 프로젝트입니다. 최종 목표는 4대4 모의 경기이며, 현재 실물 시험은 robo04 한 대를 기준으로 진행했습니다.

## 내려받기

```bash
git clone https://github.com/hion89/MX3-HSL-Robot-Soccer.git
cd MX3-HSL-Robot-Soccer
```

비공개 저장소이므로 접근 권한이 있는 GitHub 계정으로 인증해야 합니다. GitHub의 Code → Download ZIP으로도 받을 수 있습니다. 이후 갱신은 작업 파일을 먼저 커밋하거나 보관한 뒤 `git pull --ff-only`로 받습니다.

## 들어 있는 것

- `MX3_Match_Sim/src/mx3_src`: 실물 하드웨어, 정책 러너, 하위 제어와 bringup.
- `MX3_Match_Sim/src/hsl_src`: 자기위치 추정, 공 추적, 행동·역할, 팀 통신. B-Human 방식과의 대응은 아래 감사 기록을 참고합니다.
- `MX3_Match_Sim/src/humanoid_sim`: 기존 MuJoCo 경기 시뮬레이터.
- `XM_Walking_Policy`, `XM_Kick_Policy`, `XM_Getup_Policy`: 제공된 정책, 체크포인트, 학습 설정과 overlay.
- `sim2real_evidence/deploy`: 기존 로봇에 적용하는 배포 묶음과 수정 스크립트.
- `sim2real_evidence/patches`: 배포 묶음의 펼쳐진 소스. 초기 위치 최신 파일도 여기에 있습니다.
- `sim2real_evidence`: 실패 분석, 검증 코드, RViz/지도 수정, 마찰·지연 재학습 계획과 과거 진행 자료.
- `Manual`: Word/PDF 운영 매뉴얼. 작성일 이후 변경은 최신 상태 문서가 우선합니다.
- `docs`: 작업 진행 기록과 현재 상태.

최신 상태는 [CURRENT_STATUS_KO.md](docs/CURRENT_STATUS_KO.md), 전체 이력은 [진행 기록](docs/MX3_HSL_DEPLOYMENT_PROGRESS.md)을 확인하세요. B-Human 이식 범위는 [검토 기록](sim2real_evidence/manuals/BHUMAN_PORT_AUDIT_KO_20261109.md), 재학습 계획은 [RETRAINING_PLAN_KO.md](sim2real_evidence/policy_package_audit_20261112/RETRAINING_PLAN_KO.md)에 있습니다.

내려받은 뒤 `python scripts/verify_checkout.py`로 선택된 정책 해시·연습장·초기 위치 파일을 확인할 수 있습니다. Python 표준 라이브러리만 사용하며 모터 명령을 보내지 않습니다.

## 실물과 시뮬레이션 실행 구분

실물은 ROS 2 Humble/Jetson Orin의 `/home/robo04/sim2real_ws`를 기준으로 통합했습니다. 기존 `incoming/real_robot_20261109/base.sh`, `start.sh`, `role.sh`, `init_pose.sh`로 운영합니다. 복제한 저장소를 기존 로봇의 src 위에 무조건 덮어쓰지 말고 배포 묶음의 백업·검증 절차를 사용하세요.

원래 시뮬레이터의 [README](MX3_Match_Sim/README.md)와 setup.sh는 Jazzy 환경을 대상으로 작성됐습니다. 이를 Humble 실물 설치 절차로 사용하면 안 됩니다. 기존 MuJoCo 물리 경기장도 이번 practice 지도 수정과 별개이며 자동으로 연습장 규격으로 바뀌지 않았습니다.

가상환경, 빌드 결과, 브라우저 프로필, 비밀정보, 원본 영상과 ROS bag은 제외했습니다. 분석 결과와 재현용 소스, 선택된 모델 및 제공된 학습 자료는 포함합니다. 기존 작업 기록의 Windows 절대 경로는 당시 환경의 기록이며, 일부 과거 분석·묶음 생성 도구는 다른 PC에서 경로 조정이 필요합니다.

배포 manifest의 SHA256 검증이 운영체제에 따른 줄바꿈 변환으로 깨지지 않도록 Git은 파일의 원래 바이트를 보존합니다. Linux 실행용 셸 파일을 편집할 때는 LF 줄바꿈을 유지하세요.

## 이후 작업 반영

사용자 요청에 따라 관련 변경과 검증이 끝날 때마다 GitHub에 커밋하고 푸시합니다. 실물용 소스는 이 폴더 안에서 직접 관리합니다. 별도 isaac_ws에서 만든 증거·배포 자료는 다음 명령으로 가져옵니다.

```powershell
python scripts/sync_working_copy.py --workspace "C:\Users\jes\Desktop\isaac_ws"
git status --short
```

그 뒤 이번 작업 파일을 검토해 선택적으로 스테이징하고 커밋·푸시합니다. sync 도구는 파일 복사만 하며 기존 파일을 삭제하거나 GitHub로 자동 업로드하지 않습니다.

## 출처와 이용 조건

기준 자료는 사용자가 제공한 MX3_Match_Sim 및 정책 패키지입니다. B-Human의 원본 저장소는 https://github.com/bhuman/BHumanCodeRelease 입니다. 가져온 방식·재구현·실물 통합 변경을 감사 문서에서 구분했습니다. 기존 파일에 포함된 저작권·라이선스 표시는 유지하며, 이 저장소 전체에 임의의 새 라이선스를 부여하지 않습니다.
