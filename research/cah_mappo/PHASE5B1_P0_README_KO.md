# Phase 5B-1 P0 준비 도구

이 디렉터리의 새 도구는 읽기 전용 오프라인 검사다. ROS import/launch/하드웨어 접속 기능이 없다. 기존 measurement-v1, 수집기, Phase1~5B0 구현을 수정하지 않았다. 현장 센서 실행과 원본 recorder 실행은 별도 승인·검토 대상이며 이 CLI가 시작하지 않는다.

## 구성

- `src/cah_mappo/measurement_p0/audit.py`: 필수 스트림 수·type, raw header/좌표계·단위, nested TF, 시각 순서, 관절 이름 서명별 시각 감사, P1 증거 점검, 조건부 저장량 계산.
- `src/cah_mappo/measurement_p0/static_review.py`: 선택된 실제 source 17개의 AST/줄 번호/sha256과 의존 관계. ROS 코드를 import하지 않는다. 외부 드라이버·펌웨어·실제 설치본의 안전 인증기가 아니다.
- `p0_profiles.json`: 오프라인 기대 스트림 프로파일. launch 설정이 아니며 이름의 candidate는 구동 승인을 뜻하지 않는다.
- `phase5b1_p0.py`: 원본 재추출·기존 분석 재실행·전체 테스트·P0 감사를 새 결과 경로에 저장한다.
- `tests/test_phase5b1_p0.py`: 신규 합성 검사 31개. 실측값이나 장비 안전성 증거가 아니다.
- `phase5b1_p0_outputs/study_v1`: 실제 오프라인 실행 결과. 기존 출력과 분리했다.

## 재현 명령: 로컬 PowerShell

아래는 로컬 오프라인 명령이다. 원격 접속·로봇 프로세스 실행이 없다. 기존 설치된 분석 환경과 연구 .venv를 사용하고 패키지는 설치하지 않는다.

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim\research\cah_mappo
& C:\Users\jes\Desktop\isaac_ws\sim2real_evidence\.analysis_env\Scripts\python.exe -B phase5b1_p0.py `
  --study phase5b0_outputs/study_v2 `
  --output phase5b1_p0_outputs/study_v2 `
  --test-python .venv/Scripts/python.exe
```

study_v1은 이미 생성됐다. 위 study_v2도 이미 존재하면 새 이름을 사용한다. 기존 경로는 FileExistsError로 거부한다. 실제 실행은 study_v1에서 완료했다. 디스크 free 값은 실행 시마다 달라질 수 있는 PC 스냅샷으로, 재현성 비교 대상의 통계와 분리한다.

새 합성 테스트만 실행:

```powershell
& .venv\Scripts\python.exe -B -m unittest discover -s tests -p test_phase5b1_p0.py -v
```

## 읽을 결과

`completion.json`과 `replay/verification.json`은 재추출·회귀 시험 결과다. `kick_timestamp_audit.json`은 역전 168회의 원시 record locator를 포함한다. `bag_catalogs.json`은 실제 bag 토픽과 과거 offered QoS다. `source_review.json`은 검토한 source 및 줄 번호다.

`*_stream_audit.json`은 시각을 혼합 토픽과 개별 서명으로 구분한 보완 감사다. 기존 `replay/kick_analysis/validation.json`의 FAIL/168개 역전 오류를 변경하지 않았다. 새 감사 PASS는 입력의 알려진 혼합을 설명하는 것이며 시계 동기화 PASS가 아니다.

`expected_missing_stream_demo.json`은 kick bag에 없는 /imu/data를 요구했을 때 의도대로 ERROR가 나는 사례다. `onsite_evidence_UNFILLED.json`은 현장 증거의 빈 양식이고 `readiness.json`은 P1 BLOCKED다. 체크리스트는 내용의 진위를 자동 검증하지 않으며 값과 서명 입력만으로 실물 실행이 허가되지 않는다.

현장 절차는 PHASE5B1_P0_OPERATOR_PROCEDURE_KO.md, GT는 PHASE5B1_P0_GT_REQUIREMENTS_KO.md, 필드·QoS는 PHASE5B1_P0_TOPIC_FIELDS_KO.md를 본다.
