# Phase 4C 실행 및 결과 안내

이 단계는 기존 시뮬레이터·동결 MAPPO의 **원인 진단과 연구 방향 결정**이다. 새 알고리즘 학습, 실물 제어, 환경 설치를 하지 않는다. 결과부터 보려면 `PHASE4C_RESULTS_KO.md`, 물리 근거는 `ROBOT_PARAMETER_EVIDENCE_KO.md`, 최종 방향과 이후 범위는 `RESEARCH_DIRECTION_DECISION_KO.md`를 읽는다. 표 대신 항목과 서술로 정리했다.

## 생성 파일과 역할

- `PHASE4C_PROTOCOL_KO.md`: 결과 확인 전 원인 분류·최소 실험 규약.
- 위 결과/근거/방향 문서와 이 README: 해석 범위·재현 방법·새 파일 설명.
- `phase4c.py`: protect/evidence/rules/plans/information/sensitivity/classify/audit 개별 진입점. 기존 출력 위치에 다시 실행하면 덮어쓰지 않고 실패한다.
- `phase4c_reproduce.py`: 새 출력 경로를 필수로 받는 전체 재현 진입점. 기존 전용 `.venv`와 체크포인트만 쓴다.
- `phase4c_commitment.py`: 첫 진단 규칙을 보존한 채 작업 담당자 간섭 한 조건만 수정한 비교 실행.
- `phase4c_reset_audit.py`: 기존 reset seed 차이를 제거한108쌍 재실행. `CommonResetNative`는 기존 코드를 수정하지 않고 실제 초기화 입력을7로 고정한다.
- `phase4c_service_evidence.py`: 공 timer와 킥 점유/timeout 소스의 추가 읽기 전용 근거.
- `phase4c_finalize.py`: 마지막 보존 검사, 지도 범위 검사, CGR 가중/무가중 분자·분모, CSV와 코드/문서 해시 저장.
- `src/cah_mappo/phase4c/__init__.py`: 독립 진단 패키지.
- `experiment.py`: 기존 자료 보존 snapshot과 각 진단 작업 연결.
- `evidence.py`: `collect()`가 원문 줄·SHA256·로그 값을 수집하고 `validate_item()`이 설정값의 실측 오표기와 미측정 숫자 채우기를 거부한다.
- `policies.py`: 공개 입력만 쓰는 `RiskDirectedSwap`과 담당자 보호 변형, 합성 `DelayedExternalInformation`, 특권 정보 실험용 `BranchHintPolicy`,146/10 로컬 규칙의 one-hot `LocalDistribution`. 새 신경망이 아니다.
- `runs.py`: `execute()`가 기존 Simulation/FeedbackSimulation을 호출하고 기존 지표를 계산한다. `TeamSchedule`은 제한된 미래 인지 스케줄 증인, `run_rules/run_plans/run_sensitivity`는 공통 세계 대조 실행이다.
- `information.py`: `decision_prefix/compare_histories`로 실제 입력 이력을 비교하고 동결 Actor와 정보 제공 진단을 실행한다.
- `classification.py`: `mission_failed/diagnose`로 성공 증인·불가능 근거·부분 원인·미확정을 구분한다. `nonempty_witness_ids`는 비어 있는 lookup을 성공으로 세지 않는다. `resolution_comparison`은 동일 세계의 dt 차이를 계산한다.
- `evaluation.py`: `failure_service_recovery`는 raw GT의 지점 서비스 재개/검열, `coverage_denominators`는 기존 CGR 가중치와 무가중 시간을 병기한다. 보상과 분리돼 있다.
- `audit.py`: 모든 새 JSONL의 기존 지표 재계산, Actor 표본 검사, 보존 검증, 실제 전체 unittest 실행.
- `tests/test_phase4c.py`: 설정/실측 구분, 보수적 원인 귀속, 기존 규칙의 실패 보존, 보완 규칙6세계, 정보 이력/마스크, 합성 지연, dt, 서비스 복구/검열, 가중 분모와 seed 초기화 검사.

기존 Phase1~4B 소스·보상·학습기·체크포인트는 수정하지 않았다. Git 기록과 실물 배포도 수행하지 않았다.

## 결과 위치

`phase4c_outputs/study_v1/`의 파일은 모두 이번 작업의 별도 출력이다.

- `preservation_before.json`: 기존 소스219개/출력29,786개/checkpoint5개 보존 기준. 보호 데이터 내용은 읽지 않았다.
- `evidence/`: 설정과 실측·미측정을 구분한29항목+3항목 보충, 읽은 소스 해시, SQLite topic 메타데이터와 콘솔 보고 출처.
- `rules/`:4개 방법의 전120세계 실행/JSONL/계열 요약. 첫 실패 규칙도 남겼다.
- `plans/`:30개4대 세계×제한 스케줄의2,430 raw와 개별 설정/성공 증인. 최적해가 아니다.
- `information/`:108개 기존 이력 비교,5모델120쌍 공통 난수, 미래 정보 제공, Native108쌍 공통 reset. `native_reset_results.json`의 seed API 주의사항을 함께 읽는다.
- `sensitivity/`, `commitment_sensitivity/`: 한 조건씩 바꾼 파라미터와 고정 제어 시계의 물리 dt 비교.
- `analysis_final/`: 최종 원인 집계·사례·전 방법 요약·시간 차이. 중간 `analysis/`의 비어 있는 증인 목록 오류는 이 버전에서 바로잡았다. 중간 파일을 최종 근거로 쓰지 않는다.
- `completion_final/`: 최종 실제 테스트 출력,5,022 raw 감사, 서비스 복구·가중 분모·지도 범위·계열 CSV·최종 해시. `completion/`은 초기 감사이며 이후 seed 검사 등을 추가한 최종 수와 다르다.

## 실행 환경과 재현 명령

이번에 사용한 기존 전용 환경은 Python3.12.14/PyTorch2.7.0+cpu/NumPy1.26.4다. 패키지를 설치하거나 로봇/Isaac 환경을 변경하지 않았다. 초기 원인 비교와 정보 진단은 CPU에서 실행했다.

테스트만 다시 실행:

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim\research\cah_mappo
.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

전체 새 출력 재현:

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim
$phase4cOutput = "research/cah_mappo/phase4c_outputs/replay_$(Get-Date -Format yyyyMMdd_HHmmss)"
research/cah_mappo/.venv/Scripts/python.exe -B research/cah_mappo/phase4c_reproduce.py --output $phase4cOutput
```

출력이 이미 있으면 재현기는 거부한다. 기존 `study_v1`에 덮어쓰지 않는다. 실행에는 기존5개 선택 checkpoint와 Phase4B 개발 manifest/원시 로그가 필요하다. 보호 Holdout은 필요하지 않다. 재현기는 학습·네트워크·ROS2·Git 쓰기를 실행하지 않는다. 데이터와 시드가 같으면 사건·결정·지표를 비교할 수 있지만 gzip 컨테이너의 생성 시각까지 바이트 동일하다고 주장하지 않는다.

## 해석 제한

신규 규칙은 개발 자료에서 보완한 진단이다. 모든120세계를 평가했어도 미사용 Test 검증은 아니다. 실패 원인 집계2,880회, 새 실행5,022회,120개 물리/정보 세계와5개 학습 반복은 서로 다른 단위다. 무작위로 섞은 전체 평균을 실전 빈도나 독립 표본 신뢰구간으로 표현하지 않는다.

실제 최고속도·통신 지연·절대 자기위치 오차·성공 낙상 복구 등은 미측정이다. 현재 공백 지표와 추상 위협 통과를 실제 실점률로 해석하지 않는다. 최종 권고 이후의 실물 측정·새 구현·재학습은 별도 승인 전까지 진행하지 않는다.
