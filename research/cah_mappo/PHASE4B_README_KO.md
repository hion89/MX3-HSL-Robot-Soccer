# Phase 4B 코드·지표·재현 안내

이 코드는 실물 제어와 분리된 고수준 2D 할당 연구다. 기존 CGR·Phase 1~4A 코드·보상·체크포인트는 수정하지 않는다. 신규 버전 `phase4b-validity-v1`/`phase4b-coverage-v1`을 별도로 추가했다. CAH-MAPPO·재학습·패키지 설치·원격·실물·Git 반영은 수행하지 않았다.

## 파일별 역할

- `src/cah_mappo/phase4b/model.py`: 공개 관측 계약, 직선 위협 위치/도달 예측, 이동·회전·현재 작업 완료 후 응답 ETA, 용량1 매칭.
- `engine.py`: 기존 이동/작업 엔진을 재사용하는 신규 실행기. 물리 dt와1초 판단 주기를 분리하고 위협 도착 시점의 GT로 방어/통과 이벤트를 남긴다.
- `metrics.py`: 기존 CGR 병기, 소유권/응답 가능성/위협 노출/작업 후 공백 지표. 보상과 독립적으로 raw만 사용한다.
- `scenarios.py`: A/B/C와 사전 배치·고장·위험 변화·작업 후 위협·불가능 대조군의 새 개발 집합.
- `policies.py`: 기존 Fixed/Greedy/Exact/Handover와 공개 관측용 READY/적응/관측 예비 배치/상시 예비 배치 어댑터, 첫146차원 전이 투영.
- `planning.py`: 관측된 현재 상태만 쓰는 제한된 후보 행동 예측. 별도의 `OfflineSchedule`은 미래 작업 시점/정체를 아는 계획 증인이다.
- `experiment.py`: 보존, Validation 선택, 동결, 정책 비교, 제한 계획 탐색, 시간 해상도 실험.
- `analysis.py`: 원시 로그 감사, 조건별 수치, 미래 정보/같은 prefix 검사, 지표와 실제 위협 통과 연결.
- `phase4b.py`: 주 실행 CLI.
- `phase4b_feedback.py`: 기존 Phase3A `Perception`을 그대로 쓰는 센서 피드백 전이 참고군. 첫 active-only 투영과 결과를 별도 보존한다.
- `phase4b_contingency.py`: 고장 후 알려진 위협이 없을 때 작업을 다시 진행하는 공개 정보 규칙과 추가 개발 검증. 사후 개발임을 명시한다.
- `phase4b_finalize.py`: 보조 raw 감사, 최종 조건별 자료, 예전 로그의 위협 미정의 해석, 소스·문서 해시 봉인.
- `tests/test_phase4b.py`, `test_phase4b_feedback.py`, `test_phase4b_contingency.py`: 신규37개 시험. 기존179개와 총216개.

## 위협 모델

상대 공격수와 공을 하나의 점으로 취급한다. 정해진 방향으로 일정 속도로 알려진 구역에 도달한다. 공을 차거나 튕기거나 경로 중간에서 가로채는 물리는 없다. 도달 시 가동 중이고 다른 유한 작업을 수행하지 않는 로봇이 구역 반경 안에 있으면 자동 방어, 없으면 `threat_passed`다. 동시 위협은 한 로봇이 하나만 맡을 수 있다. 이것은 위협 통과율/방어 실패율이며 **실점률이 아니다**.

속도1/0.45~0.55m/s, 반경.25m, 응답 예산2초, 상대의 직선 속도1m/s와 발생 일정은 모두 시뮬레이션 가정이다. 확인된 실측 속도값을 사용한 것이 아니다. 좌표는 추상 공간이며 연습장3.9×2.6m의 경계나 실제 RoboCup 경기장 규격을 적용한 장면이 아니다. 보행·킥·접촉·마찰도 없다.

위협은 실제로 드러난 뒤에만 정책에 전달한다. 공개 버스에는 현재 팀 위치·가동/할당 상태·활성 작업·현재 관측된 위협만 있다. 미래 작업 목록/수·고장 일정·시나리오 ID·시드는 없다. 현재 위치·방향·속도로부터의 도착 예측은 허용한다.

## 지표 정의와 분모

기존 `legacy_coverage_gap_ratio`는 이전 evaluator의 위험 가중 CGR 그대로다. 기존 결과를 덮어쓰지 않는다.

Ownership Gap은 유효한 명시 담당자(owner)가 없었던 구역-초다. 이름만 있어도 가동 불능이거나 해당 구역 할당을 유지하지 않으면 유효하지 않다. 후임이 멀리 있어도 명시적으로 배정되면 소유권은 존재할 수 있다.

Physical Response Coverage Gap은 현재 작업을 유지하면서 요구된 예산 안에 응답할 수 있는 로봇의 용량1 매칭이 없었던 구역-초다. 다른 작업을 즉시 동시에 수행한다고 가정하지 않는다. 유한 작업에 묶인 경우 그 위치까지 이동·회전, 남은 서비스, 다음 제어 주기, 구역까지 이동·회전을 더한다. 현재 작업을 그 마감 안에 완료할 수 없으면 해당 응답 후보에서 제외한다. 유휴/다른 구역 로봇의 재지정에는 다음 판단 시점까지의 지연을 포함한다. 이 ETA는 현재 상태와 유지할 작업에 대한 가능성 추정이며 미래 고장·정책의 실제 선택을 보장하지 않는다. 따라서 실제 방어 이벤트와 별도로 보고한다.

위 두 지표의 책임 활성 시간은0~32초다. 분모는 구역 수×32초로2대/1구역32,4대/2구역64 구역-초다. 작업이 끝나도 구역 책임은 종료하지 않는다.

Threat-Weighted Gap은 각 실제 위협의 발생~도달 시간 중, 남은 도달시간 안에 응답 가능한 매칭이 없는 시간을 강도 가중하여 합산한 것이다. 분모는 같은 위협의 강도×노출시간 합계(위협-초)다. 구역-초 분모와 섞지 않는다. GT 위협 전체와 실제 관측 이후 노출을 따로 제공한다. 강도는 손실 확률이 아닌 설계값이다. 진행 중인 여러 위협에 대한 현재 매칭은 보수적인 용량 제한이며 서로 다른 도달 시점을 포함하는 완전한 시간 계획이 아니다.

실제 `threat_pass_rate`는 도달한 위협 중 `threat_passed` 사건 비율이다. 정책이 움직일 기회가 있었다는 ETA 지표와 실제로 방어 위치에 도착했다는 사건은 다르다. 위협이 없는 새 세계의 위협 분모와 비율은 N/A다. 과거 엔진에 위협 변수가 없었다는 것은 ‘위협이 실제 없었다’와도 다르다. 예전 로그 해석은 `completion/legacy_unmodeled_threat_interpretation_v2.json`을 사용한다.

Post-Task Gap은 **모든 유한 작업이 성공한 경우에만** 마지막 완료 시각부터32초까지 측정한다. 분모는 그 시간×구역 수다. 실패/미완료 작업이 있으면 N/A다. 소유권·기존 위치 기반 커버·새 응답 가능 커버를 각각 제공한다.

## 비교의 정보/행동 차이

비학습9개는 같은 공개 관측과 준비 중첩 행동을 받는다. 상시 예비 배치는 경고나 미래 작업 없이도 알려진 구역에 미리 보낼 수 있다. 이를 금지해 다른 방법에 유리한 조건을 만들지 않았다.

기존 MAPPO146차원 Actor는 재학습하지 않으며 새 위협 채널과 준비 명령을 사용하지 못한다. 따라서 이를 새 위협 기반 정책과 동등한 학습 정보/행동 비교군이라고 부르지 않는다. 원래 Phase3A 센서/캐시를 사용하는 보조 전이의 결과를 최종 참고군으로 쓰며 원형 투영도 보존한다. Critic은 추론에 쓰지 않는다.

기존 Exact는 순간 할당 완전 탐색이며 Hungarian이나 시간적 최적해가 아니다. 새 lookahead는 현재 관측만 가지고 제한된 후보 행동을 유지하는 예측기다. 알려지지 않은 미래 사건은 넣지 않으며 모든 행동 순서를 탐색하지 않는다. 오프라인 스케줄은 미래를 아는 별도 참고 증인이다. 증인을 못 찾았다고 불가능하다고 판정하지 않는다. 작업량 하한 및 별도로 명시한2대 이동 하한만 불가능성 근거로 사용한다.

## 실행

기존 승인 `.venv`의 Python3.12.14, PyTorch2.7.0+cpu, NumPy1.26.4를 사용한다. 실제 출력은 `phase4b_outputs/study_v1/`이다. 모든 명령은 기존 출력을 덮어쓰지 않는다.

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim
$taskPython = 'research/cah_mappo/.venv/Scripts/python.exe'
$phase4Run = 'research/cah_mappo/phase4b_outputs/my_new_run'

& $taskPython -B -m unittest discover -s research/cah_mappo/tests -v
& $taskPython -B research/cah_mappo/phase4b.py protect --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b.py prepare --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b.py legacy --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b.py compare --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b.py offline --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b.py sensitivity --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b.py audit --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b.py report --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b_feedback.py run --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b_feedback.py audit --output $phase4Run
& $taskPython -B research/cah_mappo/phase4b_contingency.py --output $phase4Run
```

최종 감사 CLI `phase4b_finalize.py audit`는 실제 전체 회귀 로그 `regression_tests_complete.txt`도 요구한다. 이번 저장 방법과 로그는 기존 `study_v1`에 있으며 실행하지 않은 로그를 성공으로 만들어 넣으면 안 된다. `seal`은 보고서 작성 뒤 소스·문서·테스트 해시를 저장한다.

## 出力

- `design/`: Validation40·진단120 manifest, 후보 raw240, 설정 선택과 소스/모델 동결.
- `comparison/`: 주 비교2,880에피소드의 raw와 개별 결과. 주 물리dt=.25초, 판단1초.
- `offline/`: 오프라인 계획507회 raw, 가능성 증인과 제한된 불가능성 증명.
- `sensitivity/`: 같은40세계×6정책×4시간 간격=960회 raw.
- `sensor_feedback_transfer/`: 원래 Phase3A 관측을 쓰는 기존5모델·결정적/확률적1,800회. 초기 투영과의 차이 및 감사.
- `fault_contingency/`: 기존24/추가24 개발 세계의 규칙 비교144회와 고장 오프라인390회.
- `analysis/`: 조건별CSV, 같은 prefix108쌍, 시간 해상도, 지표/위협 통과 대응, 5모델 변동.
- `completion/`: 보조 감사를 포함한 최종CSV/JSON, 과거 로그의N/A 해석, 최종 해시.

설정 선택은 Validation에서만 수행했고 경고 범위4초/예측 범위4초로 고정했다. 보조 고장 규칙은 주 비교 이후 개발 진단이므로 사전 규약의 정책과 혼동하지 않는다. 추가index6~11도 개발 데이터이며 최종 미사용Test라고 부르지 않는다. 보호Holdout400은 사용하지 않는다.
