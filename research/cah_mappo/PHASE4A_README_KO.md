# Phase 4A 실행 및 코드 안내

이 단계는 완료된 Standard MAPPO의 **변경 없는 추론**, 기존 평가의 사후 분석, 규칙 기반 실행 중재 비교다. 재학습·CAH-MAPPO·새 신경망·ROS2·실물 구동을 포함하지 않는다. 기존 연구 파일을 수정하지 않고 신규 모듈과 산출물을 추가했다. Git 반영은 이번 요청에서 금지되어 수행하지 않았다.

## 주요 파일

- `src/cah_mappo/phase4a/scenarios.py`: 동일 물리 조건에 A/B/C·위험·고장 요인을 적용한다. `Witness`는 fixed/immediate/ready 결정적 실행으로 가능성의 증인을 만든다. 시간적 최적해를 계산하는 정책은 아니다.
- `rules.py`: `Gate`가 현재 중앙 관측으로 기존 담당자·후임·실제 준비·마감을 확인한다. `install_gate`는 해당 환경 인스턴스의 실행 함수만 감싼다. 기존 클래스 소스와 보상 함수는 바꾸지 않는다.
- `metrics.py`: `segments`, `score`, `aggregate`가 GT 로그에서 구역별 연속 공백·꼬리·인계·고장 복구를 계산한다. 담당자 이름과 실제 물리 커버를 구분한다.
- `evaluation.py`: 해시 확인 후 기존 Actor/Critic 가중치를 불러온다. 추론에서는 `distribution`만 사용한다. Actor의 원래 행동·확률·log-probability와 실제 실행을 분리해 기록한다.
- `legacy.py`: 기존 Fixed 400개와 MAPPO 8,000개 주 평가 raw를 분석한다.
- `experiment.py`: Validation 임계값 선택, 설정 동결, 새 공통 세계 비교, 로그 전수 감사와 분석을 수행한다.
- `phase4a.py`: 위 단계의 CLI.
- `phase4a_legacy_replay.py`: 기존 공백 사례 67개에 대한 선별된 사후 대응 재현과 이동·마감 분석. 전체 평가 집합의 개선율 추정으로 사용하면 안 된다.
- `phase4a_schedule_probe.py`: 2대 사례에서 준비 시작 시점과 출발 시점을 제한적으로 탐색한다. 완전한 시간적 최적화가 아니다.
- `phase4a_transfer_probe.py`: 기존 세계 12개에서 속도·서비스 시간·발생 순서를 한 항목씩 바꾸는 분포 이동 진단.
- `phase4a_summarize.py`: Validation 선택 결과와 일치하는 **결정적 재현 raw** 저장, 조건별·작업별·시드별 분석과 그림.
- `tests/test_phase4a.py`: 신규 27개 테스트. 기존 152개와 합계 179개다.

## 실행 환경

기존 승인된 `.venv`의 Python 3.12.14, PyTorch 2.7.0+cpu, NumPy 1.26.4를 그대로 쓴다. 새 패키지나 가상환경을 만들지 않았다. CPU1스레드 추론이며 학습 optimizer를 호출하지 않는다.

실제 산출물 루트는 `phase4a_outputs/study_v1/`이다. 출력은 배타적으로 생성하며 기존 디렉터리나 파일을 덮어쓰지 않는다. 아래 명령을 이미 완료된 `study_v1`에 반복하면 기존 산출물을 보호하기 위해 거부한다.

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim
$taskPython = 'research/cah_mappo/.venv/Scripts/python.exe'

& $taskPython -B -m unittest discover -s research/cah_mappo/tests -v

# 새 주 실행을 만들 때 기존 study_v1 대신 새 경로를 사용한다.
$phase4Run = 'research/cah_mappo/phase4a_outputs/my_new_run'
& $taskPython -B research/cah_mappo/phase4a.py legacy --output $phase4Run
& $taskPython -B research/cah_mappo/phase4a.py prepare --output $phase4Run
& $taskPython -B research/cah_mappo/phase4a.py compare --output $phase4Run
& $taskPython -B research/cah_mappo/phase4a.py audit --output $phase4Run
& $taskPython -B research/cah_mappo/phase4a.py report --output $phase4Run
```

보조 진단 스크립트는 이번 `study_v1` 분석용으로 경로를 명시했다. 이미 생성된 보조 출력도 재사용하지 않는다. 다른 실행을 분석하려면 스크립트의 `BASE`/`base`를 새 로컬 실행으로 지정하고 변경 식별자를 새로 남긴다. 이를 범용 학습 실행기로 해석하지 않는다.

## 로그와 결과를 읽는 순서

1. `PHASE4A_RESULTS_KO.md`: 측정 결과와 최종 판단.
2. `PHASE4A_PROTOCOL_KO.md`: 주 비교 전에 작성한 가설·설계·선택 규약. C의 의미와 추가 탐색에서 드러난 제한은 결과 문서가 보완한다.
3. `phase4a_outputs/study_v1/design/`: 시나리오, Validation 후보/선택, 가능성 증인, 코드·모델 해시 및 동결.
4. `legacy/`: 기존 8,400개 raw의 분석과 최악 사례 시간표.
5. `comparison/<정책>/raw/`: 새 주 비교의 모든 원시 JSONL gzip. `results.json`은 개별 지표와 집계값이다.
6. `analysis/condition_metrics.csv`: 조건·팀 규모별 기본 지표.
7. `analysis_extended_v2/condition_task_metrics.json`: 전체 작업 성공률과 중요한 신규 작업 `task0`의 성공률을 분리한다. 인계 공백 분포, 상위5%, 취소·실패 계수도 포함한다.
8. `analysis_extended_v2/five_seed_statistics.json`: 시드 5개를 반복 단위로 한 평균·SD·서술적 t구간. 확률 반복 2회를 독립 학습으로 세지 않는다.
9. `audit/audit.json`: 주 비교 전수 감사. `validation_trace_replay/audit.json`은 나중에 결정적으로 재구성한 Validation raw가 원래 저장 요약과 일치하는 검사다.

`analysis_extended`와 `analysis_extended_v2`의 수치 JSON은 같다. 그림의 중첩 점을 표현할 때 데이터 좌표를 이동하지 않도록 v2에서 원래 좌표의 동심원을 사용했다. 보고에는 v2 그림을 사용한다.

## 인계 중재와 지표 의미

READY는 후임이 가동 중이고 실제 구역 반경 안에 있으며 다른 작업을 수행하지 않는 상태다. 단순히 후임 이름을 지정한 것은 READY가 아니다. 대기 규칙은 기존 담당자를 남겨 후임의 이동을 허용한다. 적응 규칙은 다음을 모두 만족하면 조기 해제할 수 있다.

```text
t + 후임ETA + 기존담당자ETA + 남은작업시간 > 마감
t + 기존담당자ETA + 남은작업시간 <= 마감
작업우선순위 >= risk_ratio * 현재구역위험도
후임ETA <= eta_budget * 구역대응제한시간
```

Validation에서 선택된 값은 `risk_ratio=0.5`, `eta_budget=4.0`이다. 최적값이나 실물 안전값이라는 뜻은 아니다. 우선순위와 위험도가 실제 손실 확률로 보정되어 있지 않다. ETA도 현재 속도의 추정이며 향후 고장을 예견하지 않는다.

실행기는 Actor가 선택하지 않은 새 작업을 다른 로봇에게 발명해 배정하지 않는다. 대신 기존 담당자의 작업 요청을 지연시키거나, 후임의 요청을 잠시 보류하고 구역 준비를 실행한다. 원래 요청이 바뀌면 취소한다. 후임/기존 담당자 고장은 명시적인 실패, 위험 상승 및 작업 소멸은 취소로 남긴다. 마감 만료로 작업이 소멸한 취소도 별도 raw에서 확인할 수 있다. `controller_failure_count`를 모든 임무 실패의 개수라고 해석하면 안 된다.

추가 실행기는 중앙 현재 관측과 준비 중첩 명령을 가진다. 원형 Standard의 학습 MDP와 다르다. 손실 버퍼를 새로 만들지 않으며 Actor의 샘플 행동을 실행 행동으로 바꿔치기하지 않는다. 향후 알고리즘 동등 비교를 하려면 같은 실행 인터페이스에서 각 방법의 학습·평가 조건을 다시 고정해야 한다.

CGR은 기존 위험 가중 공백/위험 가중 구역시간이다. 실제 공백은 가중하지 않은 구역-초다. 분모는 2대 사례 32, 4대 사례 64 구역-초이며 위험 변형에서는 위험 가중 분모도 별도로 보고한다. 인계가 없으면 인계 공백은 N/A다. 우측 검열 공백과 복구 실패는 관측된 하한으로 표시한다. 장치 복구 사건과 다른 로봇이 대신 구역을 커버하는 복구시간은 서로 다르다.
