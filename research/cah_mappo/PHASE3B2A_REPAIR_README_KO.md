# Phase 3B-2A 보완: 슬롯·정보 전달 진단과 동일 예산 재학습

2026-11-18. 기존 Standard MAPPO의 MLP, 이산 행동 10개, PPO, 보상, 학습률을 유지했다. 기존 Phase 1~3B-2A 파일은 수정하지 않고 신규 모듈과 출력만 추가했다. CAH-MAPPO, 새 인계 구조, 패키지 설치, 원격 접속, 실물 구동, Git 커밋·푸시는 수행하지 않았다.

## 신규 코드

모든 경로는 `research/cah_mappo/` 기준이다.

- `src/cah_mappo/phase3b2a_repair/slots.py`: `SlotMap`, `PERMUTATIONS`. 작업용 논리 슬롯 2·3·4의 여섯 순열을 에피소드 내내 고정한다. 관측 벡터, 자기 할당 one-hot, job ID, 유효/신선도 mask, 행동 번호와 분포를 함께 변환한다. Critic 상태도 같은 번호 체계로 변환하며 크기는 141로 유지한다.
- `.../environment.py`: `AuditedEnv`. 기존 `ParallelCore`를 감싸서 관측·행동을 양방향 변환한다. 실제 이동·작업·고장·중재는 원래 엔진이 실행한다. 센서 관측/실제 메시지 수신/Actor 인지/정책 선택의 최초 시각을 기록한다.
- `.../scenarios.py`: 고정 → 제한 위치 변화 → 우선순위·마감 변화의 진단 환경. 학습·검증·개발·최종 holdout 시드를 분리한다.
- `.../evaluation.py`: `timelines`, `score`, `summarize`, `evaluate_one/many`, `coverage_windows`. 미관측 실패와 선택 실패를 구분하고 공백 초/구역 시간/위험 가중 분모를 저장한다.
- `.../learner.py`: `RepairLearner`. 기존 `Learner`의 optimizer 업데이트를 그대로 상속한다. 시나리오·고정 슬롯 순열 수집기와 업데이트 전후 분포 진단을 추가했다.
- `.../experiment.py`: 사전 설정, Exact 가능성 검증, 진단 순서, 두 학습 설정, 두 평가 모드의 Validation 선택, 동일 조건 사후 평가.
- `.../__init__.py`: 패키지 선언.
- `phase3b2a_repair.py`: 제한된 전체 진단/재학습 실행 CLI.
- `replay_phase3b2a_repair.py`: 동일 학습을 재생하여 전체 훈련 의사결정 로그를 복원하고 매 중간 checkpoint와 갱신 보고의 정확한 일치를 검증한다. 모델 선택이나 학습 예산 연장을 하지 않는다.
- `verify_phase3b2a_repair.py`: 모든 평가 로그에서 지표·슬롯·보상 시점·통신 시각을 검사하고 선택 모델을 재평가한다.
- `plot_phase3b2a_repair.py`: 측정 학습 곡선 및 슬롯만 바꾼 작업별 결과 SVG.
- `probe_phase3b2a_sampling.py`: 고정 B3 신경망의 확률적 평가 온도만0.5로 바꾼 추가 개발 진단. optimizer 갱신과 모델 선택은 없다.
- `tests/test_phase3b2a_repair.py`: 신규 단위·통합 테스트 15개.
- `PHASE3B2A_REPAIR_HYPOTHESES_KO.md`: 결과를 보기 전에 정리한 가설·실험 계획.
- `PHASE3B2A_REPAIR_RESULTS_KO.md`: 실제 결과와 본학습 가능 여부 판단.

신규 전체 파일 목록과 해시는 `phase3b2a_repair_outputs/new_files_manifest.json`에 기록한다. 예전 결과를 보완 문서로 덮어쓰지 않았다.

## 슬롯 변환 규칙

기본 관측은 Actor 146, Critic 141, 행동 10개다. 행동0은 해제/대기, 1은 유지, 2~9는 관측한 로컬 job 슬롯0~7에 대응한다.

4대 통제 환경은 처음에 구역2개가 알려져 논리 슬롯0·1을 차지한다. 이후 발견되는 세 작업은 논리 슬롯2·3·4다. 변환은 이 세 자리만 바꾸고 나머지는 그대로 둔다. 예를 들어 순열 `(4,3,2)`는 기존 task2를 표시 슬롯2/행동4에, task0를 표시 슬롯4/행동6에 배치한다. 물리 작업 ID·위치·발생·마감은 바꾸지 않는다.

슬롯 변환기는 아직 관측하지 않은 작업의 ID나 미래 명단을 받지 않는다. 공개된 패딩 자리만 재배치하므로 일부 자리는 작업이 발견될 때까지 비어 있다. 각 로봇의 발견 순서와 기존 ID가 유지되고, 에피소드 도중 순열을 바꾸지 않는다. D 정보 실험은 identity 순서를 사용하므로 부분 관측의 발견 순서를 ‘구역 우선’으로 강제하지 않는다.

학습기가 저장하는 행동과 log-probability는 표시 슬롯 기준이다. 엔진 입력에서만 확률·mask·행동을 역변환한다. 실제 중재 결과는 다시 표시 슬롯으로 변환해 별도 기록한다. 환경이 실행한 행동으로 PPO의 샘플 행동을 바꾸지 않는다.

기존 raw transition의 `records`와 `critic_before/after`는 엔진 좌표계 그대로 보존한다. 추가 `slot_adapter` 필드에는 표시 관측, ID/slot/mask, 표시 샘플 행동, 엔진 행동, 실행 행동, Critic 변환을 함께 저장한다. 결정적 평가의 행동 분포는 one-hot이며, 실제 신경망 확률은 `model_probabilities`에 따로 보존한다.

## 작업별 타임라인과 실패 분류

각 작업에 다음을 저장한다.

- 실제 발생/마감 시각.
- 로봇별 최초 직접 센서 관측, 실제 메시지 도착, Actor 최초 인지.
- 로봇별 최초 정책 선택 및 실제 할당 수락.
- 실제 대응 시작, 완료 또는 만료.

메시지는 측정 시각과 수신 시각을 구분한다. 수신 로그는 기존 perception의 실제 전달 경로에서 기록한다. 보내졌지만 아직 도착하지 않은 정보는 Actor나 타임라인의 수신 정보로 세지 않는다.

실패는 서로 겹치지 않는 다음 범주로 분류한다. 모든 실패는 전체 성공률에 그대로 포함된다.

1. `no_timely_information`: 마감 전에 어떤 Actor도 작업을 알지 못함. 끝까지 미관측인 경우와 기한 이후에야 안 경우를 별도로도 확인한다.
2. `known_but_not_selected`: 마감 전에 알려졌으나 기한 내 정책 선택이 없음.
3. `selected_but_no_response`: 선택했으나 실제 대응 시작이 없음. 중복 중재 거부나 도달 지연 등이 포함된다.
4. `responded_but_not_completed`: 대응은 했으나 기한 내 완료하지 못함.
5. `success`: 기한 내 완료.

`ever_known`과 `timely_known`을 구분한다. 마감 후 처음 받은 정보를 정상 대응 기회가 있었다고 세지 않는다. 작업별 선택 횟수는 요청 횟수이며 완료 횟수와 다르다.

## 정보 전달 조건

- `controlled`: 같은 물리 상태·고장·속도·위험 일정에서 센서 반경100m, 잡음0, 통신 지연0. 제한된 경기 공간의 모든 현재 작업을 정상 관측하는 통제 조건이다. 미래 작업은 제공하지 않는다. peer 관측도 좋아지므로 ‘작업 정보만’의 순수 효과 실험은 아니다.
- `realistic`: 기존 D의 센서 반경2.5m, 위치 잡음 ±0.15m, 통신 지연2초를 적용한다.
- `zero_delay`: realistic의 센서·잡음은 유지하고 통신 지연만0으로 바꾼다. 지연만의 대조군이다.

작업 정보가 없는 문제를 해결한 것으로 가장하기 위해 realistic에 숨은 GT 알림을 넣지 않았다. 정상 전달 조건을 분리하고 실제 제한 정보 조건의 실패를 그대로 보고한다. 필요하다면 향후 발생 후 공통 작업 통지 경로를 별도 설계할 수 있지만 이번 구현에는 없다.

## 재학습 통제와 평가

두 설정은 같은 기존 C 최종 가중치와 Adam 상태에서 시작한다. 보상 coverage=2, 학습률0.0003, 네트워크와 PPO 설정은 같다. 새 학습 시드7 하나, 같은 제한 위치 Training 12개를 사용한다.

- `identity`: 표시 순서를 유지한 동일 예산 재학습 대조군.
- `augmented`: 에피소드마다 세 작업 슬롯의 순열을 균형 있게 바꾼다. 한 시나리오가 항상 같은 순열만 받지 않도록 풀 순환 횟수도 사용한다.

각 설정8,192스텝/256에피소드, 총 고유 학습16,384스텝이다. Validation은 물리 시나리오2개 × 순열6개를 결정적·확률적 모드로 모두 평가한다. 이는 독립 시나리오12개가 아니다. 모델 선택은 두 모드 중 낮은 성공률 → 두 모드 평균 성공률 → 더 큰 공백의 음수 → 평균 제한 응답시간의 음수 순이다. 단일 가중합 성능 점수를 사용하지 않는다.

주 비교는 모두32초다. 구역은 처음부터 계속 활성이다. 실제 공백 구역-초, 구역 수×평가시간, 위험 가중 공백/구역 시간을 함께 저장한다. 두 구역이 동시에1초 비면 공백 구역-초는2다. 8/16초 절단은 분모 진단용이며 주 성능 평가를 바꾸지 않는다.

기존 시간 제한의 bootstrap=1/에피소드 연결=0 규약을 유지했다. 새 모델이나 새로운 MAPPO 구조는 사용하지 않았다.

## 최종 holdout

사용하지 않은 시드로 위치 변화200개와 우선순위·마감 변화200개를 준비했다. 상태는 `NOT_EVALUATED`다. `AuditedEnv`는 `final_holdout` 입력을 거부하고 학습기는 Training 이외 split을 거부한다. 준비된 설정·ID·해시만 존재하며 결과는 없다.

이전에 사용한 Test는 개발용으로 명시해 재사용했다. 새 개발 평가도 앞으로 미사용 최종 Test라고 부르면 안 된다. 최종 holdout은 별도 승인 및 설정·모델·선택 규칙 동결 후 전용 평가 경로에서만 해제해야 한다. 이번 단계에서는 해제하지 않았다.

## 실행 명령: PowerShell

기존 승인된 전용 `.venv`를 사용한다. 아래 재실행은 새 경로만 허용한다.

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim
$taskPython = Join-Path (Get-Location) 'research\cah_mappo\.venv\Scripts\python.exe'

& $taskPython -B -m unittest discover -s research/cah_mappo/tests -v

& $taskPython -B research/cah_mappo/phase3b2a_repair.py --output research/cah_mappo/phase3b2a_repair_outputs/my_run

& $taskPython -B research/cah_mappo/verify_phase3b2a_repair.py --run research/cah_mappo/phase3b2a_repair_outputs/pilot_v1 --output research/cah_mappo/phase3b2a_repair_outputs/my_audit

# 같은 계산을 재생하는 검증. 추가 수렴 실험이나 예산 연장이 아니다.
& $taskPython -B research/cah_mappo/replay_phase3b2a_repair.py --run research/cah_mappo/phase3b2a_repair_outputs/pilot_v1 --output research/cah_mappo/phase3b2a_repair_outputs/my_replay
```

원시 평가 로그는 전 과정을 직접 수집했다. 훈련은 최초 rollout/샘플 궤적·모든 에피소드 요약을 직접 저장했고, 전체 훈련 의사결정은 동일 시드 재생으로 복원해 별도 `training_replay_v1/*_training_decisions.jsonl`에 저장했다. 재생한 모든 update 보고와 2,048스텝 간격 checkpoint/RNG가 원본과 정확히 같음을 확인했다. 복원 로그를 원래 시각에 수집한 원시 로그라고 표시하지 않는다. 재생 계산16,384스텝은 학습 개선 실험과 구분해서 보고한다.

기존 checkpoint의 임의 중간 시점 resume 한계는 그대로다. 재생 검증은 초기 부모 모델부터의 재현이지, 진행 중 환경/메시지 큐를 저장한 정확한 resume 기능은 아니다.
