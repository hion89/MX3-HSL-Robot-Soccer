# Phase 3B-2B 실행·파일 안내

2026-11-19. 정상적인 작업 정보 전달 조건의 Standard MAPPO 기준선이다. 모든 신규 파일은 `research/cah_mappo/` 안에 있으며 기존 연구 코드/출력은 보존한다. 기존 Holdout 400개는 내용·ID·시드·초기 상태를 열어보지 않는다.

## 구현 파일

- `src/cah_mappo/phase3b2b/scenarios.py`: 버전 고정 생성기, 독립 학습 사건 스트림, Validation과 비최종 공통 benchmark.
- `.../learner.py`: `BaselineLearner`, 독립 초기화와 기존 PPO 업데이트 재사용, episode 경계 checkpoint/정확한 재개.
- `.../training.py`: 고정 예산, 모든 훈련 의사결정·에피소드 요약, Validation 전용 선택.
- `.../evaluation.py`: Critic 상태를 받지 않는 로컬 Actor 추론, 기존 중앙 참고 정책, 동일 물리 조건의 슬롯 순열 평가.
- `.../metrics.py`: 기존 Ground Truth 지표, 완료/응답/미관측 등 보충 지표, 중앙군의 동일 보상 재구성.
- `.../benchmark.py`: 선택 동결 후 시드별 두 모드·공통 기준선·슬롯 진단 실행.
- `.../statistics.py`: 독립 학습 시드5개를 단위로 하는 통계, 같은 시나리오 대응 차이.
- `.../io.py`: 덮어쓰지 않는 JSON 및 gzip JSONL 저장/읽기.
- `.../runner.py`: 사전 비용 측정·설정 동결·로컬 숨김 worker·선택 동결 장벽.
- `.../__init__.py`: 패키지 선언.
- `phase3b2b.py`: 실행 CLI.
- `verify_phase3b2b.py`: 기존 파일 보존, 모든 훈련 mask/log-probability, 원시 평가 지표, Actor 추론 재현, 통계·선택 동결 감사.
- `analyze_phase3b2b.py`: 시드별 학습/반복/작업 결과, 인계 검열 보충값과 전체 지표 CSV.
- `supplement_phase3b2b.py`: 평균에 가려지는 공백 꼬리와 시나리오 수준 대응 불확실성. 모델 선택이나 재학습은 없다.
- `plot_phase3b2b.py`: 실제 학습·Validation 곡선, 기준선 비교 및 공백-응답 관계 SVG.
- `tests/test_phase3b2b.py`: 신규 단위·통합 테스트. 기존 회귀 테스트도 함께 실행한다.
- `PHASE3B2B_PROTOCOL_KO.md`: 동결한 환경·알고리즘·예산·평가·관측 공정성 정의.
- `PHASE3B2B_RESULTS_KO.md`: 실제 시드별 측정값, 통계·기준선·슬롯 한계와 완료 보고.
- `PHASE4_RULE_HANDOVER_PLAN_KO.md`: 후속 규칙 인계 비교 설계. 구현/실행 승인을 대신하지 않는다.

기존 `phase3b1`의 MLP/PPO/GAE와 `phase3a`의 센서/통신·보상·행동 중재, `phase2`의 실제 이동·기준선·평가를 재사용한다. 새 신경망 구조나 인계 학습 구조는 없다. 중앙 기준선은 추가 정보/준비 행동을 가진 참고군으로 명시한다.

## 실행 경로

실제 주 출력은 `phase3b2b_outputs/main_v1/`이다. 비용 측정은 `calibration_v1/`에 별도로 둔다.

- `frozen_plan.json`, `pretraining_freeze.json`, `source_manifest.json`: 학습 전 동결과 변경 식별자.
- `future_comparison_contract.json`: 향후 비교에 보존할 공통 보상·환경·기준 인터페이스.
- `validation_manifest.json`: Validation32개. 최종 보호 Holdout과 다르다.
- `benchmark_manifest.json`: 모든 시드 선택 후 생성한 비최종 공통 benchmark400개.
- `all_seeds_selection_freeze.json`: 각 선택 모델 해시와 선택 근거.
- `seed_101/`, `seed_211/`, `seed_307/`, `seed_401/`, `seed_503/`: 개별 학습·평가.
- `references/<policy>/`: 로컬 Fixed/ETA와 기존 중앙 Fixed/Greedy/Exact/Handover 참고 평가.
- `worker_logs/`: 로컬 작업 로그. 프로세스는 숨김 창, CPU1스레드씩 사용한다.
- `statistics.json`: 시드 간 통계, 대응 비교, 슬롯 통계.

각 시드 디렉터리:

- `config.json`, `selection.json`: 학습 시드/사건 스트림/버전/예산/선택.
- `checkpoints/initial.pt`, `checkpoints/stepNNNNNN.pt`: 초기 및 8,192스텝 간격/마지막 checkpoint. 선택된 파일명은 `selection.json`을 따른다.
- `learning_curve.csv/json`, `update_*.json`, `validation_step*.json`: loss·entropy·KL·clip·gradient norm·학습/검증 성과.
- `training_decisions.jsonl.gz`: 모든 훈련 단계의 관측된 job ID·할당·샘플 행동·mask·분포·log-probability·실행 중재.
- `episodes.jsonl`: 모든 학습 에피소드의 Ground Truth 평가 요약.
- `first_rollout.pt`, `trajectories/*.jsonl.gz`: 최초 rollout과 일부 원시 훈련 궤적.
- `benchmark_results.json`, `benchmark_raw/`: 결정적400개, 확률적1,200개 결과와 전체 raw.
- `slot_results.json`, `slot_raw/`: 같은32개 물리 사례에서6순열의 진단. 원래 순서는 주 평가의 반복0을 재사용한다.

학습 곡선은 해당 rollout에서 완료된 에피소드 평균이다. 작업1~3개가 섞인 학습 Return과 항상3개인 주 평가 Return을 단순히 같은 난이도로 비교하지 않는다. checkpoint 선택은 마지막 모델을 자동 채택하지 않고, 동결한 Validation 순서를 따른다. 마지막 Validation 상태도 별도로 남긴다.

## PowerShell 재현 명령

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim
$taskPython = Join-Path (Get-Location) 'research\cah_mappo\.venv\Scripts\python.exe'

& $taskPython -B -m unittest discover -s research/cah_mappo/tests -v

# 전체 재현: 기존 출력 경로를 덮어쓰지 않도록 새 이름을 사용한다.
& $taskPython -B research/cah_mappo/phase3b2b.py calibrate --output research/cah_mappo/phase3b2b_outputs/my_calibration
& $taskPython -B research/cah_mappo/phase3b2b.py prepare --output research/cah_mappo/phase3b2b_outputs/my_run --calibration research/cah_mappo/phase3b2b_outputs/my_calibration
& $taskPython -B research/cah_mappo/phase3b2b.py train-all --run research/cah_mappo/phase3b2b_outputs/my_run
& $taskPython -B research/cah_mappo/phase3b2b.py benchmark-all --run research/cah_mappo/phase3b2b_outputs/my_run

# 이미 저장한 실제 실행의 독립 감사만 수행
& $taskPython -B research/cah_mappo/verify_phase3b2b.py --run research/cah_mappo/phase3b2b_outputs/main_v1 --output research/cah_mappo/phase3b2b_outputs/my_audit
```

새 설치는 필요하지 않다. 기존 승인된 `.venv`만 사용한다. 비용 추정이 이전8시간/20GiB 범위를 넘으면 `prepare`가 본학습을 거부한다. 이 경우 예산을 확대하기 전에 사용자 승인이 필요하다.

`BaselineLearner.restore_exact(path)`는 같은 코드/설정에서 episode 경계 재개를 지원한다. 환경의 초기 상태·관측·Critic 상태를 재구성해 저장 값과 비교한다. 이 기능은 임의 중간 단계 resume가 아니다. CLI는 이미 존재하는 실행 디렉터리를 덮어쓰지 않으며, 재개를 실제 운영에 사용할 때는 같은 source manifest를 확인하고 새 출력에 기록해야 한다.

## 해석 주의

400개 공통 benchmark는 기존 보호 Holdout 400개와 다른 집합이다. 보호 파일은 메타데이터만 확인하며 내용을 해시 재검증 목적으로도 다시 읽지 않는다. 기존 Test나 이번에 확인한 benchmark를 미래 최종 미사용 평가라고 부르지 않는다.

시드별 확률적 반복3개는 실행 변동성 평가다. 독립 학습 반복은5개다. 학습 시드 간 Student t 구간은 이 고정 benchmark에서의 시드 불확실성이며, 최종 Holdout의 성능 보증이나 다중 비교 유의성 증명이 아니다.
