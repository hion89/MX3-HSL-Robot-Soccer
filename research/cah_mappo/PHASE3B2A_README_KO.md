# Phase 3B-2A 구현과 실행 안내

2026-11-18. 기존 Phase 1~3B-1을 보존하고, Standard MAPPO의 랜덤화·분리 평가·관측 공정성·보상 민감도 예비 실험을 추가했다. 새 CAH-MAPPO 또는 인계 학습 구조는 없다. 설치, 서버 접속, 실물 구동, Git 커밋·푸시는 하지 않았다.

## 신규 파일과 재사용 범위

모든 신규 파일의 위치는 `research/cah_mappo/` 내부다. 기존 파일은 수정하지 않았다.

- `src/cah_mappo/phase3b2a/scenarios.py`: `generate`, `build_manifest`, `make_env`. 정책과 독립적인 초기 상태·사건 일정 생성, split 분리와 시나리오 ID/해시.
- `src/cah_mappo/phase3b2a/baselines.py`: `LocalPolicy`. Actor 관측 패킷만 입력받는 유지·고정 역할·ETA·무작위 기준선.
- `src/cah_mappo/phase3b2a/evaluation.py`: `actor_inputs`, `evaluate_one`, `evaluate_many`, `aggregate`. Critic 정보를 받지 않는 추론 경로, 동일 시나리오 평가, JSONL 저장.
- `src/cah_mappo/phase3b2a/learner.py`: `PoolLearner`. 기존 `Learner`를 상속하고 수집기만 시나리오 순환용으로 분리했다. 실제 update/tensors/minibatch/GAE/네트워크는 기존 코드 그대로다.
- `src/cah_mappo/phase3b2a/experiment.py`: 제한된 CPU 실험, 사전 설정 저장, Validation 선택 후 동결, 이후 Test 평가.
- `src/cah_mappo/phase3b2a/prior_analysis.py`: 기존 체크포인트의 저장된 궤적에서 보상 기여, 종료 시간, 역할 변경·중복 요청 등을 계산한다.
- `src/cah_mappo/phase3b2a/__init__.py`: 패키지 선언.
- `phase3b2a.py`: 이번 예비 실험 CLI.
- `verify_phase3b2a.py`: 원시 평가 로그 재계산, 시나리오 쌍 일치, checkpoint 재평가와 이전 파일 보존 감사.
- `diagnose_phase3b2a.py`: Test 이후 수행한 명시적 사후 진단. 이전 정책을 같은 창에서 평가하고 슬롯 선택·미관측 작업을 분석한다. 학습/모델 선택에 연결되지 않는다.
- `plot_phase3b2a.py`: 기존 표준 라이브러리 SVG 출력을 읽어 두 민감도 설정의 범례를 정확히 붙인 신규 그림 생성. 원본 그림을 덮어쓰지 않는다.
- `tests/test_phase3b2a.py`: 신규 테스트 19개.
- `PHASE3B2A_RESULTS_KO.md`: 측정 결과·실패 원인 분석.
- `PHASE3B2B_PLAN_KO.md`: 미실행 본학습 설계와 계산 예산.
- `phase3b2a_outputs/`: 이전 출력과 분리된 신규 결과. 상세 파일 목록/해시는 `new_files_manifest.json`에 기록한다.

`phase2/model.py`, `engine.py`, `metrics.py`, `policies.py`와 Phase 3A의 관측·보상·중재 구현을 그대로 사용한다. 기존 중앙 기준선의 입력 정보나 알고리즘을 조용히 변경하지 않았다.

## 시나리오와 분리 규칙

사건 시드는 `split 시작값 + stage 번호*1000 + index`다. 시작값은 Training=10000, Validation=20000, Test=30000이며 index는 0~999만 허용한다. 생성기 내부 난수는 `scenario-generator-v1:seed`의 전용 RNG를 사용한다. 학습 신경망 시드 7, 정책 평가 RNG와 분리된다.

각 단계에 Training 24개, Validation 8개, Test 12개를 미리 생성했다. 총 Training 120개, Validation 40개, Test 60개다. 모든 단계가 생성됐지만 모든 단계에서 학습한 것은 아니다.

- A: 2대, 책임 구역 1개, 동적 작업 1개. 로봇 초기 위치, 작업 위치·발생 시점·마감·우선순위와 구역 우선순위를 랜덤화한다.
- B2: 2대, 구역 1개, 순차/동시 활성 동적 작업 2개. 난이도 평가만 수행했다.
- B3: 3대, 구역 1개, 동적 작업 2개. A의 Validation 기준을 통과한 뒤 별도 신경망으로 학습했다.
- C: 4대, 구역 2개, 동적 작업 3개. B3의 Validation 기준을 통과한 뒤 별도 신경망으로 학습했다.
- D: C 구성에 로봇별 속도 차이, 5초 C 고장/10초 복구, 7초 D 감속, 6초 구역 위험도 증가를 추가했다. 센서 반경 2.5m, 위치 잡음 ±0.15m, 통신 지연 2초다. 이번에는 난이도 평가만 했다.

A~C는 센서 반경 10m, 잡음·통신 지연 0이다. 이 작은 공간에서는 대부분의 현재 정보를 관측할 수 있는 쉬운 조건이다. D가 제한 관측 조건을 점검한다. 로봇 용량은 1이다. 장해물·충돌·실물 운동학 모델은 없는 추상적인 2D 환경이다.

시나리오 ID는 단계/split/index와 생성 payload 해시를 포함한다. 사건 계획과 초기 상태는 정책 호출 전에 생성하며 정책의 난수 소비나 선택 행동에 의해 바뀌지 않는다. 정책이 고장 로봇의 위치를 달리 만들 수는 있지만 고장 시점·대상 자체는 같다.

## 평가 구간 변경 — 이전 결과와 구분

새 실험은 모든 정책을 **32초 고정 구간**에서 평가한다. 작업 완료 뒤에도 책임 구역 보호를 평가한다. 기존 Phase 3B-1의 조기 종료 결과는 보존하며 직접적인 숫자 우열 비교에 사용하지 않는다.

환경의 외부 시간 제한으로 처리하므로 32초에 `truncated=True`, bootstrap=1, continuation=0이다. reset 전 마지막 Critic 상태로 bootstrap하고 다음 에피소드 GAE는 연결하지 않는다. 구역 책임이 지속되는 과정의 시간 절단이라는 규약이다. 향후 진짜 유한 임무의 정상 종료로 정의한다면 bootstrap=0인 다른 프로토콜을 별도로 고정해야 한다.

Coverage Gap Ratio는 기존 평가기를 그대로 사용한다. 가동 가능하고 구역 반경 안에 있으며 해당 구역 또는 대기 상태인 로봇을 용량 1의 매칭으로 배치한다. 다른 작업에 이미 할당된 로봇은 같은 순간의 구역 대응자로 세지 않는다. 단순 담당자 이름만 있는 상태는 물리적 커버가 아니다. 반대로 담당자 이름이 없어도 그 위치의 대기 로봇은 커버 가능하다. 실제 접촉·자세·제동·통신 반응 시간까지 판정하는 실물 지표는 아니다.

## 관측 공정성

주 비교군은 동일한 Phase 3A Actor 패킷을 사용하는 다음 정책들이다.

- `local_hold`: 현재 할당을 유지한다. 새 작업을 받지 않는 통제군이다.
- `local_fixed`: 자기 관측의 기본 역할을 유지한다. guard는 알려진 구역, worker는 알려진 작업을 슬롯 순서로 처리한다. 다른 로봇 역할이나 사전의 전역 fixed_owner 정보는 받지 않는다.
- `local_eta`: 관측한 작업의 우선순위와 자기 ETA를 사용한다. 신선한 peer 패킷의 위치·속도로 다른 로봇의 도달시간을 대략 추정한다. peer의 작업 할당·회전 속도는 제공되지 않아 사용하지 않는다. 동일 정보에서 모든 로봇의 합의가 보장되는 분산 최적 할당기는 아니다.
- `local_random`: 현재 유효 행동 중 균등 샘플링한다.
- MAPPO: 동일 패킷의 146차원 벡터와 action mask를 사용한다. Actor 추론 함수는 global state나 privileged infos를 인자로 받지 않는다.

로컬 기준선은 관측한 job ID/slot, 자신의 controller 할당과 기본 역할을 사용한다. 공개 로봇 ID는 ETA 동률 해소에 사용한다. 미래 작업 명단, 미수신 메시지, peer Ground Truth를 받지 않는다. MAPPO는 job ID 문자열을 입력받는 대신 동일 로컬 슬롯의 특징을 받는다.

별도 참고군 `central_fixed/greedy/exact/handover`는 **기존 Phase 2 정책 그대로**다. 현재 팀 전체 상태와 활성 작업, 작업 fixed_owner, 실제 고장·속도 등을 중앙에서 안다. D에서도 로컬 센서 잡음과 무선 지연을 받지 않는다. 따라서 로컬 관측군과 동일 정보의 알고리즘 비교가 아니며, 추가 정보를 허용한 참고군이다. 수학적인 성능 상한을 보장하는 정책은 아니다.

`central_handover`는 기존 준비 행동도 사용할 수 있다. Standard MAPPO의 10개 행동에는 이 준비 행동이 없다. 따라서 이 참고군과의 차이에는 **관측 정보와 행동 표현 차이가 함께 포함**된다. 신규 인계 학습을 구현한 것은 아니다. 중앙 참고군의 보상 Return은 계산하지 않고 `null`로 명시했다. 평가 지표는 같은 Ground Truth 평가기를 사용한다.

## 학습·보상 통제

기존 146차원 Actor, 141차원 Critic, 10개 행동, 4개 로봇 패딩/8개 작업 슬롯을 유지했다. 작업이 나타나면 해당 로봇의 로컬 슬롯 뒤에 추가되고 기존 슬롯은 재번호화하지 않는다. 코드는 실제 로컬 job ID로 정확히 디코딩한다. 단, flat MLP가 특정 슬롯 번호에 과적합하는 문제와 구현상의 잘못된 인덱스 연결은 서로 다르다.

PPO clipping, GAE, 활성 로봇만의 advantage normalization·Actor loss, 팀 상태 한 개당 Critic loss, entropy, gradient clipping과 optimizer는 Phase 3B-1을 재사용한다. 같은 batch를 기존 Learner와 새 PoolLearner에 넣었을 때 손실과 갱신된 모든 파라미터가 정확히 같음을 테스트했다. 수집기는 설정 풀을 32초 단위로 순환하며 원래 샘플링 행동/log-probability/mask와 중재 결과를 구분한다.

보상은 기존 Phase 3A 구현을 그대로 쓴다. 진행·완료·첫 대응·기한 만료·대기·책임 공백·역할 변경 항을 Ground Truth/이벤트에서 계산한다. 평가기는 보상과 별개다. 기본 가중치는 progress=1, completion=10, response=1, failure=10, waiting=1, coverage=2, switching=0.1이다. 실패/대기/공백/전환 항 자체는 음수다. 상세 식은 기존 `PHASE3B1_README_KO.md` 및 변경하지 않은 `phase3a/rewards.py`에 있다.

민감도 실험은 **coverage 계수만 2와 8**로 바꾼다. 알고리즘 변경이 아니다. 신경망 초기값, 학습 시드, 시나리오 순서, PPO 설정과 예산은 같다. 첫 rollout의 관측·행동·log-probability는 같고 보상만 달라짐을 테스트했다. 서로 다른 보상 설정의 Return 수치로 알고리즘 순위를 매기지 않는다.

## 사전 계획과 Test 차단

`predeclared_plan.json`을 학습 전에 저장했다. A는 두 보상 설정 각각 8,192스텝, B3는 기본 A의 선택된 Validation 성공률이 75% 이상일 때 4,096스텝, C는 B3 성공률 60% 이상일 때 2,048스텝이다. D는 학습하지 않는다. 최대 예산 22,528스텝을 실제로 지켰다.

신경망은 학습 시드 하나로 시작하며 256스텝 rollout, 4 epochs, minibatch 64, CPU 1스레드를 사용한다. Validation은 1,024스텝마다 결정적으로 평가한다. 선택은 성공률 우선 → 동률이면 공백이 작은 것 → 다시 동률이면 제한 응답시간이 작은 것의 사전 정의 순서다. 가중합 종합점수는 없다. 이 선택 규칙도 응답/공백의 가치 판단을 포함하므로 전체 성능 순위와 동일시하지 않는다.

모든 학습과 선택이 끝난 뒤 `selection_freeze.json`에 시나리오/설정/선택된 checkpoint 해시를 저장했다. 그 뒤 Test를 평가한다. Test 결과를 보고 가중치를 조정하거나 추가 학습하지 않았다. 사후 진단 파일에는 사후 분석임을 명시했다. 다음 단계에서 지금 Test를 개발에 활용한다면 이 집합은 더 이상 최종 holdout으로 부르면 안 된다.

## PowerShell 실행

이미 승인된 `.venv`를 사용한다. 새 패키지 설치는 필요하지 않다. 새 출력 경로가 이미 존재하면 덮어쓰지 않고 중단한다.

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim\research\cah_mappo
$taskPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'

& $taskPython -B -m unittest discover -s tests -v

# 새 예비 실험 전체 재현: 최대 22,528스텝. 기존 pilot_v1을 덮어쓰지 않는다.
& $taskPython -B phase3b2a.py --output phase3b2a_outputs\my_pilot

# 기존 실제 결과를 읽는 감사
& $taskPython -B verify_phase3b2a.py --run phase3b2a_outputs\pilot_v1 --output phase3b2a_outputs\my_audit
```

실제 결과는 `phase3b2a_outputs/pilot_v1/`에 있다. 각 run에 전체 학습 에피소드 요약, 첫 rollout, 초기/중간/최종 checkpoint, Validation 개별 결과, update별 손실, CSV/JSON 곡선을 저장했다. 일부 학습 원시 궤적과 **전체 Test 원시 궤적**을 저장했다. 모든 학습 스텝의 전체 raw rollout을 저장하는 구성은 아니다.

체크포인트의 임의 시점 학습 resume 한계는 Phase 3B-1과 같다. 가중치·optimizer·RNG는 저장하지만 진행 중 환경과 시나리오 커서는 완전 직렬화하지 않는다. 평가 복원은 검증했다. 다음 대규모 실행 전 정확한 resume 기능은 별도 보완 대상이다.
