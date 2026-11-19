# Phase 2 — 비학습 할당 기준선과 다중 로봇 확장

Python 3.10 이상 표준 라이브러리만 사용한다. Phase 1의 파일·출력은 수정하지 않았으며, 새 모듈과 새 출력 경로만 추가했다. MAPPO/CAH-MAPPO, 외부 라이브러리, ROS·실물·원격 통신은 사용하지 않는다.

## 실행

모듈 루트 `C:/Users/jes/Desktop/MX3_Match_Sim/research/cah_mappo`에서:

```powershell
python -B -m unittest discover -s tests -v
python -B phase2.py run --scenario single --policy handover --output phase2_outputs/my_run.jsonl
python -B phase2.py evaluate phase2_outputs/my_run.jsonl
python -B phase2.py suite --seeds 0 1 2 3 4 --jitter 0.15 --output phase2_outputs/my_paired_suite
python -B phase2.py evaluate outputs/phase1_final/normal__ready.jsonl
```

출력 파일·suite 폴더는 새 이름을 써야 한다. 기존 자료는 덮거나 지우지 않는다. pip install 없이 실행한다. 검증에 사용한 Python은 기존 `.analysis_env/Scripts/python.exe` 3.12.14다.

Phase 1의 `run.py`는 그대로 schema 1을 처리한다. 새 `phase2.py evaluate`는 schema 1을 기존 평가 함수에 위임하고 schema 2를 확장 평가기로 처리한다. **옛 run.py가 새 로그 형식을 지원하도록 바꾼 것은 아니다.**

사용자 설정 예:

```powershell
python -B phase2.py run --config configs/phase2_four_robots.json --policy exact --output phase2_outputs/custom_new.jsonl
```

`--config` 사용 시 환경 설정은 파일이 기준이고 `--scenario/--seed/--jitter`는 적용하지 않는다. 정책은 `--policy`로 선택한다.

## 새 파일과 주요 함수

- `src/cah_mappo/phase2/model.py`: RobotSpec, JobSpec, EventSpec, Config, RobotView, JobView, Observation, Action. `eta()`와 JSON 설정 변환.
- `policies.py`: Policy 프로토콜, FixedRole, GreedyETA, ExactAssignment, RuleBasedHandover, 내부 할당 비용 함수.
- `engine.py`: Environment의 사건·이동·서비스·관측·실행 중재·원시 로그 생성.
- `metrics.py`: 버전별 evaluate와 용량 제약을 반영한 물리적 구역 커버 판정.
- `scenarios.py`: 공통 외생 사건 시나리오 8개.
- `phase2.py`: 단일 실행, 비교 suite, 독립 로그 재계산 명령.
- `tests/test_phase2.py`: 추가 단위 테스트. 기존 test_phase1.py는 그대로다.
- `report_phase2.py`: 저장된 측정값으로 한국어 결과 문서를 생성한다.

## 환경 정의

- 로봇 2~4대, 2D 연속 좌표, 로봇별 선속도·회전속도·가용 상태.
- 지속 책임 구역(base)과 유한 서비스 작업(task)을 구분한다. 지속 구역도 여러 개를 표현할 수 있다.
- 작업마다 위치·우선순위·발생시각·마감·서비스 시간·고정 정책 담당자를 정의한다.
- 이번 엔진의 용량은 **각 로봇당 정확히 1개 서비스**다. capacity=2 이상의 동시 수행은 아직 지원하지 않고 설정 오류로 거부한다. 서로 다른 위치의 작업을 무제한 동시 수행하는 모델이 아니다.
- 작업 마감은 hard deadline이다. 마감에 서비스가 완료되면 성공, 미완료면 만료다. 마감 이후 서비스를 계속하여 tardiness를 재는 모델은 아니다.
- 위치에 도착해 배정된 작업을 실제 서비스하는 때만 응답으로 기록한다. 배정 자체는 응답이 아니다. 서비스 중에는 이동하지 않는다.
- 같은 구역에 free 로봇이 있어도 한 로봇을 여러 구역에 중복 계수하지 않는다. 구역 위험도 합이 최대인 용량 1 매칭으로 물리적 커버를 계산한다.
- 로봇이 고장 나면 배정이 해제되고 움직임·서비스가 정지된다. 복구나 속도 변경을 사건으로 표현할 수 있다.
- 기본 dt=1초, horizon=24초. 사건·서비스 시간은 dt의 배수다. 회전 후 남은 시간에 직선 이동하고, 도착은 경계에서 판정하므로 dt 규모의 이산화 오차가 있다.

## 공통 Policy 인터페이스

```text
reset(seed) -> None
decide(Observation) -> Action
```

Observation은 현재 시각, 관측 시각, 로봇 상태, 공개된 활성 작업만 담은 불변 dataclass다. 미래 작업과 미래 고장·위험도 변화 목록은 정책에 주지 않는다. 모든 정책은 같은 관측 구조를 받는다.

Action은 로봇→작업 배정, 준비 중 겹침 허용 쌍, 설명 notes, 선택적 할당 후보 비용을 담는다. 작업 위치가 이동 목표다. 임의 모터·속도 명령 인터페이스는 없다.

현재는 **팀 전체를 보는 중앙 계획 기준선**이다. 관측 지연 시 전체 팀 상태를 과거 스냅샷으로 제공한다. 이는 분산 메시지 수신·손실·ACK 프로토콜의 완전한 구현이 아니며, 기존 Phase 1의 요청/READY 전송 모델도 별도로 보존돼 있다. Phase 3에서 decentralized actor 관측을 별도로 정의해야 한다.

## 네 기준선

### Fixed Role (`fixed`)

초기 구역 담당자는 계속 해당 구역을 지킨다. 작업자는 작업별 fixed_owner로 정해진 자기 작업을 FIFO로 수행한다. 진행 중인 자기 작업은 유지한다. 고장난 구역 담당자를 다른 역할 로봇으로 대체하지 않는다.

동적 작업이 발생해도 모든 로봇을 무조건 정지시키는 정책은 아니다. 기본 역할·고정 담당표를 유지하는 기준선이다. 고정 작업 담당표는 시나리오 설정이며 모든 정책의 활성 작업 관측에서 똑같이 제공된다.

### Greedy / ETA (`greedy`)

현재 우선순위가 높은 작업부터 정렬한 뒤, 아직 배정되지 않은 행동 가능한 로봇 중 ETA 최소를 선택한다. 동률은 로봇 ID다. 전역 최소 비용을 보장하지 않는다.

ETA는 `(거리-허용 반경)/선속도 + 방향 변화/회전속도`를 시뮬레이션 tick으로 올림한 근사다. 기존 strategy.py의 이동·회전 시간 분리 원리를 참고했지만 기존 파일은 변경하지 않았다.

### Exact Assignment + Switching Cost (`exact`)

idle을 포함한 유효한 배정을 전부 탐색한다. 로봇별 1개, 작업별 1명의 제약을 지킨다. **Hungarian 알고리즘이 아니다.**

내부 비용은 다음과 같다.

- 배정된 작업: `priority * (ETA + 남은 서비스 시간)`.
- 마감 초과 예상 배정: 추가 `priority * unassigned_cost`.
- 미배정 작업: `priority * unassigned_cost`.
- 활성 작업에서 다른 작업 또는 idle로 바꾸는 로봇: switching_cost 추가.
- 구역 priority는 현재 위험도이고 지속 구역의 서비스 시간 항은 0이다.
- 기본 switching_cost=2, unassigned_cost=50이며 config에 공개했다.

이 비용은 **그 시점 할당의 목적함수**이며 전체 실험에서 정책 우열을 정하는 종합점수가 아니다. 완전한 미래 일정 최적화도 아니며, 이 비용 선택에 따른 민감도와 휴리스틱 ETA 오차가 남아 있다.

### Rule-Based Handover (`handover`)

동일한 Exact 할당기에 다음 실행 규칙을 더했다.

- 기존 구역 담당자가 새 작업을 맡고 후임이 구역을 맡는 배정이면, 기존 담당자는 구역을 지키고 후임부터 이동시킨다.
- 후임이 준비되면 책임을 넘긴다. 대기 중 위치 변화로 같은 비용의 배정이 뒤집히지 않도록 인계 계획을 유지한다.
- 후임을 기다리면 마감이 위협받고 새 작업 우선순위가 구역 위험도 이상이면 조기 해제를 허용한다.
- 인계 중 구역 위험도가 상승해 새 작업보다 중요해지면 보류 중인 인계를 취소하고 다시 할당한다.
- 후임·기존 담당자가 고장 나거나 작업이 끝나면 보류 계획을 폐기한다.

준비 판단은 다른 정책과 동일한 관측을 사용하므로 지연된 정보로 잘못 해제할 수도 있다. 새 학습 정책이나 고장에 대한 완전한 해결책이 아니다. 로그의 allocation_cost는 **게이트 적용 전 Exact 후보 비용**이며 최종 겹침 배정을 채점한 값이 아니다.

## 공통 실행 중재와 중복

작업 중복 요청은 로봇 ID의 결정적 순서로 1명만 수락한다. 비가용 로봇·만료/미발생 작업의 요청도 거부한다. 이 중재는 모든 정책에 같다.

구역 소유자가 남아 있는 상태에서 후임이 준비하는 의도적 겹침은 별도 preparations로 허용한다. 이는 불필요한 중복 배정과 구분한다.

평가에는 요청 중복률과 실행된 불필요 중복률을 따로 기록한다. 실행된 중복률이 0인 것은 상당 부분 공통 중재의 보장이므로 정책이 중복 문제를 학습·해결했다고 주장하지 않는다.

## 비교 시나리오

- single: 2대·단일 작업.
- simultaneous: 4대·작업 3개 동시 발생.
- priorities: 같은 4대 구성에서 작업 우선순위와 마감 차이.
- slow_robot: t=3에 C의 이동 속도가 1에서 .25m/s로 저하.
- owner_failure: t=1에 초기 구역 담당자 A 고장.
- urgent: t=2 발생 작업의 마감이 t=8.
- rising_risk: 인계를 준비하는 중 t=3에 구역 위험도 1→10.
- communication_delay: 공개 관측 스냅샷에 1초 지연.

초기 위치·작업 위치의 변동은 exogenous 전용 RNG로 정책 실행 전에 모두 생성한다. 정책 RNG는 별도 seed 공간을 사용한다. 동일 시나리오/seed/config에서는 네 정책의 exogenous_plan이 동일하다. 정책마다 RNG 소비를 다르게 해도 외생 사건이 바뀌지 않는 테스트를 추가했다.

## 지표 정의

모든 수치는 reward가 아닌 원시 GT와 사건 로그에서 재계산한다. reward 자체를 구현하지 않았다.

- `coverage_gap_ratio`: 위험도 가중 미커버 구역·시간 / 전체 위험도 가중 구역·시간. 구역 1개·위험도 1이면 Phase 1 CGR과 같다.
- `coverage_gap_s`: 실제 미커버 구역·초 합. 여러 구역이면 horizon보다 클 수 있다.
- `unweighted_coverage_gap_ratio`: 위험도 가중치를 제외한 보조 지표.
- `handovers`: 책임 해제부터 지정 후임의 실제 대응까지 초기 공백. 관측 중 다음 해제 사건이나 종료가 오면 검열한다. 후임이 지정되지 않은 고장 해제는 구역 커버 회복까지 계산한다.
- `physical_gap_union_s`: 해당 해제부터 같은 구역의 다음 해제/종료까지 실제 공백 구간 합. 사건 간 이중 귀속을 피한다.
- `response_time_mean_s`: 실제 서비스가 시작된 작업의 응답 지연 평균. 무응답 작업은 별도 수와 restricted_response_mean_s를 함께 본다.
- `task_success_rate`: 마감 내 완료 수/전체 발생 작업 수.
- `deadline_miss_rate`: 마감 내 완료되지 않은 작업 수/전체 작업 수. hard deadline이므로 이후 지각 완료 시간은 따로 없다.
- `all_tasks_makespan_s`: **모든 작업이 성공한 경우에만** 마지막 완료 시각−첫 작업 발생 시각. 하나라도 실패하면 null이다.
- `last_successful_completion_at`, `completed_task_flow_time_sum_s`: 완료된 작업만의 보조 시간이며 실패율과 함께 보고한다.
- `role_switch_count`: 로봇의 guard↔worker 서비스 역할 전환 수. idle은 역할 변경으로 세지 않는다. 고정 base_role 라벨 자체를 바꾼 횟수는 아니다.
- `assignment_change_count`: 완료/고장 해제까지 포함한 배정 변경 수.
- `task_duplicate_rate`: 활성 작업·구역 시간 중 의도적 준비 겹침을 제외한 실행 중복 비율.
- `requested_duplicate_rate`: 같은 분모에 대한 요청 단계의 중복 비율.

성공률·공백·응답 지연을 각각 보고하며 단일 종합점수나 순위는 만들지 않는다. 응답 평균에서 빠지는 무응답 작업 수, makespan이 정의된 실행 수를 숨기지 않는다. 준비된 후임이 다시 떠나거나 고장나는 경우는 초기 공백만으로 평가하지 않는다.

## 로그와 검증 자료

schema 2 header에는 전체 설정, 정책 이름, 실제 외생 계획, 새 모듈별 소스 SHA256이 있다. 매 tick에는 policy_selection(실제 입력 관측·요청·수락·거부·준비·설명), 실제 위치·속도·가용 상태·배정·작업 진행량을 기록한다. 미래 사건은 header/GT에만 있고 정책 관측에는 없다.

- `phase2_outputs/preservation_before.json`: 작업 시작 전 Phase 1 파일들과 기존 사용자 수정 파일의 SHA256.
- `test_results_final.txt`: 기존 26개 + 새 29개 = 55개 테스트 실제 출력.
- `final_seed0`: 위치 변동 없는 8×4=32개 해석적 비교 실행.
- `paired_5seeds`: seed 0~4, 위치 변동 ±.15m의 8×4×5=160개 비교 실행.
- 각 suite의 summary.json: 개별 작업·인계까지 포함한 수치.
- 각 suite의 aggregates.json: 지표별 mean/min/max/정의된 실행 수. 통계적 유의성이나 학습 시드 신뢰구간이 아니다.
- `validation_record_final.json`: 실행/재평가/보존 검증 기록.
- `diagnostic_seed0`: 개발 중 최초 진단 자료이며 최종 비교 결과로 사용하지 않는다. 기존 출력 보호 원칙에 따라 보존했다.

## 남은 한계와 Phase 3 준비

지금은 용량 1, 소규모 완전탐색, 중앙 팀 관측, 직선 이동·순차 회전, 결정적 서비스다. 충돌·경로 장애물·에너지·실제 무선 패킷·센서 잡음 분포·로봇 동역학은 없다. 위치 변동을 넣은 5개 seed는 비학습 시나리오 반복이며 독립 학습 seed가 아니다.

Phase 3 전에 다음 계약을 확정해야 한다.

1. 현재 run 루프에서 reset/step 기반 학습 환경으로 분리. 시뮬레이터와 평가기는 유지하되 이를 이번 단계에서 구현하지는 않았다.
2. actor별 자기 상태·수신 팀원 상태·현재 보류 인계를 정의하고 centralized critic 입력과 분리. 현재 전체 Observation을 decentralized actor에 그대로 주면 안 된다.
3. 작업·로봇 수 패딩, action mask, 고장 로봇의 inactive mask, 중복 중재 후 실제 실행 action과 log-prob 처리 원칙 확정.
4. reward 설계는 별도이며 로그 평가 지표와 분리. Standard MAPPO에도 필요한 공통 정보/보상을 주고 CAH 전용 상태·해제 행동의 효과를 분리해야 한다.
5. 훈련/검증/평가 시드 및 사건 계획 분리. 다음의 후보 의존성은 별도 환경에서 승인 후 설치 검토: NumPy·PyTorch, Gymnasium/PettingZoo 어댑터를 채택한다면 해당 패키지. 기존 RSL-RL 보행 학습기를 MAPPO라고 재사용하지 않는다.
6. 공식 MAPPO 구현을 재현할지 독립 구현할지 정하고 버전·라이선스·버퍼·재귀 정책·종료/시간제한 처리를 검토한다.

이번 단계에서 Phase 3 코드나 의존성을 추가하지 않았다. 커밋·푸시도 수행하지 않는다.
