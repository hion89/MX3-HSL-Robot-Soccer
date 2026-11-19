# Phase 3A — CTDE 환경 코어와 검증 계약

학습 알고리즘 구현이 아니다. Phase 1·2 파일은 변경하지 않고 phase3a 패키지, CLI, 테스트, 문서 및 별도 출력만 추가했다. Python 3.10 이상 표준 라이브러리로 실행한다. 이번 검증은 기존 Python 3.12.14 분석 환경에서 했고 설치·환경 변경·서버 접속·실물 구동·커밋/푸시는 하지 않았다.

## 실제 데이터 흐름과 재사용

기존 phase2/model.py의 Config/Action, engine.py의 boundary/apply/integrate/update_owners/truth, metrics.py의 독립 평가를 재사용한다. 기존 정책과 run 루프는 수정하지 않았다.

기존 engine.observe()는 전체 팀 상태를 보여주므로 Actor에 쓰지 않는다. 새 ParallelCore가 한 구간씩 실행하고, 센서/전송 모듈과 캐시 전용 Actor 생성 경로를 따로 사용한다.

```text
실제 상태 → 센서 범위/측정 잡음 → 로봇별 캐시 → Actor observation/mask
로컬 센서와 자기 상태 → 송신 큐 → 실제 도착/수신 → 수신 캐시
실제 현재 상태 → 별도 Critic 생성기 → Global State
샘플 행동 → 검증 → 공통 실행 중재 → 기존 물리·사건 갱신
원시 GT/사건 → 학습 보상(별도 계산) 및 저장 로그 → 독립 논문 지표 평가
```

## 실행

모듈 루트에서 실행한다. 출력 이름은 새 것을 사용한다.

```powershell
cd C:\Users\jes\Desktop\MX3_Match_Sim\research\cah_mappo
python -B -m unittest discover -s tests -v
python -B phase3a.py inspect
python -B phase3a.py smoke --output phase3a_outputs/my_smoke --seeds 0 1
python -B phase3a.py goal-demo --output phase3a_outputs/my_goal.jsonl
python -B phase3a.py evaluate phase3a_outputs/my_goal.jsonl
```

사용 가능한 기존 Python 경로는 `C:/Users/jes/Desktop/isaac_ws/sim2real_evidence/.analysis_env/Scripts/python.exe`다. 새 라이브러리나 venv가 필요하지 않다. 로그 생성은 기존 write_jsonl의 배타적 파일 생성을 사용한다.

## API

`ParallelCore(config, settings, weights)`:

- `reset(seed, options) -> (observations, infos)`.
- `step(actions) -> (observations, rewards, terminations, truncations, infos)`.
- `state()`: critic용 현재 Global State의 독립 복사본.
- `log_rows()/export_log(path)/metrics()`: 완료된 에피소드의 로그와 독립 평가.
- `agents`: 에피소드에 존재하는 로봇. 고장 시 제거하지 않고 에피소드 종료 때 비운다.
- `possible_agents`: 공개 roster의 최대 4개 슬롯.
- reset options는 config 교체와 terminate_on_tasks_resolved만 허용한다. reset은 이전 에피소드 메모리를 버리므로 필요한 완료 로그는 먼저 export한다.

관측·정보는 독립 복사본이다. info에는 훈련/디버깅용 critic 정보가 있으며 privileged_info=true다. **info/state/log를 Actor 입력으로 붙이지 않는다.** Actor 입력은 observations[agent]의 지정 필드뿐이다.

## Actor와 Critic

기본 상한은 로봇 4, 에피소드 전체 작업/구역 8이다. 더 많으면 설정을 거부한다. 개수와 관계없이:

- Actor 수치 벡터 146, 행동 마스크 10.
- Critic 수치 벡터 141.
- 실제 로봇이 2대이면 나머지는 배치 어댑터에서 0 패딩한다. 패딩 행동 마스크도 NOOP 한 개는 유효하다.

Actor의 자기 상태는 이상적인 자기 오도메트리/컨트롤러 피드백으로 가정한다. 동료·작업은 기본 반경 5m에서 관측하며 위치 측정에 ±.02m 잡음을 준다. 현재 작업 상태·중요도·마감·남은 서비스가 현장에서 읽힌다는 단순 센서 가정이다. 실물 비전이 이 정보를 모두 제공한다는 주장은 아니다.

무선은 자기 상태와 **직접 관측한 작업만** 전송한다. 받은 동료 정보를 다시 전달하거나 보이지 않는 GT를 송신하지 않는다. 기본 전송 지연은 2초, 캐시 TTL은 4초다. smoke의 communication_delay 조건은 전송 지연 3초를 사용한다. 기존 Config.observation_delay는 Phase 2의 중앙 관측 지연 필드이며 새 Actor 생성에는 쓰지 않는다. Phase 3의 지연은 Settings.radio_delay로 명시한다.

Actor는 미수신 메시지·미발생 작업·미래 사건·다른 로봇의 비가시 GT를 받지 않는다. 알려진 작업도 오래되면 fresh mask가 꺼진다. 숨은 작업이 종료됐다고 캐시나 action mask가 자동 갱신되지는 않는다.

작업 슬롯은 **각 Actor가 실제로 알게 된 순서**로 부여하고 에피소드 동안 재사용하지 않는다. 전역 작업 목록의 빈 슬롯 위치로 숨은 작업 수가 누출되는 것을 피한다. 공개 roster 마스크, 측정의 신선도, 알려진 작업 존재 마스크를 구분한다. job_ids는 Actor가 이미 아는 로컬 매핑을 위한 디버깅 필드다.

Critic은 모든 현재 로봇의 실제 위치·가용 상태·배정과 발생한 모든 작업의 현재 GT를 받는다. 미발생 작업·미래 사건 목록은 넣지 않는다. Critic의 작업 슬롯은 전역이며 Actor 슬롯과 다르다. 현재 벡터는 **물리적 전역 상태 표현**이다. 전체 메시지 큐/로봇별 belief를 모두 담은 엄밀한 Markov 상태는 아니다. 통신까지 포함한 가치 추정에 RNN/추가 상태가 필요한지는 Phase 3B에서 결정해야 한다.

벡터 정의는 perception.py에 명시했다. 자기 상태·자기 배정 one-hot, 동료 상대 위치/측정 나이, 작업 상대 위치·마감·남은 서비스·나이 및 패딩/신선도 마스크를 이어 붙인다. Actor 크기는 50+12*max_jobs, Critic 크기는 37+13*max_jobs다.

## 이산 행동과 마스크

- 0: 대기/현재 배정 해제. 항상 유효한 NOOP다.
- 1: 현재 자기 배정 유지. 자기 컨트롤러에 배정이 있을 때만 유효하다.
- 2~9: Actor 자신의 알려진 작업 슬롯 0~7 선택.

선택 마스크는 로컬 캐시의 활성 상태·신선도와 자기 가용성으로만 만든다. 다른 로봇의 실제 배정·보이지 않는 작업 종료로 oracle mask를 만들지 않는다. 현재 자기 배정은 캐시가 오래돼도 KEEP할 수 있다. 실패 로봇은 0만 유효하다. 유효하지 않은 숫자나 마스킹된 선택은 모든 환경 변이 전에 오류로 거부한다.

## 중재와 PPO 기록

공통 중재는 **환경 전이의 일부**다. Standard 행동은 별도 준비 겹침 명령 없이 Phase 2 Action으로 변환한다. 겹치는 작업 요청, 만료 작업, 비가용 요청은 기존 apply()에서 중재한다.

PPO용 SampledAction에는 행동, 당시 mask, 행동 분포, log-probability, policy_version, observation_token이 필요하다. 토큰은 공개된 Actor 관측/에이전트/시각에서 만들며 숨은 GT를 해시하지 않는다.

- 분포 합=1, 마스킹 행동 확률=0, 선택 행동 확률>0, log(p[a]) 일치를 검증한다.
- 마스크가 같아도 다른 시각/다른 로봇의 샘플은 토큰으로 거부한다.
- transition에는 sampled_action과 log_probability/action_mask/분포/Actor 입력을 보존한다.
- executed_assignment/executed_action/arbitration_changed는 별도다.
- 중재로 선택 작업이 거절돼도 PPO에서 쓰는 행동은 **원래 sampled_action**이다. 실행된 NOOP의 log-probability로 바꾸지 않는다.
- 정수 행동은 스크립트 실행만 가능하고 log-probability를 만들지 않는다. validate_ppo_record는 이를 학습 표본으로 인정하지 않는다.

기록 검사는 호출자가 제출한 분포와 입력의 구조적 일관성을 확인한다. 실제 신경망이 같은 weight/RNN 상태에서 그 분포를 냈는지는 아직 검증하지 않았다. Phase 3B에서는 같은 forward에서 행동·확률·log-probability를 얻어야 하며, RNN을 한 번 더 진행해 분포를 재생성하면 안 된다.

## 보상: GT 기반 shared team reward, 평가와 독립

max_jobs=J=8, 팀 로봇 수 N, 계획 시간 H, 구간 dt를 사용한다. 미래 작업 수에 따라 보상 크기가 달라지지 않도록 실제 총 작업 수 대신 고정 J를 분모로 쓴다.

- progress: 구간의 실제 서비스 진행량/각 작업의 필요 서비스 합계를 J로 나눔. 기본 가중치 +1.
- completion: 해당 구간 완료 사건 수/J. +10.
- response: 해당 구간 유효 서비스 시작 사건 수/J. +1.
- failure: 해당 구간 만료 사건 수/J. -10.
- waiting: 구간 시작의 활성 작업 priority를 각각 min(priority,10)/10으로 제한해 합산하고 J로 나눈 후 dt/H를 곱함. -1.
- coverage: 구간 시작의 실제 용량 제약 커버에서 위험도 가중 공백 비율 * dt/H. -2.
- switching: 정책 적용으로 이전 비어 있지 않은 배정을 바꾸는 수/N. -.1. 완료/고장에 의한 자동 해제는 제외한다.

동일한 팀 보상을 실제 에이전트들에 전달한다. 실패 로봇도 팀 보상은 받지만 정책 손실 마스크는 0이다. 배치의 로봇 수만큼 보상을 합쳐 팀 보상 크기를 늘리지 않는다.

reward_terms/가중 기여/합계를 매 transition에 기록한다. 보상은 reward 모듈의 자체 GT 판정으로 계산하고 evaluate 결과를 참조하지 않는다. weights를 바꿔도 같은 행동의 논문 지표가 같다는 테스트를 했다. coverage만 최소화하도록 설계한 것이 아니며, 고정 목표 서비스가 정지 정책보다 높은 보상을 받는 통제 사례를 검증했다. 이것으로 모든 상황의 reward hacking이 해결됐다는 뜻은 아니다.

가중치는 초기안이며 최적값이 아니다. 매 단계 절댓값 100, 에피소드 절댓값 누적합 1000을 기본 경보 한도로 두고 NaN/무한대/초과를 오류로 검출한다. 오류가 난 에피소드는 이어서 학습하지 말고 폐기·진단해야 한다.

## 종료·고장·return 의미

기본 모드는 계획 horizon까지 구역 책임을 유지하는 continuing-coverage 관측 구간이다.

- horizon 도달: terminated=false, truncated=true.
- 옵션 terminate_on_tasks_resolved=true에서 모든 동적 작업이 완료/만료되고 미발생 작업도 없으면: terminated=true, truncated=false. 미발생 작업이 남으면 조기 종료하지 않는다.
- 작업이 없는 경우에는 horizon까지 진행한다. 고장 자체는 에피소드나 개별 에이전트를 종료시키지 않는다.

기록의 bootstrap_mask는 진짜 종료에서 0, 시간 제한에서 1이다. continuation_mask는 종료/시간 제한 모두 0이므로 다른 에피소드의 advantage를 이어 붙이지 않는다. 예를 들어 r=1, gamma=.9, 실제 최종 상태 가치가 5이면 time-limit target은 5.5, true-terminal target은 1이다. 실제 return/GAE 학습기는 구현하지 않았다.

final_observation/final_state를 보존한다. bootstrap에 자동 reset 이후의 새 초기 상태를 쓰면 안 된다. 수명이 유한한 workload를 연구 목표로 택한다면 종료 옵션과 가치 목적을 먼저 고정해야 한다. 기본 continuing-coverage와 finite-workload를 섞어 비교하지 않는다.

고장 다음 관측은 active_mask=0이고 NOOP만 가능하다. 고장 후 복구하면 다시 1이다. transition의 policy_loss_mask는 **행동을 선택한 시점**의 가용성이다. critic_loss_mask는 실제 로봇이면 실패 상태에서도 1이며, 없는 패딩 슬롯은 present mask=0이다. 공유 critic 표본을 로봇 수만큼 중복 가중하지 않는 처리는 runner에서 필요하다.

RNN reset 표시는 고장/복구/에피소드 경계에서 켠다. critic return의 bootstrap/continuation과 RNN reset은 서로 다른 마스크다. 고장으로 정책을 멈췄다고 팀 가치의 return을 끊지 않는다.

## 로그·평가·검증 한계

schema 2를 유지해 기존 확장 평가기가 그대로 읽는다. actor 관측·원래 행동·중재 결과·보상·전역 상태는 별도 레코드다. 조기 자연 종료 시 평가 horizon은 실제 관측 구간으로 바꾸고 planned_horizon은 원래 값으로 보존한다. 이 로그를 재현 설정으로 쓸 때 planned_horizon을 복원해야 한다.

최종 결과는 PHASE3A_RESULTS_KO.md와 phase3a_outputs의 complete 검증 기록을 참고한다. smoke_final/smoke_bound 및 이전 test_results는 개발 단계 자료로 보존했고 완료 보고의 최종 결과와 구분한다.

선택적 make_pettingzoo()는 호출할 때만 NumPy/Gymnasium/PettingZoo를 import한다. 현재 설치하지 않았고 공식 parallel_api_test는 실행하지 않았다. 코어의 반환 형태·패딩·mask 검증을 통과한 것이 PettingZoo 배포 호환성의 인증은 아니다.

실제 torch forward/optimizer/rollout buffer/학습은 없다. 공식 구현 연결 검토와 Phase 3B 전제는 MAPPO_CONNECTION_REVIEW_KO.md에 정리했다.
