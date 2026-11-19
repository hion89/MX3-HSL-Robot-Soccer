# Phase 3B-1: CPU Standard MAPPO 최소 학습 기반

2026-11-18. Phase 3A의 2D 환경에 실제 PyTorch Actor/Critic과 PPO 업데이트를 연결했다. 이번 구현은 작은 시나리오에서 학습 데이터와 최적화가 정상 동작하는지 확인하기 위한 것이다. CAH-MAPPO, 본학습, 실물 보행·킥 학습은 포함하지 않는다.

## 파일과 연결 구조

기존 Phase 1~3A 파일은 수정하지 않았다. 신규 파일은 모두 `research/cah_mappo/` 안에 있다.

- `src/cah_mappo/phase3b1/math_utils.py`: 표준 라이브러리 기반 GAE, return 및 종료 마스크 검증.
- `src/cah_mappo/phase3b1/networks.py`: 별도 Actor/Critic MLP, 마스킹된 Categorical 분포, 행동 샘플링.
- `src/cah_mappo/phase3b1/learner.py`: `TrainConfig`, `minimal_env`, `Learner.collect/tensors/update/minibatch/checkpoint`.
- `src/cah_mappo/phase3b1/experiment.py`: 공통 조건 평가 `evaluate_policy`, 집계 `aggregate`, 제한된 실험 `run`.
- `src/cah_mappo/phase3b1/__init__.py`: 독립 패키지 진입점.
- `phase3b1.py`: CPU 최소 학습 CLI. 한 실행은 최대 16,384 환경 스텝으로 제한한다.
- `verify_phase3b1.py`: 저장 JSONL 재평가, 체크포인트 재평가, 파라미터 변화 및 이전 파일 보존 확인.
- `plot_phase3b1.py`: 실제 측정 CSV/JSON에서 독립 SVG 그래프 생성.
- `tests/test_phase3b1.py`: 신규 15개 테스트. PyTorch가 없는 환경에서는 Torch 테스트를 건너뛰도록 되어 있으나, 최종 검증에서는 전부 실제 실행했다.
- `requirements-phase3b1.txt`: 직접 의존성 및 CPU wheel 인덱스.
- `requirements-phase3b1-resolved.txt`: 실제 설치된 전체 패키지 버전.
- `UPSTREAM_MAPPO_LICENSE.txt`: 참고 공식 구현의 MIT 라이선스 원문.
- `PHASE3B1_RESULTS_KO.md`: 실제 측정값과 해석.
- `phase3b1_outputs/`: 두 실험, 원시 로그, 체크포인트, 테스트 기록, 감사 기록, 그래프. 과거 출력과 분리했다.

기존 흐름은 `phase2/model.py`의 구성·할당 자료형 → `phase2/engine.py`의 이동·고장·용량·중재 → `phase2/metrics.py`의 원시 로그 평가다. Phase 3A의 `ParallelCore`가 이 엔진 위에서 관측, 보상, 종료와 행동 감사 기록을 제공한다. 새 학습기는 기존 휴리스틱 `policies.py`를 호출하지 않고 `SampledAction`을 `ParallelCore.step`에 전달한다. 기존 평가기는 그대로 사용한다.

## 승인된 격리 환경

사용자가 전용 가상환경 설치를 승인한 뒤 `research/cah_mappo/.venv`에 설치했다.

- Windows, Python 3.12.14, PyTorch 2.7.0+cpu, NumPy 1.26.4.
- CPU: Ryzen 5 5500GT, 6코어/12스레드, RAM 약 31.3 GiB.
- 학습은 CPU 한 스레드. CUDA 런타임은 `None`. RTX 3080은 장치 목록만 확인했다.
- torch가 요구하는 부수 의존성은 잠금 파일에 기록했다. PettingZoo, Gym, ROS2 및 upstream의 오래된 의존성 묶음은 설치하지 않았다.
- 기존 Python/ROS2/Isaac Lab 환경을 변경하지 않았다. `.venv`는 기존 Git ignore에 의해 제외된다.

설치 조합은 [PyTorch 공식 이전 버전 안내](https://pytorch.org/get-started/previous-versions/)의 2.7.0 CPU 배포 방식에 맞췄다. 현재 설치가 이미 끝났으므로 실행을 위해 재설치할 필요는 없다.

## 관측·행동·Critic 분리

실제 코드의 기본 설정은 Actor 입력 146개, Critic 입력 141개, 이산 행동 10개다.

- Actor: 146 → 64 → 64 → 10, Tanh, 모든 로봇이 같은 가중치 공유. 자신의 Phase 3A 관측만 입력받는다.
- Critic: 141 → 64 → 64 → 1, Tanh. 별도로 생성된 전체 상태에서 팀의 가치를 추정한다.
- 파라미터 수: Actor 14,218개, Critic 13,313개.
- 로봇은 최대 4개 슬롯, 작업은 최대 8개 슬롯으로 패딩한다. 패딩과 고장은 `active_mask=0`으로 Actor 손실에서 제외한다.
- 행동 0은 대기/현재 할당 해제, 1은 현재 작업 유지, 2~9는 **그 로봇이 알고 있는 로컬 작업 슬롯** 선택이다.
- 작업 슬롯은 로봇별로 다를 수 있다. Critic의 전역 슬롯 번호를 Actor 행동 번호로 사용하지 않는다.
- 고장·패딩 슬롯에도 NOOP는 항상 남는다. 유효 행동이 하나뿐이면 확률 1, log-probability 0, entropy 0이다. 전부 마스킹된 분포는 오류로 거부한다.

Actor 입력 생성은 기존 `perception.actor`, Critic은 `build_critic`으로 분리되어 있다. 학습기는 `infos`의 privileged state를 Actor에 넣지 않는다. 미래 사건, 미수신 메시지, 관측되지 않은 작업을 Actor에 제공하지 않는 기존 회귀 테스트를 유지했다.

최소 실험은 센서 반경 10m, 센서 잡음 0, 통신 지연 0이다. 이 작은 공간에서는 대부분을 관측할 수 있는 쉬운 조건이다. 제한 관측·고장·통신 지연 환경에서의 학습 성능을 검증한 결과는 아니다.

## 샘플링, 실행 중재, PPO 버퍼

한 분포에서 행동을 샘플링하고 **동일한 분포의** log-probability와 당시 mask를 저장한다. 네트워크 가중치는 float32이며, 확률 감사의 오차 한계를 만족시키기 위해 Categorical logits/확률 계산은 float64로 수행한다.

버퍼에는 `obs`, `state`, `actions`, `old_logp`, `mask`, `active`, `reward`, `value`, `next_value`, `bootstrap`, `continuation`을 저장한다. `executed_actions`, `arbitration_changed`는 별도 감사 필드다. PPO 손실은 이 실행 결과 필드를 참조하지 않는다.

중복 할당 방지 중재는 환경 전이의 일부다. 거부된 행동도 정책이 실제 샘플링한 행동이므로 원래 행동/log-probability로 학습한다. 업데이트 전 같은 관측과 mask로 log-probability를 다시 계산하여 오차 1e-6을 넘으면 중단한다. 이 조건에서 importance ratio는 1이다.

원시 로그의 `policy_version=update_steps_N`은 수집 시점의 환경 스텝 표식이다. 가중치 해시나 optimizer update 번호는 아니다. 한 rollout 동안 가중치는 바뀌지 않으며 해당 rollout이 끝난 뒤에만 업데이트한다.

## 손실과 GAE

`b_t`는 bootstrap mask, `c_t`는 에피소드 연결 mask다.

```text
delta_t = r_t + gamma * b_t * V(s_next) - V(s_t)
A_t = delta_t + gamma * lambda * c_t * A_(t+1)
return_t = A_t + V(s_t)
ratio = exp(logp_new - logp_old)
L_actor = -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))
L_actor_total = L_actor - entropy_coef * mean(entropy)
L_critic = 0.5 * mean((V - return)^2)
```

Actor 평균과 advantage 평균·표준편차는 활성 로봇에만 적용한다. 모든 로봇이 비활성이면 Actor optimizer를 건너뛴다. 이때도 팀 환경은 진행하고 Critic은 환경 스텝 하나당 표본 하나로 학습한다. 로봇 수만큼 중앙 상태를 복제하여 Critic 손실의 가중치를 늘리지 않는다.

설정은 gamma=0.99, lambda=0.95, clip=0.2, entropy 계수=0.01, Critic 계수=0.5, Adam lr=0.0003/eps=1e-5, gradient norm 제한=0.5다. rollout 256스텝, 4 epochs, minibatch 64 환경 스텝이다. 최적값이라는 의미는 없다. loss/gradient/입력의 유한성과 shape, mask, 행동 범위를 검사한다.

## 보상과 평가

Phase 3A 보상 구현을 그대로 사용했다. 각 항의 상태·이벤트는 환경 내부 Ground Truth에서 계산하며 Actor의 관측 입력으로 넣지 않는다. 아래 괄호의 실패·대기·공백·전환 항은 이미 음수다.

```text
r = progress + 10*completion + response + 10*failure
    + waiting + 2*coverage + 0.1*switching
```

- 진행: 새로 수행한 작업량/필요 작업량을 합산하고 고정 작업 슬롯 수 8로 나눈다.
- 완료, 첫 대응, 실패: 해당 스텝의 완료/대응 시작/기한 만료 이벤트 수를 8로 나눈다. 실패 항은 음수다.
- 대기: 활성 동적 작업의 `min(priority,10)/10` 합을 8로 나눈 뒤 `dt/horizon`만큼 음수 부과한다.
- 공백: 시작 상태에서 물리적으로 대응 가능한 로봇이 없는 책임 구역의 우선순위 비율에 `-dt/horizon`을 곱한다. 용량 1 제약을 적용한다.
- 전환: 정책이 기존 비어 있지 않은 할당을 다른 것으로 바꾼 횟수/현재 로봇 수의 음수다.

평가 성공률, 응답시간, 책임 공백 비율과 인계 지표는 보상 합에서 추정하지 않는다. 별도 `phase2.metrics.evaluate`가 원시 상태와 이벤트에서 다시 계산한다. Return은 학습 진단용이며 논문 알고리즘 순위를 정하는 종합점수가 아니다.

## 종료와 복구

- 이번 비교는 모든 정책에 `terminate_on_tasks_resolved=True`를 동일하게 적용한다. 미래 작업까지 완료 또는 기한 만료로 해결되면 `terminated=True`, bootstrap=0, continuation=0이다.
- 따라서 `terminated` 자체가 작업 성공을 뜻하지 않는다. 기한 만료로 실패한 경우도 포함하며 실제 성공률을 따로 확인해야 한다.
- 시간 제한은 `truncated=True`, bootstrap=1, continuation=0이다. reset 전의 마지막 상태 가치로 bootstrap한다.
- 개별 고장은 에피소드 종료가 아니다. 고장 로봇의 Actor 손실만 제외하고, 복구 시 다시 활성화한다.
- rollout 경계에서 에피소드가 계속되면 다음 상태 가치로 bootstrap한다. 서로 다른 에피소드의 GAE는 이어지지 않는다.
- 모든 작업 완료 시 즉시 종료하므로 책임 공백 비율의 분모는 실제 관측된 에피소드 길이다. 과거의 고정 24초 전체 구간 평가와 직접 숫자를 비교하면 안 된다.

## 공식 구현을 참고한 범위와 차이

[marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)의 다음 파일과 MIT 라이선스를 읽었다. 읽은 blob SHA와 URL은 `phase3b1_outputs/upstream_review.json`에 남겼다.

- `rMAPPOPolicy.py`: Actor/Critic 및 optimizer 분리.
- `r_actor_critic.py`: 분리된 actor 관측/중앙 관측, action mask, active mask.
- `r_mappo.py`: clipped surrogate, entropy, advantage normalization, gradient clipping.
- `shared_buffer.py`: 관측·행동·log-probability·가치·종료 저장 및 GAE.
- `mpe_runner.py`: 수집 → 환경 실행 → 버퍼 → return → 업데이트 구조.

이번 코드는 이를 참고한 **독립적인 소규모 feed-forward 구현**이다. 공식 코드를 통째로 이식하거나 공식 runner를 그대로 실행한 것이 아니다. 주요 차이는 다음과 같다.

- RNN과 hidden state가 없다.
- PopArt/ValueNorm, 관측 정규화, clipped value loss, Huber loss는 사용하지 않는다.
- Critic은 복제된 에이전트별 상태 대신 환경 스텝당 한 팀 상태를 학습한다.
- Phase 3A의 두 종료 mask를 명시적으로 사용하고 실행 중재 기록을 분리한다.
- 연속 행동, 병렬 vector env, GPU 학습은 없다.
- 오래된 upstream 설치 환경을 적용하지 않고 승인된 최소 CPU 의존성만 사용한다.

출처의 MIT 고지를 `UPSTREAM_MAPPO_LICENSE.txt`에 보존했다. 이 차이로 인해 결과를 공식 저장소의 모든 기본 옵션을 재현한 결과로 부르면 안 된다.

## 실행 명령: PowerShell

아래의 새 출력 경로는 아직 존재하지 않아야 한다. 프로그램은 기존 실험 디렉터리를 덮어쓰지 않는다. 다음 명령은 설명용이며 이 문서를 작성하면서 추가 본학습을 실행하지 않았다.

```powershell
Set-Location C:\Users\jes\Desktop\MX3_Match_Sim\research\cah_mappo
$taskPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'

& $taskPython -B -m unittest discover -s tests -v

& $taskPython -B phase3b1.py --output phase3b1_outputs\my_seed7_run --steps 4096 --seed 7

& $taskPython -B verify_phase3b1.py --experiment phase3b1_outputs\debug_seed7_steps4096 --experiment phase3b1_outputs\debug_seed11_steps4096 --output phase3b1_outputs\my_audit

& $taskPython -B plot_phase3b1.py --experiment phase3b1_outputs\debug_seed7_steps4096 --experiment phase3b1_outputs\debug_seed11_steps4096 --output phase3b1_outputs\my_curves.svg
```

전체 checkpoint는 `torch.load(path, map_location='cpu', weights_only=True)`로 읽고 `ActorCritic().load_state_dict(checkpoint['network'])`로 복원한다. 저장된 코드 버전과 설정을 함께 사용해야 한다. 실제 재평가는 `verify_phase3b1.py`가 수행한다.

## 재현성과 저장 범위

학습 시드는 신경망 초기화/샘플링/shuffle에 사용한다. 환경 시드는 별도 100, 평가 환경 시드는 1000~1019다. 평가 정책 난수는 환경 사건 난수와 분리한다. CPU deterministic algorithms를 사용한다. 같은 설치 조합 안에서 동일 시드 수집·업데이트가 같음을 테스트했다. 다른 PyTorch/플랫폼 버전까지 비트 단위 재현을 보장하지 않는다.

실험마다 `config.json`, `baselines.json`, `learning_curve.csv/json`, `update_*.json`, `train_episodes.jsonl`, `evaluation.json`, `checkpoints/`, `first_rollout.pt`를 저장한다. 전체 평가 에피소드의 원시 JSONL과 일부 학습 에피소드 원시 JSONL을 저장한다. 모든 학습 에피소드의 요약은 저장하지만, 모든 학습 스텝의 전체 rollout 파일을 저장하는 구성은 아니다.

체크포인트에는 가중치, optimizer, 샘플링/shuffle RNG와 스텝 수가 들어 있다. **진행 중 환경 상태는 직렬화하지 않는다.** 평가 재시작은 검증했지만 임의 스텝에서 중단한 학습을 정확히 이어가는 resume 기능은 아직 없다.

Phase 3B-2는 별도 승인 후 진행한다. 현재 단계에서는 Git 커밋·푸시, 원격 서버 접속, 실물 구동을 하지 않았다.
