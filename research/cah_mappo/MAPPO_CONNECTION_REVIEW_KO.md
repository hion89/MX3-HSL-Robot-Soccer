# 공식 MAPPO 연결 검토 — Phase 3A

2026-11-18 공식 공개 저장소 파일을 읽어 대조했다. 저장소를 설치·실행하거나 로컬 작업 내용을 전송하지 않았다. 파일별 blob SHA는 `phase3a_outputs/mappo_review_sources.json`에 기록했다. 단일 commit SHA를 확보한 것은 아니므로 전체 저장소의 고정 커밋 검증이라고 주장하지 않는다.

## 확인한 공식 경로와 대응

1. [base_runner.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/runner/shared/base_runner.py)
   Actor observation_space와 centralized share_observation_space를 구분한다. 새 환경은 Actor 146, Critic 141이며 이 두 공간을 각각 연결할 수 있다. 현재 PettingZoo 어댑터의 state_space와 공식 runner가 요구하는 share_observation_space 목록은 동일 API가 아니므로 변환 wrapper가 필요하다.

2. [mpe_runner.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/runner/shared/mpe_runner.py)
   기본 MPE runner는 관측을 연결해 share_obs를 만들고 구형 obs/reward/done/info 반환을 사용한다. collect/insert에 이 환경의 available_actions·active_masks·두 종료 종류를 그대로 전달하는 처리가 없다. Discrete action도 one-hot으로 환경에 보내므로 정수 행동을 받는 현재 코어와 바로 연결되지 않는다. 그대로 학습을 실행하면 안 된다.

3. [rMAPPOPolicy.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py), [r_actor_critic.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/algorithm/r_actor_critic.py)
   cent_obs와 obs, RNN states, masks, available_actions를 구분하는 진입점이 있다. critic 입력은 state(), actor 입력은 로봇별 벡터로 연결한다. 현재 info의 privileged 정보를 actor에 넣지 않는다. SampledAction의 분포·log-probability는 동일 forward에서 추출하도록 runner 경로를 수정해야 한다.

4. [shared_buffer.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/utils/shared_buffer.py)
   obs/share_obs/available_actions/actions/action_log_probs/active_masks/bad_masks를 저장한다. 첫 두 공간은 현재 padded_batch에 대응하지만 기록할 actions는 실행된 중재 결과가 아닌 sampled_action이다. 구버전 compute_returns는 masks/bad_masks를 조합하는 방식이다. 현재의 bootstrap_mask/continuation_mask를 단순히 done 하나로 합칠 수 없다. 실제 최종 관측의 V(s_final)을 이용하는 별도 연결 검증이 필요하다.

5. [r_mappo.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py)
   old/new log-probability 비율과 active mask를 이용한다. 모든 로봇이 고장 나 active mask 합이 0이면 그대로 나눗셈하는 경로가 있으므로 정책 업데이트를 건너뛰는 guard가 필요하다. 현재의 actor 비활성화와 team critic 유지, 패딩 로봇 제외를 별도 mask로 연결해야 한다. RNN reset mask와 return 종료 mask를 같은 것으로 취급하지 않는다.

6. [distributions.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/distributions.py), [act.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/act.py)
   available_actions로 categorical 출력을 제한하는 경로가 있다. 새 환경은 비활성·패딩에서도 NOOP 한 개를 남긴다. action mask를 현재 관측에서 재생성하지 말고 해당 표본의 것을 buffer에 저장해야 한다.

## 연결 텐서 초안

N개 병렬 환경, 최대 로봇 슬롯 4, 기본 작업 상한 8일 때:

- actor obs: [N,4,146].
- critic share_obs: [N,4,141] 또는 환경당 critic 표본 1개를 두는 구조로 재설계.
- available_actions: [N,4,10].
- sampled actions/log-probs/reward: [N,4,1].
- active/present/critic mask: [N,4,1], 서로 다른 의미를 유지.
- RNN masks, bootstrap masks, GAE trace masks는 별도로 관리.

없는 로봇은 배치의 present mask=0, 실패한 실제 로봇은 present=1/actor active=0이다. Actor 로컬 작업 슬롯과 Critic 전역 슬롯의 번호가 같다고 가정하지 않는다.

실행 중재는 환경 전이다. PPO ratio 계산에는 원래 샘플을 쓴다. 중재 결과를 행동으로 갈아 끼우고 원래 log-probability를 쓰는 buffer는 금지한다. observation_token/분포/정책 버전 구조는 검증했지만 실제 신경망·RNN이 같은 정책에서 산출했는지는 아직 미검증이다.

## 환경·버전 호환성

- 로컬 코어 검증: Python 3.12.14, 표준 라이브러리만 사용.
- 기존 분석 환경 조회: NumPy 1.26.4. torch, Gymnasium, PettingZoo, tensorboardX는 미설치.
- 공식 [environment.yaml](https://github.com/marlbenchmark/on-policy/blob/main/environment.yaml)은 Python 3.6.2, NumPy 1.18.1, torch 1.5.1+cu101 등 오래된 조합을 기록한다. [README](https://github.com/marlbenchmark/on-policy)는 Python 3.6.1 설치 예를 제시한다.
- 공식 [requirements.txt](https://github.com/marlbenchmark/on-policy/blob/main/requirements.txt)는 NumPy 1.18.5, gym 0.17.2 등 광범위한 과거 의존성을 고정한다. 이를 현재 Python 3.12 분석 환경에 그대로 설치하는 방법은 제안하지 않는다. 새 코어는 gym API를 직접 사용하지 않는다.
- [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서 현재 지원 Python/OS/CPU·CUDA 조합을 확인한 뒤 별도 환경에 정확한 버전을 고정해야 한다. 기존 Isaac/Jetson 환경과 같은 버전으로 맞춰야 하는 연구는 아니다.

## Phase 3B 제안 — 아직 실행하지 않음

1. 별도 Python 3.11 또는 3.12 환경에서 CPU forward/buffer 검증부터 시작한다. 정확한 torch/NumPy 버전은 설치 가능성과 공식 코드 포팅 테스트 후 lock한다. 이번 단계에서 호환 조합을 실제 설치·검증한 것은 아니다.
2. NumPy·PyTorch와 선택한 runner의 실제 최소 의존성만 설치한다. PettingZoo/Gymnasium을 사용할 때에만 어댑터 의존성을 추가하고 parallel_api_test를 실행한다. 공식 과거 requirements 전체를 무조건 설치하지 않는다.
3. 전용 runner/wrapper를 작성한다. actor/critic 공간, 원래 sampled action, available_actions, 실패/패딩, RNN reset, final-state bootstrap을 연결한다.
4. 신경망 동일 forward 확률 검증, active 합=0, 중재 표본, 시간 제한 숫자 예제, optimizer 1회 smoke를 별도로 검증한 뒤 실제 학습을 승인받는다.
5. CPU 학습량·속도를 측정한 후 필요하면 서버 `bami122@163.180.145.122`의 별도 연구 환경을 검토한다. 이번에는 서버에 접속하지 않았으며 GPU·CUDA·가상환경 버전은 미확인이다. 기존 env_isaaclab/unitree 학습 환경을 변경하지 않는다.

## 아직 결정하지 않은 문제

- continuing coverage 또는 finite workload 중 가치 목적과 에피소드 정의 고정.
- 공유 critic 표본을 로봇 수만큼 중복 가중할지, 환경당 하나로 학습할지.
- 통신 지연까지 Markov성을 높이기 위한 belief/queue 요약 또는 critic RNN 입력.
- 센서의 작업 진행·마감 관측 가정과 실제 로봇/메시지 정보의 차이.
- CAH 확장 전 Standard MAPPO와 공유할 관측·보상·중재 규칙의 고정.
- reward scale·TTL·센서 범위 민감도 및 reward hacking 검증.

결론: 환경 코어와 기록 계약은 오프라인 검증됐지만 **공식 MAPPO와 즉시 학습 가능한 통합 상태는 아니다.** 실제 연결/의존성 검증과 학습은 별도 Phase 3B 승인 대상이다.
