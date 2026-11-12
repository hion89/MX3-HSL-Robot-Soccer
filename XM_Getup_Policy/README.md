# XM (MX3) 기상 정책 — Getup V1-2 / model_53400

XM_robot04 (MX3, 다이나믹셀 XM540-W150) 이 넘어진 상태 (등을 대고 누움 / 엎드림) 에서 일어서는 정책과, 그 정책을 학습한 환경 코드.
unitree_rl_lab 기반, Isaac Lab 2.3.2 / Isaac Sim 4.5.0 / rsl_rl 3.1.2 / torch 2.7.0 / Python 3.10 에서 학습했다.
Isaac Sim 5.x 에서도 4.5 와 같은 관절 마찰로 돌도록 호환 코드가 들어 있다 (아래 '알아둘 것').

- 태스크: `Unitree-MX3-Getup-V1-2` (재생: `Unitree-MX3-Getup-V1-2-Play`)
- 체크포인트: `model_53400` (학습 런 `unitree_mx3_getup_v1_2/OK6_2026-09-15_19-14-05_armpose`)
- 2026-09-16 실기 검증을 통과한 최종 기상 정책이다. 양방향 기상과 기립 후 안정성을 확인했다.

## 폴더
```
policy/
  policy.onnx          배포용 (obs[1,300] -> actions[1,18])
  policy.pt            같은 정책의 TorchScript
  model_53400.pt       rsl_rl 체크포인트 (actor+critic, 재학습·파인튜닝용)
  deploy.yaml          관절 순서, 기본자세, kp/kd, 액션 스케일, 관절 목표 한계, 관측 구성 (학습 코드가 자동 생성)
training/
  env.yaml, agent.yaml                학습 당시 저장된 전체 설정 (원본 기록)
  getup_v1_2_env_cfg_at_training.py   학습 당시 환경 코드 스냅샷
  RUN_NOTES.txt                       학습 기록 (변경점, 체크포인트별 실측, 선택 근거)
unitree_rl_lab_overlay/               unitree_rl_lab 레포 루트에 그대로 덮어쓰는 코드 (경로 유지)
  source/unitree_rl_lab/unitree_rl_lab/
    assets/robots/mx_bot.py                    로봇 설정. 기상은 "XM_robot04_getup" 항목 (팔 포함 18 관절)
    tasks/locomotion/robots/MX3/
      getup_v1_2_env_cfg.py                    ★ 환경 (관측·리워드·리셋·역방향 커리큘럼)
      __init__.py                              태스크 등록 (Getup-V1-2 / -Play 만 발췌)
    tasks/locomotion/mdp/                      MDP 함수 전체. 기상 전용은 getup_v1_mdp.py, getup_v1_rewards.py
    tasks/locomotion/agents/                   PPO 설정 (rsl_rl_ppo_cfg.py 의 GetupV12PPORunnerCfg)
    utils/                                     deploy.yaml 생성, play 용 cfg 파서,
                                               joint_friction_compat.py (Isaac Sim 5.x 관절 마찰을 4.5 와 같게)
  scripts/rsl_rl/  train.py, play.py, cli_args.py, snapshot_rollback.py,
                   eval_getup_recovery.py (기립 후 방향별 밀림 회복)
  scripts/list_envs.py
  robotmodels/mx_bot/usd/XM_robot04/          로봇 USD (메시 포함, 폴더 안 상대경로라 그대로 동작)
rsl_rl_patch/ppo_huber_nan_guard.diff         학습에 쓴 rsl_rl 로컬 수정분 (아래 참고)
```

## 사용법
unitree_rl_lab 레포가 이미 있다는 가정이다. 덮어쓰기 전에 기존 파일을 백업할 것.
`mdp/`, `agents/`, `mx_bot.py`, `scripts/rsl_rl/train.py`·`play.py` 가 교체되고 `utils/joint_friction_compat.py` 가 추가된다
(`mx_bot.py` 가 이 파일을 import 하므로 같이 복사해야 한다).
이미 `robots/MX3/` 에 다른 태스크가 있으면 `__init__.py` 는 덮어쓰지 말고 등록 블록만 붙여 넣는다.
보행 패키지 (`XM_Walking_Policy`) 와 공통 파일은 같은 내용이라 둘 다 덮어써도 된다.

```bash
cp -r unitree_rl_lab_overlay/. <unitree_rl_lab 경로>/
```

재생 (항상 등을 대고 누운 90° 에서 시작한다. `--fall_angle` 로 바꿀 수 있다):
```bash
python scripts/rsl_rl/play.py --task Unitree-MX3-Getup-V1-2-Play --use_train_cfg off \
    --checkpoint <이 폴더>/policy/model_53400.pt
```

이 정책은 처음부터 학습한 게 아니다. 여러 판을 이어서 학습했다
(OK3 `model_48600` 에서 resume → 복와위 추가 + 팔 기본자세 변경). 이어받기 경위는 `training/RUN_NOTES.txt` 에 있다.
같은 환경으로 처음부터 학습하는 명령:
```bash
python scripts/rsl_rl/train.py --task Unitree-MX3-Getup-V1-2 --headless --num_envs 16384
```

## 정책 입출력 (실기 러너용, 자세한 값은 `policy/deploy.yaml`)
| 항목 | 값 |
|---|---|
| 제어 | 50 Hz (step_dt 0.02), 위치 PD, `q_target = default + 0.25 * action`, 관절별 목표 한계로 clip (`target_limits`) |
| 관절 | 18 = 다리 12 (r_pelvis_pitch … r_ankle_roll, l_ 같은 순서) + 팔 6 (r_shoulder_pitch, r_shoulder_roll, r_elbow_pitch, l_ 같은 순서) |
| 게인 | 전 관절 kp 50 / kd 2.5 |
| 기본자세 (다리) | pelvis_pitch ±0.62, knee ±1.2, ankle_pitch ±0.62 (r +, l −), 나머지 0 |
| 기본자세 (팔) | r_shoulder_pitch −0.3, r_shoulder_roll 0, r_elbow +1.0 / l_shoulder_pitch +0.3, l_shoulder_roll 0, l_elbow −1.0 |
| 관측 | 60 × 5 프레임 = 300. 한 프레임 = 각속도 3 (×0.2) / 중력벡터 3 / 관절각(기본자세 기준) 18 / 관절속도 18 (×0.05) / 직전 액션 18. 히스토리는 term 단위로 묶이고 오래된 것이 먼저. 명령 입력은 없다 |
| 액션 지연 | 학습 시 1~2 스텝 랜덤 |

## 성능 (sim, 2026-10-29 이 패키지 코드로 재측정)
256 env, 절반은 등을 대고 누운 +90°, 절반은 엎드린 −90° 에서 시작, 외란 끔, 20 s:
`기립 256/256 (100 %) · 기상 시간 평균 0.87 s (p95 0.90~1.00 s) · 기립 높이 0.3947 m · 끝까지 서 있음 100 % · 10 s 정지 드리프트 3~4 mm`
학습 당시 기록 (여섯 각도 +90 ~ −90 전부 100 %, 기상 0.77~0.92 s, 기립 높이 0.3936~0.3948) 과 같다.

## 알아둘 것
- **정책을 켜기 전에 관절을 기본자세로 옮겨야 한다.** 액션이 기본자세 기준 오프셋이라, 시작 자세가 다르면 액션 원점이 어긋난다.
  특히 팔 기본자세가 이전 판 (어깨 0 / 팔꿈치 ±1.5) 과 다르다.
- **보행 정책과 팔 처리가 다르다.** 보행 정책 (12 관절) 은 팔을 제어하지 않는다. 기상 → 보행으로 넘길 때 팔 목표를 따로 잡아야 한다.
- 엎드린 상태에서는 몸을 270° 돌리지 않고 90° 만 굴러서 일어난다.
- 바닥 마찰 0.25~1.4 범위로 학습했다 (맨바닥에서 기상 확인).
- 토크·속도 한계 랜덤화 (7 판) 는 sim 지표는 좋았지만 화면으로 보면 떨림과 복와위 동작이 나빠 채택하지 않았다.
  `getup_v1_2_env_cfg.py` 에 주석으로 남아 있다.
- 충돌 모델은 `XM_robot04` (수동 캡슐) 이고 `self_collisions=True` 다. convex hull 버전이 아니다.
- **Isaac Sim 5.x 에서 돌릴 때 (2026-11-01 추가):** 이 정책은 Isaac Sim 4.5 에서 학습했고, `mx_bot.py` 의 관절 마찰 `friction=1.1` 은
  4.5 의 마찰 "계수" (관절 하중에 비례, 움직이는 동안에도 걸림) 기준 값이다. Isaac Sim 5.x 는 관절 마찰 모델이 (정지·운동·점성) 으로 바뀌어,
  그대로 두면 이 값이 정지 마찰 1.1 N·m 로만 들어가고 **움직이는 관절의 마찰은 0** 이 된다 (= 4.5 에서 마찰을 끈 로봇).
  `utils/joint_friction_compat.py` 가 5.x 에서 4.5 계수 모델로 되돌린다. `mx_bot.py` 를 import 할 때 자동 적용되고 (학습·재생·평가 전부),
  시작할 때 `[joint_friction_compat] ... 4.5 계수 모델로` 가 찍힌다. 4.x 에서는 아무것도 안 한다. 5.x 의 새 모델을 쓰려면 `UNITREE_JOINT_FRICTION=native`.
  기상은 마찰 영향이 작다: Isaac Sim 5.1 (호환 적용 / 미적용) 과 4.5 모두 기립 256/256, 10 s 기립 높이 0.395 m,
  `eval_getup_recovery.py` 밀림 회복 12 조건 전부 넘어짐 0 % · 회복 100 % (최대 기울기 차 0.5° 이내). 보행 정책은 차이가 커서 (밀림 복귀율 93 % → 70 %) 같은 코드를 넣었다.
- **rsl_rl 패치:** value loss 를 Huber 로 바꾸고 NaN gradient 때 optimizer step 을 건너뛰는 수정이 들어간 rsl_rl 3.1.2 로 학습했다.
  추론에는 영향이 없다. 재학습할 때 결과를 맞추려면 `rsl_rl_patch/` 의 diff 를 `rsl_rl/algorithms/ppo.py` 에 적용한다.
- `train.py` 는 value loss 발산 때 이전 스냅샷으로 되돌리는 `snapshot_rollback.py` 를 쓴다 (같이 들어 있다).
- `mdp/` 는 프로젝트 전체가 공유하는 패키지라 이 태스크와 무관한 파일 (보행, 킥, RS01 용) 도 들어 있다.
  `mdp/__init__.py` 가 전부 import 하므로 지우지 말 것.
