# XM (MX3) 보행 정책 — Velocity V0-6 / model_13500

XM_robot04 (MX3, 다이나믹셀 XM540-W150) 의 속도 추종 보행 정책과, 그 정책을 학습한 환경 코드.
unitree_rl_lab 기반, Isaac Lab 2.3.2 / Isaac Sim 4.5.0 / rsl_rl 3.1.2 / torch 2.7.0 / Python 3.10 에서 학습했다.
Isaac Sim 5.x 에서도 4.5 와 같은 관절 마찰로 돌도록 호환 코드가 들어 있다 (아래 '알아둘 것').

- 태스크: `Unitree-MX3-Velocity-V0-6` (재생: `Unitree-MX3-Velocity-V0-6-Play`)
- 체크포인트: `model_13500` (학습 런 `unitree_mx3_velocity_v0_6/2026-08-23_14-45-11`)
- 실기에서 쓰던 최종 보행 정책이다.

## 폴더
```
policy/
  policy.onnx          배포용 (obs[1,240] -> actions[1,12])
  policy.pt            같은 정책의 TorchScript
  model_13500.pt       rsl_rl 체크포인트 (actor+critic, 재학습·파인튜닝용)
  deploy.yaml          관절 순서, 기본자세, kp/kd, 액션 스케일, 관측 구성 (학습 코드가 자동 생성)
training/
  env.yaml, agent.yaml                 학습 당시 저장된 전체 설정 (원본 기록)
  velocity_v0_6_env_cfg_at_training.py 학습 당시 환경 코드 스냅샷
  RUN_NOTES.txt                        학습 기록 (변경점, 지표, 판단 근거)
unitree_rl_lab_overlay/               unitree_rl_lab 레포 루트에 그대로 덮어쓰는 코드 (경로 유지)
  source/unitree_rl_lab/unitree_rl_lab/
    assets/robots/mx_bot.py                    로봇 설정 (게인, 기본자세, USD 경로)
    tasks/locomotion/robots/MX3/
      velocity_v0_6_env_cfg.py                 ★ 환경 (관측·리워드·이벤트·커리큘럼)
      __init__.py                              태스크 등록 (V0-6 / -Play / -Metric 만 발췌)
    tasks/locomotion/mdp/                      리워드·관측·보행위상·리셋홀드·미러 등 MDP 함수 전체
    tasks/locomotion/agents/                   PPO 설정 (rsl_rl_ppo_sym_cfg.py 의 SymmetryPPOMirror3RunnerCfg)
    utils/                                     deploy.yaml 생성, play 용 cfg 파서,
                                               joint_friction_compat.py (Isaac Sim 5.x 관절 마찰을 4.5 와 같게)
  scripts/rsl_rl/  train.py, play.py, cli_args.py, snapshot_rollback.py,
                   eval_push_recovery.py (정지 밀림 회복), eval_march_drift.py (제자리걸음 횡드리프트)
  scripts/list_envs.py
  robotmodels/mx_bot/usd/XM_robot04/          로봇 USD (메시 포함, 폴더 안 상대경로라 그대로 동작)
rsl_rl_patch/ppo_huber_nan_guard.diff         학습에 쓴 rsl_rl 로컬 수정분 (아래 참고)
```

## 사용법
unitree_rl_lab 레포가 이미 있다는 가정이다. 덮어쓰기 전에 기존 파일을 백업할 것.
`mdp/`, `agents/`, `mx_bot.py`, `scripts/rsl_rl/train.py`·`play.py` 가 교체되고 `utils/joint_friction_compat.py` 가 추가된다
(`mx_bot.py` 가 이 파일을 import 하므로 같이 복사해야 한다).
이미 `robots/MX3/` 에 다른 태스크가 있으면 `__init__.py` 는 덮어쓰지 말고 등록 블록만 붙여 넣는다.

```bash
cp -r unitree_rl_lab_overlay/. <unitree_rl_lab 경로>/
```

재생:
```bash
python scripts/rsl_rl/play.py --task Unitree-MX3-Velocity-V0-6-Play --use_train_cfg off \
    --checkpoint <이 폴더>/policy/model_13500.pt
```
명령을 고정하려면 `--cmd_vx 0.3 --cmd_vy 0 --cmd_wz 0` 처럼 준다.

학습 (같은 조건 재현):
```bash
python scripts/rsl_rl/train.py --task Unitree-MX3-Velocity-V0-6 --headless --num_envs 16384 --max_iterations 15000
```

밀림 회복 평가:
```bash
python scripts/rsl_rl/eval_push_recovery.py --task Unitree-MX3-Velocity-V0-6 --headless --num_envs 512 \
    --checkpoint <이 폴더>/policy/model_13500.pt
```

## 정책 입출력 (실기 러너용, 자세한 값은 `policy/deploy.yaml`)
| 항목 | 값 |
|---|---|
| 제어 | 50 Hz (step_dt 0.02), 위치 PD, `q_target = default + 0.25 * action` |
| 관절 | 다리 12 (r_pelvis_pitch, r_pelvis_roll, r_pelvis_yaw, r_knee_pitch, r_ankle_pitch, r_ankle_roll, 그다음 l_ 같은 순서). **팔은 제어하지 않는다** |
| 게인 | 전 관절 kp 50 / kd 2.5 |
| 기본자세 | pelvis_pitch ±0.62, knee ±1.2, ankle_pitch ±0.62 (r +, l −), 나머지 0 |
| 관측 | 48 × 5 프레임 = 240. 한 프레임 = 각속도 3 (×0.2) / 중력벡터 3 / 속도명령 3 / 보행위상 3 (cos, sin, 주파수) / 관절각(기본자세 기준) 12 / 관절속도 12 (×0.05) / 직전 액션 12. 히스토리는 term 단위로 묶이고 오래된 것이 먼저 |
| 명령 범위 | vx −0.5 ~ 0.7 m/s, vy ±0.5 m/s, wz ±1.5 rad/s, 보행 주파수 1.7 ~ 2.4 Hz |
| 액션 지연 | 학습 시 1~2 스텝 랜덤 |

## 성능 (sim, 2026-10-29 이 패키지 코드로 재측정)
정지 밀림 회복, 512 env, 방향 4 × 세기 3 (0.20 / 0.35 / 0.50 m/s):
`peak_tilt 5.22° · 복귀율 92.6 % · 평균 복귀 0.55 s · 전도 0.0 % · 스텝 전략 43.2 %`
학습 당시 기록 (5.12~5.31°, 93.2 %, 0.58 s, 전도 0~0.4 %) 과 노이즈 범위 안에서 같다.

학습 지표 (model_13500): action_rate −0.150 · pitch RMS 2.57° · 전도 1.2 % · 에피소드 길이 997 · 대칭 손실 0.0017

## 알아둘 것
- **전진 0.62 m/s 가 모터 한계다.** 명령 0.7 을 주면 0.617 m/s 가 나오고 hip pitch 토크가 5.0 N·m 로 포화한다.
  명령 범위만 올려서 더 빨라지지 않는다. 회전은 여유가 크다.
- **시뮬레이션은 토크를 속도와 무관하게 5.0 N·m 로 둔다.** 실기 XM540-W150 은 12 V 에서
  `τ(ω) ≈ 7.3·(1 − ω/5.55)` 로 떨어진다. 고속 전진일수록 실기 여유가 적다.
- 충돌 모델은 `XM_robot04` (수동 캡슐) 이다. convex hull 버전 (`XM_robot04_convexhull`) 이 아니다.
- **Isaac Sim 5.x 에서 돌릴 때 (2026-11-01 추가):** 이 정책은 Isaac Sim 4.5 에서 학습했고, `mx_bot.py` 의 관절 마찰 `friction=1.1` 은
  4.5 의 마찰 "계수" (관절 하중에 비례, 움직이는 동안에도 걸림) 기준 값이다. Isaac Sim 5.x 는 관절 마찰 모델이 (정지·운동·점성) 으로 바뀌어,
  그대로 두면 이 값이 정지 마찰 1.1 N·m 로만 들어가고 **움직이는 관절의 마찰은 0** 이 된다 (= 4.5 에서 마찰을 끈 로봇).
  `utils/joint_friction_compat.py` 가 5.x 에서 4.5 계수 모델로 되돌린다. `mx_bot.py` 를 import 할 때 자동 적용되고 (학습·재생·평가 전부),
  시작할 때 `[joint_friction_compat] ... 4.5 계수 모델로` 가 찍힌다. 4.x 에서는 아무것도 안 한다. 5.x 의 새 모델을 쓰려면 `UNITREE_JOINT_FRICTION=native`.
  Isaac Sim 5.1 에서 이 패키지 코드로 잰 정지 밀림 회복: `peak_tilt 5.28° · 복귀율 93.4 % · 평균 복귀 0.57 s · 전도 0.2 % · 스텝전략 45.3 %` (위 4.5 값과 같다).
  호환 없이 5.1 기본으로 돌리면 복귀율 70.1 % — 관절 마찰이 없어 기울기가 2° 안으로 잘 가라앉지 않는다.
- **rsl_rl 패치:** value loss 를 Huber 로 바꾸고 NaN gradient 때 optimizer step 을 건너뛰는 수정이 들어간 rsl_rl 3.1.2 로 학습했다.
  추론에는 영향이 없다. 재학습할 때 결과를 맞추려면 `rsl_rl_patch/` 의 diff 를 `rsl_rl/algorithms/ppo.py` 에 적용한다.
- `train.py` 는 value loss 발산 때 이전 스냅샷으로 되돌리는 `snapshot_rollback.py` 를 쓴다 (같이 들어 있다).
- `mdp/` 는 프로젝트 전체가 공유하는 패키지라 이 태스크와 무관한 파일 (킥, 기상, RS01 용) 도 들어 있다.
  `mdp/__init__.py` 가 전부 import 하므로 지우지 말 것.
