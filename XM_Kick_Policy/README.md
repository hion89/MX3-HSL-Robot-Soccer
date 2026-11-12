# XM (MX3) 킥 정책 — WalkKick V1-5B / model_22300

XM_robot04 (MX3, 다이나믹셀 XM540-W150) 의 킥 정책과, 그 정책을 학습한 환경 코드.
보행 정책이 공 앞까지 데려다주면 이 정책이 넘겨받아 **공까지 몇 걸음 걸어가 목표 방향으로 차고, 그대로 서 있는다**.
킥 세기 입력 (0~1) 으로 공 속도를 조절할 수 있다.
unitree_rl_lab 기반, Isaac Lab 2.3.2 / Isaac Sim 4.5.0 / rsl_rl 3.1.2 / torch 2.7.0 / Python 3.10 에서 학습했다.
Isaac Sim 5.x 에서도 4.5 와 같은 관절 마찰로 돌도록 호환 코드가 들어 있다 (아래 '알아둘 것').

- 태스크: `Unitree-MX3-WalkKick-V1-5B` (재생: `Unitree-MX3-WalkKick-V1-5B-Play`)
- 체크포인트: `model_22300` (학습 런 `unitree_mx3_walkkick_v1_5b/2026-11-01_06-55-23`)
- ROS 4v4 경기 시뮬 (hsl, MuJoCo) 의 기본 킥이다 (`humanoid_policy_runner/models/walkkick_v1_policy.onnx` 와 같은 파일, md5 일치).
- **실기에서는 아직 검증하지 않았다.** 이전 실기 킥은 이 정책의 출발점인 v1_4 W90B70 model_18200 이다.

## 폴더
```
policy/
  policy.onnx          배포용 (obs[1,265] -> actions[1,12])
  policy.pt            같은 정책의 TorchScript (ONNX 와 출력 차 6e-6)
  model_22300.pt       rsl_rl 체크포인트 (actor+critic, 재학습·파인튜닝용)
  deploy.yaml          관절 순서, 기본자세, kp/kd, 액션 스케일·지연, 관측 구성 (학습 코드가 자동 생성)
training/
  env.yaml, agent.yaml                  학습 당시 저장된 전체 설정 (원본 기록)
  walk_kick_v1_5_env_at_training.py     학습 당시 환경 코드 스냅샷 (지금 코드와 같다)
  RUN_NOTES.txt                         학습 기록. 계보 (W90B70 18200 → v1_5 1a → 1b 시도 1~5) 를 run 마다 그대로 붙였다
  MATCH_FALLS_22300.md                  경기 시뮬에서 킥 중 넘어진 조건 분석 (아래 '알아둘 것' 의 근거)
unitree_rl_lab_overlay/               unitree_rl_lab 레포 루트에 그대로 덮어쓰는 코드 (경로 유지)
  source/unitree_rl_lab/unitree_rl_lab/
    assets/robots/mx_bot.py                    로봇 설정. 킥은 "XM_robot04_convexhull_kick" 항목 (MX_BOT_KICK_CFG)
    tasks/locomotion/robots/MX3/
      walk_kick_v1_5_env.py                    ★ 환경 (v1_5 경기 넘겨받기 조건 + v1_5B 킥 세기)
      walk_kick_v1_4_env.py                    부모 환경 (킥 레이어 보상·관측·종료 조건 대부분이 여기)
      data/handoff_bank_v0_6_13500.pt          넘겨받기 시작 상태 은행 12 만 개 (학습에 필요, 24 MB)
      __init__.py                              태스크 등록 (V1-5 / V1-5-Play / V1-5B / V1-5B-Play 만 발췌)
    tasks/locomotion/mdp/                      MDP 함수 전체. 킥 전용은 kick_mdp.py, kick_inside_mdp.py, kick_stability_mdp.py,
                                               kick_handoff_mdp.py (넘겨받기·공 믿음 오차·Profile Velocity·킥 세기), kick_clock.py
    tasks/locomotion/agents/                   PPO 설정 (rsl_rl_ppo_sym_cfg.py 의 WalkKickPPORunnerCfg)
    utils/                                     deploy.yaml 생성, play 용 cfg 파서,
                                               joint_friction_compat.py (Isaac Sim 5.x 관절 마찰을 4.5 와 같게)
  scripts/rsl_rl/  train.py, play.py, cli_args.py, snapshot_rollback.py
  scripts/kick_v1_5/
    eval_strength_isaac.py   세기별 공속·킥 성공·넘어짐·방향 오차 (아래 '성능' 표를 만든 도구)
    view_strength.py         세기별로 로봇 5 대를 나란히 세워 계속 차게 하는 GUI 보기
    export_onnx.py           체크포인트 → ONNX (Isaac 없이 torch 만)
    eval_belief_isaac.py     공 믿음 오차 종류별 (정상 / 오차 / 굴러감 / 헛발질 / 다툼) 넘어짐
    probe_sim_version.py     Isaac Sim 4.5 ↔ 5.x 관절 마찰 차이 확인용
    trace_isaac.py           한 장면 궤적을 npz 로 (MuJoCo 와 비교용)
    calib_ball_roll.py       공 구름 저항 보정 (MuJoCo 기준값은 주석에 적혀 있다)
    gen_handoff_bank.py      넘겨받기 상태 은행 다시 만들기 — XM_Walking_Policy 의 보행 V0-6 / model_13500 이 필요하다
  scripts/list_envs.py
  robotmodels/mx_bot/usd/XM_robot04_convexhull/   로봇 USD (메시 볼록 껍질 충돌, 폴더 안 상대경로라 그대로 동작)
rsl_rl_patch/ppo_huber_nan_guard.diff         학습에 쓴 rsl_rl 로컬 수정분 (보행·기상 패키지와 같은 파일)
```

## 사용법
unitree_rl_lab 레포가 이미 있다는 가정이다. 덮어쓰기 전에 기존 파일을 백업할 것.
`mdp/`, `agents/`, `mx_bot.py`, `scripts/rsl_rl/train.py`·`play.py` 가 교체되고 `utils/joint_friction_compat.py` 가 추가된다
(`mx_bot.py` 가 이 파일을 import 하므로 같이 복사해야 한다).
이미 `robots/MX3/` 에 다른 태스크가 있으면 `__init__.py` 는 덮어쓰지 말고 등록 블록만 붙여 넣는다.
보행 패키지 (XM_Walking_Policy) 와 같이 쓰려면 보행을 먼저 덮고 킥을 나중에 덮는다 (킥 쪽 `mdp/`·`mx_bot.py` 가 더 새 것이고 보행 것을 포함한다).
이때 `robots/MX3/__init__.py` 는 두 패키지의 등록 블록을 합친다. (두 패키지를 함께 덮은 상태의 보행 재현은 따로 확인하지 않았다.)

```bash
cp -r unitree_rl_lab_overlay/. <unitree_rl_lab 경로>/
```

눈으로 보기 (세기 0 / 0.25 / 0.5 / 0.75 / 1 로봇 5 대, 머리 위 공 크기·색 = 세기):
```bash
python scripts/kick_v1_5/view_strength.py --checkpoint <이 폴더>/policy/model_22300.pt
```
`--bank` 를 주면 서서가 아니라 걷던 자세 (넘겨받기 상태) 에서 시작한다. `--levels 0,0.5,1` 처럼 세기를 고를 수 있다.

재생 (학습 환경 그대로, 공 위치·목표·세기 무작위):
```bash
python scripts/rsl_rl/play.py --task Unitree-MX3-WalkKick-V1-5B-Play --checkpoint <이 폴더>/policy/model_22300.pt
```

세기 평가 (아래 '성능' 표):
```bash
python scripts/kick_v1_5/eval_strength_isaac.py --headless --checkpoint <이 폴더>/policy/model_22300.pt
```

학습 (이 체크포인트에서 이어서). `logs/rsl_rl/unitree_mx3_walkkick_v1_5b/<런 이름>/` 에 `model_22300.pt` 를 넣고:
```bash
python scripts/rsl_rl/train.py --task Unitree-MX3-WalkKick-V1-5B --headless --num_envs 16384 \
    --resume --load_run <런 이름> --checkpoint model_22300.pt --max_iterations 1000
```
원래 학습은 `--load_run 2026-11-01_04-45-51 --checkpoint model_21500.pt --max_iterations 3000` (시도 2 의 21500 에서 이어서) 이었다.
킥 세기 하한 커리큘럼 (0.8 → 0.35) 은 체크포인트에 저장되지 않아 이어 학습할 때마다 0.8 에서 다시 시작한다 (보통 500 iter 안에 다 내려온다).

## 정책 입출력 (실기 러너용, 자세한 값은 `policy/deploy.yaml`)
| 항목 | 값 |
|---|---|
| 제어 | 50 Hz (step_dt 0.02), 위치 PD, `q_target = default + 0.25 * action` |
| 관절 | 다리 12 (r_pelvis_pitch, r_pelvis_roll, r_pelvis_yaw, r_knee_pitch, r_ankle_pitch, r_ankle_roll, 그다음 l_ 같은 순서). **팔·목은 제어하지 않는다** (실기처럼 기본자세 유지: 어깨 피치 ∓0.3 · 팔꿈치 ±1.0, 학습에선 팔 게인 20 / 1, 토크 0.3~0.8 N·m 무작위) |
| 게인 | 다리 kp 50 / kd 2.5 |
| 기본자세 | pelvis_pitch ±0.62, knee ±1.2, ankle_pitch ±0.62 (r +, l −), 나머지 0 |
| 관측 | 53 × 5 프레임 = 265. 한 프레임 = 각속도 3 (×0.2) / 중력벡터 3 / 속도명령 3 / 보행위상 3 (cos, sin, 주파수/2) / 관절각(기본자세 기준) 12 / 관절속도 12 (×0.05) / 직전 액션 12 / **공 위치 2** / **목표 방향 2** / **킥 세기 1**. 히스토리는 term 단위로 묶이고 오래된 것이 먼저 |
| 공 위치 | 몸통 기준 공 (x, y) [m]. **킥 시작 순간 한 번 읽고 고정** (그 뒤 로봇이 걸어가도 안 바뀜) |
| 목표 방향 | 몸통 기준 (cos, sin). 시작 순간의 세계 고정 방향을 자이로 적분한 yaw 로 계속 다시 투영 |
| 킥 세기 | 0~1. 목표 공속 = 최대 킥 × (0.35 + 0.65 s). 학습은 40 % 가 s = 1, 나머지 U(0, 1). 경기 시뮬은 1 고정 |
| 속도명령 | 공까지 걸어가는 전진 명령, 킥 동안 고정. 학습 vx 0.1~0.5 m/s · vy 0 · wz 0. **0.15 이상 권장** (0.1 은 보행위상 정지 문턱과 같아 걷지 않을 수 있다). 러너 기본 0.30 |
| 보행위상 | 2.0 Hz 고정. 킥 시작 = 위상 0, 홀드 동안 주파수 0 |
| 학습 분포 | 공 거리 0.20~0.50 m, 공 방위 ±70°, 목표 방향 ±90° (공·목표 위치 노이즈 ±5 cm / ±3°) |
| 액션 지연 | 학습 시 1~4 스텝 랜덤 |
| 관절 목표 추종 | Profile Velocity 40~50 rpm 으로 목표를 따라감 (실기 다이나믹셀 45 rpm 설정과 같게) |
| 넘겨받기 | 시작 0.3 s 기본자세 홀드 → 정책. 학습 85 % 는 보행 V0-6 13500 이 걷다가 명령 0 을 0.04 s 받은 직후 상태에서 시작 (은행) |

실기 러너는 `humanoid_policy_runner/actor_onnx_walkkick_v1_runner.py` (mx3_new_src) 이고, 런치 기본값은
`kick_duration 2.0 s` (트리거 뒤 2.0 s 에 보행 러너로 넘김), `kick_start_hold 0.3 s`, `kick_cmd_vx 0.30`, `kick_ball_x 0.30`, OOD 게이트 켬 이다.
러너의 공 방위 게이트 기본값 (±90°) 은 학습 분포 (±70°) 보다 넓다.

## 성능 (sim)
**Isaac (이 패키지 코드로 2026-11-12 재측정, 서서 시작, 공 정상, 외란 끔, 500 env):**

| 세기 s | 목표 공속 | 공속 (4.5) | 공속 (5.1 호환) | 학습 당시 기록 (4.5) | 킥 성공 | 넘어짐 | 방향 오차 |
|---|---|---|---|---|---|---|---|
| 0    | 0.98 | 1.21 | 1.22 | 1.22 | 93~98 % | 0 %   | 14~16° |
| 0.25 | 1.43 | 1.57 | 1.58 | 1.58 | 99~100 % | 0 %  | 10~11° |
| 0.5  | 1.89 | 2.12 | 2.18 | 2.13 | 99~100 % | 0 %  | 8~10° |
| 0.75 | 2.34 | 2.48 | 2.56 | 2.47 | 99~100 % | 0 %  | 6~9° |
| 1    | 2.80 | 2.65 | 2.69 | 2.56 | 98~100 % | 0~1 % | 6~7° |
(공속 = 목표 방향 최고 공속 [m/s]. 표준편차 0.14~0.41)

**MuJoCo (경기 시뮬과 같은 브리지 물리, 서서, 공 이동 거리는 중앙 (10~90 %))** — 이 패키지에 MuJoCo 도구는 없다 (아래 '알아둘 것'):

| 세기 s | 공속 | 공 이동 거리 | 방향 오차 | 넘어짐 |
|---|---|---|---|---|
| 0    | 1.94 m/s | 0.83 m (0.60~1.11) | 21.1° | 0 % |
| 0.25 | 2.44 m/s | 1.23 m (0.98~1.54) | 14.3° | 0 % |
| 0.5  | 3.28 m/s | 2.10 m (1.70~2.57) | 6.5°  | 0 % |
| 0.75 | 4.09 m/s | 3.15 m (2.66~3.78) | 2.2°  | 0 % |
| 1    | 4.49 m/s | 3.73 m (3.09~4.49) | 3.9°  | 0.3 % |
MuJoCo 경기 넘겨받기 조건 묶음 (서서 / 옆걸음 / 회전 / 공이 +3·+4 cm 멂 / 굴러감) 최대 넘어짐 0.7 %, 헛발질 넘어짐 0.7 %.

**ROS 4v4 경기 시뮬 (세기 1 고정, 킥 527 회):** 킥 중 넘어짐 5.3 % (이전 18200 은 8.5 %), 공 3 s 이동 중앙 2.20 m (1.85 m), 골 수는 같음.

## 알아둘 것
- **실기 미검증.** 위 수치는 전부 시뮬이다.
- **IMU 관측 지연에 약하다.** 학습엔 IMU 지연이 없다 (액션 지연 1~4 스텝만). MuJoCo 경기 넘겨받기 재생에서 킥 러너의 자이로·중력만 늦추면
  넘어짐이 0 / 5 / 10 / 20 ms → 0.2 / 1.3 / 3.2 / 19.7 % 로 는다 (차점 체크포인트 22800 은 2.1 / – / 2.4 / 3.8 %). 실기 IMU 지연을 재 보고 쓸 것.
- **위험 조건: 오른쪽으로 돌며 (wz < −0.3) 넘겨받아 오른쪽으로 차는 왼발 킥.** 경기 시뮬에서 이 조건만 7 %, 거울 조건 0 %.
  원인은 킥 시작 보행위상이 늘 0 이라 좌우가 비대칭이 되는 것 (정책·모델 자체는 대칭). 러너에서 "목표가 오른쪽이면 시작 위상 0.5" 로만 바꿔도
  MuJoCo 에서 IMU 20 ms 19.7 → 10.6 %, 10 ms 3.2 → 1.4 % 로 준다 (재학습 없이). 지금 러너에는 들어가 있지 않다. 자세한 건 `training/MATCH_FALLS_22300.md`.
- **Isaac 에서 오래 학습할수록 MuJoCo 강건성을 잃는다.** 같은 run 의 24000 이후는 Isaac 지표는 멀쩡한데 MuJoCo 에서 넘어짐 15~49 % 로 무너졌다.
  22300 은 21600~24400 을 MuJoCo 로 전부 훑어 고른 것이다 (차점 23900 · 22100 · 22800). 이어 학습하면 체크포인트를 sim2sim 으로 다시 고를 것.
- **킥 길이:** 러너는 트리거 2.0 s 뒤 보행으로 넘긴다. 경기 시뮬에서 남은 넘어짐은 킥 시작 뒤 1.6~2.6 s 에 몰리는데, 3.0 s 로 늘려도 줄지 않았다 (킥 동작 후반 자체 문제).
- **Isaac Sim 5.x 에서 돌릴 때:** 이 정책은 Isaac Sim 4.5 에서 학습했고, `mx_bot.py` 의 관절 마찰 `friction=1.1` 은 4.5 의 마찰 "계수"
  (관절 하중에 비례, 움직이는 동안에도 걸림) 기준 값이다. Isaac Sim 5.x 는 관절 마찰 모델이 (정지·운동·점성) 으로 바뀌어,
  그대로 두면 이 값이 정지 마찰 1.1 N·m 로만 들어가고 **움직이는 관절의 마찰은 0** 이 된다. 킥에서는 다리가 더 빨리 돌아
  s = 1 공속이 2.83 → 3.22 m/s, 넘어짐 0 → 2 % 가 된다. `utils/joint_friction_compat.py` 가 5.x 에서 4.5 계수 모델로 되돌린다
  (`mx_bot.py` 를 import 할 때 자동 적용, 시작할 때 `[joint_friction_compat] ... 4.5 계수 모델로` 가 찍힌다). 4.x 에서는 아무것도 안 한다.
  5.x 의 새 모델을 쓰려면 `UNITREE_JOINT_FRICTION=native`.
- **시뮬 공속은 시뮬마다 다르다.** 같은 정책이 Isaac s = 1 에서 2.7 m/s, MuJoCo 에서 4.5 m/s 로 찬다 (공 물리 차이). 그래서 킥 세기는 절대 공속이 아니라
  "최대 킥 대비 비율" 로 정의했다. 실기에서 쓰려면 실기에서 s → 공 이동 거리 표를 다시 잴 것.
- 학습의 공은 실제로 구르도록 했고 (공 회전 상한 해제 + 구름 저항 1.6 m/s², MuJoCo 경기 공에 맞춤), 공 믿음 오차 (정책이 아는 공과 실제 공이 어긋남:
  고정 오차 45 % / 굴러가는 중 10 % / 헛발질 5 % / 다툼 5 %) 를 넣었다. 강건성 수치는 이 조건들을 포함한다.
- 시작할 때 `[mx_bot] Active robot: XM_robot04 → .../XM_robot04.usd` 가 찍히지만 킥 태스크는 `XM_robot04_convexhull_kick` 를 쓴다
  (기본 보행 로봇 경로를 알려 주는 로그일 뿐이고, 이 패키지에 XM_robot04 USD 는 없다).
- MuJoCo 도구 (경기 넘겨받기 재생 `kick_match_cond.py`, `replay_delay.py`, 체크포인트 훑기 `sweep_ckpts.sh` 등) 는 원래 레포의 `scripts/mujoco_sim/`
  (MuJoCo 로봇 모델·브리지 물리) 에 기대고 있어 넣지 않았다. 위 MuJoCo 수치는 그 도구로 잰 기록이다.
- **rsl_rl 패치:** value loss 를 Huber 로 바꾸고 NaN gradient 때 optimizer step 을 건너뛰는 수정이 들어간 rsl_rl 3.1.2 로 학습했다.
  추론에는 영향이 없다. 재학습할 때 결과를 맞추려면 `rsl_rl_patch/` 의 diff 를 `rsl_rl/algorithms/ppo.py` 에 적용한다.
- `train.py` 는 value loss 발산 때 이전 스냅샷으로 되돌리는 `snapshot_rollback.py` 를 쓴다 (같이 들어 있다).
- `mdp/` 는 프로젝트 전체가 공유하는 패키지라 이 태스크와 무관한 파일 (보행, 기상, RS01·RS02 용) 도 들어 있다.
  `mdp/__init__.py` 가 전부 import 하므로 지우지 말 것. 학습 뒤에 바뀐 공유 파일이 있지만 (킥 시계 `kick_clock.py`, 보행위상 옵션 등)
  이 태스크에는 기존 동작 그대로다 — 지금 코드로 만든 설정을 학습 당시 `training/env.yaml`·`agent.yaml` 과 비교해 차이가 없음을 확인했다.
