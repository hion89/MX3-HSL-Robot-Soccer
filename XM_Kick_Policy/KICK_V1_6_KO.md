# WalkKick V1-6 — 디딤발 강건화 이어 학습 (2026-11-16)

현재 상태: 코드 작성과 로컬 오프라인 검증까지 했다. 서버 설치·학습, 새 정책 생성, 로봇 배포는 아직 하지 않았다. 근거 측정은 `sim2real_evidence/kick_robust_20261116/FINDINGS_KO.md`에 있다.

## 1. 왜, 무엇을 바꾸나

킥 후 디딤발이 버티지 못하고 돌거나 밀리며 무너지는 현상을 MuJoCo에서 원인별로 쟀다. 22300 정책은 학습 조건에서 2~3%만 넘어진다. 하지만 아래 조건 각각에서 20~45%가 넘어지고, 겹치면 80% 안팎이 된다.
- 실제 XM540 속도-토크 곡선과 전압 저하.
- 관측 지연(실물 러너의 프레임 순서와 IMU).
- 낮은 미끄럼 마찰.

그래서 바닥 마찰 하나가 아니라 다섯 갈래를 학습에 넣었다. 관측 265, 행동 12, 관절 순서, action_scale 0.25는 그대로라 러너는 ONNX 파일만 바꿔 쓸 수 있다.

1. **ACT — 구동기.** 다리 6그룹을 `VoltageDCMotor`로 바꾼다. 12V 기준 스톨 7.3N·m, 무부하 5.55rad/s, 상한 5N·m에, 에피소드마다 전압 10.6~12.4V를 뽑는다.
   22300에서 바로 실기 곡선을 주면 넘어짐이 커서, 평평한 5N·m와 곡선을 β 비율로 섞는다. 넘어짐 EMA가 12% 아래로 내려갈 때마다 β를 0.25에서 0.25씩 올려 1.0까지 간다.
   팔·목과 `mx_bot.py` 전역 설정(보행·기상 태스크)은 바꾸지 않는다.
2. **OBS — 관측 지연.** 자이로·중력·관절 관측을 env별로 늦춘다.
   - 에피소드의 50%에 러너 프레임 순서와 같은 1스텝(20ms) 지연을 준다.
   - IMU에는 여기에 0~1.5스텝(0~30ms)을 더한다.
   - 기존 행동 지연 1~4스텝은 유지한다.
3. **SURF — 바닥.**
   - 기존 전신 마찰 랜덤(0.45~1.35) 위에 발바닥만 정지 0.30~1.30 / 운동 0.25~1.05로 다시 뽑는다.
   - Isaac의 패치 마찰은 디딤발 비틀림을 거의 다 잡는다. 원 작성자도 torsional_patch_radius 등을 바꿔 봤지만 재현이 안 됐다.
     그래서 접지한 발에 "잔디가 못 잡는 만큼"의 yaw 토크를 외력으로 준다: τ = c·Fn·tanh(ω_z/0.5), c ∈ 0~0.03m, 에피소드의 30%는 0.
4. **CTR — 러너 계약.** 시작 hold를 0.3~0.6초로 뽑고(실물 0.6), 에피소드의 50%에 러너와 같은 행동 저역 필터(α = exp(−dt/τ), τ 0.015~0.03초)를 건다. last_action 관측은 러너처럼 필터 전 값이다.
5. **REW — 보상.** 접지한 발의 yaw 각속도 제곱 벌점(-0.05, 킥 후 1초 ×3)을 추가한다. 기존 보상과 종료 조건은 그대로다.

등록 태스크:
- `Unitree-MX3-WalkKick-V1-6A`: 구동기만.
- `Unitree-MX3-WalkKick-V1-6B`: A + 관측 지연.
- `Unitree-MX3-WalkKick-V1-6C`: 전체(B + 바닥·계약·보상).
- 재생: `-V1-6-Play`, `-Play-Turf`, `-Play-IMU20`, `-Play-Volt106`. 커리큘럼 없이 β=1이다.

새 파일:
- `unitree_rl_lab_overlay/source/.../mdp/kick_robust_mdp.py`
- `.../robots/MX3/walk_kick_v1_6_env.py`
- `scripts/kick_v1_6/prepare_server_v16.sh`, `scripts/kick_v1_6/train_v16.sh`

수정 파일은 `robots/MX3/__init__.py` 하나다(V1-6 등록 블록 추가, 기존 V1-5/V1-5B 등록은 그대로). 기존 mdp 공유 파일은 고치지 않았다.

## 2. 로컬에서 확인한 것과 못 한 것

확인한 것:
- 새 Python 파일 5개의 Python 3.10 AST·컴파일.
- `kick_robust_mdp.py`를 그대로 불러온 단위 시험 13개(`sim2real_evidence/kick_robust_20261116/test_kick_robust_math.py`). 대상은 다음과 같다.
  - DC 사분면 규칙, 전압 배율, β 혼합. MuJoCo 평가기 규칙과 200개 무작위 값으로 대조했다.
  - 지연 보간, DelayedObs의 같은 스텝 중복 push 방지와 리셋 후 다시 채우기.
  - 비틀림 토크 부호와 비접지 0, 러너 필터 식 일치, hold 범위, 전압 샘플, 커리큘럼 단계 상승과 정지.
- 서버 준비 스크립트를 가짜 대상 폴더로 시험했다. 미커밋 대상 거부, 덮어쓸 파일 백업과 복원, overlay 복사, 재개 체크포인트 배치, 재실행 거부를 확인했다. bash 문법과 LF도 확인했다.

못 한 것: 이 PC에는 Isaac Sim/Lab이 없어 태스크 생성, 물리 실행, 학습 속도는 확인하지 못했다. Isaac Lab 2.3 API에 맞춰 썼지만 서버 첫 실행(아래 SMOKE)에서 확인해야 한다.

## 3. 서버 실행 절차

### 3-0. 기본 경로: Isaac-work-space push → 서버 pull (2026-11-16 사용자 지정)

예전처럼 PC에서 `hion89/Isaac-work-space`에 push하고, 서버가 `~/isaac_ws`에서 pull해 학습한다.

PC에서 할 일:
- `python scripts/assemble_kick16_isaac_ws.py --isaac-repo C:/Users/jes/Desktop/isaac_ws/isaac_ws`로 `unitree_rl_lab_kick16/`을 만든다. 구성은 기존 `unitree_rl_lab` 커밋 파일 + 킥 overlay + 22300 체크포인트 + 설정 스크립트다.
- 스크립트가 출력한 `git add` 명령으로 그 폴더만 스테이징한 뒤 commit·push한다. Isaac-work-space에 이미 있던 미커밋 변경은 함께 올리지 않는다.
- 첫 반영은 Isaac-work-space `0965cb8`이다. GitHub에서 새로 받은 사본으로 체크포인트 SHA256, 실제 바이너리(LFS 포인터 아님), LF를 확인했다.
- Isaac-work-space는 Git LFS를 쓰지 않으므로, 조립 폴더의 USD와 체크포인트는 일반 blob으로 저장했다.
- 원본을 고친 뒤 다시 반영할 때도 같은 두 단계(조립 → 그 폴더만 commit·push)를 반복한다.

서버에서 할 일:
```bash
cd ~/isaac_ws && git status --short | head && git pull --ff-only
conda create -n env_kick16 --clone env_isaaclab      # 처음 한 번 (기존 env 는 그대로)
conda activate env_kick16
pip install -e ~/isaac_ws/unitree_rl_lab_kick16/source/unitree_rl_lab   # 처음 한 번
cd ~/isaac_ws/unitree_rl_lab_kick16 && bash kick16_setup.sh             # KICK16_SETUP_DONE
SMOKE=1 bash scripts/kick_v1_6/train_v16.sh C
tmux new -s kick16 && bash scripts/kick_v1_6/train_v16.sh C
```

서버의 `~/isaac_ws`에 로컬 수정이 있으면 `git pull --ff-only`가 거부될 수 있다. 이때는 `git status` 출력을 보고 판단하고, reset으로 지우지 않는다.

#### 컨트롤센터(GUI)로 학습하기 (Isaac-work-space `robotmodel/XM540_robot/scripts/control_center.sh`)

- Task ID가 `Unitree-MX3-`로 시작하면, 컨트롤센터는 `<unitree_rl_lab Path>/scripts/rsl_rl/train.py`와 `play.py`를 `isaaclab.sh -p`로 그 폴더에서 실행한다. 킥 패키지의 스냅샷 되돌림과 체크포인트 탐색이 그대로 유지된다.
- XM540 태스크는 기존 IsaacLab 스크립트 + `launch_external.py` 경로 그대로다.
- `isaaclab.sh -p`는 활성화된 conda 환경의 python을 쓰므로, `conda activate env_kick16` 상태에서 `control_center.sh`를 실행한다. 화면이 필요하므로 MobaXterm X11로 접속한다.
- Settings 탭 값:
  - IsaacLab Path: `/home/bami122/IsaacLab`
  - unitree_rl_lab Path: `/home/bami122/isaac_ws/unitree_rl_lab_kick16`
  - TensorBoard Logs: `/home/bami122/isaac_ws/unitree_rl_lab_kick16/logs/rsl_rl/unitree_mx3_walkkick_v1_6c`
  - Task ID: `Unitree-MX3-WalkKick-V1-6C`
  - Experiment Name: `unitree_mx3_walkkick_v1_6c`
- XM540 값은 먼저 프로필(`xm540`)로 저장해 둔다.
- Training 탭: Resume을 체크하고 체크포인트로 `init_22300/model_22300.pt`를 고른다(`kick16_setup.sh`가 배치한다). 이어 학습 체크포인트는 `logs/rsl_rl/<실험>/<run>/` 안의 것만 허용한다. Parallel Environments는 16384, Maximum Iterations는 3000, Headless를 켠다.
- 실행 전에 Preview Command로 `.../unitree_rl_lab_kick16/scripts/rsl_rl/train.py ... --load_run init_22300 --checkpoint model_22300.pt`가 나오는지 확인한다. 짧은 확인은 Parallel Environments 64, Maximum Iterations 5로 먼저 돌린다.

아래 3-1 이후는 MX3-HSL-Robot-Soccer를 서버에 직접 clone하는 대체 경로다.

### 3-1. 대체 경로 (MX3 저장소 직접 clone)

서버 정보(인수인계 기록 기준, 현재 상태 미확인): `ssh bami122@163.180.145.122`, `/home/bami122/isaac_ws`, conda `env_isaaclab`. 기존 `unitree_rl_lab` 폴더와 과거 XM540 작업은 덮어쓰지 않는다.

1. 접속 후 상태를 기록한다.
   ```bash
   cd ~/isaac_ws && ls
   git -C unitree_rl_lab remote -v && git -C unitree_rl_lab log -1 --oneline && git -C unitree_rl_lab status --short | head
   conda activate env_isaaclab && python -c "import isaaclab,rsl_rl,torch;print(torch.__version__)"; nvidia-smi
   ```
2. 별도 복사본을 만든다(기존 폴더는 그대로 둔다). 미커밋 변경이 있으면 clone 대신 `cp -a`를 쓴다.
   ```bash
   cd ~/isaac_ws && git clone unitree_rl_lab unitree_rl_lab_kick16
   ```
   패키지 설치 경로는 conda 환경에 하나만 등록된다. 기존 환경을 건드리지 않으려면 환경을 복제해 쓴다.
   ```bash
   conda create -n env_kick16 --clone env_isaaclab && conda activate env_kick16
   pip install -e ~/isaac_ws/unitree_rl_lab_kick16/source/unitree_rl_lab
   ```
3. 이 저장소를 서버로 가져온다. 공개 저장소라 서버에서 clone할 수 있다. 서버가 GitHub에 접근하지 못하면 Windows에서 `XM_Kick_Policy` 폴더를 scp로 보낸다.
   ```bash
   cd ~/isaac_ws && git clone https://github.com/hion89/MX3-HSL-Robot-Soccer.git
   bash MX3-HSL-Robot-Soccer/XM_Kick_Policy/unitree_rl_lab_overlay/scripts/kick_v1_6/prepare_server_v16.sh \
        MX3-HSL-Robot-Soccer/XM_Kick_Policy unitree_rl_lab_kick16
   ```
   `PREPARE_V16_DONE`이 나와야 한다. 출력에 rsl_rl 패치가 "NOT applied"로 나오면, 안내된 patch 명령을 검토한 뒤 복제한 환경에서만 적용한다.
4. 태스크가 만들어지는지 짧게 확인한다(64 env, 5 iteration).
   ```bash
   cd ~/isaac_ws/unitree_rl_lab_kick16 && SMOKE=1 bash scripts/kick_v1_6/train_v16.sh C
   ```
   로그에서 확인할 것:
   - `[kick_robust] VoltageDCMotor` 6줄.
   - `[kick_robust] dc_blend 커리큘럼 시작 β=0.25`.
   - `[kick_handoff] 넘겨받기 상태 120000 개`.
   - 오류 없이 5 iteration이 진행됨.

   실패하면 학습을 시작하지 말고 오류 원문을 보관한다.
5. 본 학습. C를 먼저 돌리고, 자원이 되면 A와 B를 비교용으로 돌린다.
   ```bash
   bash scripts/kick_v1_6/train_v16.sh C          # 기본 16384 env, 3000 iteration
   ```
   TensorBoard에서 볼 것:
   - `Curriculum/dc_blend`: 1.0까지 올라가는지.
   - `Episode_Termination/fall_*`.
   - `Episode_Reward/ball_velocity`, `stance_foot_yaw`, `metric_*`.

   β가 몇백 iteration 넘게 같은 값에 머물면 `walk_kick_v1_6_env.py`의 `DC_FALL_TH`(0.12)를 0.15로 올리는 것을 검토한다.
6. 체크포인트를 고른다. Isaac 지표만으로 고르지 않는다(원 작성자 기록: 오래 학습할수록 MuJoCo 강건성이 무너졌다). 런 폴더의 `model_*.pt`를 Windows PC로 복사한 뒤 실행한다.
   ```powershell
   cd C:\Users\jes\Desktop\MX3_Match_Sim\sim2real_evidence\kick_robust_20261116
   & C:\Users\jes\Desktop\isaac_ws\sim2real_evidence\.mujoco_env\Scripts\python.exe gate_checkpoints.py --ckpt-dir <복사한 런 폴더> --every 500 --trials 300
   ```
   `gate_out/ranking.md`에서 최악 조건 전도율이 가장 낮고 경고(킥 성공 < 95%, 공속 < 기준 85%)가 없는 후보를 고른다. 1차 목표는 다음과 같다.
   - `real_like` 전도율을 22300(78%)의 절반 이하로.
   - 학습 순서 조건 3% 이하 유지.
   - 세기 1 공속 4.0m/s 이상(MuJoCo).

   300회 기준 95% 구간이 ±3~6%p이므로 차이가 그보다 작으면 같은 수준으로 본다.

## 4. 실물 적용 (후보가 생긴 뒤, 사용자 승인 후)

- 새 ONNX는 기존 파일을 덮지 않고 `hsl_bringup/models/walkkick_v1_6c_<iter>.onnx`처럼 새 이름으로 넣는다. `control.launch.py`의 `kick_actor_path`로 선택한다. 22300은 보존한다.
- 배포 계약은 학습 분포 안에 둔다. 현재 hold 0.6, 접근 0.20, 필터 τ 0.02는 모두 v1_6 학습 범위 안이다. kick_duration 2.0초는 hold를 포함하므로 정책 구간이 1.4초다. 후보의 킥 후 회복 시간을 MuJoCo 궤적으로 보고 함께 정한다.
- 순서는 다음과 같다.
  1. 사람이 지지할 준비를 한 상태에서 텔레옵 F 킥으로 연구실 바닥과 잔디를 비교한다.
  2. 그다음 attacker 역할 시험.
  3. 그다음 여러 대 경기.

  ONNX 교체는 텔레옵 F 킥과 자율 경기 모두에 적용된다.

## 5. 재학습 없이 먼저 시험해 볼 수 있는 것 (제안, 아직 적용 안 함)

- **킥 세기 낮추기.** MuJoCo에서 22300을 세기 0.5로 차면 실물 유사 복합 조건 전도율이 78%에서 0%로 줄고, 공속은 4.7에서 3.25m/s로 준다(원 기록상 이동 거리 약 2.1m).
  현재 `kick_strength`는 러너가 시작할 때 한 번만 읽고, `control.launch.py`가 넘기지 않아 항상 1.0이다. 시험하려면 launch에 인자를 추가하고 러너를 재시작해야 한다. 텔레옵과 자율 경기에 모두 적용되고 공 이동 거리가 짧아진다.
- **러너 관측 순서 맞추기.** 킥 러너가 추론 직전에 현재 센서 프레임을 history에 넣도록 바꾸면 학습 순서와 같아진다(MuJoCo 13%→2.3%).
  다만 실물 러너 코드 변경이라 텔레옵과 자율 경기 모두에 영향이 있다. 보행 러너도 같은 패턴인지 함께 확인해야 한다. v1_6은 두 순서를 모두 학습하므로, 수정하지 않아도 동작하도록 만들었다.

## 학습 환경 보정 확인 — V1-6-Play-RealLike (2026-11-19)

학습 환경에서 22300이 실물처럼 "차고 나서 넘어지는" 실패를 보여야 학습이 그 실패를 고친다. 첫 V1-6C 학습(22300→30600)은 Isaac 넘어짐이 출발점 수준에서 줄지 않았다. 모터 커리큘럼은 시작 직후 1.0이 되었다(min_episodes 20000이 16384 env에서 금방 참). `metric_*` 항목은 weight 0이라 TensorBoard에 항상 0으로 나온다.

`Unitree-MX3-WalkKick-V1-6-Play-RealLike`는 MuJoCo real_like와 같은 조건을 고정한다(11.1 V, 러너 프레임 1스텝+IMU 10 ms, hold 0.6 s, 필터 tau 0.02 s, 발 마찰 약 0.6, 비틀림 부족 c=0.03). 서버에서 22300과 새 체크포인트를 같은 조건으로 잰다.

    python scripts/kick_v1_5/eval_strength_isaac.py --headless --task Unitree-MX3-WalkKick-V1-6-Play-RealLike \
      --checkpoint kick16_init/model_22300.pt --levels 0.5,1

22300이 여기서도 많이 넘어지면 Isaac이 실물 실패를 재현하는 것이므로 학습 분포를 이 조건 쪽으로 옮긴다. 거의 넘어지지 않으면 Isaac 물리가 실패 원인을 재현하지 못하는 것이므로 원인 조건을 하나씩 바꿔 찾은 뒤 학습한다.

## 22300이 V1-6-Play에서 공을 안 찬 문제 — 원인 분리 (2026-11-19)

사용자 GUI 재생에서 22300이 V1-6 재생 환경에서 공을 차지 않았다. 원래 22300의 다리 모터는 ImplicitActuator(PhysX 내부 PD)인데, V1-6의 VoltageDCMotor는 Explicit PD라서 곡선을 끈 β=0에서도 동역학이 달라진다. MuJoCo 평가기는 위치 액추에이터에 힘 범위만 잘랐으므로 이 전환을 시험하지 않았다. 아직 확인 전의 유력 후보다.

변경을 하나씩만 켠 재생 태스크로 확인한다. `Unitree-MX3-WalkKick-V1-6-Play-Diag-ActFlat`(Explicit, β=0), `-Diag-Act`(β=1+전압), `-Diag-Obs`, `-Diag-Surf`, `-Diag-Ctr`가 있다. 기준은 `Unitree-MX3-WalkKick-V1-5B-Play`다. 각 태스크에서 `eval_strength_isaac.py --levels 1`로 22300의 킥 성공·넘어짐을 비교한다.

### 진단 결과와 V1-6D (2026-11-19, 서버 실측)

22300, 세기 1, `eval_strength_isaac.py` 500 env 결과는 다음과 같다. V1-5B는 킥 성공 99%, 넘어짐 0.2%다. Diag-ActFlat(Explicit 모터, 곡선 끔)과 Diag-Act, 전체 V1-6은 킥 성공 0%다. Diag-Obs는 99%/1.4%, Diag-Ctr는 99%/0.2%, Diag-Surf는 82%/6.0%다.

결론: Implicit→Explicit 모터 전환만으로 22300이 공을 차지 못한다. 그래서 V1-6A/B/C 학습(22300→30600)은 잘못된 모터 위에서 진행된 것이라 쓰지 않는다. 바닥 조건은 "차지만 넘어지는" 실물 실패를 Isaac에서 재현한다.

`Unitree-MX3-WalkKick-V1-6D`는 C에서 Explicit DC 모터만 뺀 학습 태스크다(22300과 같은 ImplicitActuator). 모터 한계는 v1_4의 torque_envelope 벌점이 그대로 맡는다. 재생은 `V1-6D-Play`, 실물 유사 고정 조건은 `V1-6D-Play-RealLike`(전압 제외)다. 학습 전에 22300이 D-Play에서 공을 차는지(킥 성공 90% 이상) 먼저 확인한다.

22300 확인 결과(서버, 2026-11-19, 500 env, 세기 0.5/1, 킥 성공/넘어짐):
- V1-5B-Play: 100%/0.0%, 100%/0.0%.
- V1-6D-Play: 81%/8.0%, 78%/6.8%. 공을 차면서 넘어짐이 생기므로 학습 환경으로 쓴다.
- V1-6D-Play-RealLike: 세기 0.5에서 27%/35.6%(방향 오차 30°). 실물 유사 고정 조건이 22300을 크게 무너뜨리므로 평가용으로 쓴다.

같은 실행에서 `kick16_setup.sh`의 rsl_rl 패치 확인이 "dry-run failed"로 나왔다. 패치 적용 여부를 직접 확인해야 한다.

### V1-6D 학습에서 차기 포기 → V1-6E (2026-11-19)

V1-6D 학습(22300→22880, 8192 env)의 TensorBoard 값을 보면 정책이 차기를 포기했다. `kick_resolved`는 37%에서 1%로, `no_kick_deadline`은 45%에서 94%로 바뀌었는데 `mean_reward`는 3.8에서 6.0으로 올랐다. 기존 보상에는 안 차는 것 자체의 비용이 없다. 찬 뒤 넘어짐만 −6이라 바닥 조건이 어려워지면 안 차는 쪽이 유리하다.

`Unitree-MX3-WalkKick-V1-6E`는 D에 두 가지를 바꾼다. 하나는 안 차기 벌점 `no_kick`이다. 데드라인까지 안 차면 1회 −400×dt=−8을 주고, 헛발질·다툼 에피소드는 제외한다. 다른 하나는 `stance_foot_yaw` 가중치를 −0.05에서 −0.02로 낮춘 것이다. 22300에서 다시 시작하고, `kick_resolved`·`no_kick_deadline`·`ball_velocity`를 먼저 감시한다. 재생·평가는 V1-6D-Play와 -RealLike를 그대로 쓴다.
