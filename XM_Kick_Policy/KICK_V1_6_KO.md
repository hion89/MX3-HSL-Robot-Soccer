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

### 대조 실험과 V1-6F (2026-11-19)

V1-5B를 그대로 22300에서 250 iteration 이어 학습하면 결과가 좋아진다. `kick_resolved`는 45%에서 78%로, `no_kick_deadline`은 37%에서 11%로 바뀌고 노이즈도 0.35에서 0.36으로 안정적이다. 서버 학습 설정(rsl_rl 3.0.1, 8192 env, 패치)은 원인이 아니다.

V1-6E를 다시 학습하면 약 700 iteration 동안 공 속도와 성공 보상이 0 근처에서 회복하지 않는다. V1-6D가 실패한 원인은 시작 시 디딤발 yaw 벌점(−0.049)이 킥 보상 합(약 +0.031)보다 커서 차는 쪽이 손해였기 때문으로 본다. V1-6E에서는 안 차기 벌점(−8)을 넘어져서 피했다(넘어짐 4배). 넘어짐 자체에는 벌점이 없다.

`Unitree-MX3-WalkKick-V1-6F`는 보상을 22300(V1-5B)과 완전히 같게 두고 환경만 바꾼다(관측 지연, 바닥, hold·필터). 대조 실험처럼 짧게 돌려 22300이 공을 계속 차는지 먼저 본다.

### V1-6F 결과와 체크포인트 평가 (2026-11-19)

V1-6F는 22300에서 25300까지 학습했다(런 `2026-11-19_17-28-39_f_from22300`). 23500 무렵까지는 공을 계속 차면서 넘어짐이 절반으로 줄었다. 24000부터는 차기가 줄었다. `kick_resolved`는 0.32에서 0.09로, `no_kick_deadline`은 0.57에서 0.84로 바뀌었다. 그래서 32300까지 연장하던 학습을 25222에서 멈췄다. 체크포인트는 모두 보존했다.

Isaac 평가(`eval_strength_isaac.py`, 500 env, 세기 0.5/1)에서 22300과 F를 비교했다.
- V1-6D-Play: 22300은 킥 79%/71%, 넘어짐 6.4%/10.8%다. F 23400은 킥 62%/51%, 넘어짐 0.4%/0%이고, F 23500은 킥 55%/42%, 넘어짐 0%/0.4%다.
- V1-6D-Play-RealLike: 22300은 킥 31%/32%, 넘어짐 47%/39%, 방향 오차 30° 안팎이다. F 23000~23500은 킥 16~26%, 넘어짐 0.4~3.6%, 방향 오차 17~22°이고, 찬 공의 속도는 1.5~1.7 m/s로 22300(0.8~1.1)보다 빠르다.
- 여기서 킥 성공은 평가 시간 안에 공 속도가 1 m/s를 넘은 비율이다.

MuJoCo 게이트(`results_v16f_gate`, 조건마다 300회):
- 22300은 넘어짐 최악 89%, 평균 31.5%, real_like 78%다.
- F 23500은 최악 11%, 평균 3.0%, real_like 1.0%다. 다만 킥은 65% 안팎, 공 속도는 3.9 m/s(22300의 85%)다. 23000은 최악 20%, 23400은 최악 46%로 체크포인트마다 차이가 크다.
- 찬 시행에서는 킥 시점이 22300과 같다(중앙값 1.1~1.5 s). 안 찬 시행은 망설임이 아니라 거부다. 공이 몸 옆(방위 20° 밖)이나 멀리 있는 배치에 몰려 있다(킥 56~61%).

해석: 보상에 '안 차는 비용'이 없으니, 정책은 넘어질 위험이 큰 배치에서 차기를 건너뛰는 쪽을 배웠다. 찬 시행만 놓고 본 넘어짐(MuJoCo real_like)은 22300 78%, F 23500 1.5%, 23000 17%, 23400 58%다. 23500의 개선은 안 찬 시행 덕만이 아니지만, 체크포인트마다 편차가 커서 하나만 믿기 어렵다. 하지만 실전에서 35%를 건너뛰면 쓸 수 없다.

### V1-6G (2026-11-19)

`Unitree-MX3-WalkKick-V1-6G`는 F에 `end_no_kick` 벌점을 하나 더한다. 에피소드가 공을 안 찬 채 끝나면 종료 원인(데드라인, 넘어짐, 시간 초과)과 상관없이 1회 −200×dt=−4를 준다. 헛발질·다툼 에피소드는 제외한다. E는 데드라인에만 벌점을 줘서, 일찍 넘어지면 벌점을 피할 수 있었다. G는 원인과 무관하게 같은 값이라 그 탈출구가 없다. −4는 찬 뒤 넘어짐(−6)보다 작아서 "차고 넘어짐"을 "안 참"보다 낫게 만들지는 않는다. F model_23500에서 이어 학습한다(서버 `logs/rsl_rl/unitree_mx3_walkkick_v1_6g/init_23500`).

V1-6G 결과(23500→25114, 런 `2026-11-19_20-20-32_g_from23500`): 넘어짐은 F 수준(정상 약 2%)을 유지하고 E 같은 일찍 넘어짐은 없었다. 하지만 차기는 회복하지 않았다. `kick_resolved`는 0.15~0.26 사이를 오가다 0.16, `no_kick_deadline`은 0.63~0.75였고, 탐색 노이즈는 0.35에서 0.42로 올랐다. 1600 iteration을 보고 멈췄다.
- Isaac V1-6D-Play 킥은 G 24000~25100이 26~43%로, F 23500(55%/42%)보다 낮다. RealLike는 9~24%로 F와 비슷하다.
- MuJoCo 게이트(`results_v16g_gate`): G 24700은 최악 22%, real_like 18%, 킥 43%, 공 속도 3.6 m/s다. G 24000은 최악 27%, 킥 39%다. 넘어짐·킥·공 속도 모두 F 23500보다 나쁘다.
- 판단: −4 정도의 '안 차고 끝남' 벌점은 어려운 배치를 차게 만들지 못했고, 탐색만 늘려 정책을 흐트러뜨렸다. 지금 후보 중 가장 나은 것은 F model_23500이지만 킥 65%라 배포 기준(95%)에 못 미친다.

### V1-6H — 차기 전 플러스 보상 게이트 (2026-11-20)

사용자 제안: 공을 못 차면 점수를 비율로 깎아 곱하면, 공차기가 다른 보상의 전제 조건이 된다.

F·G에서 안 차는 에피소드는 데드라인까지 `alive`, `gait`, `capture_point`, `joint_pose_to_default` 같은 플러스 보상을 계속 받았다. 차면 `kick_resolved`로 에피소드가 일찍 끝나 그 수입을 잃었다. 그래서 넘어질 위험이 있는 배치는 건너뛰는 편이 이득이었다. G의 −4는 이 수입보다 작았던 것으로 본다.

H는 차기 전의 플러스 보상에 배율 g를 곱한다.
- 찼거나, 헛발질·다툼 에피소드면 g = 1이다.
- 차기 전에는 g = 1 − α·clamp(t/ramp, 0, 1)이다.
- 대상 항목은 `alive`, `gait`, `stop_posture`, `joint_pose_to_default`, `capture_point`, `track_lin_vel_x`다. 벌점, `ball_approach`, 킥 보상은 그대로 둔다.

구현할 때 지킬 점.
- 벌점까지 곱하면 안 찰수록 벌점이 줄어 거꾸로 된다. 그래서 플러스 값만 깎는다.
- 배율은 로봇마다 따로 계산한다. 전체 평균 킥 비율을 곱하면 개별 로봇에 학습 신호가 없다.
- 게이트 항 `kick_gate`는 보상 목록의 마지막이다. Isaac Lab RewardManager는 항을 순서대로 계산하며 `_step_reward`에 weight를 곱한 값을 넣는다. 마지막 항이 이번 스텝의 앞 항들 값을 읽어 −(1−g)·Σmax(항, 0)을 돌려준다.
- 텐서보드의 각 항목은 원래 값 그대로 보이고, 깎인 양은 `Episode_Reward/kick_gate`에 음수로 따로 보인다.

계수 시험: F model_23500에서 각 400 iteration을 돌린다.
- H는 α=1, ramp=에피소드 데드라인이다.
- H2는 α=1, ramp=1.5 s다.
- H3는 α=0.5, ramp=1.5 s다.
- 서버 `~/kick16_runs/sweep_h.sh`, 결과는 `sweep_h_summary.txt`.

V1-6H 계수 시험 결과(2026-11-20 학습, 평가는 11-23):
- 학습 400 iteration 동안 `kick_resolved`가 H 0→0.54, H2 0.54, H3 0.43으로 올랐다(F 23500 근처는 0.3 안팎). 넘어짐(정상)은 2~3.5%였다.
- Isaac V1-6D-Play 킥(세기 0.5/1, model_final 기준): H 82%/80%, H2 78%/78%, H3 63%/70%다. 22300은 79%/71%, F 23500은 55%/42%였다.
- Isaac RealLike 킥: 29~36%(22300 31%, F 16~26%), 넘어짐 1~9%(22300 39~47%)다.
- 점수(킥 평균 − 3×넘어짐 평균): H 48.6, H3 45.2, H2 41.5. 그래서 H로 장기 학습한다.
- MuJoCo 게이트(`results_v16h_sweep_gate`, H 23800/23899):
  - 킥은 95%/94%로 F 65%에서 크게 올랐다.
  - real_like 넘어짐은 14.7%로, 22300(78%)보다 훨씬 낮고 F 23500(1%)보다는 높다.
  - 최악 조건(low_mu, 미끄러운 바닥)은 34~38%이고, 공 속도는 3.7~3.9 m/s(22300의 82~85%)다.
- 해석: 게이트로 '안 차고 버티기' 수입을 없애자 정책이 다시 찬다. 대신 F가 피하던 어려운 배치도 차니 넘어짐이 일부 돌아왔다. 장기 학습에서 넘어짐이 다시 줄어드는지 본다.

운영 교훈: Isaac Sim이 시작 단계에서 멈추면 SIGTERM을 무시한다. `timeout 900`만 쓴 평가가 11-20 18:55부터 3일 동안 멈춰 있었고, 그 뒤 장기 학습도 시작되지 않았다. 평가와 학습은 `timeout -k 30 600`(SIGTERM 30초 뒤 SIGKILL)과 재시도로 감싼다.

장기 학습: H, F model_23500 → 29500(8192 env), 2026-11-23 15:17 시작. 서버 `~/kick16_runs/long_h.sh`가 끝난 뒤 500 간격 체크포인트를 Isaac으로 평가해 `long_h_summary.txt`에 남긴다.

### H 장기 학습 중 롤백과 액션 클리핑 (2026-11-23)

장기 학습 중 value loss가 2000을 넘어 자동 롤백이 반복됐다. 23673, 23675, 24126, 24133에서 일어났고 값은 최대 8.9e22였다. 롤백하면 모든 로봇을 리셋하고 에피소드 시계를 무작위로 다시 정하므로, 몇 iteration 동안 '안 차고 데드라인'이 86~97%, 에피소드 길이가 약 20으로 찍힌다. 정책이 무너진 것이 아니라 집계 왜곡이고, 약 10 iteration 뒤 원래 값으로 돌아온다.

전체 로그에서 롤백은 E 7회, H 6회(시험 2회 + 장기 4회)였다. 22300 대조, D, F, G, H2, H3에는 없었다.

유력한 원인은 킥 러너의 `clip_actions: null`이다. 정책 출력에 상한이 없어 가끔 폭주하고, 그러면 `action_rate`(차이의 제곱)가 폭발한다. 기상 V2에서 같은 원인이 추적된 적이 있다(2026-09-11, action_rate 2e15).

조치: `WalkKickClipPPORunnerCfg`(clip_actions=100)를 만들어 H, H2, H3 등록에 연결했다. scale 0.25 기준 관절 목표 25 rad라 정상 동작에는 영향이 없다. `kick_gate` 안에 진단을 넣어, 어떤 보상 항목이든 |값|이 1e4를 넘으면 `[REWARD_BLOWUP]`으로 항 이름·env·행동 크기를 출력한다.

장기 학습을 model_24600에서 다시 시작했다(16:21, 런 `2026-11-23_16-2x_long_from24600`, seed 43). 16:17에 잘못 시작했다 멈춘 같은 이름의 짧은 런 폴더도 남아 있다.

24500 MuJoCo 게이트: 킥 95~98%, real_like 넘어짐 14.7%(시험 23800과 같음)였다. 지연 없는 train_fresh_obs는 9%에서 47%로, baseline은 19%에서 31%로 나빠졌다. 지연이 있는 조건은 유지되고 깨끗한 입력에서만 나빠졌으니 경고 신호로 기록한다.

### H 장기 학습 결과와 최종 후보 (2026-11-23 종료, 11-24 정리)

롤백 없이 29500까지 끝났다. 클리핑 뒤 `[REWARD_BLOWUP]`이 130번 찍혔다(대부분 27200~27400의 action_rate, |행동|이 상한 100에 닿음). 하지만 value loss 발산이나 롤백은 없었다.

MuJoCo 게이트(300회 × 11조건, `results_v16h_long_N`)의 넘어짐:
- real_like: 24500 14.7%, 25500 8.7%, 26500 10.7%, 27500 0.7%, 28500 5.0%, 29500(model_final) 25.3%. 22300은 78.3%다.
- real_like_low_mu: 27500 4.0%, 28500 27.3%, 29500 59.7%다.
- 최악 조건: 27500 18.3%(imu_delay_20ms), 28500 27.3%, 29500 59.7%다. 22300은 89%다.
- 킥은 모든 체크포인트에서 97~100%다.
- 공 속도는 3.5~3.8 m/s로, 22300(4.55 m/s)의 78~84%다. 85% 기준에 못 미친다.

Isaac 평가(`~/kick16_runs/long_h_summary.txt`):
- 일반 조건(Play) 킥은 81~90%다.
- RealLike 킥은 30%에서 42~49%로 올랐고, 넘어짐은 27500·28000에서 0~0.4%다.
- 방향 오차는 15~23°로, 22300보다 크다.

결론: 마지막 체크포인트가 가장 좋지 않다. 28500 이후 미끄러운 바닥에서 다시 나빠진다. 최종 후보는 **H long model_27500**(런 `2026-11-23_16-20-57_long_from24600`)이고, 대안은 25500이다. 실물 시험은 아직 하지 않았다. 남은 과제는 공 속도(85% 미만)와 방향 오차다.

### V1-6I — 방향 배율을 곱한 공 속도 보상 (2026-11-24)

목표: H 27500의 안정성은 유지하고 공 속도(22300의 78%)와 방향 오차(16~22°)를 되찾는다.

원인: 기존 방향 항 `ball_offaxis`(−0.3 × 수직성분²)는 20° 오차에서 킥당 약 −0.06으로, 속도 보상(킥당 약 +3)에 비해 거의 영향이 없었다. 속도 보상의 투영 proj²도 20°에서 12%만 깎는다.

변경:
- `ball_velocity`를 `ball_velocity_dir_gated`로 바꿨다. 기존 세기 보상(`ball_velocity_strength`)에 방향 배율 dir = 0.3 + 0.7·exp(−(θ/20°)²)를 곱한다. 배율은 10°에서 0.82, 20°에서 0.56, 30°에서 0.38이다. 공속이 0.3 m/s 미만이면 배율은 1이다.
- 가중치는 I가 2.0 → 2.5, I2가 2.0 → 3.0이다.
- 나머지(kick_gate, fall_after_kick −6, 환경, clip_actions 100)는 H와 같다.

시험: H long model_27500에서 I, I2를 400 iteration씩 학습한다(서버 `~/kick16_runs/sweep_i.sh`, 결과 `sweep_i_summary.txt`). 그다음 Isaac 평가와 MuJoCo 게이트(`results_v16i_sweep_gate`)를 거쳐 장기 학습할 쪽을 고른다.

처음 실행은 보상 함수에 `**kwargs`를 써서 Isaac Lab 인자 검사에 걸려 시작하지 못했다. 그래서 인자를 그대로 나열하도록 고쳤다. 실패한 빈 런 폴더(체크포인트 없음)는 지웠다.

V1-6I 계수 시험 결과(400 iteration, 27500 → 27900):
- I(가중치 2.5)는 차기를 피했다. 학습 kick_resolved가 0.6에서 0.3으로 떨어졌고, 정상 넘어짐은 3%에서 8%로 늘었다.
  - Isaac Play 킥은 57%(27500은 88%)였다.
  - MuJoCo 킥은 74~80%, 공 속도는 2.97 m/s였다.
  - 넘어짐은 낮았다(real_like 2.3%, 최악 10.3%). 하지만 안 차서 생긴 결과다.
- I2(가중치 3.0)는 차기를 유지했다. kick_resolved는 0.57~0.61, 정상 넘어짐은 3.3%였다.
  - Isaac Play 방향 오차는 11°로 줄었다(27500은 16°, 22300은 11~14°). 공 속도(s=1)는 2.05에서 2.18 m/s로 올랐다. RealLike 방향 오차는 17° 정도였다.
  - MuJoCo 킥은 99~100%, 공 속도는 3.53에서 3.73 m/s로 올랐다(22300의 82%).
  - 넘어짐은 늘었다. real_like 0.7%에서 17%, low_mu 4%에서 45.7%, 최악 18.3%에서 45.7%다. H 시험 400 iteration 시점(real_like 14.7%)과 비슷한 수준이다.
- 판단: H도 장기 학습에서 넘어짐이 다시 줄었으므로 I2를 3000 iteration 이어 학습한다(27900 → 30900, 서버 `~/kick16_runs/long_i2.sh`, 결과 `long_i2_summary.txt`). 1000마다 MuJoCo 게이트로 확인하고, 넘어짐이 27500 수준으로 내려오지 않으면 27500을 유지한다.

I2 장기 학습 28900 MuJoCo 게이트:
- 공 속도는 3.89 m/s로, 22300의 85.5%다. 처음으로 85% 기준을 넘었다. 킥은 97~99%다.
- 넘어짐은 더 늘었다. real_like 24.7%, real_like_low_mu 50.3%, 최악 50.3%, 평균 20.9%다(27900은 17%, 45.7%, 45.7%, 24.9%).
- 학습 정상 넘어짐도 3.25%에서 3.9%로 조금씩 오른다.
- 29900에서 넘어짐이 줄지 않으면 I2를 멈춘다. 그 경우 속도 가중치를 2.0으로 되돌리고 방향 배율만 남기거나, 찬 뒤 넘어짐 벌점을 함께 올리는 쪽을 검토한다.

I2 장기 학습 결과(2026-11-24 18:05 종료, 30899, 롤백 없음):
- MuJoCo 넘어짐(29900 → 30500 → 30899): real_like 6.7% → 5.0% → 1.7%, real_like_low_mu 29.3% → 22.7% → 6.3%다.
  - 30899의 최악 조건은 imu_delay_20ms 24.0%이고, 평균은 10.2%다.
  - 킥은 96% 이상이고, 공 속도는 3.77 m/s(22300의 83%)다.
- Isaac 방향 오차는 Play 7.6~12.5°, RealLike 12.7~16.3°다. 30899는 Play 10~12.5°, RealLike 14°다.
  - 27500은 Play 16°, RealLike 21°였고, 22300은 Play 11~14°였다.
- 27500과 30899 비교:
  - 방향이 22300 수준으로 돌아왔다.
  - 공 속도는 7% 빨라졌다(3.53 → 3.77 m/s).
  - 넘어짐은 조금 늘었다. real_like 0.7 → 1.7%, low_mu 4.0 → 6.3%, 최악 18.3 → 24.0%다.
- 후보: I2 model_final(30899). 런은 `2026-11-24_15-29-42_long_from27900`이고, PC 사본은 `candidate_v16i2_30899/`이다. 실물 설치는 사용자 결정에 맡긴다. 로봇에는 27500이 설치돼 있다.

### 실물 킥 낙상 진단 (2026-11-25, robo04)

증상: 찰 때 디딤발(왼쪽)이 안쪽으로 접히고, 차는 다리(오른쪽)가 밖으로 크게 벌어진다. 그대로 디딤발 쪽 roll 방향으로 굴러 넘어진다. 22300, 27500, 30899 모두 같다(11-25 오전 13회 중 11회 넘어짐).

시험(공 0.35 m 정면, ros2 bag으로 /joint_goals·/joint_states·/imu·/motor_diagnostics 기록):
- 1회: 30899, 세기 1.0. 버팀(최대 roll 23.6°). 차는 다리 골반 roll 명령이 +55~58°다. 1.7 s에 디딤발 골반 roll 명령은 −30°, 실제는 +12°다.
- 2회: 공 y −0.05(차는 발 앞). 넘어짐. 효과 없음.
- 3회: 세기 0.6(launch에 `kick_strength` 인자 추가). 넘어짐. 다리 벌림 폭은 그대로다.
- 4회: 다리 게인 P800/D500 → P1200/D800. 넘어짐. 1.7 s에 디딤발 골반 roll 명령 −47°, 실제 +14°다. 시험 뒤 원래 게인으로 되돌렸다.

결정적 관측(4회, +1.67 s):
- 디딤발 l_pelvis_roll의 PWM이 −885(=100%, 최대 출력)이고 전류는 4.2 A다. 같은 순간 r_pelvis_pitch도 −885다.
- 전압은 10.1 V까지 떨어졌다(서 있을 때 10.7~10.9 V).

결론: 정책이 디딤발 골반 roll 모터가 낼 수 있는 토크보다 큰 회복 토크를 요구한다. 모터는 포화된 채로 못 따라간다. 게인(포화 전 기울기)으로는 해결되지 않는다. 시뮬레이션(ImplicitActuator)은 이 토크 한계를 실물보다 넉넉하게 본다.

다음:
- 배터리 완충(12 V대) 상태로 재시험한다. 토크는 전압에 비례한다.
- 재학습 때 골반 roll·pitch에 실측 기반 토크·속도 한계(10~11 V XM540)를 넣고, 포화 근처 토크에 벌점을 준다.
- 차는 다리 골반 roll 벌림을 제한한다.
- MuJoCo 게이트에도 같은 토크 한계를 넣는다.

### 실물 조건 MuJoCo 재현 비교 (2026-11-25)

`sim2real_evidence/kick_robust_20261116/kick_real_compare.py`로 실물 시험 조건을 재현했다. 조건은 공 0.35 m 정면(놓는 오차 ±3 cm), 목표 정면, cmd 0.2, hold 0.6 s, 필터 0.02, 11.1 V·10.3 V 모터 한계, 2.0 s 뒤 기본 자세 hold다. 결과는 `results_real_compare_20261125/summary.json`(각 10회)에 있다.

정책 동작은 실물과 같다:
- 차는 다리 골반 roll이 30899는 61°(오른발), 27500은 59~65°(왼발)다. 실물은 55~63°였다.
- 디딤발 골반 roll도 시뮬레이션에서 토크 한계(5 Nm)에 걸린다. 95% 이상인 시간이 18~23%이고, 명령-실제 차이는 20~26°다.

그런데 시뮬레이션은 넘어지지 않는다:
- 30899·27500의 넘어짐은 0%다(세기 0.5·1.0, 2.0 s 절단, 10.3 V 모두).
- 몸 roll 최대는 7~15°다. 실물은 버틴 시행에서도 23°, 넘어진 시행은 체중 이동 때 이미 18~23°였다.
- 22300만 세기 1.0에서 40~60% 넘어진다.

실물과 시뮬레이션의 차이 후보(측정값):
- 서 있을 때 골반 roll 모터가 좌우 반대 방향으로 ±1.2~1.45 A, 골반 yaw가 ±1.0~1.2 A를 쓴다(4·5회 시행 전 평균). 두 다리가 서로 밀고 있는 상태이고, 시뮬레이션에서는 이 토크가 0이다. 정지 토크 4.4 A 기준으로 약 30%를 서 있는 데 이미 쓴다.
- 서보 강성: XM540 P800은 약 12.5° 오차에서 PWM 100%다. 12 V 기준 약 33 Nm/rad, 10.5 V 기준 약 29 Nm/rad다. MuJoCo는 kp 50 Nm/rad다. 실물이 약 40% 무르다.
- 질량: MuJoCo 로봇은 5.70 kg이다. 실물 질량은 아직 측정하지 않았다.

다음: 골반 roll·yaw 영점(캘리브레이션)과 서 있는 자세를 확인하고, 실물 질량을 잰다. 그 결과를 학습·MuJoCo 모델(서보 강성 ~30 Nm/rad, 토크 한계, 질량)에 반영한다.
