"""MX3 supine get-up — v1_2. **v1 최고 판에서 출발한 수정본.**

★ 출발점 ★
  logs/rsl_rl/unitree_mx3_getup_v1/OK_16-27-54_smooth_stage3/
    (2026-09-02, "부드러움 3 단계판", model_44400)
  이 판의 params/getup_v1_env_cfg.py 를 **그대로** 복사했다.
  아카이브본과 당시 작업본이 동일했으므로(diff 0) 재현성이 보장된다.

★ 왜 여기로 돌아왔나 ★
  V2(보행 베이스)와 V3(팔 거울상 구조 강제)를 거쳤지만, 실기에서 이 판이
  가장 낫다는 판단. V3 는 중지하고 이 판을 조금씩 고치는 방향으로 간다.

★ 공유 모듈 주의 ★
  getup_v1_mdp / getup_v1_rewards 는 V2·V3 작업 중 확장됐다. v1 동작에
  영향을 주는 변경은 전부 **기본값을 예전 동작으로** 되돌려 두었다:
    - reset_root_supine 의 prone_height : None 이면 예전과 동일
    - walk_pose_deadline 의 gate        : 기본 "height" (V3 만 "latch")
  새로 추가된 함수들(getup_overshoot, joint_deviation_standing,
  arm_lift_power, *_gated)은 이 cfg 가 참조하지 않는다.

--- 이하 v1 원문 ---

MX3 supine get-up (등을 대고 누운 상태에서 일어나기) — v1.

★ 보행/킥과의 분리 ★
  - 로봇 자산 : mx_bot.ROBOT_CONFIGS["XM_robot04_getup"] (팔 구동 ON)
                보행/킥은 "XM_robot04" 를 그대로 쓴다.
  - MDP 항목  : mdp.getup_v1_mdp / mdp.getup_v1_rewards
                reset_hold.py 와 getup_rewards.py 는 건드리지 않는다.
  - 기존 getup_env_cfg.py (face-down, 12 DOF) 도 그대로 남겨 둔다.

★ 액션 공간 ★
  다리 12 + 팔 6 = 18 DOF. 스케일은 전 관절 0.25 균일(보행과 동일)이다.
  보행과 다른 점이 "팔이 구동된다" 하나뿐이라 문제 원인을 좁히기 쉽다.

★ 리셋 절차 ★
  1) 안착 높이보다 3~6 cm 위에 supine 자세로 배치 (지면 관통 방지)
  2) 관절에 덧셈 노이즈
  3) 0.5 s 홀드 — 스폰 자세를 유지한 채 낙하·정착. 리워드 0
  4) 홀드 해제, 에피소드 시작
  홀드 동안 스폰 자세를 유지하는 것이 핵심이다. 기본각으로 끌려가면
  (부모 클래스 동작) 관절 랜덤화가 0.25 s 만에 지워진다.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.mx_bot import (
    ACTIVE_ROBOT,
    MX_ARM_JOINTS,
    MX_LEG_JOINTS,
    build_robot_cfg,
)
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import (
    getup_v1_mdp,
    getup_v1_rewards,
    kick_stability_mdp,
)

# ── 팔 구동 로봇. 보행/킥이 쓰는 MX_BOT_CFG 와 별개 객체다 ──
ROBOT_CFG = build_robot_cfg("XM_robot04_getup")

# ── 최종 목표 자세 = **보행 기본각** ──
#   getup 항목은 팔꿈치 원점을 ±1.5 로 옮겼지만, 일어선 뒤의 자세는 보행이
#   정지 상태에서 잡는 그 자세(팔꿈치 ±2.967 접힘)여야 한다. 그래야 보행
#   정책으로 넘길 때 불연속이 없다.
#   값을 여기 적지 않고 레지스트리에서 직접 읽는다 — 보행 기본각을 바꾸면
#   이쪽도 자동으로 따라온다.
_WALK_READY_POSE = dict(build_robot_cfg(ACTIVE_ROBOT).init_state.joint_pos)

# ★ 2026-09-15 — 팔 종료 목표를 기상 로봇 기본각과 같게 맞춘다 (사용자 지정) ★
#   원래는 보행 로봇(XM_robot04) 기본각을 그대로 썼다. 이유는 "일어선 자세가
#   곧 보행 시작 자세여야 핸드오프에 불연속이 없다" 였고, 다리는 그대로 둔다.
#   다만 팔은 보행 기본각이 팔꿈치 ±2.967(170도, 완전히 접힘)이라 기상 원점
#   (±1.5)과 1.5 rad 이나 벌어져 있었다. walk_ready_pose / walk_pose_deadline
#   이 그 벌어진 값을 목표로 삼아 기립 후 팔을 계속 접으라고 요구한다.
#
#   초기 위치와 종료 위치를 같은 값으로 두면 기립 후 팔에 남는 압력이 없다.
#   다리·목은 건드리지 않는다.
_WALK_READY_POSE.update({
    "r_shoulder_pitch_joint": -0.3,
    "r_shoulder_roll_joint":   0.0,
    "r_elbow_pitch_joint":     1.0,
    "l_shoulder_pitch_joint":  0.3,
    "l_shoulder_roll_joint":   0.0,
    "l_elbow_pitch_joint":    -1.0,
})

MX_BASE_LINK = "base_link"

# 액션/관측 관절 순서 — mx_bot 의 joint_sdk_names 와 동일하게 유지할 것.
GETUP_JOINTS = MX_LEG_JOINTS + MX_ARM_JOINTS          # 18

# 기립 목표 몸통 높이 [m].
#   보행/킥이 쓰는 값과 동일하게 맞춘다 — 일어선 자세가 곧 보행 시작
#   자세여야 정책 핸드오프에 불연속이 없다.
#     velocity_v0_6_env_cfg.py:522   target_height 0.39
#     walk_kick_v1_4_env.py:931      target_height 0.39
#   실측 기립 높이도 0.396 이라 일치한다 (보행 스모크 3 태스크).
_STAND_HEIGHT = 0.39

# supine 안착 높이 [m]. mx_bot 의 XM_robot04_getup init_pos z 와 같다.
# 실측: scripts/rsl_rl/check_getup_spawn.py 3 회 → 0.0568 / 0.0571 / 0.0580
_SUPINE_HEIGHT = 0.057

# 복와위(엎드림) 안착 높이 [m]. 앙와위보다 5.2 cm 높다 — 가슴/골반이
# 먼저 닿고 팔이 몸 옆에 깔리기 때문이다.
# 실측: probe_settle.py (액션 0, 256 env) → 앙와위 0.0577 / 복와위 0.1099
# supine_height 로 양쪽을 보간하면 복와위가 바닥 **아래**에 스폰돼
# PhysX depenetration 이 공중으로 날린다 (V3 1 판에서 발사율 20.9%).
# ★ 2026-09-15 — 팔 기본각 변경으로 재측정: 0.1099 -> 0.0815 ★
#   팔꿈치 ±1.5 -> ±1.0, 어깨피치 0 -> ∓0.3 으로 바꾸니 팔이 덜 접혀
#   몸통이 2.8 cm 더 내려앉는다. 앙와위(0.0567)는 등이 바닥이라 변화 없다.
#   실측 smoke_prone_v12.py, 512 env, 액션 0, 55도/90도에서 0.0816/0.0815.
_PRONE_HEIGHT = 0.0815

# 발 링크 — 하중 비율 리워드에 쓴다.
MX_FEET = ["r_ankle_roll_link", "l_ankle_roll_link"]

# 관절 가동 한계 — USD 실측 (scripts/rsl_rl/check_getup_spawn.py 계열로 덤프).
# soft limit == hard limit 이라 이 값을 그대로 target 클램프로 쓴다.
_TARGET_LIMITS = {
    "r_pelvis_pitch_joint":   (-1.5708, +1.5708),
    "r_pelvis_roll_joint":    (-0.4363, +1.3963),
    "r_pelvis_yaw_joint":     (-1.5708, +1.5708),
    "r_knee_pitch_joint":     (+0.0000, +2.4435),
    "r_ankle_pitch_joint":    (-0.8727, +1.8326),
    "r_ankle_roll_joint":     (-0.5236, +0.5236),
    "l_pelvis_pitch_joint":   (-1.5708, +1.5708),
    "l_pelvis_roll_joint":    (-1.3963, +0.4363),
    "l_pelvis_yaw_joint":     (-1.5708, +1.5708),
    "l_knee_pitch_joint":     (-2.4435, +0.0000),
    "l_ankle_pitch_joint":    (-1.8326, +0.8727),
    "l_ankle_roll_joint":     (-0.5236, +0.5236),
    "r_shoulder_pitch_joint": (-2.6180, +2.6180),
    "r_shoulder_roll_joint":  (-0.2618, +2.8798),
    "r_elbow_pitch_joint":    (+0.0000, +2.9671),
    "l_shoulder_pitch_joint": (-2.6180, +2.6180),
    "l_shoulder_roll_joint":  (-2.8798, +0.2618),
    "l_elbow_pitch_joint":    (-2.9671, +0.0000),
}

# ── 커리큘럼 단계 ──
# 문턱은 common_step_counter 기준이다. 이 값은 env.step() 호출당 1 씩
# 오르므로 (manager_based_rl_env.py:202), num_steps_per_env=24 기준
# **iteration 당 24** 다. 전체 50,000 iteration = 1,200,000 step.
#
#   ⚠ 첫 시도에서 4,000 / 12,000 으로 잡았다가 실패했다. 각각
#     iteration 167 / 500 — 전체 학습의 0.3% / 1% 지점이다. "좁게 시작해
#     넓힌다" 가 사실상 작동하지 않았다 (실측: iter 296 에 이미 2단계).
_CURR_STAGE_STEPS = (100_000, 400_000)       # ≈ iteration 4,167 / 16,667

# 리셋 랜덤화 — 좁게 시작해 넓힌다. yaw/xy 는 1단계부터 전범위라 여기 없다.
_CURR_SIDE_ROLL = (0.15, 0.50, 0.90)         # 어깨 쪽으로 기울어 눕는 정도
_CURR_TILT = (0.15, 0.30, 0.50)              # 머리 위/아래 끄덕임
_CURR_JOINT_NOISE = (0.10, 0.25, 0.40)       # 관절 덧셈 노이즈

# ── 자기충돌 판정 상대 링크 ──
#   팔이 부딪혀서 곤란한 부위. 실측에서 r_elbow ↔ r_pelvis_roll 가
#   에피소드 내내 최대 35 N 으로 밀고 있었다 (체중 55.9 N).
_SELF_COLL_TARGETS = [
    "{ENV_REGEX_NS}/Robot/" + b
    for b in ("base_link",
              "r_pelvis_roll_link", "l_pelvis_roll_link",
              "r_pelvis_pitch_link", "l_pelvis_pitch_link",
              "r_knee_pitch_link", "l_knee_pitch_link")
]
_SELF_COLL_SENSORS = ("arm_r_elbow", "arm_l_elbow", "arm_r_shoulder", "arm_l_shoulder")

# ── 모터 속도-토크 포락선 (walk_kick_v1_4 와 동일 값) ──
#   sim 의 ImplicitActuator 는 ω 무관 5.0 N·m 를 평평하게 준다. 실기
#   XM540-W150 은 τ=6.9(1-ω/5.24) 라, 기상 실측 ω=4.57 에서 1.29 N·m 뿐이다
#   (sim 의 3.9 배 과대). 걷기는 ω<2 라 갭이 작지만 기상은 정확히 그 구간이다.
#
#   DCMotorCfg 로 바꾸면 물리가 달라져 재학습이 필요하다. 이 항들은 물리를
#   그대로 두고 **실현 불가 영역에만 비용**을 매기므로 기존 체크포인트에서
#   파인튜닝으로 넘어갈 수 있다. mx_bot.py 주석의 권장 순서이기도 하다.
#
#   ★ 킥과 다른 점: 다리 12 개가 아니라 **팔 포함 18 개**에 건다.
#     실측에서 팔꿈치·어깨도 ω 4.51 로 한계를 넘고 있었다.
MOTOR_STALL_TORQUE = 6.9      # [N·m] 11.1V 기준 (배터리 새그 고려, 킥과 동일)
MOTOR_NO_LOAD_SPEED = 5.24    # [rad/s] 50 rpm
MOTOR_EFFORT_CAP = 5.0        # sim actuator effort 와 일치
MOTOR_ENVELOPE_MARGIN = 0.8   # 곡선의 80% 안에서 놀게
JOINT_VEL_SOFT_LIMIT = 3.5    # 속도만 보는 강건 버전의 문턱
# ── 실기 이관용 페널티의 램프인 ────────────────────────────────────
#   레벨 4(90 deg, 완전 supine) 에 **도달한 뒤부터** 0 -> 절반 -> 전량.
#   근거는 getup_v1_mdp.penalty_rampin_curriculum docstring 참조.
#   문턱은 레벨 4 체류 step 수 (16384 env 에서 24 step/iter):
#       24_000 step = 1,000 iter,  72_000 step = 3,000 iter
# 2026-09-01: (24_000, 72_000) = 1,000/3,000 iter 에서 줄였다.
#   실측 적응 속도가 훨씬 빨랐다 — 램프 1단계(iter 24,003)를 켠 뒤
#       +87  iter  reward 118.9 -> 116.5
#       +430 iter  reward 117.2 (거의 회복)
#   100~400 iter 면 흡수하는데 1,000/3,000 은 3~7 배 여유였다.
#   300/900 iter 로 줄이면 램프 구간이 4,000 -> 1,200 iter, 판당 약 2시간 20분
#   절약된다. 처음 도입 때는 적응 속도를 몰라 크게 잡았던 값이다.
# ★ 2026-09-14 — 이어받기 시 램프인 재시작을 막는다 ★
#   커리큘럼 상태는 체크포인트에 안 담긴다. 페널티가 **전량 켜진 채**
#   학습된 정책(model_44400 계보)을 이어받으면 램프인이 0 부터 다시
#   시작해 껐다 켜는 과정에서 정책이 흔들린다.
#   V1-2 1 판 실측 — 램프인 구간에서:
#       arate  -0.168 -> -2.793  (16 배 악화)
#       stand  +4.599 -> +3.912
#       reward +117.81 -> +66.08,  롤백 2 회
#   stage_steps=(0,0) + from_level=0 이면 **처음부터 전량**이라 그 전이가
#   아예 없다. v1 기록에도 "이미 일어설 줄 아는 정책에는 페널티가 무해하다"
#   고 남아 있다 (OK_03-32-02 RUN_NOTES).
#
# ★ 2026-09-15 — 복와위 추가 판에서 (7_200, 21_600) / 5 로 되돌린다 ★
#   이 인자는 penalty_rampin 과 disturbance_rampin **둘 다** 제어한다.
#   from_level=0 이면 넉다운 푸시(1.5 m/s)까지 1 스텝부터 전량이 되는데,
#   복와위는 이 정책이 한 번도 본 적 없는 과제라 커리큘럼을 레벨 0(15°)
#   부터 다시 올라야 한다. 그 구간에 외란을 전량으로 깔면 복와위 학습
#   자체가 방해받는다.
#
#   from_level=5 = "복와위·앙와위 모두 90° 를 증명한 뒤에 켠다".
#   stage_steps 를 (0,0) 이 아니라 원래의 점진 램프로 두는 이유는 A 판
#   실측 때문이다 — 레벨 5 도달 직후 페널티와 외란이 함께 들어오면서
#   mean_reward 117.8 -> 39.9, stand_success 4.60 -> 2.16 으로 내려갔다.
#   (정책 붕괴가 아니라 외란이 켜져 환경이 어려워진 것이지만, 전이를
#    급하게 만들 이유는 없다.)
# ★ 2026-09-15 — model_48600 이어받기에 맞춰 (0,0)/0 으로 ★
#   48600(OK3)은 이 값으로 학습됐다. 페널티가 전량 켜진 정책을 이어받으면서
#   from_level=5 를 주면 레벨 5 까지 페널티가 꺼졌다가 다시 켜진다. 그
#   껐다 켜는 전이가 정책을 흔든다(V1-2 1 판에서 arate 16 배 악화 실측).
#   (0,0) + from_level=0 이면 처음부터 전량이라 전이가 아예 없다.
_CURR_PENALTY_STEPS = (0, 0)
_CURR_PENALTY_FROM_LEVEL = 0
_CURR_WALK_POSE = (0.0, -0.5, -1.0)
_CURR_ENVELOPE = (0.0, -0.01, -0.02)
_CURR_JOINTVEL_RAMP = (0.0, -0.01, -0.02)
_CURR_HEIGHT_WELL = (0.0, -75.0, -150.0)

# ── 외란 램프인 (밀기 + 외력) ──────────────────────────────────────
#   레벨 5 관문 이후에만 켠다. 부트스트랩에 외란을 섞으면 레벨 1 이 막힌다.
#   목적은 손상되지 않는다 — 레벨 5 에서도 매 에피소드가 supine 시작이라
#   그때부터 **일어나는 구간에** 밀기가 들어간다.
_CURR_PUSH_VEL = (0.0, 0.2, 0.4)      # m/s, xyz 각 축
_CURR_EXT_FORCE = (0.0, 1.5, 3.0)     # N, base_link (체중 55.9 N 의 5%)
# 넘어뜨리기 — 기립 후 세게 밀어 쓰러뜨리고 다시 일어나게 한다.
#   ★ 세기는 실측으로 정했다 (2026-09-02, model_27200, 256 env) ★
#     "시도 횟수 - 1" 이 실제로 넘어진 횟수다 (10 초 에피에 밀기 기회 2~3 회):
#         세기    넘어진 횟수   최종 기립률   t_settle
#         ±0.8      0.02          95.7 %      0.98 s   <- 버틴다. 너무 약함
#         ±1.5      1.0           74.2 %      6.13 s   <- 채택
#         ±2.5      1.6           58.6 %      8.24 s   <- 에피 끝에 몰려
#                                                          못 일어난 채 종료 증가
#     ±1.5 = 확실히 넘어뜨리면서 다시 일어날 시간이 남는 지점.
_CURR_KNOCKDOWN = (0.0, 0.8, 1.5)

_W_ENVELOPE = -0.02           # 제곱합이라 스케일 작게 (킥과 동일)
_W_JOINTVEL = -0.02

# 기립 후 안정화 항이 공통으로 쓰는 높이 인자.
#   ⚠ 리워드 configclass **안에** 두면 IsaacLab 이 리워드 항으로 해석해서
#     "Configuration for the term '_H' is not of type RewardTermCfg" 로 죽는다.
#     클래스 속성은 전부 항으로 간주되므로 상수는 모듈 레벨에 둔다.
_H = {"supine_height": _SUPINE_HEIGHT, "stand_height": _STAND_HEIGHT}

# ── 역방향 커리큘럼 단계 (넘어진 정도, deg) ──
#   0 = 직립, 90 = supine. 거의 서 있는 상태에서 시작해 점점 눕힌다.
#   전진은 step 이 아니라 **성공률** 기반이다 (getup_v1_mdp 참조).
# ★ 레벨 5 는 각도가 레벨 4 와 같은 90 deg 다 (2026-08-30) ★
#   난이도를 더 올리는 단계가 아니라, "90 deg 를 성공률로 증명했다" 는
#   **관문**이다. 실기 이관용 페널티(penalty_rampin)를 여기서 켠다.
#   step 카운트로 켜면 90 deg 를 실제로 해내는지와 무관하게 켜지는데,
#   이렇게 하면 EMA >= 0.5 를 통과해야만 켜진다.
_CURR_FALL_ANGLES = (15.0, 35.0, 55.0, 75.0, 90.0, 90.0)

# ★ 2026-09-15 — 복와위 도입과 함께 0.5 -> 0.8 ★
#   【이걸 안 바꾸면 복와위를 아예 안 배운다. V2 12 판이 그랬다.】
#   prone_ratio=0.5 면 어느 레벨이든 env 의 약 50 % 가 앙와위이고,
#   이어받는 정책은 앙와위를 이미 완벽히 한다. 문턱이 정확히 0.5 면
#   **앙와위 성공만으로 승격**되어 복와위를 하나도 못 해도 커리큘럼이
#   끝까지 올라간다.
#   V2 실측 — 문턱 0.5(12 판) vs 0.8(13 판), 복와위 기립률:
#       각도    12 판    13 판
#       -30      9.4 %   100 %
#       -60      0.0 %   100 %
#       -90      8.6 %   100 %
#   0.8 이면 앙와위 50 % 로는 불가능해 복와위를 최소 60 % 는 해야 한다.
#   13 판은 이 한 줄만 바꿔 여섯 각도(+90~-90) 전부 100 % 가 됐다.
#   ※ 앙와위 전용으로 되돌릴 때(prone_ratio=0.0)는 0.5 로 되돌릴 것.
_CURR_SUCCESS_THRESHOLD = 0.8

# 부드러움 페널티 — 약하게 시작해 실기 수준으로 조인다.
#   실기 검증된 값: velocity_v0_6 -0.15 / walk_kick_v1_4 -0.6
#   자세한 근거는 getup_v1_mdp.regularization_curriculum docstring 참조.
# 2026-09-02: 3 단계(-0.60) 추가. 실기에서 "일어날 때 너무 격하다".
#   실측 (model_40700, 90 deg, 256 env) — 구간을 나눠 재보니:
#       기립 중   action_rate_l2 1.211   joint_acc_l2 22,949
#       서 있음   action_rate_l2 0.036   joint_acc_l2    852
#   기립 구간이 33 배 / 27 배 격하다. 서 있을 때는 이미 충분히 조용하다.
#   느려져도 좋다는 판단 하에 둘 다 2 배로 올린다(사용자 지정).
_CURR_ACTION_RATE = (-0.05, -0.15, -0.30, -0.60)
# joint_acc 는 커리큘럼으로 올리지 않고 **처음부터 킥 값**으로 간다.
#   velocity_v0_6(보행) -2.5e-7  /  walk_kick_v1_4(킥) -2.5e-6  ← 10 배
#   기상은 킥처럼 크고 빠른 동작이라 킥 값이 맞다. 이어받기로
#   common_step_counter 가 0 부터라 커리큘럼을 두면 100,000 step
#   (약 4,167 iteration) 을 기다려야 해서 효과가 늦다.
_CURR_JOINT_ACC = (-2.5e-7, -1.0e-6, -2.5e-6, -5.0e-6)   # 3 단계 추가 (2026-09-02)


@configclass
class GetupV12SceneCfg(InteractiveSceneCfg):
    """평지만. 기상 자체에 집중시킨다."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # ── 자기충돌 감지용 바디쌍 센서 ──
    #   filter_prim_paths_expr 는 "센서 prim 이 환경당 단일 prim 일 때만"
    #   동작한다(contact_sensor_cfg.py). 그래서 팔 링크마다 하나씩 둔다.
    #   상대는 몸통·골반·허벅지 — 팔이 부딪혀서 곤란한 부위들이다.
    #   바닥은 필터에 없으므로 지면 접촉이 섞이지 않는다.
    arm_r_elbow = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_elbow_pitch_link",
        history_length=3,
        filter_prim_paths_expr=_SELF_COLL_TARGETS,
    )
    arm_l_elbow = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_elbow_pitch_link",
        history_length=3,
        filter_prim_paths_expr=_SELF_COLL_TARGETS,
    )
    arm_r_shoulder = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_shoulder_roll_link",
        history_length=3,
        filter_prim_paths_expr=_SELF_COLL_TARGETS,
    )
    arm_l_shoulder = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_shoulder_roll_link",
        history_length=3,
        filter_prim_paths_expr=_SELF_COLL_TARGETS,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                f"kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


@configclass
class GetupV12EventCfg:
    # ★ 2026-09-14 — 바닥 미끄럼 학습 ★
    #
    #   【실기 관찰】 잔디에서는 잘 되는데 맨바닥(대리석보다 조금 덜 미끄러운
    #   정도)에서 과하게 민다.
    #
    #   【원인 — 실측 (probe_fric_v12.py, 512 env, 형상 42 개)】
    #     기존: randomize_rigid_body_material, static (0.6,1.4) / dynamic (0.6,1.2)
    #       전체 형상   static 최소 0.603  평균 0.961  최대 1.378
    #       env 내 산포 평균 0.732        <- 한 env 안에서도 형상마다 제각각
    #       env 평균    최소 0.866  평균 0.961  최대 1.056
    #       env 평균 <= 0.6 인 비율  0.00 %
    #       env 평균 <= 0.5 인 비율  0.00 %
    #
    #     문제가 둘이다:
    #       ① 하한 0.6 이 실기 맨바닥(mu 0.3~0.5 추정)을 안 덮는다.
    #       ② **형상마다 독립 추출**이라 42 개 평균이 대수의 법칙으로 1.0
    #          근처에 수렴한다. 산포가 0.73 이나 되는데도 env 평균이 0.866
    #          아래로 내려간 적이 한 번도 없다. 즉 정책이 겪은 "가장 미끄러운
    #          바닥" 이 0.866 이고, **바닥 전체가 미끄러운 상황은 본 적이 없다.**
    #
    #   【처방】 env 당 값 하나를 뽑아 전 형상에 적용한다 (V2 에서 같은 문제를
    #   겪고 만든 함수). env 평균 분포가 0.866~1.056 -> 0.25~1.40 균일이 되고,
    #   실기 맨바닥 구간(0.3~0.5)이 전체의 약 17 % 를 차지한다.
    #
    #   ⚠ 지형은 static/dynamic 1.0 에 combine_mode="multiply" 이므로
    #     실효 마찰 = 이 값 그대로다.
    #   ⚠ 하한을 크게 낮추면 기상이 어려워진다. model_44400 이어받기로
    #     시작해 적응만 시킨다. 처음부터는 커리큘럼 초반이 막힐 수 있다.
    physics_material = EventTerm(
        func=getup_v1_mdp.randomize_friction_per_env,
        mode="startup",
        params={
            "static_friction_range":  (0.25, 1.4),
            "dynamic_friction_range": (0.20, 1.2),
        },
    )

    # ★ 2026-09-16 — 토크/속도 한계 랜덤화 (신규) ★
    #   sim 은 ω 와 무관하게 평평한 effort 5.0 N·m / vel 4.5 rad/s 를 준다.
    #   실기 XM540-W150 은 τ(ω)=7.3(1−ω/5.55) 로 떨어진다. 기상 동작의
    #   N-T 곡선 위반률은 7.6~8.4 % 로 보행 vx0.7(13.2 %, 실기 동작함)보다
    #   낮지만, 기상은 한 번에 밀어올리는 동작이라 결정적 순간에 모자라면
    #   아예 못 일어난다. 개체차·전압강하·발열 여유를 보험으로 넣는다.
    #
    #   effort   0.80~1.00 -> 4.0~5.0 N·m
    #   velocity 0.78~1.00 -> 3.5~4.5 rad/s
    #
    #   ⚠ mode="startup" 이다. 매 에피소드 바뀌는 DR 은 정책을 반응형으로
    #     만들어 action_rate 를 올리고 실기 휘적거림이 된다
    #     ([[dr-intensity-action-rate-tradeoff]]). env 당 한 번만 뽑아
    #     에피소드 내내 상수로 둔다 — 마찰에서 쓴 것과 같은 방식이다.
    #   ⚠ env 당 배율 하나를 전 관절에 적용한다. 관절마다 따로 뽑으면
    #     env 평균이 1.0 으로 수렴해 "로봇 전체가 약한 개체" 가 안 나온다
    #     (randomize_rigid_body_material 에서 겪은 그 함정).
    # ★ 2026-09-16 — 7 판에서 넣었다가 **껐다** ★
    #   sim 지표는 좋았다: 커리큘럼 비용 없음(1081 vs 1066 iter), 약한
    #   개체(effort 4.0 N·m)도 앙와위·복와위 100 % 기립(기상 6 % 증가),
    #   드리프트 0.0045 로 6 판(0.0049)보다 낮음.
    #   그런데 **사용자가 화면으로 보고 6 판 model_53400 이 더 안정적이라고
    #   판정했다** — 복와위 기상 동작과 기립 후 떨림 둘 다. 이 세션에서
    #   sim 지표와 실제 동작이 여러 번 갈렸고, 눈으로 본 쪽을 따른다.
    #
    #   ⚠ 되살릴 때 알아둘 것: mode="startup" 이라 **재생(play)에서도 돈다.**
    #     --num_envs 1 이면 로봇 한 대가 4.0~5.0 중 하나를 무작위로 뽑으므로,
    #     고정 5.0 인 정책과 눈으로 비교하면 불공정하다. 공정 비교를 하려면
    #     effort/velocity_scale_range 를 (1.0, 1.0) 으로 고정할 것.
    #
    #   함수(randomize_actuator_limits_per_env)와 검증 프로브
    #   (probe_actlimit.py, probe_weakbin.py)는 그대로 둔다. 되살리려면
    #   아래 주석을 풀면 된다.
    # actuator_limits = EventTerm(
    #     func=getup_v1_mdp.randomize_actuator_limits_per_env,
    #     mode="startup",
    #     params={
    #         "effort_scale_range":   (0.80, 1.00),
    #         "velocity_scale_range": (0.78, 1.00),
    #     },
    # )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_BASE_LINK),
            "mass_distribution_params": (-0.4, 0.6),
            "operation": "add",
        },
    )

    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_link"),
            "mass_distribution_params": (0.93, 1.07),
            "operation": "scale",
        },
    )

    # ── 1) 루트: supine 자세로 살짝 띄워 배치 ──
    #    회전 인자 이름이 곧 의미다 (월드 yaw / 바디 옆구르기 / 바디 끄덕임).
    #    자세한 축 설명은 getup_v1_mdp.reset_root_supine docstring 참조.
    reset_base_supine = EventTerm(
        func=getup_v1_mdp.reset_root_supine,
        mode="reset",
        params={
            "xy_range": (-0.2, 0.2),
            # ★ 2026-09-15 — 3~6 cm -> 8~12 cm (복와위 도입) ★
            #   prone_height 로 방향별 안착 높이를 맞춰도 3~6 cm 로는 모자란다.
            #   side_roll / tilt / 관절노이즈가 env 마다 최저점을 바꿔서
            #   복와위 안착이 평균 0.1090, **최소 0.0947** 로 1.6 cm 퍼지기
            #   때문이다. 발사율 실측 (probe_margin.py, 512 env, v_z>0.3):
            #       height_range     복와위    앙와위
            #       (0.03, 0.06)     20.9 %     0.0 %
            #       (0.05, 0.09)      1.8 %       -
            #       (0.08, 0.12)      0.0 %     0.0 %   <- 채택
            #   0.12 m 낙하는 착지속도 1.53 m/s 로 아래 주석의 1.08 보다 크지만
            #   실측 튐은 최대 +0.08 이고 홀드 0.5 s 안에 정착한다.
            "height_range": (0.08, 0.12),
            "yaw_range": (-math.pi, math.pi),        # 1단계부터 전범위
            "side_roll_range": (-_CURR_SIDE_ROLL[0], _CURR_SIDE_ROLL[0]),
            "tilt_range": (-_CURR_TILT[0], _CURR_TILT[0]),
            # 역방향 커리큘럼이 매 리셋마다 덮어쓴다. 여기 값은 0 단계 초기값.
            "fall_angle_deg": _CURR_FALL_ANGLES[0],
            # ★ 2026-09-14 — 공유 함수 기본값 오염 차단 ★
            #   reset_root_supine 은 V2 작업 중 확장되면서 기본값이
            #       fall_angle_min_deg = 0.0   (밴드 샘플링)
            #       prone_ratio        = 0.5   (복와위 절반)
            #   로 생겼다. 이 판(2026-09-02)이 돌 때는 두 인자가 아예 없었고
            #   **앙와위 고정 각도** 전용이었다. 명시하지 않으면 조용히
            #   다른 태스크가 된다 — v1 설계 노트에도 "prone 은 별도 태스크로.
            #   supine 과 섞으면 롤아웃 간섭으로 성능이 나빠진다" 고 적혀 있다.
            #   (복와위는 스폰 침투 문제도 있다. height_range 3~6 cm 는
            #    앙와위 안착 0.057 기준이고 복와위 안착은 0.110 이라
            #    바닥 아래에 스폰돼 PhysX 가 공중으로 날린다.)
            #
            # ★ 2026-09-15 — 복와위 도입: 0.0 -> 0.5 ★
            #   v1 설계 노트는 "prone 은 별도 태스크로" 라고 적었지만,
            #   V3 에서 앙와위·복와위 반반으로 **둘 다** 학습됐다
            #   (복와위 기상 1.0 s). 실기에서 앞으로 넘어지는 경우가
            #   실제로 나오므로 한 정책이 둘 다 해야 한다.
            #   역방향 커리큘럼이 레벨 0(15°)부터 올라가므로 복와위도
            #   거의 서 있는 상태에서 점진적으로 도입된다.
            "prone_ratio": 0.5,
            "prone_height": _PRONE_HEIGHT,
            "fall_angle_min_deg": 0.0,
            "supine_height": _SUPINE_HEIGHT,
            "stand_height": _STAND_HEIGHT,
        },
    )

    # ── 2) 관절: 기본각 기준 덧셈 노이즈 ──
    #    reset_joints_by_scale(곱셈) 은 안 쓴다 — 기본각 0 인 관절은 아예
    #    안 흔들리고 큰 관절만 비례해서 흔들려 폭이 자의적이다.
    #    reset_joints_additive_noise 도 안 쓴다 — 그건 *현재* 관절각에
    #    노이즈를 더하는 함수라, 앞선 관절 리셋이 없으면 직전 에피소드
    #    끝 자세가 그대로 이월된다(실측: 기본각 대비 0.56 rad 편차).
    #    reset_joints_by_offset 은 default_joint_pos 기준이라 이게 맞다.
    reset_joint_noise = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-_CURR_JOINT_NOISE[0], _CURR_JOINT_NOISE[0]),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ── 3) 홀드: 스폰 직후 전이(낙하·바운스)만 흡수 ──
    #    GetupJointPositionAction 이 이 구간 동안 스폰 자세를 유지한다.
    #
    #    ★ 0.5 s → 0.1 s (2026-08-28) ★
    #      홀드 동안 액션이 0 이라 기울어진 로봇은 그냥 넘어진다. 역방향
    #      커리큘럼의 중간 단계가 에피소드 시작 전에 전부 supine 으로
    #      무너져 커리큘럼이 무의미해졌다. 실측(해제 시점 upright):
    #          홀드     15°    35°    55°    75°
    #          0.3 s   0.96   0.76   0.51   0.47   ← 55°+ 이미 누움
    #          0.2 s   0.97   0.85   0.63   0.47
    #          0.1 s   0.98   0.90   0.76   0.58   ← 의도한 자세 유지
    #      (기대값 0.98 / 0.91 / 0.79 / 0.63)
    #      supine 낙하는 3~6 cm 라 자유낙하 0.096 s. 0.1 s 로 충분하다.
    reset_hold = EventTerm(
        func=mdp.set_reset_hold_steps,
        mode="reset",
        params={"hold_time_s": 0.1},
    )

    # 🆕 밀기 외란 (2026-08-30)
    #   ★ 보행의 간격(2~4 초)을 그대로 쓰면 안 된다 ★
    #     기상은 실측 0.83 초에 끝난다. 2~4 초 간격이면 첫 밀기가 이미
    #     다 선 뒤에 들어와, 검증이 안 된 **일어나는 구간**을 못 건드린다.
    #     서 있는 상태는 이미 보행 수준이다(base_z 산포 0.23 mm, 양발
    #     100 % 접지). 거기를 흔들어봐야 얻을 게 없다.
    #     -> 0.3~1.0 초로 좁혀 기립 구간에 1~2 회 들어가게 한다.
    #
    #   ★ 넘어짐 종료가 없다는 점이 여기서 이득이 된다 ★
    #     밀려서 무너지면 에피소드가 끝나는 게 아니라 **거기서 다시
    #     일어나야 한다**. 인위적인 supine 스폰이 아니라 실제로 넘어진
    #     자세에서 기상을 배우게 된다.
    #
    #   축별 효과 (0.4 m/s 기준):
    #     x, y   마찰 상대로 실제로 미끄러진다. 주 외란.
    #     +z     탄도 상승 8 mm / 체공 40 ms. 거의 논다.
    #     -z     누워 있을 땐 바닥이 막아 무효. 하지만 **일어나는 중간
    #            (브릿지 자세, base 0.2~0.3 m)에 걸리면 반쯤 올린 몸을
    #            다시 주저앉힌다.** z 를 넣는 이유는 이 경우다.
    #   Critic 이 외란을 볼 수 있도록 기록형 래퍼를 쓴다
    #   (getup_v1_mdp.push_recorded / apply_base_force_recorded).
    #   범위는 disturbance_rampin 커리큘럼이 레벨 5 이후 올린다.
    push_robot = EventTerm(
        func=getup_v1_mdp.push_recorded,
        mode="interval",
        interval_range_s=(0.3, 1.0),
        params={
            "velocity_range": {
                "x": (-_CURR_PUSH_VEL[0], _CURR_PUSH_VEL[0]),
                "y": (-_CURR_PUSH_VEL[0], _CURR_PUSH_VEL[0]),
                "z": (-_CURR_PUSH_VEL[0], _CURR_PUSH_VEL[0]),
            },
            # 🆕 기립 전에만 민다 (2026-09-01). 서 있는 동안 계속 밀면
            #    정책이 "항상 대비" 상태가 되어 실기에서 진동한다.
            #    외란의 목적은 일어나는 구간의 강건성이다.
            "gate_below_height_frac": 0.85,   # _stand_gate 와 같은 문턱
            **_H,
        },
    )

    # 🆕 넘어뜨리기 (2026-09-02) — 기립 후 세게 밀어 쓰러뜨린다.
    #   목적 두 가지:
    #     (a) 기상 샘플이 2~3 배 늘어난다. 지금은 10 초 에피에 기상이
    #         한 번(0.8 초)이고 나머지 9.2 초가 서 있기다.
    #     (b) **실제로 넘어진 자세**에서 일어나게 된다. 인위적 supine
    #         스폰이 아니라 옆으로/앞으로/비틀린 채 넘어진다.
    #         "앞뒤 상관없이 일어나기" 로 가는 자연스러운 경로다.
    #
    #   ★ push_robot(약+잦음)과 성질이 다르다 ★
    #     약하고 잦으면 "버티면 이득" 이라 정책이 항상 대비하게 되고,
    #     그게 실기 진동이 됐다(2026-09-01). 세고 드물면 저항이 무의미해
    #     그냥 넘어지고 다시 일어나는 게 최적이라 그 부작용이 없다.
    #
    #   수평(x, y)만 준다. z 는 튕기기만 하고 쓰러뜨리지 못한다.
    push_knockdown = EventTerm(
        func=getup_v1_mdp.push_recorded,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params={
            "velocity_range": {
                "x": (-_CURR_KNOCKDOWN[0], _CURR_KNOCKDOWN[0]),
                "y": (-_CURR_KNOCKDOWN[0], _CURR_KNOCKDOWN[0]),
            },
            "gate_above_height_frac": 0.85,   # 다 일어선 env 만
            **_H,
        },
    )

    # 🆕 지속 외력 — 밀기(순간 충격)와 역할이 다르다.
    #   기울어진 바닥 / 끌리는 케이블 / 모델 오차를 흉내내며, 기상 동작
    #   전체를 편향에 맞서 해내게 만들어 **토크 여유**를 시험한다.
    #   torque_envelope 페널티가 켜져 있으므로, 모터 한계 안에서 그걸
    #   해내는 법을 배우게 된다.
    #   ★ base_link 하나에만 건다 — 원본은 바디마다 힘을 따로 뽑는다.
    push_force = EventTerm(
        func=getup_v1_mdp.apply_base_force_recorded,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MX_BASE_LINK),
            "force_range": (-_CURR_EXT_FORCE[0], _CURR_EXT_FORCE[0]),
            "torque_range": (0.0, 0.0),
        },
    )


@configclass
class GetupV12ActionsCfg:
    JointPositionAction = getup_v1_mdp.GetupJointPositionActionCfg(
        asset_name="robot",
        joint_names=GETUP_JOINTS,
        # 전 관절 균일. 보행과 다른 변수를 늘리지 않는다.
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        target_limits=_TARGET_LIMITS,
        action_delay_range=(1, 2),
    )


@configclass
class GetupV12ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.3, n_max=0.3)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=GETUP_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=GETUP_JOINTS, preserve_order=True)},
            scale=0.05,
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=GETUP_JOINTS, preserve_order=True)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(
                "robot", joint_names=GETUP_JOINTS, preserve_order=True)},
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)

        # ── 특권 정보 (sim 에서만 얻을 수 있는 것) ──
        #   Policy 에는 절대 넣지 않는다. 실기에 외부 트래킹이 없어서
        #   몸통 높이도 접촉력도 측정할 수 없다. Critic 은 sim 전용이라
        #   써도 sim2real 비용이 0 이다.
        #
        #   왜 필요한가 — 이 태스크 리워드의 대부분(base_height 3.0 +
        #   stand_success 5.0 = 11.0 중 8.0)이 root_pos_w[:,2] 에서
        #   나온다. Critic 이 그 높이를 못 보면 관절각과 중력벡터로
        #   추론해야 하고, 가치 추정이 흐려지면 어드밴티지도 흐려진다.
        #   희소 보상 태스크에서 이건 그대로 돌파 실패로 이어진다.
        base_height = ObsTerm(func=mdp.base_pos_z)                      # 1

        # 발이 받치는 체중 비율. feet_load 리워드와 같은 계산부를 쓴다.
        feet_load = ObsTerm(
            func=getup_v1_mdp.feet_load_obs,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET)},
        )                                                                # 1

        # 바디별 접촉 여부. 기상은 본질적으로 접촉 모드 전환(등 → 팔/
        # 엉덩이 → 발)이라, 어느 부위가 닿아 있는지가 곧 진행 단계다.
        # 관절각만으로는 같은 자세가 공중인지 바닥에 눌린 건지 모른다.
        # ── 외란 관측 (ext_force / last_push) 은 **뺐다** (2026-08-30) ──
        #   원리상 무해해야 했다. 실측으로 값이 정확히 0 이고
        #   (empirical_normalization=false 라 선형층 기여도 0), 이벤트도
        #   완전히 무력화해서 물리를 안 건드리게 만들었다.
        #   그런데도 레벨 1 을 못 넘겼다:
        #       Critic 관측 없음  통과 2/2  (+261 에 sr 0.999 / 0.917)
        #       Critic 관측 있음  정체 0/3  (+261 에 sr 0.066 / 0.047 / 0.052)
        #   입력 차원이 430 -> 460 이 되면 초기화가 난수를 더 소비해 RNG
        #   스트림 전체가 어긋나는 것 외에 알려진 메커니즘이 없는데,
        #   3/3 이면 그것만으로는 설명이 약하다. **원인 미규명**이다.
        #   규명되면 다시 넣는다. 함수(ext_force_obs / last_push_obs)는
        #   getup_v1_mdp 에 그대로 남겨뒀다.

        body_contact = ObsTerm(
            func=getup_v1_mdp.body_contact_obs,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
                "threshold": 1.0,
            },
        )                                                                # 바디 수

        # ※ episode_phase(진행도) 관측을 넣었다가 뺐다.
        #   시간 게이트 리워드 때문에 V(s) 가 흐려질까 봐 넣었는데,
        #   (1) critic 이 430 -> 435 로 바뀌어 기존 체크포인트를 못 읽고
        #   (2) 기상이 1.2 초에 끝나고 기한이 1.5 초라 두 구간의 상태가
        #       거의 안 겹친다 (겹치는 건 "서 있는데 t=1.4 vs 1.6초" 정도).
        #   이론적 우려만으로 좋은 정책을 버릴 근거가 못 된다고 판단했다.

        def __post_init__(self):
            self.history_length = 5

    critic: CriticCfg = CriticCfg()


@configclass
class GetupV12RewardsCfg:
    """전 항이 리셋 홀드 동안 0 이다 (getup_v1_rewards.hold_masked).

    【시작 상태(supine)에서의 기여값 — 설계 시 손으로 맞춰 본 값】
      base_height   0.00 × 3.0 = 0.00   ← 정규화라 시작이 정확히 0
      height_rising 0.00 × 0.5 = 0.00
      upright       0.00 × 1.0 = 0.00
      feet_load     ~0   × 1.0 = ~0     ← 등이 체중을 받으므로
      walk_ready    0.00 × 1.0 = 0.00   ← 게이트가 닫혀 있음
      stand_success 0.00 × 5.0 = 0.00
      ─────────────────────────────
      합계 ≈ 0.00

    시작 리워드가 0 이어야 한다. 구 설계는 여기서 0.785 였고 그중 63% 가
    "가만히 있어라" 신호였다. 모든 항이 supine 에서 0, 기립에서 최대가
    되도록 정규화·게이트를 걸었다.

    【기립 상태에서의 기여값】
      base_height 1.0×3.0 + upright 1.0×1.0 + feet_load ~1.0×1.0
      + walk_ready ~1.0×1.0 + stand_success 1.0×5.0  ≈ 11.0
    """

    # ── 주 신호: 높이 ──
    base_height = RewTerm(
        func=getup_v1_rewards.base_height_progress,
        weight=3.0,
        params={"supine_height": _SUPINE_HEIGHT, "stand_height": _STAND_HEIGHT},
    )

    # 절대 높이가 평평한 구간(구르기·팔 짚기)에서도 신호를 남긴다.
    height_rising = RewTerm(func=getup_v1_rewards.height_rising, weight=0.5)

    # ── 자세 ──
    #   ⚠ 첫 시도에서 1.0 으로 뒀다가 "브릿지" 국소해에 빠졌다. iter 296 실측:
    #       feet_load 0.69 (발이 체중 69% 지지)  ↔  upright 0.008 (안 일어섬)
    #     발만 짚고 엉덩이를 든 자세로 base_height(3.0)+feet_load(1.0)를
    #     챙기면서 몸통 회전은 하지 않는다. HumanUP 이 자세를 0.25 로 낮게
    #     둔 걸 따라갔는데, 그쪽은 높이 항이 머리+몸통 2 개(합 10)라 균형이
    #     달랐다. base_height(3.0)에 준하는 값으로 올린다.
    upright = RewTerm(
        func=getup_v1_rewards.upright_progress,
        weight=2.5,
        params={"supine_height": _SUPINE_HEIGHT, "stand_height": _STAND_HEIGHT},
    )

    # ── 접촉 모드 전환: 발이 체중을 받는가 ──
    feet_load = RewTerm(
        func=getup_v1_rewards.feet_load_fraction,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=MX_FEET),
            "supine_height": _SUPINE_HEIGHT,
            "stand_height": _STAND_HEIGHT,
        },
    )

    # ── 최종 형태: 기본자세로 서 있기 ──
    # 직립 정도로 게이트가 걸려 있어 supine 에서는 0 이다.
    walk_ready_pose = RewTerm(
        func=getup_v1_rewards.walk_ready_pose,
        weight=1.0,
        params={
            "target_pose": _WALK_READY_POSE,
            "std": 0.5,
            # _upright_frac 스케일 변경(clamp → (1-g_z)/2)에 맞춰 재조정.
            #   새 스케일: prone 0 / supine 0.5 / 직립 1
            #   0.75 → g_z = -0.5, 몸통이 수평에서 60° 이상 일어선 상태.
            #   0.5 로 두면 supine 을 넘자마자 열려서 "가만히 있어라" 신호가
            #   너무 이르게 들어온다.
            "gate_from": 0.75,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=GETUP_JOINTS, preserve_order=True),
        },
    )

    # 희소 성공 보너스. 조건을 겹겹이 쌓지 않는다 (자세는 위 항이 안내).
    stand_success = RewTerm(
        func=getup_v1_rewards.stand_success,
        weight=5.0,
        params={
            "supine_height": _SUPINE_HEIGHT,
            "stand_height": _STAND_HEIGHT,
            "height_frac_threshold": 0.85,
            # 스케일 변경 반영. 구 스케일 0.90 = g_z -0.90 이었는데
            # 새 스케일에서 0.90 은 g_z -0.80 으로 느슨해진다. 0.95 가
            # 같은 엄격도(g_z -0.90)다.
            "upright_threshold": 0.95,
        },
    )

    # ── 과회전(넘어감) 벌점 ── ★ 2026-09-15 복와위 도입과 함께 신설 ★
    #   V3 실측(model_7000): 복와위가 **직립을 지나쳐 등으로 넘어간다**.
    #       t=0.80  base_z 0.260  g_x +1.00      상체 들어올림
    #       t=1.20         0.268  g_x +0.38      회전 중
    #       t=1.60         0.086  g_x -0.98      128 env 전부 뒤집힘
    #       t=3.00         0.371  g_x  0.00      앙와위로 다시 기상 (3.0 s 소요)
    #   보상은 1.20 s 에 upright 1.368 / base_height 1.904 였다가 1.60 s 에
    #   0.124 / 0.261 로 떨어진다. **이득을 노린 게 아니라 실패다.**
    #   이 항을 넣고 복와위 기상이 3.0 s -> 1.0 s 가 됐다.
    #
    #   clamp(g_x * fall_sign, min=0)^2 — 출발 방향을 지나친 정도만 센다.
    #   기상에 필요한 회전은 전혀 방해하지 않는다. 앙와위에도 대칭 적용돼
    #   앞으로 고꾸라지는 것도 막는다.
    #   게이트 없음: 과회전 구간 base_z 0.24~0.27 은 어떤 기립 게이트보다
    #   낮고, 기립 후에는 g_x ~= 0 이라 자동으로 0 이 된다.
    # ★ 2026-09-15 — 가중치 -10 -> -60 (복와위 270도 우회 교정) ★
    #   4 판 실측(model_55300, 복와위 90도, 128 env): **전 env 가 앙와위를
    #   거쳐서 일어난다.** 몸 장축 기준 270 도를 돈다.
    #       t=0.00  g_x +0.992  g_y +0.011   복와위
    #       t=0.20  g_x +0.981            base_z 0.12 -> 0.21 (팔 푸시 시작)
    #       t=0.60  g_x +0.513  g_y -0.717  0.2135 에서 포기하고 구르기 전환
    #       t=0.70  g_x -0.096  g_y -0.925  왼쪽 측와위
    #       t=0.90  g_x -0.905  g_y -0.107  앙와위 (높이 0.10 으로 주저앉음)
    #       t=1.10  g_x -0.376  g_y +0.903  오른쪽 측와위
    #       t=1.50                          직립 (1.47 s)
    #   대조 — 앙와위는 90 도만 돈다:
    #       g_x -0.99 -> g_y +0.83 (오른쪽 측와위) -> 직립 (0.88 s)
    #
    #   【무엇이 문제인가】 팔 푸시업을 못 배운 게 아니다. 정책은
    #   "오른쪽 측와위에서 일어나기" 하나만 알고, 앙와위 전문가
    #   (model_46400)를 이어받아 **그때 익힌 회전 방향 하나만** 쓴다.
    #   복와위에서 같은 자리까지는 반대로 90 도면 되는데 270 도를 돈다.
    #   목표 자세는 이미 할 줄 아는 것이므로 **방향 선택만 고치면 된다.**
    #
    #   【왜 이 항으로 되는가】 clamp(g_x * fall_sign, 0)^2 이 두 경로를
    #   정확히 가른다. 270 도 경로는 g_x 가 반드시 -1 까지 가고, 90 도
    #   경로는 g_x 가 +1 -> 0 으로만 내려간다. 몸통 방향이라 조작 가능한
    #   양이 아니고(피치로 굴러도 g_x 는 음수를 지난다) 우회로가 원리적으로
    #   없다 ([[gate-must-not-be-manipulable]] 의 반대 사례).
    #   앙와위 경로는 실측 g_x 최대 +0.028 -> 벌점 0.0008 로 무영향이다.
    #
    #   -10 에서 Episode_Reward -0.20 (에피소드당 약 -2.0) 이었는데
    #   0.6 초 단축 이득보다 작아서 정책이 벌점을 감수했다. -60 이면
    #   -1.2 급이 되어 행동을 바꾼 항들과 같은 자리에 온다.
    #   ※ 위험: 지역해에 갇혀 반대 방향을 못 찾으면 복와위 성공률이 떨어진다.
    #     그 경우 대안은 미러 손실(symmetry_cfg)인데 v1 계보엔 없던 것이다.
    # ★ 2026-09-15 — -60 -> -10 되돌림 ★
    #   실기 검증 결과로 원인이 이 가중치로 좁혀졌다:
    #       model_48600 (항 없음)   실기 정상
    #       model_55300 (-10)       실기 정상
    #       model_61700 (-60)       "일어나고 과하게 움직여 균형을 깬다"
    #   sim 60 초 정지 측정도 같은 순서다 (수평 드리프트):
    #       48600 0.0049 / 55300 0.0085 / 61700 0.0255 m
    #   -10 은 270 도 우회를 못 막지만 실기에서 문제가 없다. 우회 교정은
    #   실기가 안정된 뒤에 다시 본다.
    getup_overshoot = RewTerm(
        func=getup_v1_rewards.getup_overshoot,
        weight=-10.0,
    )

    # ── 모터 포락선 ──
    #   torque_envelope : **비활성**. 물리가 이미 실기 곡선(use_dc_motor)이라
    #                     토크 기반 페널티는 진짜 중복이다.
    #   joint_vel_limit : **활성**. 이건 중복이 아니다.
    #
    #   ★ 껐다가 실패해서 되살렸다 (2026-08-29) ★
    #     DC 모터로 바꾸면서 둘 다 껐더니 iter 200 에서 학습이 깨졌다:
    #         action_rate  -32.68  (raw 653, 평평한 토크 시절의 24 배)
    #         value_loss    47.2   (정상 0.1 수준)
    #         reward      -290.48
    #     원인: 모터가 명령 타깃을 못 따라가니 정책이 더 큰 액션을 내고,
    #     그래도 안 움직이니 더 키운다 — **액션 폭주**. action_rate 가 그
    #     결과를 벌주지만 원인을 막지는 못한다.
    #
    #     물리가 속도를 제한하는 것과, 정책에게 "그 속도를 요구하지 말라"고
    #     알려주는 것은 다르다. 물리만 제한하면 정책은 벽에 계속 부딪히며
    #     액션만 키운다. joint_vel_limit 은 applied_torque 에 의존하지 않고
    #     관절 속도만 보므로 DC 물리에서도 그대로 유효하다.
    #   mx_bot 의 XM_robot04_getup 이 use_dc_motor=True 로 물리 자체가
    #   실기 토크 곡선이 됐다. 포락선 페널티는 "물리는 그대로 두고
    #   실현 불가 영역에 비용을 매기는" 우회책이라 이제 중복이다.
    #   이중 제약은 학습만 어렵게 한다. 되돌릴 때를 대비해 항은 남긴다.
    #
    #   ※ 우회책이 왜 부족했는지 (2026-08-29 실측)
    #     에피소드 평균이라 기상 구간(13%)보다 정지 구간(74%)을 개선하는
    #     쪽이 쉬웠다. 평균은 -0.232 -> -0.082 로 좋아졌지만
    #         정지 ω 2.13 -> 0.61  (해결)
    #         기상 ω 4.57 -> 4.57  (변화 없음)
    #     결국 그 정책은 DC 물리에서 기립 성공률 0% 였다.
    #   walk_kick_v1_4 와 같은 함수·같은 상수·같은 weight. 관절만 18 개다.
    torque_envelope = RewTerm(
        func=kick_stability_mdp.torque_envelope_penalty,
        weight=_CURR_ENVELOPE[0],   # 램프인: penalty_rampin 이 레벨 4 이후 올린다
        params={
            "stall_torque": MOTOR_STALL_TORQUE,
            "no_load_speed": MOTOR_NO_LOAD_SPEED,
            "effort_cap": MOTOR_EFFORT_CAP,
            "margin": MOTOR_ENVELOPE_MARGIN,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=GETUP_JOINTS, preserve_order=True),
        },
    )
    # 포락선의 강건 버전 — applied_torque 구현에 안 의존하고 속도만 본다.
    joint_vel_limit = RewTerm(
        func=kick_stability_mdp.joint_vel_over_limit_penalty,
        weight=_CURR_JOINTVEL_RAMP[0],   # 램프인: penalty_rampin 이 레벨 4 이후 올린다
        params={
            "vel_limit": JOINT_VEL_SOFT_LIMIT,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=GETUP_JOINTS, preserve_order=True),
        },
    )

    # 기한이 지나도 보행 기본자세가 아니면 벌점.
    #   walk_ready_pose(지수 보너스)는 0.83 에서 포화해 1,300 iteration 동안
    #   거의 안 움직였다. 보행의 stop_default_posture_v01 과 같은
    #   sum(square(dev)) x -1.0 형태라 포화 없이 끝까지 민다.
    #   기한 1.5 초 — 실측 기상 완료가 1.0~1.2 초라 여유를 뒀다. 1.0 으로
    #   잡으면 "더 빨리 일어나라" 압력이 되는데, 기상 관절속도가 이미
    #   4.57 rad/s 로 액추에이터 한계(4.5)를 넘어 sim2real 이 악화된다.
    walk_pose_deadline = RewTerm(
        func=getup_v1_rewards.walk_pose_deadline,
        weight=_CURR_WALK_POSE[0],   # 램프인: penalty_rampin 이 레벨 4 이후 올린다
        params={
            "target_pose": _WALK_READY_POSE,
            **_H,                      # 높이 게이트용 (2026-09-02, 시간 게이트 대체)
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=GETUP_JOINTS, preserve_order=True),
        },
    )

    # ── 기립 후 안정화: "보행에서 cmd=0" 과 같은 상태 ──
    #   실측(model_2300, step 120~470): 수평 0.102 m/s, 회전 0.381 rad/s 로
    #   계속 흔들린다. 높이는 0.34 mm 로 완벽한데 몸통 운동을 벌주는 항이
    #   하나도 없어서다. 가중치는 전부 velocity_v0_6(실기 검증)에서 그대로
    #   가져왔고, height_frac 곱으로 기립 후에만 걸린다.
    # 🆕 높이 우물 — 기립 후 몸통 높이를 0.39 에 이차로 고정.
    #   보행 base_height_l2(target 0.39, weight -150) 을 그대로 이식했다.
    #   겟업의 base_height 보상은 기립하면 1.0 으로 포화해 **높이 기울기가
    #   0** 이 된다. 그래서 까딱거려도 손해가 없다. 실측 격차:
    #       base_z 표준편차  보행 0.15 mm  vs  겟업 5.78 mm  (39 배)
    #       평균 base_z      보행 0.395 m  vs  겟업 0.401 m  (6 mm 높음)
    #   게이트(0.85)가 걸려 있어 다 일어선 뒤에만 작동한다.
    base_height_well = RewTerm(         # 보행 base_height (-150.0)
        func=getup_v1_rewards.base_height_well,
        weight=_CURR_HEIGHT_WELL[0],   # 램프인: penalty_rampin 이 레벨 4 이후 올린다
        params={**_H, "target_height": _STAND_HEIGHT},
    )

    # ── 기립 후 안정화 (2026-09-01, 실기 관찰 반영) ────────────────────
    #   실기 증상: 기상 자체는 잘 되는데, **일어난 뒤 기본자세로 수렴하지
    #   못하고 과격한 동작이 나온다.** sim 에서는 안 보인다(base_z 산포
    #   0.46 mm, 발 접촉 0.23 회/s). 모델 오차가 없어서다.
    #
    #   원인: 보행에 있는 damping 항이 여기엔 전부 꺼져 있었다.
    #       항                  보행 v0_6   여기(이전)
    #       base_ang_vel_xy      -0.05       0.0
    #       base_roll            -5.0        0.0
    #       base_lin_vel_z       -2.0        0.0
    #   실기의 작은 오차를 잡아줄 것이 없으니 진동으로 번진다.
    #
    #   ★ 페널티 셋만 켠다. 양수 보너스(stand_still_lin/ang)는 계속 끈다 ★
    #     10차(FAILED_23-46-50)가 다섯 개를 한꺼번에 켰다가 성공률
    #     0.9995 -> 0.06 으로 무너졌는데, 기록상 범인은 **양수 보너스**가
    #     중간 높이에 국소해를 만든 것이었다(가만히 있으면 보너스 +0.76/초).
    #     페널티 쪽은 무해했다(합 -0.19, action_rate -2.21 대비 미미).
    #   모두 _stand_gate(0.85) 가 걸려 있어 다 일어선 뒤에만 작동한다.
    stand_still_lin = RewTerm(          # 보행 track_lin_vel_xy (+1.0)
        func=getup_v1_rewards.stand_still_lin_vel, weight=0.0, params=_H,
    )
    stand_still_ang = RewTerm(          # 보행 track_ang_vel_z (+1.0)
        func=getup_v1_rewards.stand_still_ang_vel, weight=0.0, params=_H,
    )
    base_lin_vel_z = RewTerm(           # 보행 base_linear_velocity (-2.0)
        func=getup_v1_rewards.base_lin_vel_z, weight=-2.0, params=_H,   # 2026-09-01 켬
    )
    base_ang_vel_xy = RewTerm(          # 보행 base_angular_velocity (-0.05)
        func=getup_v1_rewards.base_ang_vel_xy, weight=-0.05, params=_H,  # 2026-09-01 켬
    )
    base_roll = RewTerm(                # 보행 base_roll_l2 (-5.0)
        func=getup_v1_rewards.base_roll, weight=-5.0, params=_H,   # 2026-09-01 켬
    )
    foot_flat_roll = RewTerm(           # 보행 foot_flat (-0.3), roll 만
        func=getup_v1_rewards.foot_flat_roll,
        weight=-0.3,
        params={**_H, "asset_cfg": SceneEntityCfg("robot", body_names=MX_FEET)},
    )
    # ★ 기상 전용 ★ 보행은 toe-off/heel-strike 때문에 pitch 를 일부러 뺐지만
    #   기상의 최종 상태는 정지라 발바닥이 앞뒤로도 평평해야 한다.
    #   관측된 "발꿈치를 들었다 놨다" 가 이 성분이다. weight 는 roll 과 동일.
    foot_flat_pitch = RewTerm(
        func=getup_v1_rewards.foot_flat_pitch,
        weight=-0.3,
        params={**_H, "asset_cfg": SceneEntityCfg("robot", body_names=MX_FEET)},
    )

    # ── 자기충돌: 팔이 몸통/골반/허벅지를 미는 힘 ──
    #   ContactSensor 의 바디쌍 필터로 직접 읽는다. 바닥 접촉은 필터에
    #   없으므로 섞이지 않는다. 실측(model_1900 기립 롤아웃):
    #       r_elbow_pitch_link <-> r_pelvis_roll_link
    #       step 60: 35.34 N   step 180: 17.66 N   step 469: 5.53 N
    #       (다른 링크쌍은 전부 0.00 N)
    #   체중 55.9 N 대비 35 N 이면 상당하다. 스케일이 크므로 weight 를
    #   작게 잡는다 — 35 N x 0.01 = 0.35/초.
    self_collision = RewTerm(
        func=getup_v1_rewards.self_collision,
        weight=-0.01,
        params={"sensor_names": _SELF_COLL_SENSORS, "force_threshold": 1.0},
    )

    # ── 정규화 (기상은 크게 휘저어야 하므로 약하게) ──
    # alive 는 뺐다 — 종료 조건이 time_out(truncated) 뿐이라 terminated 가
    # 항상 False 이고, 상수 리워드는 어드밴티지에서 상쇄돼 그래디언트가 0 이다.
    joint_vel = RewTerm(func=getup_v1_rewards.joint_vel_l2, weight=-1e-4)
    # action_rate / joint_acc 는 regularization_curriculum 이 매 스텝
    # 덮어쓴다. 여기 값은 0 단계와 같게 맞춰 둔 초기값이다.
    joint_acc = RewTerm(func=getup_v1_rewards.joint_acc_l2, weight=_CURR_JOINT_ACC[0])
    action_rate = RewTerm(func=getup_v1_rewards.action_rate_l2, weight=_CURR_ACTION_RATE[0])
    dof_pos_limits = RewTerm(func=getup_v1_rewards.joint_pos_limits, weight=-5.0)
    joint_torques = RewTerm(func=getup_v1_rewards.joint_torques_l2, weight=-1e-5)


@configclass
class GetupV12TerminationsCfg:
    # 정상 종료는 time_out 뿐이다.
    #   성공 종료를 넣지 않는다 — 일어섰다고 끊으면 정책이 그 상태를
    #   순간적으로 찍는 것만 배우고 유지를 안 배운다. 계속 굴리면
    #   stand_success(0.1/스텝)가 쌓여서 "빨리 일어나 계속 서 있기"가
    #   자연스럽게 유도된다 (3 초 기립 35.0 vs 8 초 기립 10.0).
    #
    #   높이 하한 / 자세 이상 종료도 넣을 수 없다. 누운 자세가 시작
    #   상태라 매 에피소드가 첫 스텝에 끝난다. (보행은 이 둘을 쓴다)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 수치 폭주 안전장치 — 명백히 비정상인 경우만 잡는다.
    #   실측 base_z (scripts/rsl_rl 계열 측정, 2026-08-28):
    #       보행 기본각(크라우치)  0.3957   ← 기립 목표
    #       다리 완전 신전         0.4431   ← 물리적 상한에 가까움
    #   0.45 로 잡으면 완전 신전 대비 여유가 7 mm 뿐이다. 발목을 세우거나
    #   (ankle_pitch 가동범위 ±1.83/0.87), 팔로 밀 때 오버슈트하거나,
    #   도메인 랜덤화(질량 ±7%, 마찰 0.6~1.4)로 정착 높이가 흔들리면
    #   정상 동작이 종료로 처리된다. 그러면 원인 파악도 어렵다.
    #
    #   0.50 = 완전 신전 대비 +0.057 m, 정상 기립의 1.26 배.
    #   depenetration 임펄스로 튀는 경우는 수십 cm 단위로 날아가므로
    #   이 값으로도 폭주 감지 목적은 그대로 달성된다.
    flew_away = DoneTerm(
        func=getup_v1_mdp.root_height_above_maximum,
        params={"maximum_height": 0.50},
    )


@configclass
class GetupV12CurriculumCfg:
    # ★ 역방향 커리큘럼 — 이 태스크의 핵심 ★
    #   거의 서 있는 상태(15°)에서 시작해 성공률 50% 를 넘을 때마다
    #   한 단계씩 눕힌다. 최종 단계 90° 가 실제 과제(supine)다.
    #   로그: Curriculum/reverse_getup 의 level / fall_angle / success_rate
    reverse_getup = CurrTerm(
        func=getup_v1_mdp.reverse_getup_curriculum,
        params={
            # v1 원판은 고정 각도다. V2 가 넣은 밴드 샘플링을 끈다.
            "band": False,
            "fall_angles": _CURR_FALL_ANGLES,
            "success_threshold": _CURR_SUCCESS_THRESHOLD,
        },
    )

    regularization = CurrTerm(
        func=getup_v1_mdp.regularization_curriculum,
        params={
            "stage_steps": _CURR_STAGE_STEPS,
            "action_rate": _CURR_ACTION_RATE,
            "joint_acc": _CURR_JOINT_ACC,
            # 🆕 이어받기 판은 3단계에서 시작 (2026-09-02).
            #    action_rate -0.60 = 보행(-0.15)의 4배, joint_acc -5.0e-6.
            #    이어받으면 common_step_counter 가 0 부터라 가장 약한 값으로
            #    되돌아가는데, 실기 이관 단계에서는 역행이다.
            #    처음부터 학습할 때는 0 으로 되돌릴 것.
            "start_stage": 3,
        },
    )

    # 실기 이관용 페널티 램프인 — **레벨 5(90 deg 증명 관문)** 도달 후에 켠다.
    #   2026-08-30 확정: 팔 1.5 + 페널티 없음만 완주(199/410/610/810).
    #   팔 1.5 라도 페널티를 처음부터 켜면 레벨 1 에서 막힌다(02-54-38).
    #   반면 이미 일어설 줄 아는 정책에는 무해하다(19-35-26 이 전량으로 완주).
    #   그래서 커리큘럼 완주 후에 켠다.
    #   로그: Curriculum/penalty_rampin (0 -> 1 -> 2)
    penalty_rampin = CurrTerm(
        func=getup_v1_mdp.penalty_rampin_curriculum,
        params={
            "stage_steps": _CURR_PENALTY_STEPS,
            "walk_pose_deadline": _CURR_WALK_POSE,
            "torque_envelope": _CURR_ENVELOPE,
            "joint_vel_limit": _CURR_JOINTVEL_RAMP,
            "base_height_well": _CURR_HEIGHT_WELL,
            "from_level": _CURR_PENALTY_FROM_LEVEL,
        },
    )

    # 외란 램프인 — 레벨 5 이후 0 -> 절반 -> 전량
    #   로그: Curriculum/disturbance_rampin
    disturbance_rampin = CurrTerm(
        func=getup_v1_mdp.disturbance_rampin_curriculum,
        params={
            "stage_steps": _CURR_PENALTY_STEPS,
            "push_vel": _CURR_PUSH_VEL,
            "ext_force": _CURR_EXT_FORCE,
            "knockdown_vel": _CURR_KNOCKDOWN,
            "from_level": _CURR_PENALTY_FROM_LEVEL,
        },
    )

    reset_range = CurrTerm(
        func=getup_v1_mdp.supine_reset_curriculum,
        params={
            "stage_steps": _CURR_STAGE_STEPS,
            "side_roll": _CURR_SIDE_ROLL,
            "tilt": _CURR_TILT,
            "joint_noise": _CURR_JOINT_NOISE,
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """MX3 supine get-up 환경 (18 DOF)."""

    scene: GetupV12SceneCfg = GetupV12SceneCfg(num_envs=16384, env_spacing=2.5)

    observations: GetupV12ObservationsCfg = GetupV12ObservationsCfg()
    actions: GetupV12ActionsCfg = GetupV12ActionsCfg()
    commands = None                       # 속도 명령 없음

    rewards: GetupV12RewardsCfg = GetupV12RewardsCfg()
    terminations: GetupV12TerminationsCfg = GetupV12TerminationsCfg()
    events: GetupV12EventCfg = GetupV12EventCfg()
    curriculum: GetupV12CurriculumCfg = GetupV12CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        # 0.5 s 는 정착에 쓰이므로 실제 학습 구간은 9.5 s 다.
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # ★ 10 * 2**15 (327,680) 이었다가 넘쳤다 (2026-08-29) ★
        #   level 4 도달 = 16,384 대가 전부 바닥에 누운 시점(iter 6156)부터
        #   "Patch buffer overflow, need at least 343,686" 이 964,857 회 발생.
        #   초과분 접촉은 버려지므로 발이 바닥을 안 짚은 것처럼 계산된다.
        #
        #   기상은 프로젝트에서 접촉이 가장 많은 태스크다 — 누워서 시작해
        #   몸 전체가 지면에 닿고, self_collisions=True 로 링크끼리도 닿는다.
        #   그런데 삭제한 구 getup_env_cfg.py 의 값을 그대로 물려받아
        #   프로젝트에서 **가장 낮은** 값을 쓰고 있었다:
        #       walk_kick_v1_4   24 * 2**15 = 786,432
        #       velocity v0_5~   20 * 2**15 = 655,360
        #       getup (구)       10 * 2**15 = 327,680  ← 부족
        #   킥과 같은 값으로 올린다. 요구값 343,686 의 2.3 배로 여유가 있다.
        self.sim.physx.gpu_max_rigid_patch_count = 24 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """추론용. **항상 supine(90°)에서 시작한다.**

    역방향 커리큘럼은 env 인스턴스에 상태(_getup_level)를 들고 있어서,
    새로 만든 env 는 항상 level 0(15°, 거의 서 있음)부터 시작한다. 그대로
    추론하면 학습된 기상 능력이 아니라 "살짝 기운 상태에서 균형 잡기" 만
    보게 된다. 여기서는 커리큘럼을 끄고 최종 단계를 직접 박는다.

    다른 단계를 보고 싶으면 아래 PLAY_FALL_ANGLE_DEG 만 바꾸면 된다.
    """

    PLAY_FALL_ANGLE_DEG = 90.0      # 0=직립, 90=supine(원래 과제)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.curriculum.reverse_getup = None

        # ★ 2026-09-16 — 재생 전용 바닥 색 ★
        #   【왜 필요한가】 terrain_type="plane" 은 GroundPlaneCfg 를 쓰고
        #   그 USD 가 격자 환경이다
        #   (ISAAC_NUCLEUS_DIR/Environments/Grid/default_environment.usd).
        #   import_ground_plane 은 visual_material 에서 **diffuse_color 한 값만**
        #   뽑아 그 격자에 tint 로 곱한다. 기본 tint 가 (0,0,0) 이라 재질을
        #   안 주면 **새까만 바닥**이 되고, 돔 라이트의 하늘 HDR 만 보여
        #   로봇이 허공에 뜬 것처럼 나온다.
        #
        #   ⚠ MdlFileCfg 는 이 경로에서 **무시된다** — diffuse_color 필드가
        #     없어서 "no diffuse color found" 경고 후 검정이다. 보행
        #     (velocity_v0_6)은 terrain_type="generator" 라 import_mesh 로
        #     가고 MDL 이 제대로 적용돼 격자 없는 회색이다. 그 구성도
        #     시험해 봤지만(물리 동일, 격자 없음) **격자 쪽이 낫다는
        #     사용자 판단으로 plane 을 유지한다.**
        #
        #   시각 전용이라 물리·정책에 영향이 없다. 학습 cfg(RobotEnvCfg)는
        #   건드리지 않는다 — model_53400 계보와 설정을 계속 일치시킨다.
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.45, 0.45, 0.48),
        )
        self.events.reset_base_supine.params["fall_angle_deg"] = self.PLAY_FALL_ANGLE_DEG
