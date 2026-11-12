# walk_kick_v1_5_env.py
# ============================================================================
# ★★★ v1_5 — 경기에서 넘겨받는 조건으로 학습 (2026-10-31) ★★★
#   W90B70 (목표 ±90° / 공 방위 ±70°) 위에서 이어 학습한다. 관측 265 / 행동 12 그대로라
#   러너·실기 쪽 수정 없이 ONNX 만 바꿔 끼운다. 킥 레이어 보상 (파워×품질 곱, 1 회 래치,
#   접촉 예산, 강타 문턱) 과 v1_4 안정성 항은 손대지 않는다.
#
#   【왜】 scripts/mujoco_sim/kick_match_cond.py — 4v4 시뮬 경기 넘겨받기 615 건을 브리지 물리로
#     W90B70 18200 에 다시 차게 하면: 서서·공 정확 0.2 % / 걷다가 넘겨받기 4~6 % /
#     공이 추정보다 +3·+4 cm 멂 6 · 19 % / 0.3 m/s 로 굴러 멀어짐 19 % / 헛발질 3 %.
#     넘어진 판 95 % 가 오른발·공 맞힌 뒤 0.3~0.5 s·앞으로. 경기 실측 (kick_handoff_probe.py):
#     넘겨받을 때 46 % 가 0.5 rad/s 넘게 회전, 30 % 가 옆걸음 중.
#
#   【바뀌는 것 — 전부 "학습이 경기와 다른 곳" 을 메운다】
#     ① 로봇: XM_robot04_convexhull_kick — 메시 볼록 껍질 충돌 + 실기 기본자세 (팔 내림,
#        어깨 피치 ∓0.3 · 팔꿈치 ±1.0, 위치 유지). W90B70 은 캡슐 + 팔 접음 (±2.967, 늘어진 팔).
#        팔은 실기처럼 유연하게 (게인 20 / 1, 토크 0.3~0.8 N·m) — 2026-11-01 수정, 처음엔 기상용 50 / 2.5 / 5.
#     ② 행동: 다리 목표를 Profile Velocity 40~50 rpm 으로 따라감 (실기 45 rpm).
#     ③ 시작 상태: 85 % 는 최종 보행 v0_6 13500 이 넘겨받기 명령으로 걷다가 명령 0 을 0.04 s
#        받은 직후 상태 (은행). 그 뒤는 실기 러너와 같다 — 0.3 s 기본자세 홀드, 위상 0.
#     ④ 공 믿음 오차: 정책이 아는 공 (frozen) 과 실제 공을 에피소드마다 어긋나게 —
#        고정 오차 45 % / 굴러가는 중 10 % / 헛발질 5 % / 다툼으로 튕겨 나감 5 %.
#     ⑤ 공 물리: max_angular_velocity 50 (deg/s 라 사실상 못 구름) → 해제 + 구름 저항 외력.
#     ⑦ (2026-11-01, 19500 부터) 몸통 yaw 각속도 상한 제곱 벌점 + 팔 유연하게 (① 참고).
#        19500~20900: 4.0 rad/s / −0.5,  20900~: 3.5 rad/s / −2.0
#   【안 바뀌는 것】 관측·행동 차원, 홀드 0.3 s, 1.0 s 전 킥 금지, 목표·공 분포, 대칭 (증강+미러).
#
#   【킥 세기】 1b 단계에서 (이 학습이 끝난 뒤 이어서). 관측 kick_strength 자리는 이미 있다.
#
#   gym id: Unitree-MX3-WalkKick-V1-5 / -Play. experiment=unitree_mx3_walkkick_v1_5.
#   resume: unitree_mx3_walkkick_v1_4_w90b70/2026-08-19_14-51-43/model_18200.pt
# ============================================================================

import math
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.mx_bot import MX_BOT_KICK_CFG
from unitree_rl_lab.tasks.locomotion.mdp import kick_handoff_mdp as kh

from . import walk_kick_v1_4_env as v14

HANDOFF_BANK = os.environ.get(
    "KICK_HANDOFF_BANK",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "handoff_bank_v0_6_13500.pt"),
)

STANDING_START_PROB = 0.15
BELIEF_PROBS = {"nominal": 0.35, "bias": 0.45, "roll": 0.10, "whiff": 0.05, "duel": 0.05}
BIAS_X_RANGE = (-0.05, 0.06)        # + = 실제 공이 더 멀다 (MuJoCo 에서 +3 cm 부터 넘어짐이 급증)
BIAS_Y_RANGE = (-0.04, 0.04)
ROLL_SPEED_RANGE = (0.05, 0.40)     # 행동 노드 KickEnvelope.max_ball_speed 0.4 까지 넘겨준다
ROLL_AWAY_PROB = 0.5
DUEL_TIME_RANGE = (0.4, 1.2)        # [s] 에피소드 시각 (홀드 0.3 s 포함). 킥은 1.0~1.3 s 에 난다
DUEL_SPEED_RANGE = (0.5, 2.0)       # 경기 기록: 다툼 중 넘겨받은 순간 공 0.9~2.0 m/s
PROFILE_RPM_RANGE = (40.0, 50.0)
ARM_EFFORT_RANGE = (0.3, 0.8)       # [N·m] 팔 관절 최대 힘 (실기 goal PWM 5.6 %, 브리지 0.5)
# [rad/s] 몸통 yaw 각속도 상한 (W90B70 헛발질 4.4, v1_5 19200 6.3).
#   19500~20900 은 4.0 / −0.5 — 약해서 헛발질 yaw 5.55 → 5.47 밖에 안 줄었다 (MuJoCo 헛발질 넘어짐 14~19 %).
#   20900~21400 은 3.5 / −2.0 — 실패 (회전은 4.3 으로 줄었는데 헛발질 46 %, 다른 조건도 5~7 %). 20500 값으로 되돌림.
YAW_RATE_LIMIT = 4.0
YAW_RATE_WEIGHT = -0.5
# 공 구름 저항 [m/s²] — scripts/kick_v1_5/calib_ball_roll.py 로 MuJoCo 경기 공 (convexhull, rolling 0.030) 에 맞춤.
#   1.6: 회전 없이 출발 (킥에 가까움) 2/3/4/5 m/s → 0.63/1.34/2.25/3.36 m (MuJoCo 0.63/1.32/2.24/3.34),
#        구르며 4 m/s → 3.80 m (MuJoCo 3.11). 2.0 이면 굴림 출발은 맞지만 킥 쪽이 15 % 짧다.
BALL_ROLL_DECEL = float(os.environ.get("KICK_BALL_ROLL_DECEL", "1.6"))


def _apply_v1_5(cfg):
    # ① 로봇
    cfg.scene.robot = MX_BOT_KICK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # ⑤ 공: PhysX max_angular_velocity 는 deg/s. 4.5 m/s 로 구르려면 69 rad/s ≈ 4000 deg/s.
    cfg.scene.ball.spawn.rigid_props.max_angular_velocity = 10000.0

    # ② 행동 (지연·홀드·클램프는 v1_4 그대로)
    a = cfg.actions.JointPositionAction
    cfg.actions.JointPositionAction = kh.ProfileVelocityJointPositionActionCfg(
        asset_name=a.asset_name,
        joint_names=a.joint_names,
        scale=a.scale,
        use_default_offset=a.use_default_offset,
        preserve_order=a.preserve_order,
        target_limits=a.target_limits,
        action_delay_range=a.action_delay_range,
        profile_rpm_range=PROFILE_RPM_RANGE,
    )

    # ③ 시작 상태 — reset_robot_joints 자리 (reset_base 바로 뒤, reset_ball 앞) 를 덮어써 순서를 지킨다
    old = cfg.events.reset_robot_joints.params
    cfg.events.reset_robot_joints = EventTerm(
        func=kh.ResetFromHandoffBank,
        mode="reset",
        params={
            "bank_path": HANDOFF_BANK,
            "standing_prob": STANDING_START_PROB,
            "position_range": old["position_range"],
            "velocity_range": old["velocity_range"],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 은행 상태는 이미 걷던 속도를 갖고 있으니 reset_base 의 작은 속도 랜덤은 서서 시작에만 의미가 있다 (그대로 둠)

    # ④ 공 믿음 오차 — 새 항은 이벤트 목록 끝에 붙으므로 reset_kick_state (frozen 관측 기록) 뒤에 돈다
    cfg.events.ball_belief = EventTerm(
        func=kh.perturb_ball_belief,
        mode="reset",
        params={
            "probs": BELIEF_PROBS,
            "bias_x_range": BIAS_X_RANGE,
            "bias_y_range": BIAS_Y_RANGE,
            "roll_speed_range": ROLL_SPEED_RANGE,
            "roll_away_prob": ROLL_AWAY_PROB,
            "whiff_distance": 3.0,
            "duel_time_range": DUEL_TIME_RANGE,
        },
    )
    cfg.events.duel_knock = EventTerm(
        func=kh.duel_knock_ball,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"speed_range": DUEL_SPEED_RANGE},
    )
    cfg.events.ball_rolling_resistance = EventTerm(
        func=kh.ball_rolling_resistance,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"decel": BALL_ROLL_DECEL},
    )

    # ①' 팔 토크 한계 0.3~0.8 N·m (기본 0.5 = 브리지). 실기 goal PWM 5.6 % 라 정확한 값을 모른다.
    cfg.events.arm_effort_limit = EventTerm(
        func=kh.randomize_joint_effort_limit,
        mode="startup",
        params={"effort_range": ARM_EFFORT_RANGE,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])},
    )

    # ⑦ 몸통 yaw 각속도 상한 (2026-11-01 추가 — 디딤발이 도는 바닥에서 헛발질 시 뒤로 넘어짐)
    cfg.rewards.base_yaw_rate_limit = RewTerm(
        func=kh.base_yaw_rate_over_limit, weight=YAW_RATE_WEIGHT, params={"limit": YAW_RATE_LIMIT})

    # ⑥ 넘어짐 종료를 종류별로 (임계는 v1_4 STAGE_HARD 그대로: 0.6 rad / 0.28 m)
    lim = cfg.terminations.bad_orientation.params["limit_angle"]
    hmin = cfg.terminations.base_height.params["minimum_height"]
    cfg.terminations.bad_orientation = None
    cfg.terminations.base_height = None
    for kind in ("stand", "nominal", "bias", "roll", "whiff", "duel"):
        setattr(cfg.terminations, f"fall_{kind}", DoneTerm(
            func=kh.fall_by_type, params={"kind": kind, "limit_angle": lim, "minimum_height": hmin}))


@configclass
class RobotV15EnvCfg(v14.RobotW90B70EnvCfg):
    """학습용 — W90B70 + 넘겨받기 조건 (convexhull·팔 내림·Profile Velocity·은행 시작·공 믿음 오차·구르는 공)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_5(self)


@configclass
class RobotV15PlayEnvCfg(v14.RobotW90B70PlayEnvCfg):
    """추론용 — 학습과 같은 조건 (조건별 넘어짐 확인용). 공 오차 없이 보려면 BELIEF_PROBS 를 바꿀 것."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_5(self)


# ============================================================================
# ★ v1_5b — 킥 세기 (2026-11-01). v1_5 (20500 구성) 위에 관측 kick_strength = s ∈ [0, 1].
#   목표 공속 (목표방향) v*(s) = V_FULL · (LO + (1 − LO)·s). s = 1 은 기존 보상 그대로.
#   러너는 이미 kick_strength 파라미터 (0~1, 트리거 때 고정) 를 관측에 넣는다 → 입출력·러너 변경 없음.
#   resume: unitree_mx3_walkkick_v1_5/2026-11-01_01-24-08/model_20500.pt
# ============================================================================
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402

from unitree_rl_lab.tasks.locomotion.mdp import kick_mdp  # noqa: E402

STR_V_FULL = 2.8          # [m/s] Isaac 서서 최대 킥 목표방향 공속 (20500 2.72, W90B70 2.84 — 공이 구르게 된 뒤)
STR_LO = 0.35             # s = 0 → 최대의 35 % (≈ 1.0 m/s)
STR_FULL_PROB = 0.4       # 에피소드의 40 % 는 s = 1 (슛 유지)
STR_SIGMA_FRAC, STR_SIGMA_MIN = 0.12, 0.15
STR_LIN_ERR, STR_LIN_W = 2.0, 0.5   # 시도 2~: 목표에서 2 m/s 까지 직선으로 줄어드는 항을 반 섞음 (시도 1 은 0)
STR_SOFT_SCALE = 0.5      # 시도 5~: s < 1 맞춤 보상 크기 (시도 1~4 는 1.0)
FALL_AFTER_KICK_W, FALL_AFTER_KICK_S = -300.0, 2.0  # 시도 5~: 찬 뒤 2 s 안 넘어지면 1 회 −300 × dt 0.02 = −6 (킥 1 회 보상 ≈ +7)
STR_OVER_MARGIN = 0.25    # [m/s] s < 1 에서 v* 를 이만큼 넘으면 벌점
STR_EVAL_LEVELS = None    # 평가 때 (0.0, 0.25, 0.5, 0.75, 1.0) 처럼 env 순서대로
STR_LO_START = 0.8        # 시도 3~: 하한 커리큘럼 0.8 → 0.35 (오차 EMA < 15 % 마다 0.1). None = 끔 (시도 1·2)


def _apply_v1_5b(cfg):
    for grp in (cfg.observations.policy, cfg.observations.critic):
        grp.kick_strength = ObsTerm(func=kick_mdp.kick_strength_obs)
    cfg.events.kick_strength = EventTerm(
        func=kh.sample_kick_strength_v15,
        mode="reset",
        params={"full_prob": STR_FULL_PROB, "lo": STR_LO, "strike_full": v14.INSIDE_MIN_STRIKE_SPEED,
                "eval_levels": STR_EVAL_LEVELS, "lo_start": STR_LO_START, "err_threshold": 0.20},
    )
    bv = cfg.rewards.ball_velocity
    cfg.rewards.ball_velocity = RewTerm(
        func=kh.ball_velocity_strength, weight=bv.weight,
        params={**bv.params, "v_full": STR_V_FULL, "lo": STR_LO, "sigma_frac": STR_SIGMA_FRAC,
                "sigma_min": STR_SIGMA_MIN, "lin_err": STR_LIN_ERR, "lin_w": STR_LIN_W, "soft_scale": STR_SOFT_SCALE},
    )
    cfg.rewards.fall_after_kick = RewTerm(
        func=kh.fall_after_kick, weight=FALL_AFTER_KICK_W,
        params={"window_s": FALL_AFTER_KICK_S, "limit_angle": v14.TerminationsCfg().bad_orientation.params["limit_angle"],
                "minimum_height": v14.TerminationsCfg().base_height.params["minimum_height"]})
    ov = cfg.rewards.ball_overspeed
    cfg.rewards.ball_overspeed = RewTerm(
        func=kh.ball_overspeed_strength, weight=ov.weight,
        params={"target_full": ov.params["target_speed"], "v_full": STR_V_FULL, "lo": STR_LO,
                "margin": STR_OVER_MARGIN, "post_kick_window_s": ov.params["post_kick_window_s"]},
    )


@configclass
class RobotV15BEnvCfg(RobotV15EnvCfg):
    """학습용 — v1_5 + 킥 세기."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_5b(self)


@configclass
class RobotV15BPlayEnvCfg(RobotV15PlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_5b(self)
