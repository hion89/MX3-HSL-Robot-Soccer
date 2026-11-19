# walk_kick_v1_6_env.py
# ============================================================================
# ★ v1_6 — 디딤발이 버티지 못하는 원인을 여러 갈래로 나눠 넣은 강건화 이어 학습 (2026-11-16) ★
#   v1_5b (model_22300) 위에서 이어 학습한다. 관측 265 / 행동 12 / 관절 순서 / action_scale 0.25 그대로라
#   러너·실기 코드 수정 없이 ONNX 만 바꿔 끼울 수 있다 (러너 쪽 변경 제안은 별도 문서).
#
#   【근거 — MuJoCo sim2sim, 22300, 서서 시작, 조건당 300 회】 sim2real_evidence/kick_robust_20261116/
#     학습과 같은 관측 순서 2.3 % / 실물 러너 관측 순서 13 %
#     XM540 속도-토크 곡선 12 / 11.1 / 10.6 V → 27 / 34 / 33 %
#     IMU 지연 10 / 20 / 40 ms → 22 / 44 / 82 %,  행동 지연 +1 / +2 / +3 스텝 → 24 / 28 / 35 %
#     미끄럼 마찰 0.6 / 0.4 → 23 / 67 %
#     비틀림 마찰을 낮추면 디딤발이 27~46° 도는 대신 MuJoCo 에선 오히려 안 넘어진다 → "바닥 마찰" 하나로 설명 안 됨
#
#   【갈래 — 스위치로 켜고 끈다 (A/B/C 비교용)】
#     ACT  : 다리 구동기 = 전압 배율 DC 속도-토크 곡선 (stall 7.3 N·m · 무부하 5.55 rad/s @12V, 10.6~12.4 V).
#            평평한 5 N·m 와 섞는 β 를 0.25 → 1.0 커리큘럼 (넘어짐 EMA < 12 % 마다 +0.25).
#     OBS  : 관측 지연 — 50 % 에피소드는 러너처럼 전 프레임 1 스텝, IMU 는 여기에 0~1.5 스텝 (0~30 ms) 추가.
#     SURF : 발바닥 마찰 정지 0.30~1.30 / 운동 0.25~1.05 (기존 전신 0.45~1.35 위에 발만 덮어씀),
#            잔디 비틀림 부족 토크 c ∈ [0, 0.03] m (30 % 는 0) — Isaac 패치 마찰이 비틀림을 다 잡는 것을 보정.
#     CTR  : 러너 계약 — 시작 hold 0.3~0.6 s, 행동 저역 필터 τ 0.015~0.03 s (50 % 에피소드).
#     REW  : 접지 발 yaw 각속도 제곱 벌점 (킥 창 ×3) + 로그 메트릭.
#   【안 바뀌는 것】 관측·행동 차원, 보상 (REW 추가 1 항 외), 종료, 공·목표 분포, 대칭 증강, 킥 세기.
#
#   gym id: Unitree-MX3-WalkKick-V1-6A / -V1-6B / -V1-6C (+ -Play, -Play-Turf, -Play-IMU20, -Play-Volt106)
#   resume: unitree_mx3_walkkick_v1_5b/2026-11-01_06-55-23/model_22300.pt  (실행 절차: KICK_V1_6_KO.md)
# ============================================================================

import inspect

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.mx_bot import build_robot_cfg
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import kick_robust_mdp as kr

from . import walk_kick_v1_4_env as v14
from . import walk_kick_v1_5_env as v15

# ── 구동기 (XM540-W150, ROBOTIS 공식 @12V) ──────────────────────────────
DC_STALL_12V = 7.3          # [N·m]
DC_NOLOAD_12V = 5.55        # [rad/s] 53 rpm
VOLT_RANGE = (10.6, 12.4)   # [V] 실물 bag 최저 10.6 V, 완충 12.4 V 가정
DC_BLEND_START, DC_BLEND_STEP, DC_FALL_TH = 0.25, 0.25, 0.12

# ── 관측 지연 [정책 스텝 = 20 ms] ─────────────────────────────────────────
FRAME_DELAY_PROB, FRAME_DELAY_STEPS = 0.5, 1.0
IMU_EXTRA_RANGE = (0.0, 1.5)
OBS_MAX_DELAY = 3

# ── 바닥 ────────────────────────────────────────────────────────────────
FEET_STATIC_FR = (0.30, 1.30)
FEET_DYNAMIC_FR = (0.25, 1.05)
FEET_RESTITUTION = (0.0, 0.1)
TURF_C_RANGE = (0.0, 0.03)  # [m] τ = c·Fn·tanh(ω/0.5). Fn 55 N 에서 최대 1.65 N·m
TURF_ZERO_PROB = 0.3

# ── 러너 계약 ───────────────────────────────────────────────────────────
HOLD_RANGE_S = (0.3, 0.6)
FILTER_PROB, FILTER_TAU = 0.5, (0.015, 0.03)

# ── 보상 ────────────────────────────────────────────────────────────────
W_STANCE_YAW = -0.05

_FEET = v14.MX_FEET


def _feet_asset():
    return SceneEntityCfg("robot", body_names=_FEET, preserve_order=True)


def _feet_sensor():
    return SceneEntityCfg("contact_forces", body_names=_FEET, preserve_order=True)


def _robot_with_dc(dc_blend_init: float):
    """kick 로봇 cfg 를 새로 빌드하고 다리 그룹만 VoltageDCMotorCfg 로 바꾼다 (mx_bot.py 전역은 그대로)."""
    rob = build_robot_cfg("XM_robot04_convexhull_kick").replace(prim_path="{ENV_REGEX_NS}/Robot")
    acts = {}
    for name, a in rob.actuators.items():
        if name in ("arm", "neck"):
            acts[name] = a
            continue
        acts[name] = kr.VoltageDCMotorCfg(
            joint_names_expr=a.joint_names_expr,
            effort_limit=a.effort_limit_sim,          # 5.0 — 짧은 시간 상한 (기존 평평한 값)
            saturation_effort=DC_STALL_12V,
            velocity_limit=DC_NOLOAD_12V,
            effort_limit_sim=a.effort_limit_sim,      # 물리 솔버 안전망 (mx_bot.py 2026-08-29 주의 참고)
            velocity_limit_sim=a.velocity_limit_sim,
            stiffness=a.stiffness,
            damping=a.damping,
            armature=a.armature,
            friction=a.friction,
            dc_blend_init=dc_blend_init,
        )
    rob.actuators = acts
    return rob


def _material_kwargs():
    params = {
        "asset_cfg": _feet_asset(),
        "static_friction_range": FEET_STATIC_FR,
        "dynamic_friction_range": FEET_DYNAMIC_FR,
        "restitution_range": FEET_RESTITUTION,
        "num_buckets": 64,
    }
    try:
        sig = inspect.signature(mdp.randomize_rigid_body_material.__call__)
        if "make_consistent" in sig.parameters:
            params["make_consistent"] = True       # 운동 ≤ 정지 마찰
    except (TypeError, ValueError):
        pass
    return params


def _delayed(term: ObsTerm, inner: str, kind: str, asset_cfg=None) -> ObsTerm:
    p = {"inner": inner, "kind": kind, "max_delay_steps": OBS_MAX_DELAY,
         "asset_cfg": asset_cfg if asset_cfg is not None else SceneEntityCfg("robot")}
    return ObsTerm(func=kr.DelayedObs, params=p, scale=term.scale, noise=term.noise, clip=term.clip)


def _apply_v1_6(cfg, act=True, obs=True, surf=True, ctr=True, rew=True, curriculum=True, dc_blend=None):
    # ── ACT ──
    if act:
        cfg.scene.robot = _robot_with_dc(DC_BLEND_START if (curriculum and dc_blend is None) else
                                         (1.0 if dc_blend is None else dc_blend))
        cfg.events.motor_voltage = EventTerm(func=kr.randomize_motor_voltage, mode="reset",
                                             params={"volt_range": VOLT_RANGE})
        if curriculum:
            cfg.curriculum.dc_blend = CurrTerm(func=kr.dc_motor_blend_curriculum,
                                               params={"start": DC_BLEND_START, "step": DC_BLEND_STEP,
                                                       "fall_threshold": DC_FALL_TH})
        cfg.rewards.metric_motor_volt = RewTerm(func=kr.metric_motor_volt, weight=0.0)

    # ── OBS ──
    if obs:
        pol = cfg.observations.policy
        jcfg = pol.joint_pos_rel.params["asset_cfg"]
        pol.base_ang_vel = _delayed(pol.base_ang_vel, "ang_vel", "imu")
        pol.projected_gravity = _delayed(pol.projected_gravity, "gravity", "imu")
        pol.joint_pos_rel = _delayed(pol.joint_pos_rel, "joint_pos", "joint", jcfg)
        pol.joint_vel_rel = _delayed(pol.joint_vel_rel, "joint_vel", "joint", jcfg)
        cfg.events.obs_delays = EventTerm(func=kr.sample_obs_delays, mode="reset",
                                          params={"frame_delay_prob": FRAME_DELAY_PROB,
                                                  "frame_delay_steps": FRAME_DELAY_STEPS,
                                                  "imu_extra_range": IMU_EXTRA_RANGE})
        cfg.rewards.metric_imu_delay_ms = RewTerm(func=kr.metric_imu_delay_ms, weight=0.0)

    # ── SURF ── (events 는 선언 순서대로 돈다: 기존 physics_material (전신) 다음에 발만 덮어쓴다)
    if surf:
        cfg.events.feet_material = EventTerm(func=mdp.randomize_rigid_body_material, mode="reset",
                                             params=_material_kwargs())
        cfg.events.turf_torsion_sample = EventTerm(func=kr.sample_turf_torsion, mode="reset",
                                                   params={"coef_range": TURF_C_RANGE, "zero_prob": TURF_ZERO_PROB})
        cfg.events.turf_torsion = EventTerm(func=kr.apply_turf_torsion_deficit, mode="interval",
                                            interval_range_s=(0.02, 0.02),
                                            params={"wz_ref": 0.5, "fn_threshold": 5.0,
                                                    "sensor_cfg": _feet_sensor(), "asset_cfg": _feet_asset()})
        cfg.rewards.metric_turf_torque = RewTerm(func=kr.metric_turf_torque, weight=0.0)

    # ── CTR ──
    if ctr:
        cfg.events.reset_hold = EventTerm(func=kr.set_reset_hold_random, mode="reset",
                                          params={"hold_range_s": HOLD_RANGE_S})
        # hold 가 길어진 만큼 no-kick 데드라인 여유를 늘린다 (기존 margin 2.0 s 는 hold 0.3 기준)
        cfg.events.reset_kick_state.params["margin_s"] = (cfg.events.reset_kick_state.params["margin_s"]
                                                          + HOLD_RANGE_S[1] - 0.3)
        a = cfg.actions.JointPositionAction
        cfg.actions.JointPositionAction = kr.FilteredProfileVelocityJointPositionActionCfg(
            asset_name=a.asset_name, joint_names=a.joint_names, scale=a.scale,
            use_default_offset=a.use_default_offset, preserve_order=a.preserve_order,
            target_limits=a.target_limits, action_delay_range=a.action_delay_range,
            profile_rpm_range=a.profile_rpm_range, filter_prob=FILTER_PROB, filter_tau_range=FILTER_TAU)

    # ── REW ──
    if rew:
        cfg.rewards.stance_foot_yaw = RewTerm(
            func=kr.stance_foot_yaw_rate_penalty, weight=W_STANCE_YAW,
            params={"post_kick_window_s": v14.KICK_STAB_WINDOW_S, "boost": 3.0, "force_threshold": 1.0,
                    "sensor_cfg": _feet_sensor(), "asset_cfg": _feet_asset()})
    cfg.rewards.metric_post_kick_yaw_rate = RewTerm(func=kr.metric_post_kick_yaw_rate, weight=0.0,
                                                    params={"window_s": 1.0})


# ============================================================================
# 학습용 — A / B / C
# ============================================================================
@configclass
class RobotV16AEnvCfg(v15.RobotV15BEnvCfg):
    """A: 구동기만 (DC 곡선 + 전압 + β 커리큘럼)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6(self, act=True, obs=False, surf=False, ctr=False, rew=False)


@configclass
class RobotV16BEnvCfg(v15.RobotV15BEnvCfg):
    """B: A + 관측 지연 (러너 프레임 순서 + IMU)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6(self, act=True, obs=True, surf=False, ctr=False, rew=False)


@configclass
class RobotV16CEnvCfg(v15.RobotV15BEnvCfg):
    """C: 전체 (B + 발바닥 마찰·비틀림 부족 + 러너 계약 + 디딤발 yaw 벌점)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6(self, act=True, obs=True, surf=True, ctr=True, rew=True)


# ============================================================================
# 재생·평가용 — 커리큘럼 없이 β = 1 (실기 곡선 그대로)
# ============================================================================
@configclass
class RobotV16PlayEnvCfg(v15.RobotV15BPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6(self, curriculum=False, dc_blend=1.0)


@configclass
class RobotV16PlayTurfEnvCfg(RobotV16PlayEnvCfg):
    """미끄럽고 비틀림을 못 잡는 잔디 고정: 발 마찰 0.35~0.45, c = 0.03 (항상)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.feet_material.params["static_friction_range"] = (0.35, 0.45)
        self.events.feet_material.params["dynamic_friction_range"] = (0.30, 0.40)
        self.events.turf_torsion_sample.params["coef_range"] = (0.03, 0.03)
        self.events.turf_torsion_sample.params["zero_prob"] = 0.0


@configclass
class RobotV16PlayIMU20EnvCfg(RobotV16PlayEnvCfg):
    """러너 프레임 1 스텝 + IMU 1 스텝 (20 ms) 고정."""

    def __post_init__(self):
        super().__post_init__()
        self.events.obs_delays.params["frame_delay_prob"] = 1.0
        self.events.obs_delays.params["imu_extra_range"] = (1.0, 1.0)


@configclass
class RobotV16PlayVolt106EnvCfg(RobotV16PlayEnvCfg):
    """배터리 10.6 V 고정 (실물 bag 최저)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.motor_voltage.params["volt_range"] = (10.6, 10.6)


@configclass
class RobotV16PlayRealLikeEnvCfg(RobotV16PlayEnvCfg):
    """실물 유사 조건 고정 (MuJoCo real_like 와 같은 값; 2026-11-19 추가).

    목적: 학습 환경이 실물 실패 ("차고 나서 넘어짐") 를 재현하는지 22300 으로 먼저 확인한다.
    MuJoCo real_like 에서 22300 은 78 % 넘어졌다. Isaac 에서도 비슷하게 넘어져야 이 조건이 학습 신호가 된다.
      전압 11.1 V, 러너 프레임 1 스텝 + IMU 0.5 스텝 (10 ms), hold 0.6 s, 출력 필터 tau 0.02 s,
      발 마찰 정지 0.55~0.65 / 운동 0.50~0.60, 잔디 비틀림 부족 c = 0.03 (항상).
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.motor_voltage.params["volt_range"] = (11.1, 11.1)
        self.events.obs_delays.params["frame_delay_prob"] = 1.0
        self.events.obs_delays.params["imu_extra_range"] = (0.5, 0.5)
        self.events.reset_hold.params["hold_range_s"] = (0.6, 0.6)
        self.actions.JointPositionAction.filter_prob = 1.0
        self.actions.JointPositionAction.filter_tau_range = (0.02, 0.02)
        self.events.feet_material.params["static_friction_range"] = (0.55, 0.65)
        self.events.feet_material.params["dynamic_friction_range"] = (0.50, 0.60)
        self.events.turf_torsion_sample.params["coef_range"] = (0.03, 0.03)
        self.events.turf_torsion_sample.params["zero_prob"] = 0.0


# ============================================================================
# 진단용 재생 — V1-6 변경을 하나씩만 켠다 (2026-11-19). 22300 이 V1-6-Play 에서 공을 안 찬 원인 분리용.
#   Diag-ActFlat : 다리 모터를 Explicit(DCMotor) 로 바꾸되 곡선은 끔 (β = 0) → Implicit→Explicit 전환만의 영향
#   Diag-Act     : 모터 곡선 β = 1 + 전압 (A 와 같은 구동기)
#   Diag-Obs / Diag-Surf / Diag-Ctr : 관측 지연 / 발 마찰·비틀림 / hold·필터 만
# ============================================================================
def _diag(cfg, **flags):
    base = dict(act=False, obs=False, surf=False, ctr=False, rew=False, curriculum=False, dc_blend=1.0)
    base.update(flags)
    _apply_v1_6(cfg, **base)


@configclass
class RobotV16DiagActFlatEnvCfg(v15.RobotV15BPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _diag(self, act=True, dc_blend=0.0)


@configclass
class RobotV16DiagActEnvCfg(v15.RobotV15BPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _diag(self, act=True)


@configclass
class RobotV16DiagObsEnvCfg(v15.RobotV15BPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _diag(self, obs=True)


@configclass
class RobotV16DiagSurfEnvCfg(v15.RobotV15BPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _diag(self, surf=True)


@configclass
class RobotV16DiagCtrEnvCfg(v15.RobotV15BPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _diag(self, ctr=True)


# ============================================================================
# V1-6D (2026-11-19) — Explicit DC 모터를 뺀 학습 태스크.
#   진단 (22300, s = 1, eval_strength_isaac): V1-5B 킥 99 % / 넘어짐 0.2 %, Diag-ActFlat (Explicit, β = 0) 킥 0 %,
#   Diag-Obs 99 % / 1.4 %, Diag-Ctr 99 % / 0.2 %, Diag-Surf 82 % / 6.0 %.
#   → Implicit → Explicit 모터 전환만으로 22300 이 공을 못 찬다 (A/B/C 는 이 모터로 학습됐다).
#   D = 22300 과 같은 ImplicitActuator + 관측 지연 + 바닥 (마찰·비틀림) + hold·필터 + 디딤발 yaw 벌점.
#   모터 토크-속도 한계는 v1_4 의 torque_envelope 벌점 (그대로) 이 맡는다.
# ============================================================================
def _real_like(cfg, volt: bool):
    if volt:
        cfg.events.motor_voltage.params["volt_range"] = (11.1, 11.1)
    cfg.events.obs_delays.params["frame_delay_prob"] = 1.0
    cfg.events.obs_delays.params["imu_extra_range"] = (0.5, 0.5)
    cfg.events.reset_hold.params["hold_range_s"] = (0.6, 0.6)
    cfg.actions.JointPositionAction.filter_prob = 1.0
    cfg.actions.JointPositionAction.filter_tau_range = (0.02, 0.02)
    cfg.events.feet_material.params["static_friction_range"] = (0.55, 0.65)
    cfg.events.feet_material.params["dynamic_friction_range"] = (0.50, 0.60)
    cfg.events.turf_torsion_sample.params["coef_range"] = (0.03, 0.03)
    cfg.events.turf_torsion_sample.params["zero_prob"] = 0.0


@configclass
class RobotV16DEnvCfg(v15.RobotV15BEnvCfg):
    """D: 학습용 — C 에서 Explicit DC 모터만 뺀 것."""

    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6(self, act=False, obs=True, surf=True, ctr=True, rew=True)


@configclass
class RobotV16DPlayEnvCfg(v15.RobotV15BPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6(self, act=False, obs=True, surf=True, ctr=True, rew=True)


@configclass
class RobotV16DPlayRealLikeEnvCfg(RobotV16DPlayEnvCfg):
    """D 의 실물 유사 고정 조건 (전압 제외 — D 에는 전압 모델이 없다)."""

    def __post_init__(self):
        super().__post_init__()
        _real_like(self, volt=False)


# ============================================================================
# V1-6E (2026-11-19) — D 에 "안 차기 벌점" 추가, 디딤발 yaw 벌점 완화.
#   D 학습 (22300 → 22880): kick_resolved 37 → 1 %, no_kick_deadline 45 → 94 %, mean_reward 3.8 → 6.0 — 차기 포기.
#   ① no_kick: 데드라인까지 안 차면 1 회 W_NO_KICK × dt (−400 × 0.02 = −8). 찬 뒤 넘어짐 (−6) 보다 크게 →
#      "차고 가끔 넘어짐" 이 "안 참" 보다 항상 낫다. 헛발질·다툼 에피소드 (공을 찰 수 없음) 는 제외.
#   ② stance_foot_yaw −0.05 → −0.02 (D 시작 시 가장 큰 킥 관련 벌점 −0.049 였다).
# ============================================================================
W_NO_KICK = -400.0
W_STANCE_YAW_E = -0.02


def _apply_v1_6e(cfg):
    _apply_v1_6(cfg, act=False, obs=True, surf=True, ctr=True, rew=True)
    cfg.rewards.stance_foot_yaw.weight = W_STANCE_YAW_E
    cfg.rewards.no_kick = RewTerm(func=kr.no_kick_penalty, weight=W_NO_KICK)


@configclass
class RobotV16EEnvCfg(v15.RobotV15BEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6e(self)


@configclass
class RobotV16EPlayEnvCfg(v15.RobotV15BPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6e(self)


# ============================================================================
# V1-6F (2026-11-19) — 보상은 22300 (V1-5B) 과 완전히 같고 환경만 바꾼다 (관측 지연 + 바닥 + hold·필터).
#   대조 실험: V1-5B 그대로 22300 → 22550 은 kick_resolved 45 → 78 %, 안정.
#   D (−0.05 디딤발 yaw, 킥 직후 ×3) 는 시작 시 그 벌점 (−0.049) 이 킥 보상 합 (~+0.031) 보다 커서 차기를 포기했다.
#   E (−0.02 + 안 차기 −8) 는 넘어져서 안 차기 벌점을 피했다 (넘어짐 ×4). → 보상 추가 없이 환경만 먼저 확인한다.
# ============================================================================
@configclass
class RobotV16FEnvCfg(v15.RobotV15BEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6(self, act=False, obs=True, surf=True, ctr=True, rew=False)
