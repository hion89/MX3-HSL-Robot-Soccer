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


# ============================================================================
# V1-6G (2026-11-19) — F + '안 차고 끝남' 벌점. model_23500 (F) 에서 이어 학습한다.
#   F 결과: 넘어짐은 크게 줄었다 (MuJoCo real_like 78 → 1 %, 최악 89 → 11 %) 하지만 킥 100 → 65 %.
#   안 찬 시행은 공이 옆 (방위 20° 밖) 이나 멀리 있을 때 몰렸다 → 어려운 배치를 건너뛰는 법을 배웠다.
#   E 의 실패 (데드라인 벌점만 → 일찍 넘어져서 회피) 를 막으려고, 종료 원인과 상관없이 안 차고 끝나면 −4.
#   −4 는 찬 뒤 넘어짐 (−6) 보다 작다: "차고 넘어짐" 보다는 "안 참" 이 덜 나쁘지만, 둘 다 공짜는 아니다.
# ============================================================================
W_END_NO_KICK_G = -200.0  # × dt 0.02 = −4 per episode


@configclass
class RobotV16GEnvCfg(RobotV16FEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.end_no_kick = RewTerm(func=kr.ended_without_kick_penalty, weight=W_END_NO_KICK_G)


# ============================================================================
# V1-6H (2026-11-20) — F + '차야 플러스 보상' 게이트 (사용자 제안: 공차기를 전제 조건으로 곱한다).
#   F·G 에서 안 차는 에피소드도 데드라인까지 alive·gait·capture_point 등 플러스 보상을 계속 받았고,
#   차면 kick_resolved 로 일찍 끝나 그 수입을 잃었다. G 의 −4 는 이 수입보다 작았던 것으로 본다.
#   차기 전 플러스 보상에 g = 1 − α·clamp(t/ramp) 를 곱하고, 차면 g = 1. 벌점·ball_approach·킥 보상은 그대로.
#   계수 (α, ramp) 는 짧은 시험으로 고른다. 변형: H (α1, 데드라인), H2 (α1, 1.5 s), H3 (α0.5, 1.5 s).
# ============================================================================
def _apply_v1_6h(cfg, alpha: float, ramp_s: float):
    cfg.rewards.kick_gate = RewTerm(func=kr.kick_gated_positive, weight=1.0, params={"alpha": alpha, "ramp_s": ramp_s})


@configclass
class RobotV16HEnvCfg(RobotV16FEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6h(self, alpha=1.0, ramp_s=-1.0)


@configclass
class RobotV16H2EnvCfg(RobotV16FEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6h(self, alpha=1.0, ramp_s=1.5)


@configclass
class RobotV16H3EnvCfg(RobotV16FEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6h(self, alpha=0.5, ramp_s=1.5)


# ============================================================================
# V1-6I (2026-11-24) — H + 방향 배율 곱한 공 속도 보상 (+ 속도 가중치). H long model_27500 에서 이어 학습.
#   I: 가중치 2.0 → 2.5, I2: 2.0 → 3.0. 둘 다 dir σ 20°, floor 0.3. 넘어짐 벌점 (fall_after_kick −6) 은 그대로.
#   kick_gate 는 마지막 항이어야 한다 — 기존 ball_velocity 항을 바꿔 끼우므로 순서는 유지된다.
# ============================================================================
def _apply_v1_6i(cfg, weight: float, sigma_deg: float = 20.0, floor: float = 0.3):
    bv = cfg.rewards.ball_velocity
    cfg.rewards.ball_velocity = RewTerm(func=kr.ball_velocity_dir_gated, weight=weight,
                                        params={**bv.params, "dir_sigma_deg": sigma_deg, "dir_floor": floor})


@configclass
class RobotV16IEnvCfg(RobotV16HEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6i(self, weight=2.5)


@configclass
class RobotV16I2EnvCfg(RobotV16HEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6i(self, weight=3.0)


# ============================================================================
# V1-6J (2026-11-25) — I2 + 골반 roll 벌림 상한 벌점. I2 long model_final (30899) 에서 이어 학습.
#   실물에서 차는 다리 골반 roll 55~63° 직후 디딤발이 모서리로 들려 넘어졌다 (KICK_V1_6_KO.md 실물 진단).
#   계수 시험: J1 35°·−20, J2 35°·−60, J3 45°·−60, J4 25°·−40. kick_gate 는 마지막 항으로 다시 붙인다.
# ============================================================================
def _apply_v1_6j(cfg, threshold_deg: float, weight: float):
    kg = cfg.rewards.kick_gate
    delattr(cfg.rewards, "kick_gate")
    # ⚠ SceneEntityCfg 는 params 에 넣어야 Isaac Lab 이 관절 번호로 바꾼다. 함수 기본값으로 두면 전체 관절이 된다
    #   (2026-11-25 발견: J1~J6 은 이 때문에 모든 관절의 35~45° 넘는 움직임에 벌점을 줬다).
    cfg.rewards.hip_roll_excess = RewTerm(func=kr.hip_roll_excess, weight=weight,
                                          params={"threshold_rad": threshold_deg * 3.14159265 / 180.0,
                                                  "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_pelvis_roll_joint"])})
    cfg.rewards.metric_hip_roll_max = RewTerm(func=kr.metric_hip_roll_max_deg, weight=0.0,
                                              params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_pelvis_roll_joint"])})
    cfg.rewards.kick_gate = kg


@configclass
class RobotV16J1EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6j(self, 35.0, -20.0)


@configclass
class RobotV16J2EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6j(self, 35.0, -60.0)


@configclass
class RobotV16J3EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6j(self, 45.0, -60.0)


@configclass
class RobotV16J4EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6j(self, 25.0, -40.0)


# V1-6J5·J6 (2026-11-25): J1~J4 는 큰 벌점을 한 번에 넣어 킥이 무너졌다. 같은 벌점 (35°) 을 0 에서 천천히 올린다.
#   iteration 당 24 스텝 → ramp 12000 스텝 = 약 500 iteration. J5 목표 −5, J6 목표 −10.
J_RAMP_STEPS = 12000


def _apply_v1_6j_ramp(cfg, threshold_deg: float, target: float):
    _apply_v1_6j(cfg, threshold_deg, 0.0)
    cfg.curriculum.hip_roll_excess_w = CurrTerm(func=kr.ramp_reward_weight,
                                                params={"term_name": "hip_roll_excess", "target": target,
                                                        "ramp_steps": J_RAMP_STEPS})


@configclass
class RobotV16J5EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6j_ramp(self, 35.0, -5.0)


@configclass
class RobotV16J6EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6j_ramp(self, 35.0, -10.0)


# ============================================================================
# V1-6K (2026-11-25) — '공을 차되 넘어지지 않게' 후보 6개. 모두 I2 long model_final (30899) 에서 700 iteration.
#   J1~J6 (골반 roll 벌림 벌점 35~45°) 는 벌림을 줄이지 못하고 킥만 포기했다.
#   ① 각도 여유: K1 50°·−10, K2 55°·−30 (벌점 램프)
#   ② 디딤발 쪽 외력: K3 (0~12 N, 에피소드 절반, 0.5 s, 램프). MuJoCo 에서 30899 는 8 N, 27500 은 4 N 부터 넘어진다.
#   ③ 몸 roll 벌점: K4 (10° 넘는 만큼 제곱, −20, 램프). 넘어짐 종료는 기존 기울기 0.6 rad (34°) 그대로.
#   조합: K5 = ② + ③, K6 = ② + ③ + ① (50°·−10).
#   새 항은 모두 0 에서 J_RAMP_STEPS (약 500 iteration) 동안 키운다. kick_gate 는 마지막 항으로 다시 붙인다.
# ============================================================================
K_PUSH_N = (0.0, 12.0)
K_PUSH_P = 0.5
K_PUSH_S = 0.5
K_ROLL_DEG, K_ROLL_W = 10.0, -20.0


def _k_reattach_gate(cfg):
    kg = cfg.rewards.kick_gate
    delattr(cfg.rewards, "kick_gate")
    cfg.rewards.kick_gate = kg


def _apply_k_push(cfg, force_range=K_PUSH_N, probability=K_PUSH_P, ramp_steps=None):
    ramp_steps = J_RAMP_STEPS if ramp_steps is None else ramp_steps
    base = lambda: SceneEntityCfg("robot", body_names=v14.MX_BASE_LINK)
    cfg.events.kick_support_push_reset = EventTerm(
        func=kr.kick_support_push_reset, mode="reset",
        params={"force_range": force_range, "probability": probability, "asset_cfg": base()})
    cfg.events.kick_support_push = EventTerm(
        func=kr.kick_support_push, mode="interval", interval_range_s=(0.02, 0.02),
        params={"duration_s": K_PUSH_S, "trig_deg": 40.0, "ramp_steps": ramp_steps, "asset_cfg": base(),
                "roll_cfg": SceneEntityCfg("robot", joint_names=["r_pelvis_roll_joint", "l_pelvis_roll_joint"],
                                           preserve_order=True),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(_FEET), preserve_order=True)})
    cfg.curriculum.kick_push_max_N = CurrTerm(func=kr.kick_push_scale, params={"max_N": force_range[1]})
    cfg.rewards.metric_kick_push_N = RewTerm(func=kr.metric_kick_push_N, weight=0.0)
    _k_reattach_gate(cfg)


def _apply_k_roll(cfg, target=K_ROLL_W, ramp_steps=None):
    ramp_steps = J_RAMP_STEPS if ramp_steps is None else ramp_steps
    cfg.rewards.base_roll_excess = RewTerm(func=kr.base_roll_excess, weight=0.0,
                                           params={"threshold_rad": K_ROLL_DEG * 3.14159265 / 180.0})
    cfg.rewards.metric_base_roll = RewTerm(func=kr.metric_base_roll_deg, weight=0.0)
    cfg.curriculum.base_roll_excess_w = CurrTerm(func=kr.ramp_reward_weight,
                                                 params={"term_name": "base_roll_excess", "target": target,
                                                         "ramp_steps": ramp_steps})
    _k_reattach_gate(cfg)


@configclass
class RobotV16K1EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6j_ramp(self, 50.0, -10.0)


@configclass
class RobotV16K2EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_v1_6j_ramp(self, 55.0, -30.0)


@configclass
class RobotV16K3EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_k_push(self)


@configclass
class RobotV16K4EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_k_roll(self)


@configclass
class RobotV16K5EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_k_push(self)
        _apply_k_roll(self)


@configclass
class RobotV16K6EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_k_push(self)
        _apply_k_roll(self)
        _apply_v1_6j_ramp(self, 50.0, -10.0)


# ============================================================================
# V1-6L (2026-11-26) — K5 단기 31599 에서 외력을 낮춰 짧게 이어 학습 (조기 종료용, 100 iteration 마다 MuJoCo 게이트).
#   K5 장기 학습 (12 N 그대로 +3000) 은 Isaac 에서만 좋아지고 MuJoCo 넘어짐이 9 % → 21~62 % 로 늘었다.
#   이미 외력·벌점에 적응한 모델에서 시작하므로 램프 없이 처음부터 목표값이다.
#   L1: 0~8 N·p0.5·roll −20 / L2: 0~6 N·p0.5·roll −20 / L3: 0~8 N·p0.5·roll −10
# ============================================================================
def _apply_l(cfg, max_n, prob, roll_w):
    _apply_k_push(cfg, (0.0, max_n), prob, ramp_steps=1)
    _apply_k_roll(cfg, roll_w, ramp_steps=1)


@configclass
class RobotV16L1EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_l(self, 8.0, 0.5, -20.0)


@configclass
class RobotV16L2EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_l(self, 6.0, 0.5, -20.0)


@configclass
class RobotV16L3EnvCfg(RobotV16I2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_l(self, 8.0, 0.5, -10.0)


# ============================================================================
# V1-6M (2026-11-26) — 실물 디딤발 토크 부족 반영. L3 model_31800 (kick_push_31800, 실물 설치) 에서 이어 학습.
#   실물 1회차: 1.88 s 디딤발 골반 pitch·발목 pitch·골반 roll PWM 100 % → 1.9 s 넘어짐.
#   MuJoCo 보정: 다리 토크 상한 3 N·m·스톨 ×0.7·10.3 V 에서 L3·30899·27500 모두 1.8~2.2 s 에 넘어짐 (실물과 같다).
#   M1   : 다리 관절 effort limit 을 매 스텝 속도·전압 곡선으로 (cap 3~5 N·m, 스톨 ×0.7~1.0, 10.3~12 V). 300 iteration 램프.
#   M1P  : M1 + 상한 80 % 넘는 토크 제곱 벌점 (−2, 램프).
#   M1PH : M1P + 더 약한 모터 (cap 2.8~4 N·m, 스톨 ×0.6~0.9).
#   L3 의 외력 (0~8 N) 과 몸 roll 벌점 (−10) 은 그대로 둔다.
# ============================================================================
M_RAMP_STEPS = 7200   # 약 300 iteration


def _apply_m1(cfg, cap_range=(3.0, 5.0), stall_range=(0.7, 1.0), volt_range=(10.3, 12.0)):
    legs = lambda: SceneEntityCfg("robot", joint_names=v14.MX_LEG_JOINTS, preserve_order=True)
    cfg.events.dc_effort_limit_reset = EventTerm(
        func=kr.dc_effort_limit_reset, mode="reset",
        params={"volt_range": volt_range, "stall_scale_range": stall_range, "cap_range": cap_range,
                "asset_cfg": legs()})
    cfg.events.dc_effort_limit = EventTerm(
        func=kr.dc_effort_limit, mode="interval", interval_range_s=(0.02, 0.02),
        params={"ramp_steps": M_RAMP_STEPS, "asset_cfg": legs()})
    cfg.curriculum.dc_limit = CurrTerm(func=kr.dc_limit_stats)


def _apply_m3(cfg, weight=-2.0):
    cfg.rewards.torque_near_limit = RewTerm(func=kr.torque_near_limit, weight=0.0,
                                            params={"ratio": 0.8, "asset_cfg": SceneEntityCfg(
                                                "robot", joint_names=v14.MX_LEG_JOINTS, preserve_order=True)})
    cfg.curriculum.torque_near_limit_w = CurrTerm(func=kr.ramp_reward_weight,
                                                  params={"term_name": "torque_near_limit", "target": weight,
                                                          "ramp_steps": M_RAMP_STEPS})
    _k_reattach_gate(cfg)


@configclass
class RobotV16M1EnvCfg(RobotV16L3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_m1(self)


@configclass
class RobotV16M1PEnvCfg(RobotV16L3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_m1(self)
        _apply_m3(self)


@configclass
class RobotV16M1PHEnvCfg(RobotV16L3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_m1(self, cap_range=(2.8, 4.0), stall_range=(0.6, 0.9))
        _apply_m3(self)


# ============================================================================
# 실측 질량 (2026-11-26) — 로봇 실측 6.5 kg, 모델 5.7 kg. 지금까지 학습은 몸통 −0.3~+0.5 kg (5.4~6.2 kg) 이었다.
#   몸통 추가 질량을 +0.5~+1.1 kg (6.2~6.8 kg, 중심 6.5 kg) 으로 바꾼다. MuJoCo 실물 조건도 몸통 +0.8 kg 이다.
#   V1-6N1 = L3 (kick_push_31800, 외력 0~8 N·몸 roll −10) + 실측 질량. L3 model_31800 에서 이어 학습.
#   평가: V1-6D-Play-RealLike-Mass65 (몸통 +0.8 kg 고정).
# ============================================================================
REAL_MASS_ADD = (0.5, 1.1)


def _apply_real_mass(cfg, add_range=REAL_MASS_ADD):
    if getattr(cfg.events, "add_base_mass", None) is None:
        cfg.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=v14.MX_BASE_LINK),
                    "mass_distribution_params": add_range, "operation": "add"})
    else:
        cfg.events.add_base_mass.params["mass_distribution_params"] = add_range


@configclass
class RobotV16N1EnvCfg(RobotV16L3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_real_mass(self)


@configclass
class RobotV16DPlayRealLikeMass65EnvCfg(RobotV16DPlayRealLikeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_real_mass(self, (0.8, 0.8))


# ============================================================================
# V1-6Q (2026-11-26) — 발-바닥 마찰 상한 확대. N1 (L3 + 실측 질량 6.5 kg) 에 발 마찰 범위만 넓힌다.
#   지금까지 발 마찰: 정지 0.30~1.30, 운동 0.25~1.05 (바닥 1.0 곱하기 → 실효값 = 발 값).
#   Q15 / Q17 / Q20: 정지 상한 1.5 / 1.7 / 2.0, 운동 상한은 같은 비율 (×0.81). 하한은 그대로.
#   실물 세기 1.0 시험 (11-26 17:11): 킥은 마쳤지만 3.6 s 에 넘어짐. L3 model_31800 에서 이어 학습.
# ============================================================================
def _apply_feet_friction(cfg, static_max: float):
    p = cfg.events.feet_material.params
    p["static_friction_range"] = (FEET_STATIC_FR[0], static_max)
    p["dynamic_friction_range"] = (FEET_DYNAMIC_FR[0], round(static_max * FEET_DYNAMIC_FR[1] / FEET_STATIC_FR[1], 3))


@configclass
class RobotV16Q15EnvCfg(RobotV16N1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_feet_friction(self, 1.5)


@configclass
class RobotV16Q17EnvCfg(RobotV16N1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_feet_friction(self, 1.7)


@configclass
class RobotV16Q20EnvCfg(RobotV16N1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_feet_friction(self, 2.0)
