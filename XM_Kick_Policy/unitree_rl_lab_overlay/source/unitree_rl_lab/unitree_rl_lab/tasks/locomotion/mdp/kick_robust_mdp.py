# kick_robust_mdp.py
# ============================================================================
# walk_kick v1_6 — "킥 후 디딤발이 버티지 못하고 돌아/밀려 무너짐" 을 여러 원인으로 나눠 학습에 넣는 항 (2026-11-16)
# ----------------------------------------------------------------------------
# 근거: sim2real_evidence/kick_robust_20261116/ (MuJoCo sim2sim, 22300, 서서 시작, 조건당 200~300 회)
#   학습과 같은 관측 순서         3 %   → 실물 러너 관측 순서 (센서 1 스텝 묵음) 13.5 %
#   XM540 속도-토크 곡선 12 / 11.1 / 10.6 V   25 / 35 / 36 %   (Isaac 학습은 ω 무관 5 N·m)
#   IMU 지연 10 / 20 / 40 ms     20 / 42 / 82 %                 (학습엔 IMU 지연 없음)
#   미끄럼 마찰 0.6 / 0.4         22 / 66 %
#   비틀림 마찰을 낮추면 디딤발이 30~45° 돌지만 MuJoCo 에선 오히려 덜 넘어진다 → 마찰 하나로 설명 안 됨.
#   실물 기록 (kick_balance_20261110): 0.31 s 에 발목 roll 목표 −31.9° vs 실측 −0.7°, 모터 전압 10.6 V.
#
# 이 모듈이 제공하는 것 (기존 mdp 파일은 건드리지 않는다):
#   1. VoltageDCMotor(Cfg)          다리 구동기를 DC 속도-토크 곡선 + 전압 배율로. 평평한 5 N·m 와 섞는 비율 β (커리큘럼)
#   2. randomize_motor_voltage      에피소드마다 env 별 전압 (V/12 배율)
#   3. dc_motor_blend_curriculum    넘어짐 EMA 가 낮으면 β 를 올린다 (22300 에서 바로 β=1 이면 넘어짐이 커서)
#   4. DelayedObs                   자이로·중력·관절 관측을 env 별 (소수) 스텝만큼 늦게 — 실물 러너 순서 + IMU 지연
#   5. sample_obs_delays            위 지연을 에피소드마다 뽑는다
#   6. sample_turf_torsion / apply_turf_torsion_deficit
#                                   PhysX 패치 마찰은 디딤발 비틀림을 꽉 잡는다 (원 작성자 실측: torsional_patch_radius·
#                                   friction_correlation_distance 를 바꿔도 Isaac 헛발질 0~0.4 %) → 접지한 발에
#                                   "잔디가 못 잡는 만큼" 의 yaw 토크 (c·Fn, 몸통 회전 방향) 를 외력으로 준다.
#   7. set_reset_hold_random        킥 시작 hold 를 env 별 0.3~0.6 s (실물 kick_start_hold 0.6)
#   8. FilteredProfileVelocityJointPositionAction
#                                   실물 러너의 행동 저역 필터 (alpha = exp(−dt/τ)) 를 env 별로 켠다. last_action 관측은 raw.
#   9. stance_foot_yaw_rate_penalty 접지한 발의 yaw 각속도 제곱 (킥 창에서 강화) + 메트릭
#
# ※ Isaac Lab 2.3.x API 기준으로 썼다. 로컬 PC 에는 Isaac 이 없어 순수 텐서 부분만 단위 시험했다
#   (sim2real_evidence/kick_robust_20261116/test_kick_robust_math.py). 서버 첫 실행에서 시작 로그의
#   "[kick_robust]" 줄과 TensorBoard Curriculum/dc_blend 를 반드시 확인할 것.
# ============================================================================
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from .kick_clock import kick_clock_s
from .kick_handoff_mdp import ProfileVelocityJointPositionAction, ProfileVelocityJointPositionActionCfg

try:  # Isaac Lab 2.x
    from isaaclab.utils.math import quat_apply_inverse as _quat_apply_inverse
except ImportError:  # older name
    from isaaclab.utils.math import quat_rotate_inverse as _quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_EPS = 1e-6


# ============================================================================
# 순수 텐서 함수 (Isaac 없이 단위 시험한다)
# ============================================================================
def dc_clip_effort(effort, qd, sat, v_noload, cap, volt_scale, beta):
    """DC 모터 사분면 규칙 + 전압 배율 + 평평한 상한과의 혼합.

    구동 방향 (토크와 속도 같은 부호) 은 sat·k·(1 − qd/(v_nl·k)) 로 줄고, 제동 방향은 cap 까지 허용한다
    (Isaac Lab DCMotor._clip_effort 와 같은 식, k = V/12).
    beta = 0 → 기존 평평한 ±cap (ImplicitActuator effort_limit_sim 과 같은 결과), 1 → DC 곡선.
    effort, qd: (N, J). sat, v_noload: float. cap: float 또는 (N, J). volt_scale: (N, 1). beta: float 또는 (N, 1).
    """
    cap = torch.as_tensor(cap, dtype=effort.dtype, device=effort.device)
    s = sat * volt_scale
    vnl = torch.clamp(torch.as_tensor(v_noload, dtype=effort.dtype, device=effort.device) * volt_scale, min=_EPS)
    top = torch.minimum(s * (1.0 - qd / vnl), cap)
    bot = torch.maximum(s * (-1.0 - qd / vnl), -cap)
    dc = torch.minimum(torch.maximum(effort, bot), top)
    flat = torch.minimum(torch.maximum(effort, -cap), cap)
    return flat + beta * (dc - flat)


def interp_delayed(buf: torch.Tensor, delay_steps: torch.Tensor) -> torch.Tensor:
    """buf: (N, K, D) index 0 = 최신. delay_steps: (N,) 소수 스텝 (0 ≤ d ≤ K−1). 선형 보간."""
    k = buf.shape[1]
    d = torch.clamp(delay_steps, 0.0, float(k - 1))
    i0 = torch.floor(d).long()
    i1 = torch.clamp(i0 + 1, max=k - 1)
    f = (d - i0.float()).unsqueeze(-1)
    ar = torch.arange(buf.shape[0], device=buf.device)
    return (1.0 - f) * buf[ar, i0] + f * buf[ar, i1]


def torsion_deficit_torque_z(fn: torch.Tensor, coef: torch.Tensor, base_wz: torch.Tensor, wz_ref: float,
                             fn_threshold: float) -> torch.Tensor:
    """접지한 발마다 world z 토크 [N·m]. fn: (N, F) 수직 반력, coef: (N,) [m], base_wz: (N,) [rad/s].

    τ = c · Fn · tanh(ω_z / ω_ref) — 몸통이 도는 방향으로, 잔디가 비틀림을 못 잡는 몫만큼 발을 더 돌린다.
    마찰 토크 상한 (μ·Fn·r) 이 Fn 에 비례하므로 같은 꼴로 둔다. 접지하지 않은 발은 0.
    """
    on = (fn > fn_threshold).float()
    s = torch.tanh(base_wz / max(wz_ref, _EPS)).unsqueeze(-1)
    return coef.unsqueeze(-1) * torch.clamp(fn, min=0.0) * s * on


def lowpass_alpha(dt: float, tau: torch.Tensor) -> torch.Tensor:
    """러너와 같은 alpha = exp(−dt/τ). τ ≤ 0 이면 0 (필터 끔)."""
    return torch.where(tau > 0, torch.exp(-dt / torch.clamp(tau, min=_EPS)), torch.zeros_like(tau))


# ============================================================================
# 1. 전압 배율 DC 모터
# ============================================================================
class VoltageDCMotor(DCMotor):
    """DCMotor + env 별 전압 배율 (volt_scale) + 평평한 상한과의 혼합 β (dc_blend).

    _clip_effort 를 직접 다시 쓴다 (부모의 _vel_at_effort_lim 같은 내부 이름에 기대지 않으려고).
    """

    cfg: "VoltageDCMotorCfg"

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        n = self.computed_effort.shape[0]
        dev = self.computed_effort.device
        self.volt_scale = torch.ones(n, 1, device=dev)
        self.dc_blend = float(cfg.dc_blend_init)
        self._qd = torch.zeros_like(self.computed_effort)
        print(f"[kick_robust] VoltageDCMotor {cfg.joint_names_expr}: stall {cfg.saturation_effort} N·m @12V, "
              f"no-load {cfg.velocity_limit} rad/s, cap {cfg.effort_limit} N·m, dc_blend {self.dc_blend}")

    def compute(self, control_action, joint_pos, joint_vel):
        self._qd[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        return dc_clip_effort(effort, self._qd, float(self.cfg.saturation_effort), self.velocity_limit,
                              self.effort_limit, self.volt_scale, self.dc_blend)


@configclass
class VoltageDCMotorCfg(DCMotorCfg):
    class_type: type = VoltageDCMotor
    dc_blend_init: float = 1.0
    """시작 혼합 비율. 커리큘럼을 쓰면 dc_motor_blend_curriculum 이 덮어쓴다."""


def _voltage_motors(env):
    robot = env.scene["robot"]
    return [a for a in robot.actuators.values() if isinstance(a, VoltageDCMotor)]


# ============================================================================
# 2. 전압 무작위 (reset)
# ============================================================================
def randomize_motor_voltage(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, volt_range: tuple[float, float],
                            nominal: float = 12.0):
    """에피소드마다 env 별 배터리 전압. 실물 기록: 대기 11.9~12.2 V, 킥 중 모터별 최저 10.6 V."""
    motors = _voltage_motors(env)
    if not motors:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    v = torch.empty(len(env_ids), 1, device=env.device).uniform_(*volt_range)
    if not hasattr(env, "kick_motor_volt"):
        env.kick_motor_volt = torch.full((env.num_envs,), nominal, device=env.device)
    env.kick_motor_volt[env_ids] = v.squeeze(-1)
    for m in motors:
        m.volt_scale[env_ids] = v / nominal


# ============================================================================
# 3. DC 혼합 커리큘럼
# ============================================================================
def dc_motor_blend_curriculum(env: "ManagerBasedRLEnv", env_ids, start: float = 0.25, step: float = 0.25,
                              fall_threshold: float = 0.12, min_episodes: int = 20000, ema_window: float = 20000.0,
                              fall_prefix: str = "fall_"):
    """끝난 에피소드의 넘어짐 비율 EMA 가 fall_threshold 아래면 β 를 step 만큼 올린다 (1.0 까지).

    TensorBoard: Curriculum/dc_blend. 넘어짐은 termination 항 중 이름이 fall_ 로 시작하는 것 (v1_5 fall_by_type).
    """
    motors = _voltage_motors(env)
    if not motors:
        return 0.0
    if not hasattr(env, "_dc_cur"):
        env._dc_cur = dict(beta=float(start), ema=0.0, count=0)
        for m in motors:
            m.dc_blend = float(start)
        print(f"[kick_robust] dc_blend 커리큘럼 시작 β={start}")
    st = env._dc_cur
    if env_ids is None or len(env_ids) == 0 or env.common_step_counter == 0:
        return st["beta"]
    tm = env.termination_manager
    fell = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    for name in tm.active_terms:
        if name.startswith(fall_prefix):
            fell |= tm.get_term(name)[env_ids]
    k = len(env_ids)
    rate = fell.float().mean().item()
    w = ema_window / (ema_window + k)
    st["ema"] = w * st["ema"] + (1.0 - w) * rate if st["count"] else rate
    st["count"] += k
    if st["count"] >= min_episodes and st["ema"] < fall_threshold and st["beta"] < 1.0 - 1e-6:
        prev = st["beta"]
        st["beta"] = min(1.0, st["beta"] + step)
        st["count"] = 0
        for m in motors:
            m.dc_blend = st["beta"]
        print(f"[kick_robust] dc_blend {prev:.2f} → {st['beta']:.2f} (넘어짐 EMA {st['ema']:.3f})", flush=True)
    return st["beta"]


# ============================================================================
# 4. 관측 지연
# ============================================================================
_OBS_KINDS = ("imu", "joint")


def sample_obs_delays(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, frame_delay_prob: float,
                      frame_delay_steps: float, imu_extra_range: tuple[float, float]):
    """에피소드마다 관측 지연 [정책 스텝, 20 ms 단위, 소수 허용].

    frame: 실물 킥 러너는 history.push(one_frame()) 를 틱 끝에 해서 다음 추론이 한 틱 전 센서를 본다
           (actor_onnx_walkkick_v1_runner.py control_tick). frame_delay_prob 비율의 에피소드에 frame_delay_steps.
    imu  : frame 지연 + IMU 필터·통신 지연 U(imu_extra_range).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "kick_obs_delay"):
        env.kick_obs_delay = {k: torch.zeros(env.num_envs, device=env.device) for k in _OBS_KINDS}
    n = len(env_ids)
    frame = (torch.rand(n, device=env.device) < frame_delay_prob).float() * frame_delay_steps
    extra = torch.empty(n, device=env.device).uniform_(*imu_extra_range)
    env.kick_obs_delay["joint"][env_ids] = frame
    env.kick_obs_delay["imu"][env_ids] = frame + extra


class DelayedObs(ManagerTermBase):
    """안쪽 관측 함수 값을 env 별 지연만큼 늦춰 돌려준다 (ObsTerm 의 func 로 쓴다).

    params: inner ("ang_vel" | "gravity" | "joint_pos" | "joint_vel"), kind ("imu" | "joint"),
            max_delay_steps, asset_cfg. scale·noise 는 ObsTerm 에서 그대로 (지연된 값에 노이즈).
    같은 스텝에 여러 번 불려도 한 번만 쌓는다 (env.common_step_counter 로 판정). 리셋된 env 는 현재 값으로 채운다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.kind = p.get("kind", "imu")
        self.k = int(p.get("max_delay_steps", 3)) + 2
        self.buf = None
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_step = -1
        inner = p["inner"]
        from . import kick_stability_mdp as ks
        from isaaclab.envs import mdp as il_mdp
        self._inner = {
            "ang_vel": lambda e, a: ks.base_ang_vel_biased(e, a),
            "gravity": lambda e, a: ks.projected_gravity_biased(e, a),
            "joint_pos": lambda e, a: il_mdp.joint_pos_rel(e, a),
            "joint_vel": lambda e, a: il_mdp.joint_vel_rel(e, a),
        }[inner]

    def reset(self, env_ids=None):
        if env_ids is None:
            self.fresh[:] = True
        else:
            self.fresh[env_ids] = True

    def __call__(self, env, inner: str, kind: str = "imu", max_delay_steps: int = 3,
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        val = self._inner(env, asset_cfg)
        step = int(getattr(env, "common_step_counter", 0))
        if self.buf is None:
            self.buf = val.unsqueeze(1).repeat(1, self.k, 1).clone()
            self.fresh[:] = False
            self.last_step = step
        elif step != self.last_step:
            self.buf = torch.roll(self.buf, shifts=1, dims=1)
            self.buf[:, 0] = val
            self.last_step = step
        else:
            self.buf[:, 0] = val
        if self.fresh.any():
            ids = self.fresh.nonzero(as_tuple=False).squeeze(-1)
            self.buf[ids] = val[ids].unsqueeze(1)
            self.fresh[ids] = False
        d = getattr(env, "kick_obs_delay", None)
        if d is None:
            return val
        return interp_delayed(self.buf, d[kind])


# ============================================================================
# 6. 잔디 비틀림 부족 토크
# ============================================================================
def sample_turf_torsion(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, coef_range: tuple[float, float],
                        zero_prob: float):
    """에피소드마다 c [m] — 0 이면 단단한 바닥 (Isaac 기본). zero_prob 비율은 0."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "kick_turf_c"):
        env.kick_turf_c = torch.zeros(env.num_envs, device=env.device)
    n = len(env_ids)
    c = torch.empty(n, device=env.device).uniform_(*coef_range)
    c = torch.where(torch.rand(n, device=env.device) < zero_prob, torch.zeros_like(c), c)
    env.kick_turf_c[env_ids] = c


def apply_turf_torsion_deficit(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, wz_ref: float = 0.5,
                               fn_threshold: float = 5.0,
                               sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """interval 이벤트 (매 정책 스텝). 접지한 발에 world z 토크를 몸체 좌표로 바꿔 준다. 힘은 0."""
    if not hasattr(env, "kick_turf_c"):
        return
    robot = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    fids = asset_cfg.body_ids
    fn = sensor.data.net_forces_w[env_ids][:, sensor_cfg.body_ids, 2]          # (n, F)
    wz = robot.data.root_ang_vel_w[env_ids, 2]
    tz = torsion_deficit_torque_z(fn, env.kick_turf_c[env_ids], wz, wz_ref, fn_threshold)   # (n, F)
    tw = torch.zeros(len(env_ids), tz.shape[1], 3, device=env.device)
    tw[..., 2] = tz
    q = robot.data.body_quat_w[env_ids][:, fids]                              # (n, F, 4)
    tb = _quat_apply_inverse(q.reshape(-1, 4), tw.reshape(-1, 3)).reshape(tw.shape)
    # Isaac Lab ≤2.2 의 set_external_force_and_torque 는 넘긴 값이 전부 0 이면 has_external_wrench=False 로
    # 바꿔 몸통 외란 (base_external_force_torque) 까지 꺼 버린다 → 0 이 되지 않게 아주 작은 값을 더한다.
    tb[..., 2] += 1e-6
    robot.set_external_force_and_torque(torch.zeros_like(tb), tb, body_ids=fids, env_ids=env_ids)
    if not hasattr(env, "kick_turf_tz"):
        env.kick_turf_tz = torch.zeros(env.num_envs, device=env.device)
    env.kick_turf_tz[env_ids] = tz.abs().sum(dim=-1)


# ============================================================================
# 7. hold 무작위
# ============================================================================
def set_reset_hold_random(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, hold_range_s: tuple[float, float]):
    """reset_hold.set_reset_hold_steps 의 env 별 버전. 실물 kick_start_hold (= 러너 resume_hold_seconds) 0.3~0.6 s."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, "reset_hold_steps"):
        env.reset_hold_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    lo, hi = hold_range_s
    s = torch.empty(len(env_ids), device=env.device).uniform_(lo, hi)
    env.reset_hold_steps[env_ids] = torch.round(s / env.step_dt).long()


# ============================================================================
# 8. 러너 행동 필터
# ============================================================================
class FilteredProfileVelocityJointPositionAction(ProfileVelocityJointPositionAction):
    """정책 출력 → (env 별) 저역 필터 → hold·지연·Profile Velocity. 러너와 같은 순서.

    러너: a_used = clip(a_raw); filt = α·filt + (1−α)·a_used; q = default + 0.25·filt (hold 중에도 filt 갱신);
          관측 last_action = a_used (raw). Isaac 의 last_action 관측은 action_manager.action (raw) 이라 그대로 맞다.
    """

    cfg: "FilteredProfileVelocityJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = env.num_envs
        self._filt = torch.zeros(n, self.action_dim, device=env.device)
        self._alpha = torch.zeros(n, 1, device=env.device)
        self._sample_tau(torch.arange(n, device=env.device))

    def _sample_tau(self, env_ids):
        on = torch.rand(len(env_ids), device=self._env.device) < self.cfg.filter_prob
        tau = torch.empty(len(env_ids), device=self._env.device).uniform_(*self.cfg.filter_tau_range)
        tau = torch.where(on, tau, torch.zeros_like(tau))
        self._alpha[env_ids, 0] = lowpass_alpha(self._env.step_dt, tau)

    def process_actions(self, actions: torch.Tensor):
        self._filt = self._alpha * self._filt + (1.0 - self._alpha) * actions
        super().process_actions(self._filt)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self._env.device)
        if len(env_ids) == 0:
            return
        self._filt[env_ids] = 0.0
        self._sample_tau(env_ids)


@configclass
class FilteredProfileVelocityJointPositionActionCfg(ProfileVelocityJointPositionActionCfg):
    class_type: type = FilteredProfileVelocityJointPositionAction
    filter_prob: float = 0.5
    """필터를 켠 에피소드 비율 (실물 control.launch 는 켬, 시뮬 launch 는 끔)."""
    filter_tau_range: tuple[float, float] = (0.015, 0.03)
    """[s] 실물 tau_action_filter 0.02."""


# ============================================================================
# 9. 보상·메트릭
# ============================================================================
def stance_foot_yaw_rate_penalty(env: "ManagerBasedRLEnv", post_kick_window_s: float = 1.0, boost: float = 3.0,
                                 force_threshold: float = 1.0,
                                 sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Σ_접지 발 ω_z² (world) — 음수 weight. 디딤발이 도는 것 자체를 벌한다 (support_slip 은 병진만 본다)."""
    from . import kick_stability_mdp as ks
    robot = env.scene[asset_cfg.name]
    contact = ks._foot_contact(env, sensor_cfg, force_threshold).float()
    wz = robot.data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    return (torch.square(wz) * contact).sum(dim=-1) * ks._boost(env, post_kick_window_s, boost)


def metric_turf_torque(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """이번 스텝 잔디 비틀림 부족 토크 합 [N·m] (weight 0, 로그용)."""
    t = getattr(env, "kick_turf_tz", None)
    return t if t is not None else torch.zeros(env.num_envs, device=env.device)


def metric_motor_volt(env: "ManagerBasedRLEnv") -> torch.Tensor:
    v = getattr(env, "kick_motor_volt", None)
    return v if v is not None else torch.full((env.num_envs,), 12.0, device=env.device)


def metric_imu_delay_ms(env: "ManagerBasedRLEnv") -> torch.Tensor:
    d = getattr(env, "kick_obs_delay", None)
    if d is None:
        return torch.zeros(env.num_envs, device=env.device)
    return d["imu"] * env.step_dt * 1000.0


def metric_post_kick_yaw_rate(env: "ManagerBasedRLEnv", window_s: float = 1.0) -> torch.Tensor:
    """킥 후 window_s 안의 |몸통 ω_z| [rad/s], 그 밖은 0 (로그용)."""
    robot = env.scene["robot"]
    if not hasattr(env, "kick_time_s"):
        return torch.zeros(env.num_envs, device=env.device)
    el = kick_clock_s(env) - env.kick_time_s
    win = (env.kick_time_s >= 0) & (el <= window_s)
    return torch.where(win, robot.data.root_ang_vel_w[:, 2].abs(), torch.zeros(env.num_envs, device=env.device))


# ============================================================================
# 10. 안 차기 벌점 (V1-6E, 2026-11-19)
# ============================================================================
def no_kick_penalty(env: "ManagerBasedRLEnv", exclude: tuple[str, ...] = ("whiff", "duel"), fallback_s: float = 5.0):
    """데드라인까지 공을 안 찼으면 그 스텝에 1.0 (음수 weight). 공을 찰 수 없는 에피소드 (헛발질·다툼) 는 뺀다.

    V1-6D 에서 정책이 차기를 포기했다 (22300 → 22880: kick_resolved 37 → 1 %, no_kick_deadline 45 → 94 %,
    mean_reward 3.8 → 6.0). 기존 보상에는 안 차는 것 자체의 비용이 없고, 찬 뒤 넘어짐만 −6 이라 안 차는 쪽이 유리했다.
    """
    from . import kick_handoff_mdp as kh
    from . import kick_mdp

    out = kick_mdp.no_kick_timeout_termination(env, fallback_s=fallback_s)
    ep = getattr(env, "kick_episode_type", None)
    from_bank = getattr(env, "kick_from_bank", None)
    if ep is not None and from_bank is not None and exclude:
        skip = torch.zeros_like(out)
        for name in exclude:
            skip |= from_bank & (ep == kh.EP_NAMES[name])
        out = out & ~skip
    return out.float()


def ended_without_kick_penalty(env: "ManagerBasedRLEnv", exclude: tuple[str, ...] = ("whiff", "duel")):
    """에피소드가 공을 안 찬 채 끝나면 (데드라인·넘어짐·시간초과 무엇이든) 그 스텝에 1.0 (음수 weight).

    V1-6E 는 데드라인에만 벌점을 줘서, 정책이 데드라인 전에 넘어져 벌점을 피했다 (넘어짐 4 → 27 %).
    V1-6F 는 벌점이 없어 공이 옆·멀리 있으면 차기를 건너뛰었다 (MuJoCo 킥 65 %, 방위 20° 밖 56~61 %).
    종료 원인과 상관없이 '안 차고 끝남' 에 같은 값을 매기면 일찍 넘어지는 탈출구가 없다.
    리워드는 종료 판정 뒤·리셋 전에 계산되므로 termination_manager.dones 와 kick_time_s 가 이 스텝 값이다.
    """
    from . import kick_handoff_mdp as kh

    out = env.termination_manager.dones & (env.kick_time_s < 0)
    ep = getattr(env, "kick_episode_type", None)
    from_bank = getattr(env, "kick_from_bank", None)
    if ep is not None and from_bank is not None and exclude:
        skip = torch.zeros_like(out)
        for name in exclude:
            skip |= from_bank & (ep == kh.EP_NAMES[name])
        out = out & ~skip
    return out.float()


GATED_POSITIVE_TERMS = ("alive", "gait", "stop_posture", "joint_pose_to_default", "capture_point", "track_lin_vel_x")


def kick_gated_positive(
    env: "ManagerBasedRLEnv",
    terms: tuple[str, ...] = GATED_POSITIVE_TERMS,
    alpha: float = 1.0,
    ramp_s: float = -1.0,
    exclude: tuple[str, ...] = ("whiff", "duel"),
):
    """'공을 차야 생존·자세 보상을 받는다' 를 곱으로 만든다 (V1-6H). 반드시 보상 목록의 마지막 항이어야 한다.

    F·G 에서 안 차는 에피소드는 데드라인까지 alive·gait·capture_point 같은 플러스 보상을 계속 받았고,
    차면 kick_resolved 로 에피소드가 일찍 끝나 그 수입을 잃었다 → 어려운 배치는 건너뛰는 게 이득.
    여기서는 차기 전의 그 플러스 보상에 배율 g 를 곱한 효과를 낸다:
        g = 1                                   (찼거나, 헛발질·다툼 에피소드)
        g = 1 − alpha · clamp(t / ramp, 0, 1)    (차기 전, ramp_s < 0 이면 ramp = 에피소드 데드라인)
    RewardManager 는 항을 순서대로 계산하고 _step_reward 에 weight 곱한 값을 넣으므로, 마지막 항이 앞 항들의
    이번 스텝 값을 읽어 −(1−g)·Σmax(항, 0) 을 돌려준다 (weight 1.0). 음수 값은 깎지 않는다 (벌점을 줄이면 거꾸로 된다).
    """
    from . import kick_handoff_mdp as kh
    from . import kick_mdp

    rm = env.reward_manager
    idx = getattr(env, "_kick_gate_idx", None)
    if idx is None:
        names = rm._term_names
        me = names.index("kick_gate")
        idx = [names.index(t) for t in terms if t in names]
        assert all(i < me for i in idx), "kick_gate 는 보상 목록의 마지막에 있어야 한다"
        env._kick_gate_idx = idx
    kick_mdp._ensure_kick_state(env)
    t = kick_clock_s(env)
    ramp = env.kick_deadline_s if (ramp_s < 0 and hasattr(env, "kick_deadline_s")) else max(ramp_s, 1e-3)
    if ramp_s == 0:
        frac = torch.ones_like(t)
    else:
        frac = torch.clamp(t / ramp, 0.0, 1.0)
    g = 1.0 - alpha * frac
    open_ = env.kick_time_s >= 0
    ep = getattr(env, "kick_episode_type", None)
    from_bank = getattr(env, "kick_from_bank", None)
    if ep is not None and from_bank is not None and exclude:
        for name in exclude:
            open_ |= from_bank & (ep == kh.EP_NAMES[name])
    g = torch.where(open_, torch.ones_like(g), g)
    pos = torch.clamp(rm._step_reward[:, idx], min=0.0).sum(dim=1)
    _watch_reward_blowup(env, rm)
    return -(1.0 - g) * pos


REWARD_BLOWUP_WARN = 1.0e4


def _watch_reward_blowup(env, rm):
    """진단 (2026-11-23): 이번 스텝의 항별 |보상| 최댓값이 1e4 를 넘으면 항 이름·env·행동 크기를 출력한다.

    value loss 발산 롤백 (E 7 회, H 6 회) 의 원인 항을 기록하기 위한 것. 평소 항별 값은 1 미만이다.
    kick_gate 가 마지막 항이라 이 시점에 _step_reward 에 이번 스텝의 모든 항이 들어 있다.
    """
    sr = rm._step_reward
    bad = ~torch.isfinite(sr)
    mx = torch.where(bad, torch.full_like(sr, float("inf")), sr.abs()).amax(dim=0)
    if bool((mx > REWARD_BLOWUP_WARN).any()):
        n = int(getattr(env, "_blowup_prints", 0))
        if n < 200:
            env._blowup_prints = n + 1
            act = env.action_manager.action.abs()
            it = int(env.common_step_counter)
            hits = [(rm._term_names[i], float(mx[i]), int(torch.where(bad[:, i], 1.0, sr[:, i].abs()).argmax()))
                    for i in torch.nonzero(mx > REWARD_BLOWUP_WARN).flatten().tolist()]
            print(f"[REWARD_BLOWUP] step {it} terms {hits} | |action| max {float(act.max()):.3g} "
                  f"(env {int(act.amax(dim=1).argmax())})", flush=True)


# ============================================================================
# V1-6I (2026-11-24) — 공 속도 보상에 방향 배율을 곱한다.
#   H 27500 은 넘어짐은 줄었지만 공 속도 −22 %, 방향 오차 16~22° (22300 11~14°).
#   기존 방향 항 ball_offaxis (−0.3 × 수직성분²) 는 20° 오차에서 킥당 약 −0.06 으로 속도 보상 (~+3) 에 비해 무시할 만했다.
#   투영 proj² 도 20° 에서 12 % 만 깎는다. 그래서 속도 보상 자체에 dir = floor + (1 − floor)·exp(−(θ/σ)²) 를 곱한다:
#   σ 20° → 10° 는 ×0.82, 20° 는 ×0.56, 30° 는 ×0.38 (floor 0.3). 공이 거의 안 움직이면 (속도 < 0.3 m/s) 배율 1.
# ============================================================================
def ball_velocity_dir_gated(
    env, v_full: float, lo: float, sigma_frac: float, sigma_min: float, lin_err: float = 0.0, lin_w: float = 0.0,
    soft_scale: float = 1.0, max_speed: float = 3.0, post_kick_window_s: float = 0.4, quality_floor: float = 0.3,
    min_strike_speed: float = 1.8, window_steps: int = 20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    dir_sigma_deg: float = 20.0, dir_floor: float = 0.3,
) -> torch.Tensor:
    # Isaac Lab 은 보상 함수의 **kwargs 를 받지 않는다 → kh.ball_velocity_strength 인자를 그대로 나열한다.
    from . import kick_handoff_mdp as kh
    from . import kick_inside_mdp as kim

    base = kh.ball_velocity_strength(env, v_full, lo, sigma_frac, sigma_min, lin_err, lin_w, soft_scale, max_speed,
                                     post_kick_window_s, quality_floor, min_strike_speed, window_steps,
                                     asset_cfg, ball_cfg)
    ball = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    speed = torch.norm(v_xy, dim=-1)
    cos = ((v_xy * kim._target_dir_xy(env)).sum(dim=-1) / (speed + 1e-6)).clamp(-1.0, 1.0)
    theta = torch.rad2deg(torch.acos(cos))
    dir_q = dir_floor + (1.0 - dir_floor) * torch.exp(-torch.square(theta / dir_sigma_deg))
    dir_q = torch.where(speed > 0.3, dir_q, torch.ones_like(dir_q))
    return base * dir_q


# ============================================================================
# V1-6J (2026-11-25) — 골반 roll 벌림 상한 벌점.
#   실물 킥 (2026-11-25, robo04): 차는 다리 골반 roll 이 55~63° 까지 벌어지고, 그 직후 디딤발이 바깥 모서리로
#   들리며 넘어졌다. MuJoCo 는 같은 동작에서 버텨 재현되지 않았다 (질량·강성·지연·바닥·발 형상 모두 시험).
#   그래서 실물이 감당할 여유를 두도록 |골반 roll − 기본| 이 threshold 를 넘는 만큼 제곱 벌점을 준다 (양 다리).
# ============================================================================
def _require_resolved(cfg: SceneEntityCfg, attr: str, who: str):
    """Isaac Lab 은 params 에 넣은 SceneEntityCfg 만 resolve 한다. 함수 기본값은 slice(None) (= 전체) 로 남는다."""
    if isinstance(getattr(cfg, attr), slice):
        raise ValueError(f"{who}: {cfg.name}.{attr} 가 resolve 되지 않았다 — SceneEntityCfg 를 params 에 넣어라")


def hip_roll_excess(env, threshold_rad: float = 0.61,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_pelvis_roll_joint"])) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    _require_resolved(asset_cfg, "joint_ids", "hip_roll_excess")
    dev = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    excess = torch.clamp(dev.abs() - threshold_rad, min=0.0)
    return torch.square(excess).sum(dim=1)


def metric_hip_roll_max_deg(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_pelvis_roll_joint"])) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    dev = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    return torch.rad2deg(dev.abs().max(dim=1).values)


def ramp_reward_weight(env, env_ids, term_name: str, target: float, ramp_steps: int) -> float:
    """보상 항 가중치를 0 → target 으로 학습 진행 (env.common_step_counter) 에 비례해 올린다.

    V1-6J1~J4 (2026-11-25) 는 수렴한 정책에 큰 새 벌점을 한 번에 넣어 value 추정이 깨지고 킥이 무너졌다
    (kick_resolved 0.6 → 0.005~0.33, noise std 0.375 → 0.46). 가중치를 천천히 올려 그 충격을 피한다.
    TensorBoard: Curriculum/<이 항 이름> = 현재 가중치.
    """
    frac = min(1.0, float(env.common_step_counter) / max(1, ramp_steps))
    rm = env.reward_manager
    cfg = rm.get_term_cfg(term_name)
    cfg.weight = target * frac
    rm.set_term_cfg(term_name, cfg)
    return cfg.weight


# ============================================================================
# V1-6K (2026-11-25) — 킥 디딤발 쪽 지속 외력 + 몸 roll 벌점 (실물 '디딤발 쪽 roll 전도' 대용)
# ----------------------------------------------------------------------------
# 실물: 차는 다리 골반 roll 55~63° 벌림 직후 디딤발이 바깥 모서리로 들려 디딤발 쪽으로 넘어진다.
# 시뮬은 같은 동작에도 버틴다 (발·잔디·모터 차이를 모델로 재현 못 함, KICK_V1_6_KO.md).
# 그 차이를 '디딤발 쪽으로 미는 지속 외력' 으로 대신 넣는다.
#   시작: 한쪽 발이 떠 있고 (접촉 수직력 < 5 N) 그 다리 골반 roll 벌림 > trig_deg (40°, 걷는 중 ~30° 보다 큼)
#   방향: 몸통 좌우축 (수평), 디딤발 쪽. 크기: 에피소드마다 force_range 균등, probability 의 에피소드만.
#   시간: duration_s 동안 일정, 에피소드당 한 번. 크기는 ramp_steps 동안 0 → 1 배로 키운다.
# MuJoCo 보정 (push_calib.py, 2026-11-25): 같은 외력 0.5 s 에 30899 는 8 N, 27500 은 4 N 부터 거의 다 넘어진다.
# 같은 정의가 mujoco_kick_stress.KickPush 에 있다 (게이트 push 조건).
# ============================================================================
_KSP_ROLL = SceneEntityCfg("robot", joint_names=["r_pelvis_roll_joint", "l_pelvis_roll_joint"], preserve_order=True)
_KSP_FEET = SceneEntityCfg("contact_forces", body_names=["r_ankle_roll_link", "l_ankle_roll_link"], preserve_order=True)


def _ksp_init(env):
    if hasattr(env, "_ksp_left"):
        return
    n, dev = env.num_envs, env.device
    env._ksp_left = torch.full((n,), -1.0, device=dev)        # 남은 외력 시간 [s], −1 = 대기
    env._ksp_done = torch.zeros(n, dtype=torch.bool, device=dev)
    env._ksp_peak = torch.zeros(n, device=dev)                 # 이 에피소드의 힘 [N], 0 = 안 밂
    env._ksp_F = torch.zeros(n, 3, device=dev)
    env._ksp_scale = 0.0
    env._ksp_stat = {"eligible": 0.0, "pushed": 0.0, "force_sum": 0.0, "roll_sum": 0.0, "eps": 0.0}
    env._ksp_roll_max = torch.zeros(n, device=dev)


def kick_support_push_reset(env, env_ids, force_range: tuple[float, float], probability: float = 0.5,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link")):
    """리셋 env: 외력 끄고, 이번 에피소드 힘 크기를 뽑는다."""
    _ksp_init(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    k = len(env_ids)
    # 끝난 에피소드 통계 (Curriculum/kick_push_max_N/* 로 기록)
    st, el = env._ksp_stat, env._ksp_peak[env_ids] > 0
    pushed = el & (env._ksp_done[env_ids] | (env._ksp_left[env_ids] >= 0))
    st["eligible"] += float(el.sum()); st["pushed"] += float(pushed.sum())
    st["force_sum"] += float((env._ksp_peak[env_ids] * env._ksp_scale * pushed.float()).sum())
    st["roll_sum"] += float(env._ksp_roll_max[env_ids].sum()); st["eps"] += float(k)
    env._ksp_roll_max[env_ids] = 0.0
    lo, hi = force_range
    peak = torch.rand(k, device=env.device) * (hi - lo) + lo
    on = torch.rand(k, device=env.device) < probability
    env._ksp_peak[env_ids] = torch.where(on, peak, torch.zeros_like(peak))
    env._ksp_left[env_ids] = -1.0
    env._ksp_done[env_ids] = False
    env._ksp_F[env_ids] = 0.0
    _require_resolved(asset_cfg, "body_ids", "kick_support_push_reset")
    asset = env.scene[asset_cfg.name]
    z = torch.zeros(k, 1, 3, device=env.device)
    asset.set_external_force_and_torque(z, z, body_ids=asset_cfg.body_ids, env_ids=env_ids)


def kick_support_push(env, env_ids, duration_s: float = 0.5, trig_deg: float = 40.0, ramp_steps: int = 12000,
                      contact_n: float = 5.0,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
                      roll_cfg: SceneEntityCfg = _KSP_ROLL, sensor_cfg: SceneEntityCfg = _KSP_FEET):
    """매 스텝 (interval 0.02 s) 호출. 시작 조건을 보고 디딤발 쪽 외력을 쓴다."""
    _ksp_init(env)
    _require_resolved(asset_cfg, "body_ids", "kick_support_push")
    _require_resolved(roll_cfg, "joint_ids", "kick_support_push")
    _require_resolved(sensor_cfg, "body_ids", "kick_support_push")
    asset = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    jid = roll_cfg.joint_ids
    dev_deg = torch.rad2deg((asset.data.joint_pos[:, jid] - asset.data.default_joint_pos[:, jid]).abs())
    fn = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].abs()
    air = fn < contact_n
    idle = (env._ksp_left < 0) & ~env._ksp_done & (env._ksp_peak > 0)
    r_kick = idle & air[:, 0] & ~air[:, 1] & (dev_deg[:, 0] > trig_deg)
    l_kick = idle & air[:, 1] & ~air[:, 0] & (dev_deg[:, 1] > trig_deg) & ~r_kick
    start = r_kick | l_kick
    env._ksp_scale = min(1.0, float(env.common_step_counter) / max(1, ramp_steps))
    g = asset.data.projected_gravity_b
    env._ksp_roll_max = torch.maximum(env._ksp_roll_max, torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2]).abs()))
    if start.any():
        q = asset.data.root_quat_w                                   # (w, x, y, z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        ax = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z)], dim=-1)   # 몸 +y 축 (월드 xy)
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sgn = torch.where(r_kick, 1.0, -1.0)                         # R 차기 → 디딤발 L = 몸 +y
        mag = sgn * env._ksp_peak * env._ksp_scale
        F = torch.zeros_like(env._ksp_F)
        F[:, :2] = ax * mag.unsqueeze(-1)
        env._ksp_F = torch.where(start.unsqueeze(-1), F, env._ksp_F)
        env._ksp_left = torch.where(start, torch.full_like(env._ksp_left, duration_s), env._ksp_left)
    active = env._ksp_left >= 0
    force = env._ksp_F * active.unsqueeze(-1).float()
    asset.set_external_force_and_torque(force.unsqueeze(1), torch.zeros_like(force).unsqueeze(1),
                                        body_ids=asset_cfg.body_ids, is_global=True)
    env._ksp_left = torch.where(active, env._ksp_left - env.step_dt, env._ksp_left)
    env._ksp_done = env._ksp_done | (active & (env._ksp_left < 0))


def metric_kick_push_N(env) -> torch.Tensor:
    """지금 걸린 디딤발 쪽 외력 크기 [N] (weight 0, TensorBoard 용)."""
    if not hasattr(env, "_ksp_left"):
        return torch.zeros(env.num_envs, device=env.device)
    return env._ksp_F.norm(dim=-1) * (env._ksp_left >= 0).float()


def kick_push_scale(env, env_ids, max_N: float) -> dict:
    """커리큘럼 기록용 (가중치 0 보상 항은 Isaac Lab 이 계산하지 않아 로그가 늘 0 이다).
    TensorBoard: Curriculum/<이름>/max_N (지금 외력 최대 = max_N × 램프), pushed_frac (외력 대상 에피소드 중 실제 밀린 비율),
    mean_N (밀린 에피소드 평균 힘), roll_max_deg (에피소드 몸 roll 최대 평균). 지난 기록 뒤 끝난 에피소드 기준."""
    out = {"max_N": max_N * float(getattr(env, "_ksp_scale", 0.0))}
    st = getattr(env, "_ksp_stat", None)
    if st and st["eps"] > 0:
        out["pushed_frac"] = st["pushed"] / max(1.0, st["eligible"])
        out["mean_N"] = st["force_sum"] / max(1.0, st["pushed"])
        out["roll_max_deg"] = st["roll_sum"] / st["eps"]
        for key in st:
            st[key] = 0.0
    return out


def base_roll_excess(env, threshold_rad: float = 0.1745, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 roll 이 threshold 를 넘은 만큼의 제곱 [rad²]."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    roll = torch.atan2(g[:, 1], -g[:, 2])
    return torch.square(torch.clamp(roll.abs() - threshold_rad, min=0.0))


def metric_base_roll_deg(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.rad2deg(torch.atan2(g[:, 1], -g[:, 2]).abs())
