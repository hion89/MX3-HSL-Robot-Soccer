"""RS02 기상 (Unitree-RS02-Getup-V0) 전용 — 넘어짐 상태 은행 리셋 · '천천히' 제약 · 이름 무관 램프인 커리큘럼.

MX3 기상 (getup_v1_mdp · getup_v1_rewards, 실기 검증 v1_2) 을 그대로 쓰고 RS02 에 새로 필요한 것만 여기 둔다.
  - reset_from_fall_bank : 역방향 커리큘럼 마지막 단계에서 은행 상태로 시작 (scripts/rsl_rl/rs02_getup/gen_fall_bank.py 가 만든 .pt)
  - base_ang_vel_over / base_rise_speed_over / torque_over_budget : 2026-11-06 사용자 "천천히 안정적으로 일어나는 기상"
  - penalty_rampin_by_name : getup_v1_mdp.penalty_rampin_curriculum 은 항 이름 4 개가 박혀 있어서 이름 → 가중치 표를 받는 판
"""

from __future__ import annotations

import math

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz, quat_mul, yaw_quat

from .getup_v1_mdp import GetupJointPositionAction, GetupJointPositionActionCfg, hold_masked


# ═══════════════════════════════════════════════════════════════
#  액션 — 홀드 동안 지연 버퍼에도 스폰 자세를 채운다
# ═══════════════════════════════════════════════════════════════
class GetupSpawnHoldAction(GetupJointPositionAction):
    """GetupJointPositionAction + 홀드 동안 액션 지연 버퍼에 '스폰 자세가 되는 액션' 을 써 넣는다.

    부모들 (DelayedJointPositionAction) 은 홀드 동안 액션을 0 으로 지연 버퍼에 넣는다. 홀드가 풀리면 지연 (1~2 스텝) 때문에
    그 0 이 먼저 나와 목표 = 기본 서기 자세 로 한두 스텝 튄다. MX3 기상은 시작 자세가 기본각 + 노이즈라 티가 안 났지만
    넘어짐 상태 은행은 기본각과 멀어서 홀드 해제 순간 관절 0.17 rad · 몸 0.3~0.5 m/s 로 덜컥 당겨졌다 (probe_getup_reset.py, 2026-11-06).
    실기는 기상 정책 시작 순간 PD 목표 = 지금 관절각이므로 이쪽이 맞다.

    cfg.target_rate_limit [rad/s] 를 주면 관절 목표가 정책 스텝마다 움직일 수 있는 양을 rate × step_dt 로 자른다 (None = 안 자름).
    '천천히' 를 벌점이 아니라 구조로 강제 (9 차: 벌점 ×5 로도 관절 최고 13/22 rad/s 그대로). 실기 노드도 같은 제한을 목표에 걸어야 한다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev_target = self._q_at_reset.clone()
        self._lpf_target = self._q_at_reset.clone()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        self._prev_target[idx] = self._q_at_reset[idx]
        self._lpf_target[idx] = self._q_at_reset[idx]

    def process_actions(self, actions: torch.Tensor):
        hold = (self._env.reset_hold_steps > 0) if hasattr(self._env, "reset_hold_steps") else None
        super().process_actions(actions)
        if hold is not None and self._use_delay and bool(hold.any()):
            ids = hold.nonzero().squeeze(-1)
            a_spawn = (self._q_at_reset[ids] - self._offset[ids]) / self._scale
            slot = (self._delay_write_idx[ids] - 1) % (self._delay_max + 1)   # 방금 쓴 칸
            self._delay_buf[ids, slot] = a_spawn
        if self.cfg.target_rate_limit is not None:
            d = self.cfg.target_rate_limit * self._env.step_dt
            self._processed_actions[:] = self._prev_target + (self._processed_actions - self._prev_target).clamp(-d, d)
        self._prev_target[:] = self._processed_actions
        # 눕기 V1-3 (11-10): 목표 속도 제한 뒤 1 차 저역 통과 — 목표 ← 목표 + α (제한된 목표 − 목표). 속도 제한은 필터 전 목표 기준 (노드도 같은 순서).
        if self.cfg.target_lpf_alpha is not None:
            self._lpf_target += self.cfg.target_lpf_alpha * (self._processed_actions - self._lpf_target)
            self._processed_actions[:] = self._lpf_target


@configclass
class GetupSpawnHoldActionCfg(GetupJointPositionActionCfg):
    class_type = GetupSpawnHoldAction
    target_rate_limit: float | None = None
    target_lpf_alpha: float | None = None   # 눕기 V1-3: 관절 목표 1 차 저역 통과 (None = 안 함)


# ═══════════════════════════════════════════════════════════════
#  리셋 — 넘어짐 상태 은행
# ═══════════════════════════════════════════════════════════════
def _load_bank(env, bank_weights: dict, asset) -> dict:
    """은행 파일들을 한 번만 읽어 env 에 둔다. 출처별 가중치를 상태 하나하나의 뽑기 확률로 바꾼다."""
    env._fall_bank = _read_bank(env, bank_weights, asset)
    return env._fall_bank


def _read_bank(env, bank_weights: dict, asset) -> dict:
    """_load_bank 의 읽기 부분 (env 에 안 둠) — 11-11 SitStand 단계별 은행이 여러 개를 따로 들고 있으려고."""
    names = list(asset.data.joint_names)
    zs, qs, js, ps, srcs, cats = [], [], [], [], [], []
    for i, (path, w) in enumerate(bank_weights.items()):
        b = torch.load(path, weights_only=False)
        perm = [b["joint_names"].index(n) for n in names]
        n = len(b["root_z"])
        zs.append(b["root_z"])
        qs.append(b["root_quat"])
        js.append(b["joint_pos"][:, perm])
        ps.append(torch.full((n,), float(w) / n))
        srcs.append(torch.full((n,), i, dtype=torch.long))
        cats.append(b["cat"])
    dev = env.device
    g = quat_apply_inverse(torch.cat(qs), torch.tensor([0.0, 0.0, -1.0]).expand(sum(len(z) for z in zs), 3))
    # 과회전 벌점 (getup_v1_rewards.getup_overshoot) 의 출발 방향: 등 (g_x≈−1) +1 · 엎드림 (g_x≈+1) −1 · 옆·기타 0
    sign = torch.where(g[:, 0] < -0.5, 1.0, torch.where(g[:, 0] > 0.5, -1.0, 0.0))
    return dict(z=torch.cat(zs).to(dev), quat=torch.cat(qs).to(dev), q=torch.cat(js).to(dev),
                p=torch.cat(ps).to(dev), src=torch.cat(srcs).to(dev), cat=torch.cat(cats).to(dev), sign=sign.to(dev))


def reset_from_fall_bank(
    env,
    env_ids,
    bank_weights: dict,
    from_level: int,
    xy_range: tuple[float, float] = (-0.2, 0.2),
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    height_pad: float = 0.01,
    joint_noise: float = 0.05,
    force: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """역방향 커리큘럼 레벨 ≥ from_level (또는 force) 이면 은행에서 뽑은 상태로 루트·관절을 덮어쓴다. 아니면 아무것도 안 함.

    리셋 이벤트 **순서**: reset_root_supine → reset_joints_by_offset → **이것** → set_reset_hold_steps.
      앞의 둘이 쓴 자세를 덮어쓰고, GetupJointPositionAction.reset 이 그 뒤 관절각을 '스폰 자세' 로 잡아 홀드 동안 유지한다
      (실기: 댐핑으로 안착 → 기상 정책 시작 순간 PD 목표 = 지금 관절각).
    은행 상태는 이미 바닥에 안착해 멈춘 자세라 height_pad 만큼만 띄운다 (침투 방지). yaw 는 은행 값을 지우고 yaw_range 로 다시 뽑는다.
    env.getup_from_bank (bool) · env.getup_bank_cat (자세 종류) · env.getup_bank_src (출처) 를 남긴다 — 평가에서 쪼개 보려고.
    """
    level = int(getattr(env, "_getup_level", 0))
    asset = env.scene[asset_cfg.name]
    dev = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=dev)
    if not hasattr(env, "getup_from_bank"):
        env.getup_from_bank = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        env.getup_bank_cat = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)
        env.getup_bank_src = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)
    env.getup_from_bank[env_ids] = False
    env.getup_bank_cat[env_ids] = -1
    env.getup_bank_src[env_ids] = -1
    if not force and level < from_level:
        return
    bank = getattr(env, "_fall_bank", None) or _load_bank(env, bank_weights, asset)
    n = len(env_ids)
    idx = torch.multinomial(bank["p"], n, replacement=True)

    quat = bank["quat"][idx]
    _, _, yaw_b = euler_xyz_from_quat(quat)
    zeros = torch.zeros(n, device=dev)
    yaw = torch.empty(n, device=dev).uniform_(*yaw_range)
    quat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw - yaw_b), quat)
    pos = env.scene.env_origins[env_ids].clone()
    pos[:, 0] += torch.empty(n, device=dev).uniform_(*xy_range)
    pos[:, 1] += torch.empty(n, device=dev).uniform_(*xy_range)
    pos[:, 2] += bank["z"][idx] + height_pad
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    lim = asset.data.soft_joint_pos_limits[env_ids]
    q = bank["q"][idx] + torch.empty_like(bank["q"][idx]).uniform_(-joint_noise, joint_noise)
    q = torch.minimum(torch.maximum(q, lim[..., 0]), lim[..., 1])
    asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

    if not hasattr(env, "getup_fall_sign"):
        env.getup_fall_sign = torch.ones(env.num_envs, device=dev)
    env.getup_fall_sign[env_ids] = bank["sign"][idx]
    env.getup_from_bank[env_ids] = True
    env.getup_bank_cat[env_ids] = bank["cat"][idx]
    env.getup_bank_src[env_ids] = bank["src"][idx]


# ═══════════════════════════════════════════════════════════════
#  '천천히 · 안정적으로' 제약 (전부 홀드 구간 0, 음수 weight)
# ═══════════════════════════════════════════════════════════════
@hold_masked
def base_ang_vel_over(env, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 각속도 크기가 limit [rad/s] 를 넘은 만큼의 제곱 — 휙 구르거나 튕겨 일어나기를 막는다."""
    w = env.scene[asset_cfg.name].data.root_ang_vel_w.norm(dim=-1)
    return torch.square(torch.clamp(w - limit, min=0.0))


@hold_masked
def base_rise_speed_over(env, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 상승 속도가 limit [m/s] 를 넘은 만큼의 제곱 — 튀어 오르듯 일어나기를 막는다 (내려가는 건 안 봄).
    0.25 m/s 면 누운 높이 0.057 → 기립 0.48 의 상승만 최소 1.7 s."""
    vz = env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2]
    return torch.square(torch.clamp(vz - limit, min=0.0))


@hold_masked
def torque_over_budget(
    env,
    budgets: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절 토크가 모터별 '오래 버틸 수 있는' 값을 넘은 만큼의 제곱 합.

    budgets = {관절 이름 정규식: N·m}. 천천히 일어나면 팔·발목이 몇 초씩 힘을 준다 — RS00 은 14 N·m 3 s · 8 N·m 53 s,
    RS06 은 36 N·m 5 s · 18 N·m 70 s (rs_bot.py 과부하 표). 피크는 액추에이터가 이미 자르므로 여기선 '지속' 쪽 값을 문턱으로 쓴다.
    """
    asset = env.scene[asset_cfg.name]
    key = "_torque_budget_idx"
    if not hasattr(env, key):
        lim = torch.full((asset.num_joints,), float("inf"), device=env.device)
        for pat, v in budgets.items():
            ids, _ = asset.find_joints(pat)
            lim[ids] = float(v)
        setattr(env, key, lim)
    lim = getattr(env, key)
    tq = asset.data.applied_torque.abs()
    return torch.square(torch.clamp(tq - lim, min=0.0)).sum(dim=-1)


def set_getup_level(env, env_ids, level: int):
    """startup: 역방향 커리큘럼 시작 레벨 (이어받기용 — 커리큘럼 상태는 체크포인트에 안 담겨 새 env 는 레벨 0 부터)."""
    env._getup_level = int(level)
    env._getup_level_since = 0


@hold_masked
def joint_limit_margin(env, margin_frac: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """관절이 가동범위 양 끝 margin_frac 안에 들어간 만큼의 합 (L1, rad) — 관절 한계에 기대 잠그고 버티기를 막는다.

    3 차 (2026-11-06): 2 차 끝 자세가 무릎 0.00 (= 가동 한계, 목표 ∓0.5) · 발목 roll ±0.35 (한계 0.349) 로 관절 끝에 기대
    힘 안 들이고 버티는 넓은 자세에 갇혔다. 기존 dof_pos_limits 는 한계 '밖' 만 봐서 정확히 한계에 붙은 건 못 잡는다.
    L1 이라 경계 안으로 밀어내는 힘이 일정하다 (제곱이면 무릎 0.12 rad 침범이 0.015 로 너무 약함).
    """
    asset = env.scene[asset_cfg.name]
    lim = asset.data.soft_joint_pos_limits
    lo, hi = lim[..., 0], lim[..., 1]
    m = margin_frac * (hi - lo)
    q = asset.data.joint_pos
    return (torch.clamp(lo + m - q, min=0.0) + torch.clamp(q - (hi - m), min=0.0)).sum(dim=-1)


@hold_masked
def feet_stance_error(env, width: float, gate_from: float, asset_cfg: SceneEntityCfg, dx_weight: float = 1.0) -> torch.Tensor:
    """두 발 (asset_cfg.body_ids = [왼발, 오른발]) 의 상대 위치가 보행 기본 자세와 다른 정도 [m] (L1): |좌우 간격 − width| + |앞뒤 엇갈림|.
    몸통 yaw 좌표. 몸통이 서 있을 때만 (직립 (1 − g_z)/2 가 gate_from 부터 1 까지 선형) — 높이 아닌 방향 게이트.

    4 차 (2026-11-06): 3 차 4300 은 팔·무릎은 거의 맞췄지만 발을 41 cm 로 벌리고 9 cm 엇갈려 섬 (기본 자세 23 cm · 0).
    발을 모으려면 한 발 들어 디뎌야 해서 관절 끝 자세 보상만으로는 못 찾음 → 발 위치를 직접. 멀리서도 기울기가 일정하게 L1.
    """
    asset = env.scene[asset_cfg.name]
    err = _feet_err(env, width, asset_cfg, dx_weight)
    up = 0.5 * (1.0 - asset.data.projected_gravity_b[:, 2])
    gate = torch.clamp((up - gate_from) / (1.0 - gate_from), 0.0, 1.0)
    return err * gate


def _feet_err(env, width: float, asset_cfg: SceneEntityCfg, dx_weight: float = 1.0) -> torch.Tensor:
    """|좌우 간격 − width| + dx_weight·|앞뒤 엇갈림| [m] (몸통 yaw 좌표, asset_cfg.body_ids = [왼발, 오른발])."""
    asset = env.scene[asset_cfg.name]
    p = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), p[:, 0] - p[:, 1])
    return (rel[:, 1] - width).abs() + dx_weight * rel[:, 0].abs()


@hold_masked
def stand_success_stance(
    env,
    supine_height: float,
    stand_height: float,
    width: float,
    std: float,
    floor: float,
    feet_cfg: SceneEntityCfg,
    height_frac_threshold: float = 0.85,
    upright_threshold: float = 0.95,
    dx_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """getup_v1_rewards.stand_success × (floor + (1 − floor)·exp(−발 오차²/std²)) — 발 자리가 틀리면 서 있어도 성공 보상이 floor 까지 깎인다.

    5 차 (2026-11-06): 4 차 5400 은 일어선 뒤 발을 한 번도 안 뗌 (env 98~100 %) → 자세가 '일어나는 동작이 끝난 자리' 로 굳는다.
    가장 큰 보상 (5/s) 에 발 자리를 걸어, 일어나는 도중 (쪼그린 단계, 아직 손·무릎이 무게를 나눌 때) 에 발을 제자리에 놓게 유도.
    연속 계수라 MX3 의 '조건을 겹치면 희소 보상이 안 터진다' 함정은 없다 (floor 이상은 늘 지급).
    """
    from .getup_v1_rewards import _height_frac, _upright_frac

    h = _height_frac(env, asset_cfg, supine_height, stand_height)
    up = _upright_frac(env, asset_cfg)
    base = ((h > height_frac_threshold) & (up > upright_threshold)).float()
    q = torch.exp(-torch.square(_feet_err(env, width, feet_cfg, dx_weight)) / (std * std))
    return base * (floor + (1.0 - floor) * q)


@hold_masked
def step_to_fix_stance(
    env,
    width: float,
    err_from: float,
    err_span: float,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    supine_height: float,
    stand_height: float,
    force_threshold: float = 5.0,
    dx_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """다 일어섰는데 발 자리가 틀린 동안 (발 오차 > err_from) 한 발로 서 있으면 보상 — 서서 발을 옮겨 디디는 탐색을 돕는다.

    5 차: 4 차 5400 이 서서 발을 전혀 안 떼서 (feet_stance 벌점만으로는 '발 들기' 라는 중간 단계를 못 찾음).
    발 오차가 err_from 아래로 오면 0 → 발을 다 모은 뒤 제자리걸음으로 새지 않는다. 계수는 (오차 − err_from)/err_span 로 0~1.
    """
    from .getup_v1_rewards import _height_frac, _upright_frac

    sensor = env.scene.sensors[sensor_cfg.name]
    on = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
    single = (on.sum(dim=1) == 1).float()
    stood = ((_height_frac(env, asset_cfg, supine_height, stand_height) > 0.85) & (_upright_frac(env, asset_cfg) > 0.95)).float()
    k = torch.clamp((_feet_err(env, width, feet_cfg, dx_weight) - err_from) / err_span, 0.0, 1.0)
    return single * stood * k


@hold_masked
def leg_pose_l1_stood(
    env,
    target_pose: dict,
    supine_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg,
    height_frac_threshold: float = 0.85,
    upright_threshold: float = 0.95,
) -> torch.Tensor:
    """다 일어선 동안 (높이 비율 > 0.85 & 직립 > 0.95 — stand_success 와 같은 조건) Σ |q − 보행 기본각| [rad] (asset_cfg 관절들, L1).

    7 차 (2026-11-06): 6 차 7200 — 발은 거의 제자리 (간격 0.248 · 엇갈림 0.028) 인데 다리 관절 RMS 0.165 그대로:
    왼무릎 −0.10 (목표 −0.5, 편 채) · 오른 발목 roll −0.30 · hip pitch +0.06/−0.33 의 한쪽 다리에 기대는 비대칭 자세.
    가우시안 (walk_ready_legs) 은 이 근처에서 당기는 힘이 약하고, 관절 한계 벌점으로도 안 움직임 → 관절마다 일정한 힘 (L1).
    게이트를 높이로 피하려면 stand_success (5/s) 를 버려야 해서 조작 이득 없음 (memory gate-must-not-be-manipulable).
    """
    from .getup_v1_rewards import _height_frac, _upright_frac

    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    key = "_leg_l1_target"
    tgt = getattr(env, key, None)
    if tgt is None or tgt.shape != q.shape[1:]:
        names = [asset.data.joint_names[i] for i in asset_cfg.joint_ids]
        tgt = torch.tensor([float(target_pose.get(n, 0.0)) for n in names], dtype=q.dtype, device=q.device)
        setattr(env, key, tgt)
    stood = ((_height_frac(env, asset_cfg, supine_height, stand_height) > height_frac_threshold)
             & (_upright_frac(env, asset_cfg) > upright_threshold)).float()
    return (q - tgt).abs().sum(dim=-1) * stood


# ═══════════════════════════════════════════════════════════════
#  커리큘럼 — 역방향 커리큘럼이 from_level 에 닿은 뒤부터 가중치를 단계적으로
# ═══════════════════════════════════════════════════════════════
def penalty_rampin_by_name(
    env,
    env_ids,
    stage_steps: tuple[int, ...],
    weights: dict,
    from_level: int,
    rate_limits: tuple | None = None,
    action_name: str = "JointPositionAction",
):
    """weights = {보상 항 이름: (단계 0, 단계 1, …)}. rate_limits = (단계 0, 1, …) 면 액션 항 (GetupSpawnHoldAction) 의 관절 목표 속도 제한 [rad/s] 도 단계별로. 레벨 < from_level 이면 단계 0, 닿은 순간부터 common_step_counter 로
    stage_steps 를 하나씩 넘을 때마다 다음 단계. MX3 penalty_rampin_curriculum 과 같은 규칙인데 항 이름을 인자로 받는다.

    이유 (MX3 실측): 과제를 배우기 전에 실기 이관용 벌점을 켜면 역방향 커리큘럼이 레벨 1 에서 막혔다.
    이미 일어설 줄 아는 정책에는 벌점이 해롭지 않았다 → 은행 단계에 닿은 뒤 켠다.
    """
    level = int(getattr(env, "_getup_level", 0))
    if level >= from_level and not hasattr(env, "_rampin_t0"):
        env._rampin_t0 = env.common_step_counter
    if level < from_level:
        idx = 0
    else:
        dwell = env.common_step_counter - env._rampin_t0
        idx = sum(1 for s in stage_steps if dwell >= s)
    for name, table in weights.items():
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = float(table[min(idx, len(table) - 1)])
        env.reward_manager.set_term_cfg(name, cfg)
    if rate_limits is not None:
        env.action_manager.get_term(action_name).cfg.target_rate_limit = float(rate_limits[min(idx, len(rate_limits) - 1)])
    return torch.tensor(float(idx), device=env.device)


# ═══════════════════════════════════════════════════════════════
#  앉음 → 서기 (SitStand-V0, 11-11) 종료 조건
# ═══════════════════════════════════════════════════════════════
def lay_back(env, pitch_deg: float = -65.0, z_min: float = 0.09, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """뒤로 누움 = 몸통 pitch (atan2(g_x, −g_z), 뒤 −) 가 pitch_deg 보다 뒤 · 또는 골반 < z_min [m] (반듯이 누움 0.057 · 앉음 0.122).
    11-11 사용자 "앉은 상태로 못일어나? 추가로 학습해서?" — 기상 V0-1 은 앉음에서 뒤로 누웠다 (−106°, 등 764 N) 반동으로 선다 → 이 길을 막는다.
    손 짚고 뒤로 기대는 자세 (앉기 동작 4 s 근처 −45~−56°) 는 허용."""
    d = env.scene[asset_cfg.name].data
    g = d.projected_gravity_b
    pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
    z = d.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    out = (pitch < pitch_deg) | (z < z_min)
    hold = getattr(env, "getup_hold_active", None)
    return out & ~hold if hold is not None else out


def not_risen(env, t_s: float, z_min: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """t_s [s] 가 지났는데 골반이 z_min [m] 아래 = 아직 못 일어남 (앉은 채 버티기 구멍 막기 — 앉기 V0-1 6800 의 '쪼그린 채 버티기' 교훈)."""
    z = env.scene[asset_cfg.name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (env.episode_length_buf * env.step_dt >= t_s) & (z < z_min)


def reset_from_level_banks(
    env,
    env_ids,
    level_banks: list,
    band_prob: float = 0.25,
    xy_range: tuple[float, float] = (-0.2, 0.2),
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    height_pad: float = 0.01,
    joint_noise: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """SitStand 역순 커리큘럼 (11-11): 단계 L (env._sit_level, sit_level_curriculum 이 올림) 의 은행에서 시작 상태를 뽑는다.
    level_banks[L] = {은행 경로: 가중치} — 앉기 동작을 거꾸로: 0 쪼그림 (앉기 2.6 s) → … → 손 짚음 → 엉덩이 닿음 → 다리 뻗고 앉음 (끝 · 댐핑).
    band_prob 만큼은 아래 단계에서 고르게 (잊지 않게). 뽑은 단계는 env.sit_start_level 에 (승격 판정은 지금 단계에서 시작한 판만).
    리셋 순서 · 높이 띄움 · yaw · 관절 노이즈 · getup_fall_sign 은 reset_from_fall_bank 와 같음 (앉음은 등/엎드림 아님 → 부호는 은행대로)."""
    asset = env.scene[asset_cfg.name]
    dev = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=dev)
    if not hasattr(env, "_lv_banks"):
        env._lv_banks = [_read_bank(env, bw, asset) for bw in level_banks]
        env.sit_start_level = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        env._sit_level = int(getattr(env, "_sit_level", 0))
    if not hasattr(env, "getup_fall_sign"):
        env.getup_fall_sign = torch.ones(env.num_envs, device=dev)
    lvl = int(env._sit_level)
    n = len(env_ids)
    lv = torch.full((n,), lvl, dtype=torch.long, device=dev)
    if lvl > 0:
        low = torch.rand(n, device=dev) < band_prob
        lv[low] = torch.randint(0, lvl, (int(low.sum()),), device=dev)
    env.sit_start_level[env_ids] = lv
    for L in torch.unique(lv).tolist():
        ids = env_ids[lv == L]
        bank = env._lv_banks[L]
        m = len(ids)
        idx = torch.multinomial(bank["p"], m, replacement=True)
        quat = bank["quat"][idx]
        _, _, yaw_b = euler_xyz_from_quat(quat)
        zeros = torch.zeros(m, device=dev)
        yaw = torch.empty(m, device=dev).uniform_(*yaw_range)
        quat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw - yaw_b), quat)
        pos = env.scene.env_origins[ids].clone()
        pos[:, 0] += torch.empty(m, device=dev).uniform_(*xy_range)
        pos[:, 1] += torch.empty(m, device=dev).uniform_(*xy_range)
        pos[:, 2] += bank["z"][idx] + height_pad
        asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
        asset.write_root_velocity_to_sim(torch.zeros(m, 6, device=dev), env_ids=ids)
        lim = asset.data.soft_joint_pos_limits[ids]
        q = bank["q"][idx] + torch.empty_like(bank["q"][idx]).uniform_(-joint_noise, joint_noise)
        q = torch.minimum(torch.maximum(q, lim[..., 0]), lim[..., 1])
        asset.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        env.getup_fall_sign[ids] = bank["sign"][idx]


def sit_level_curriculum(env, env_ids, n_levels: int, success_threshold: float = 0.7, hold_frac: float = 0.25,
                         ema_alpha: float = 0.05, min_steps_per_level: int = 2400, reward_name: str = "stand_flag"):
    """SitStand 역순 커리큘럼 승격 (11-11): 지금 단계에서 시작한 판의 '서 있음' 누적 (stand_flag) 이 에피소드의 hold_frac 이상이면 성공.
    뒤로 누움 · 5 s 못 일어남 종료는 누적이 모자라 실패. EMA ≥ success_threshold 이고 min_steps 머물렀으면 다음 (더 낮은) 단계.
    getup_v1_mdp.reverse_getup_curriculum 과 같은 판정 · EMA (배치 무관 고정 alpha) · 내려가지 않음."""
    sums = getattr(env.reward_manager, "_episode_sums", None)
    level = int(getattr(env, "_sit_level", 0))
    ema = getattr(env, "_sit_ema", 0.0)
    since = getattr(env, "_sit_since", 0)
    if sums is not None and reward_name in sums and env_ids is not None and len(env_ids) > 0 and hasattr(env, "sit_start_level"):
        cur = env_ids[env.sit_start_level[env_ids] == level]
        if len(cur) > 0:
            w = env.reward_manager.get_term_cfg(reward_name).weight
            need = hold_frac * abs(w) * env.max_episode_length_s
            succ = (sums[reward_name][cur] >= need).float().mean().item()
            ema = (1.0 - ema_alpha) * ema + ema_alpha * succ
    if ema >= success_threshold and (env.common_step_counter - since) >= min_steps_per_level and level < n_levels - 1:
        level += 1
        ema = 0.0
        since = env.common_step_counter
    env._sit_level, env._sit_ema, env._sit_since = level, ema, since
    return {"level": torch.tensor(float(level), device=env.device), "success_rate": torch.tensor(float(ema), device=env.device)}


def lay_back_angle_curriculum(env, env_ids, angles: tuple, z_min: float = 0.0, success_threshold: float = 0.6, hold_frac: float = 0.25,
                              ema_alpha: float = 0.05, min_steps_per_level: int = 2400, reward_name: str = "stand_flag",
                              term_name: str = "lay_back"):
    """SitStand-V0-2 (11-12, 사용자 "처음에는 등대고 일어나게 했다가, 점점 뒤로 기대는 각도를 막는게 낫지않을까?"):
    시작은 다리 뻗고 앉음 그대로, 뒤로 누움 종료 각도를 angles[단계] 로 — 단계 0 은 사실상 없음 (기상 V0-1 의 '눕고 반동' 그대로 성공),
    '3 s 넘게 서 있음' EMA ≥ success_threshold · min_steps 머묾 → 다음 (더 좁은) 각도. 반동은 남기고 뒤로 넘어가는 폭만 줄여 간다.
    성공 판정 · EMA 는 sit_level_curriculum 과 같음 (모든 판이 같은 단계)."""
    sums = getattr(env.reward_manager, "_episode_sums", None)
    level = int(getattr(env, "_lb_level", 0))
    ema = getattr(env, "_lb_ema", 0.0)
    since = getattr(env, "_lb_since", 0)
    if sums is not None and reward_name in sums and env_ids is not None and len(env_ids) > 0:
        w = env.reward_manager.get_term_cfg(reward_name).weight
        need = hold_frac * abs(w) * env.max_episode_length_s
        ema = (1.0 - ema_alpha) * ema + ema_alpha * (sums[reward_name][env_ids] >= need).float().mean().item()
    if ema >= success_threshold and (env.common_step_counter - since) >= min_steps_per_level and level < len(angles) - 1:
        level += 1
        ema = 0.0
        since = env.common_step_counter
    env._lb_level, env._lb_ema, env._lb_since = level, ema, since
    cfg = env.termination_manager.get_term_cfg(term_name)
    cfg.params["pitch_deg"] = float(angles[level])
    cfg.params["z_min"] = z_min
    env.termination_manager.set_term_cfg(term_name, cfg)
    return {"level": torch.tensor(float(level), device=env.device), "angle": torch.tensor(float(angles[level]), device=env.device),
            "success_rate": torch.tensor(float(ema), device=env.device)}
