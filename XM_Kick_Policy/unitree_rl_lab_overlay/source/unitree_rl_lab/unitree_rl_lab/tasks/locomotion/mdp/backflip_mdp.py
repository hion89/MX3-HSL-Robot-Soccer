"""제자리 백덤블링 (RS02 Backflip-V0, 2026-11-08) — 서 있다가 → 웅크려 → 뛰어 뒤로 한 바퀴 → 두 발로 착지해 서기.

기준 궤적 (시각 t, 에피소드 시작 기준 — 공중 시작 (RSI) 에피소드는 flip_t0 만큼 앞당겨진 시각):
  골반 높이 z_ref : 서기 0.48 → (0.30~0.60) 웅크림 0.36 → (0.60~0.80) 밀기 0.50 · 2.2 m/s → (0.80~1.25) 포물선 (정점 ~0.75)
                   → (1.25~1.50) 착지 흡수 0.38 → (1.50~2.10) 일어서 0.48
  뒤로 돈 각 rho_ref : ROT_T0~ROT_T1 smoothstep 으로 0 → 2π (이륙 0.80 에 ~0.8 rad, 착지 1.25 에 ~6.2 rad)
  웅크림 tuck(t)   : 공중 (0.80~1.25) 에서 sin 으로 0 → 1 → 0 (RSI 관절 자세만)
상태 (env 에 붙임, _ensure):
  flip_rho      '뒤로 돈 각' [rad] = −∫ω_y dt (몸 좌표 y = 왼쪽, 뒤로 넘어가면 ω_y < 0). 실기에선 IMU 자이로 y 적분으로 같은 값.
  flip_t0       시각 앞당김 [s] (RSI). ep_time = 스텝 × dt + flip_t0
  flip_ok_run · flip_success  '다 돌고 두 발로 바로 섬' 이 이어진 시간 · 성공 (한 번)
  flip_assist   보조 배율 0~1 (커리큘럼이 성공률 보고 줄임)
보조 ('꼭두각시', flip_step): 보조 창 안에서 골반에 a × (높이 PD + m(g + z̈_ref) 앞먹임) 수직 힘 · a × (rho PD) 뒤로 도는 토크.
  a = 1 이면 정책이 아무것도 안 해도 기준대로 뜨고 돈다 → 처음부터 웅크리기 · 착지를 배울 기회. a 를 0 까지 줄이면 혼자 하는 것.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

TWO_PI = 2.0 * math.pi
G = 9.81
REF_DT = 0.001
T_TAKEOFF, T_LAND = 0.80, 1.25
V0 = 2.2
Z_STAND, Z_CROUCH, Z_TAKEOFF, Z_ABSORB = 0.48, 0.36, 0.50, 0.38


def _hermite(t, t0, t1, z0, v0, z1, v1):
    h = t1 - t0
    s = (t - t0) / h
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * z0 + h10 * h * v0 + h01 * z1 + h11 * h * v1


def _build_z_table(total_s: float = 4.0) -> np.ndarray:
    t = np.arange(0.0, total_s + REF_DT, REF_DT)
    z = np.full_like(t, Z_STAND)
    m = (t >= 0.30) & (t < 0.60)
    z[m] = _hermite(t[m], 0.30, 0.60, Z_STAND, 0.0, Z_CROUCH, 0.0)
    m = (t >= 0.60) & (t < T_TAKEOFF)
    z[m] = _hermite(t[m], 0.60, T_TAKEOFF, Z_CROUCH, 0.0, Z_TAKEOFF, V0)
    m = (t >= T_TAKEOFF) & (t < T_LAND)
    tau = t[m] - T_TAKEOFF
    z[m] = Z_TAKEOFF + V0 * tau - 0.5 * G * tau**2
    z_land = Z_TAKEOFF + V0 * (T_LAND - T_TAKEOFF) - 0.5 * G * (T_LAND - T_TAKEOFF) ** 2
    v_land = V0 - G * (T_LAND - T_TAKEOFF)
    m = (t >= T_LAND) & (t < 1.50)
    z[m] = _hermite(t[m], T_LAND, 1.50, z_land, v_land, Z_ABSORB, 0.0)
    m = (t >= 1.50) & (t < 2.10)
    z[m] = _hermite(t[m], 1.50, 2.10, Z_ABSORB, 0.0, Z_STAND, 0.0)
    return z


_Z = _build_z_table()
_TABLES: dict = {}


def _tables(dev):
    if dev not in _TABLES:
        z = torch.tensor(_Z, dtype=torch.float32, device=dev)
        v = torch.gradient(z, spacing=REF_DT)[0]
        a = torch.gradient(v, spacing=REF_DT)[0]
        _TABLES[dev] = (z, v, a)
    return _TABLES[dev]


def z_ref(t: torch.Tensor):
    """(z, ż, z̈) 기준 골반 높이"""
    z, v, a = _tables(t.device)
    i = (t / REF_DT).round().long().clamp(0, len(z) - 1)
    return z[i], v[i], a[i]


def rho_ref(t: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
    u = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
    return TWO_PI * u * u * (3.0 - 2.0 * u)


def rho_dot_ref(t: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
    u = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
    return TWO_PI * 6.0 * u * (1.0 - u) / (t1 - t0)


def tuck_ref(t: torch.Tensor) -> torch.Tensor:
    u = ((t - T_TAKEOFF) / (T_LAND - T_TAKEOFF)).clamp(0.0, 1.0)
    return torch.sin(math.pi * u)


def _ensure(env) -> None:
    if not hasattr(env, "flip_rho"):
        n, dev = env.num_envs, env.device
        env.flip_rho = torch.zeros(n, device=dev)
        env.flip_t0 = torch.zeros(n, device=dev)
        env.flip_maxz = torch.zeros(n, device=dev)
        env.flip_maxair = torch.zeros(n, device=dev)                     # Jump-V0: 두 발 다 뜬 최장 시간 (_jump_gate)
        env.flip_td = torch.full((n,), 1e6, device=dev)                  # Jump-V0-10: 착지 시각 (_touchdown, 없으면 1e6)
        env.flip_ok_run = torch.zeros(n, device=dev)
        env.flip_success = torch.zeros(n, dtype=torch.bool, device=dev)
        env.flip_rsi = torch.zeros(n, dtype=torch.bool, device=dev)
        env.flip_x0 = torch.zeros(n, 2, device=dev)                      # V0-3: 시작 방위 (수평 단위 벡터)
        env.flip_x0[:, 0] = 1.0
        env.flip_raw_prev = torch.zeros(n, device=dev)
        env.flip_need_init = torch.ones(n, dtype=torch.bool, device=dev)
        env.flip_assist = 1.0
        env.flip_succ_ema = 0.0
        env.flip_succ_count = 0


def ep_time(env) -> torch.Tensor:
    _ensure(env)
    return env.episode_length_buf.float() * env.step_dt + env.flip_t0


def _tilt(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0))


def _feet_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 5.0) -> torch.Tensor:
    """두 발 다 닿음 (N,) bool"""
    cs = env.scene.sensors[sensor_cfg.name]
    f = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (f > threshold).all(dim=1)


def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], dim=-1)


# ============================================================================
# 이벤트
# ============================================================================
def reset_flip_state(env, env_ids: torch.Tensor) -> None:
    _ensure(env)
    env.flip_rho[env_ids] = 0.0
    env.flip_t0[env_ids] = 0.0
    env.flip_maxz[env_ids] = 0.0
    env.flip_maxair[env_ids] = 0.0
    env.flip_td[env_ids] = 1e6
    env.flip_ok_run[env_ids] = 0.0
    env.flip_success[env_ids] = False
    env.flip_rsi[env_ids] = False
    q = env.scene["robot"].data.root_quat_w[env_ids]                  # reset_base 뒤 — 방위 (공중 시작은 yaw 를 그대로 둠)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    env.flip_x0[env_ids] = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    env.flip_need_init[env_ids] = True


def _geom_angle(q: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """몸 위 축 z_b 가 시작 방위 x0 의 시상면 (x0, 세계 위) 에서 뒤로 돈 각 [−π, π] — atan2(−z_b·x0, z_b·z_w).
    뒤로 넘어가면 z_b 가 −x0 쪽으로 → 양수, 거꾸로 = π. 몸 좌표 각속도 적분 (rho = −∫ω_y) 과 달리 지금 자세만의 함수라
    원뿔 · 대각선 회전이 '돈 각' 으로 쌓이지 않는다 (V0-2 2000: 옆으로 기운 축으로 돌아도 rho 는 2π)."""
    w, x, y, z = q.unbind(-1)
    zb = torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    return torch.atan2(-(zb[:, 0] * x0[:, 0] + zb[:, 1] * x0[:, 1]), zb[:, 2])


def lateral_axis_dev(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V0-3: 몸 옆 축 y_b 가 시작 옆 축 y0 (= x0 를 90° 돌린 수평) 에서 벗어난 정도 1 − (y_b·y0)² (음수 weight).
    깨끗한 백덤블링은 옆 축이 내내 수평 · 그대로 → 0. 옆으로 기울거나 (roll) 방위가 틀어지면 (yaw) 커짐."""
    _ensure(env)
    q = env.scene[asset_cfg.name].data.root_quat_w
    w, x, y, z = q.unbind(-1)
    yb = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
    y0 = torch.stack([-env.flip_x0[:, 1], env.flip_x0[:, 0]], -1)
    c = yb[:, 0] * y0[:, 0] + yb[:, 1] * y0[:, 1]
    return 1.0 - c * c


def target_limit_excess(env, action_name: str = "JointPositionAction", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V0-3: 관절 목표가 가동범위 (soft limit) 밖으로 나간 만큼의 제곱합 (음수 weight).
    V0-2 2000: 발목 roll (±20°) 목표를 ~2 rad 로 줘 한계에 내내 밀어붙임 (토크 최대 · 실기 포화 감지)."""
    term = env.action_manager.get_term(action_name)
    tgt = term.processed_actions
    lim = env.scene[asset_cfg.name].data.soft_joint_pos_limits[:, term._joint_ids]
    ex = (tgt - tgt.clamp(lim[..., 0], lim[..., 1])).abs()
    return torch.sum(ex * ex, dim=1)


def pose_track(env, window: tuple[float, float], delta: dict, joint_names: list, sigma: float,
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V0-4: 창 안에서 joint_names 관절이 (걷기 기본자세 + delta) 에 가까울수록 exp(−mean(오차²)/σ²).
    웅크림 (사용자 '처음 뛸 때 무릎을 굽혀 낮추는 게 아니라 이상한 자세로 낮춤' — V0-3 4000: 0.55 s 에 엉덩이 yaw 양쪽 −1.4~−1.6 ·
    왼 roll −1.1 로 다리를 비틀어 낮춤, 무릎은 −0.4~−0.7 만)."""
    t = ep_time(env)
    on = (t >= window[0]) & (t < window[1])
    return on.float() * torch.exp(-_pose_err(env, delta, joint_names, asset_cfg) / (sigma * sigma))


def _pose_err(env, delta: dict, joint_names: list, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint_names 관절이 (걷기 기본자세 + delta) 에서 벗어난 제곱 평균 (pose_track · land_squat)"""
    robot = env.scene[asset_cfg.name]
    if not hasattr(env, "_pose_track_cache"):
        env._pose_track_cache = {}
    key = (tuple(joint_names), tuple(sorted(delta.items())))
    if key not in env._pose_track_cache:
        ids = [robot.joint_names.index(n) for n in joint_names]
        tgt = robot.data.default_joint_pos[0, ids].clone()
        for n, v in delta.items():
            tgt[joint_names.index(n)] += v
        env._pose_track_cache[key] = (torch.tensor(ids, device=env.device), tgt)
    ids, tgt = env._pose_track_cache[key]
    return torch.mean(torch.square(robot.data.joint_pos[:, ids] - tgt), dim=1)


def landed_pose(env, after_t: float, margin: float, sigma: float, air_gate: tuple[float, float] | None = None,
                land_after: float | None = None) -> torch.Tensor:
    """V0-4: 다 돈 뒤 (rho ≥ 2π − margin, t ≥ after_t) 전 관절이 걷기 기본자세에 가까울수록 exp(−RMS²/σ²)
    (사용자 '최종 자세는 걷기랑 같아야')."""
    ok = _flipped(env, margin) & (ep_time(env) >= after_t) & _since_land(env, land_after)
    return ok.float() * torch.exp(-torch.square(_pose_rms(env) / sigma)) * _jump_gate(env, air_gate)


def hip_twist(env, joint_names: list, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V0-4: 엉덩이 yaw · roll 이 기본자세에서 벗어난 제곱합 (음수 weight, 에피소드 내내) — 깨끗한 백덤블링은 거의 0.
    V0-3 4000 은 웅크림에서 yaw 를 양쪽 같은 방향으로 ~90° 돌려 몸을 비틀었다 (공중 비틀기의 시작)."""
    robot = env.scene[asset_cfg.name]
    if not hasattr(env, "_hip_twist_ids"):
        env._hip_twist_ids = torch.tensor([robot.joint_names.index(n) for n in joint_names], device=env.device)
    ids = env._hip_twist_ids
    return torch.sum(torch.square(robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]), dim=1)


def feet_impact(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """V0-4: 발 접촉힘이 threshold [N] 넘은 만큼 /1000 (두 발 합, 센서 이력 = 이 스텝 물리 스텝들 중 최대) — 음수 weight.
    (사용자 '착지 시 충격 패널티') V0-3 4000 착지: 한 발 최대 중앙 4038 N · 최대 8289 N (몸무게 126 N)."""
    cs = env.scene.sensors[sensor_cfg.name]
    f = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return torch.sum((f - threshold).clamp(min=0.0), dim=1) / 1000.0


def foot_tilt(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, tol: float, contact_threshold: float = 5.0) -> torch.Tensor:
    """Jump-V0 (11-08): 발이 땅에 닿아 있는 동안 발바닥 기울기 (발 z 축 ↔ 세계 위, pitch · roll 다) 가 tol 넘은 만큼 제곱 합 (음수 weight).
    사용자 '웅크린 뒤 앞꿈치로 뛴다 — 실기에선 안 될 것': Backflip-V0-5 7000 은 0.55~0.70 s 발바닥 +19~21° (발목 0.87 = 한계) 로 이륙.
    발끝이 발목 pitch 축 앞 11.6 cm 라 앞꿈치로 이륙 추력 (~2.4 mg) 을 내면 발목 ~18 N·m > RS00 14 N·m.
    두 cfg 다 preserve_order=True 로 (접촉 센서 바디 순서 ≠ articulation 순서 — memory isaaclab-contact-sensor-body-order)."""
    q = env.scene[asset_cfg.name].data.body_quat_w[:, asset_cfg.body_ids]
    w, x, y, z = q.unbind(-1)
    tilt = torch.acos((1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0))
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    return torch.sum((f > contact_threshold).float() * torch.square((tilt - tol).clamp(min=0.0)), dim=1)


def raw_action_over_clip(env) -> torch.Tensor:
    """V0-3: 자르기 전 정책 출력이 learner_action_clip 을 넘은 만큼의 제곱합 (BackflipEnv.step 이 남김) — 음수 weight.
    (kick_switch_mdp.raw_action_over_clip 과 같음 — 자르기만 있으면 한계 밖 평균에 벌점이 없어 그쪽에 머문다)"""
    ex = getattr(env, "raw_action_excess", None)
    if ex is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(ex * ex, dim=1)


def reset_midair(env, env_ids: torch.Tensor, prob: float, t_range: tuple[float, float], rot_t0: float, rot_t1: float,
                 tuck_delta: dict, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    """RSI: prob 비율의 env 를 기준 궤적의 공중 시각 t ∈ t_range 상태로 (높이 · 수직 속도 · 뒤로 돈 각 · 회전 속도 · 웅크린 관절).
    reset_base · reset_joints · reset_flip_state · reset_hold 뒤에 둘 것 (홀드도 0 으로)."""
    _ensure(env)
    robot = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = env.device
    pick = torch.rand(n, device=dev) < prob
    ids = env_ids[pick]
    k = len(ids)
    if k == 0:
        return
    t = torch.empty(k, device=dev).uniform_(*t_range)
    z, vz, _ = z_ref(t)
    rho = rho_ref(t, rot_t0, rot_t1)
    rhod = rho_dot_ref(t, rot_t0, rot_t1)
    # 자세: 지금 (reset_base 가 준) 방위의 yaw 를 두고 몸 y 축으로 −rho (코가 들림 = 뒤로)
    q0 = robot.data.root_quat_w[ids]
    w, x, y, zq = q0.unbind(-1)
    yaw = torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
    q_yaw = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)
    q_p = torch.stack([torch.cos(-rho / 2), torch.zeros_like(rho), torch.sin(-rho / 2), torch.zeros_like(rho)], -1)
    q = _quat_mul(q_yaw, q_p)
    pos = robot.data.root_pos_w[ids].clone()
    pos[:, 2] = z
    # 몸 y 축 (세계) = q 로 돌린 (0, 1, 0)
    qw, qx, qy, qz = q.unbind(-1)
    y_axis = torch.stack([2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx)], -1)
    vel = torch.zeros(k, 6, device=dev)
    vel[:, 2] = vz
    vel[:, 3:6] = (-rhod)[:, None] * y_axis
    robot.write_root_pose_to_sim(torch.cat([pos, q], -1), env_ids=ids)
    robot.write_root_velocity_to_sim(vel, env_ids=ids)
    # 관절: 기본자세 + tuck × 변위
    jp = robot.data.default_joint_pos[ids].clone()
    s = tuck_ref(t)
    for name, dq in tuck_delta.items():
        j = robot.joint_names.index(name)
        jp[:, j] += s * dq
    jp = torch.max(torch.min(jp, robot.data.soft_joint_pos_limits[ids, :, 1]), robot.data.soft_joint_pos_limits[ids, :, 0])
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
    env.flip_rho[ids] = rho
    env.flip_t0[ids] = t
    env.flip_rsi[ids] = True
    if hasattr(env, "reset_hold_steps"):
        env.reset_hold_steps[ids] = 0


def set_assist(env, env_ids: torch.Tensor, value: float) -> None:
    """재생용 (reset 이벤트): 보조 배율 고정 — 영상에서 학습 중 보조 수준을 보려고 (BACKFLIP_PLAY_ASSIST)"""
    _ensure(env)
    env.flip_assist = float(value)


def flip_step(env, env_ids: torch.Tensor, window: tuple[float, float], rot_t0: float, rot_t1: float,
              kp_rot: float, kd_rot: float, tau_max: float, kp_z: float, kd_z: float, f_max: float,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="pelvis"), rot_mode: str = "gyro",
              z_mode: str = "pd", z_ff_k: float = 0.0) -> None:
    """매 스텝 (interval = step_dt): 뒤로 돈 각 · 최고 높이 · 꼭두각시 보조 (골반, 세계 좌표, a = env.flip_assist).
    rot_mode "gyro" (V0~V0-2): rho += −ω_y dt (몸 좌표 적분) / "geom" (V0-3): 시상면 기하 각 (_geom_angle) 을 펼쳐 쌓음.
    z_mode "pd" (V0~V0-5): 높이 = 앞먹임 m(g+z̈_ref) + PD 로 z_ref 따라감 / "ff" (11-08 시험): 일정한 위 힘 a·k·m·g (중력을 1−a·k 배로) —
      PD 는 정책이 덜 밀면 더 밀어 줘 자기 점프를 늘릴 압력을 지움 (5000~6500 정체). env.flip_assist_fz 에 이번 위 힘 (세계, N) 을 남김.
    진단용: env.flip_probe_fz (N, 세계 위) 가 있으면 더함 (backflip_assist_probe.py — '정책이 더 민' 흉내)."""
    _ensure(env)
    robot = env.scene[asset_cfg.name]
    d = robot.data
    w_y = d.root_ang_vel_b[env_ids, 1]
    if rot_mode == "geom":
        raw = _geom_angle(d.root_quat_w[env_ids], env.flip_x0[env_ids])
        init = env.flip_need_init[env_ids]
        dphi = torch.remainder(raw - env.flip_raw_prev[env_ids] + math.pi, TWO_PI) - math.pi
        env.flip_rho[env_ids] += torch.where(init, torch.zeros_like(dphi), dphi)
        env.flip_raw_prev[env_ids] = raw
        env.flip_need_init[env_ids] = False
    else:
        env.flip_rho[env_ids] += -w_y * env.step_dt
    env.flip_maxz[env_ids] = torch.maximum(env.flip_maxz[env_ids], d.root_pos_w[env_ids, 2])
    if not hasattr(env, "flip_mass"):
        env.flip_mass = robot.root_physx_view.get_masses().sum(dim=1).to(env.device)
    a = float(env.flip_assist)
    t = ep_time(env)[env_ids]
    on = ((t >= window[0]) & (t < window[1]) & ~env.flip_success[env_ids]).float() * a
    # 회전: rho PD → 몸 y 축 반대 방향 토크 (뒤로)
    rho_e = rho_ref(t, rot_t0, rot_t1) - env.flip_rho[env_ids]
    rhod_e = rho_dot_ref(t, rot_t0, rot_t1) - (-w_y)
    tau = (kp_rot * rho_e + kd_rot * rhod_e).clamp(-tau_max, tau_max) * on
    tq = torch.zeros(len(env_ids), 3, device=env.device)
    tq[:, 1] = -tau                                                  # 몸 y 축 반대 = 뒤로
    # 높이: 앞먹임 m (g + z̈_ref) + PD, 위로만 — 세계 위쪽 힘을 지금 자세로 몸 좌표로 바꿔 넣음.
    #   ⚠ is_global=True 는 못 씀: IsaacLab permanent wrench composer 가 리셋 뒤 첫 호출의 링크 자세를 캐시해 에피소드 내내 그걸로
    #   바꾼다 (wrench_composer._link_poses_updated) → 몸이 뒤집히면 '위' 힘이 아래로 민다 (11-08 추적: 거꾸로 선 순간 골반 낙하 · 발 5650 N).
    zr, vzr, azr = z_ref(t)
    m = env.flip_mass[env_ids]
    if z_mode == "ff":
        fz = (z_ff_k * m * G).clamp(0.0, f_max) * on
    else:
        fz = (m * (G + azr) + kp_z * (zr - d.root_pos_w[env_ids, 2]) + kd_z * (vzr - d.root_lin_vel_w[env_ids, 2])).clamp(0.0, f_max) * on
    if not hasattr(env, "flip_assist_fz"):
        env.flip_assist_fz = torch.zeros(env.num_envs, device=env.device)
    env.flip_assist_fz[env_ids] = fz
    fw = torch.zeros(len(env_ids), 3, device=env.device)
    fw[:, 2] = fz
    if getattr(env, "flip_probe_fz", None) is not None:
        fw[:, 2] += env.flip_probe_fz[env_ids]
    fc = quat_apply_inverse(d.root_quat_w[env_ids], fw)
    robot.set_external_force_and_torque(fc[:, None, :], tq[:, None, :], body_ids=asset_cfg.body_ids, env_ids=env_ids, is_global=False)


# ============================================================================
# 관측
# ============================================================================
def phase_obs(env, t0: float, t1: float, total_s: float) -> torch.Tensor:
    """(시각/total, 기준 회전/2π) — 정책이 언제 뛰고 언제 다 돌아야 하는지"""
    t = ep_time(env)
    return torch.stack([(t / total_s).clamp(0.0, 1.0), rho_ref(t, t0, t1) / TWO_PI], dim=-1)


def flip_progress_obs(env) -> torch.Tensor:
    """뒤로 돈 각 / 2π (실기: 자이로 y 적분)"""
    _ensure(env)
    return (env.flip_rho / TWO_PI).unsqueeze(-1)


def assist_obs(env) -> torch.Tensor:
    """비평가만: 보조 배율"""
    _ensure(env)
    return torch.full((env.num_envs, 1), float(env.flip_assist), device=env.device)


# ============================================================================
# 보상
# ============================================================================
def track_rotation(env, t0: float, t1: float, sigma: float) -> torch.Tensor:
    e = env.flip_rho - rho_ref(ep_time(env), t0, t1) if hasattr(env, "flip_rho") else torch.zeros(env.num_envs, device=env.device)
    return torch.exp(-torch.square(e / sigma))


def track_height(env, window: tuple[float, float], sigma: float) -> torch.Tensor:
    """창 안에서 골반 높이가 기준 (z_ref) 을 따라감"""
    t = ep_time(env)
    z = env.scene["robot"].data.root_pos_w[:, 2]
    on = (t >= window[0]) & (t < window[1])
    return on.float() * torch.exp(-torch.square((z - z_ref(t)[0]) / sigma))


def jump_height(env, window: tuple[float, float], stand_height: float, cap: float) -> torch.Tensor:
    """창 안에서 골반이 서 있는 높이보다 높은 만큼 (최대 cap)"""
    t = ep_time(env)
    z = env.scene["robot"].data.root_pos_w[:, 2]
    on = (t >= window[0]) & (t < window[1])
    return torch.where(on, (z - stand_height).clamp(0.0, cap), torch.zeros_like(z))


def push_up_vel(env, window: tuple[float, float], cap: float) -> torch.Tensor:
    """Jump-V0-1 (11-08): 창 안 (미는 구간) 골반 위 속도 [m/s] (0 ~ cap) — 조밀한 '세게 밀기' 기울기.
    Jump-V0 500: 첫 점프 전엔 뛰기 쪽 보상이 다 0 (jump_height 는 골반 > 서 있는 높이, track_height σ 0.08 은 정점에서 멀면 0) 이라
    '웅크리고 가만히' 에 갇힘. 웅크림에서 일어서기만 해도 받고 빨리 밀수록 커져 이륙으로 이어짐."""
    t = ep_time(env)
    vz = env.scene["robot"].data.root_lin_vel_w[:, 2]
    on = (t >= window[0]) & (t < window[1])
    return torch.where(on, vz.clamp(0.0, cap), torch.zeros_like(vz))


def _flipped(env, margin: float) -> torch.Tensor:
    return env.flip_rho >= TWO_PI - margin


_GATE_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def _jump_gate(env, gate: tuple[float, float] | None):
    """Jump-V0 (11-08): 착지 보상을 뛴 만큼 — 두 발이 다 떠 있던 최장 시간 (접촉 센서 current_air_time 의 두 발 최소, t > 0.3 s) 이
    gate[0] → gate[1] 초에서 0 → 1. None 이면 1. 착지 보상을 쓰는 항마다 매 스텝 불려 최댓값을 갱신 (같은 스텝 여러 번 불려도 같음).
    '다 돌았으면' 조건이 없는 점프에선 착지 보상 (~20 점) 이 안 뛰고 서 있어도 나와 '안 뛰기' 가 최적 (첫 판 3 iter 에서 멈춤).
    골반 최고로 거르면 다리 펴고 앞꿈치로 서기 (~0.59 m) 로 절반을 받음 → 꾸밀 수 없는 공중 시간으로 (memory gate-must-not-be-manipulable)."""
    if gate is None:
        return 1.0
    _air_now(env)
    return ((env.flip_maxair - gate[0]) / (gate[1] - gate[0])).clamp(0.0, 1.0)


def _air_now(env) -> torch.Tensor:
    """지금 두 발 다 떠 있는 시간 (current_air_time 두 발 최소, t ≤ 0.3 s 는 0) — flip_maxair 갱신"""
    _ensure(env)
    cs = env.scene.sensors["contact_forces"]
    if not hasattr(env, "_gate_feet_ids"):
        env._gate_feet_ids = [cs.body_names.index(n) for n in _GATE_FEET]
    air = cs.data.current_air_time[:, env._gate_feet_ids].min(dim=1)[0]
    air = torch.where(ep_time(env) > 0.3, air, torch.zeros_like(air))
    env.flip_maxair[:] = torch.maximum(env.flip_maxair, air)
    return air


def _touchdown(env, min_air: float = 0.1) -> torch.Tensor:
    """Jump-V0-10 (11-09): 착지 시각 — 두 발 다 뜬 시간이 min_air 넘은 뒤 한 발이라도 처음 닿은 ep_time (없으면 1e6).
    착지 보상을 고정 시각 (1.35 s) 대신 착지 기준으로 — 점프 높이 · 이륙 시각이 바뀌어도 흡수 구간이 같이 움직임.
    매 스텝 불려야 함 (land_squat 이 weight ≠ 0 이라 늘 불림). 같은 스텝 여러 번 불려도 같음."""
    air = _air_now(env)
    new = (env.flip_maxair >= min_air) & (air <= 0.0) & (env.flip_td > 1e5)
    env.flip_td[:] = torch.where(new, ep_time(env), env.flip_td)
    return env.flip_td


def _since_land(env, land_after: float | None) -> torch.Tensor:
    """land_after 가 있으면 착지 뒤 land_after 초 지났는가 (N,) bool, 없으면 늘 참"""
    if land_after is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return ep_time(env) - _touchdown(env) >= land_after


def land_squat(env, window: tuple[float, float], delta: dict, joint_names: list, sigma: float,
               air_gate: tuple[float, float] | None = None) -> torch.Tensor:
    """Jump-V0-10 (11-09): 착지 뒤 window 초 (착지 기준) 다리가 스쿼트 자세 (기본 + delta) 에 가까울수록 exp(−오차²/σ²) × 공중 시간 문.
    사용자 (11-09 아침): "착지 때 처음 뛸 때처럼 스쿼트로 완충하면 좋겠는데 거의 딱딱하게 착지, 무릎도 조금밖에 안 굽힘".
    V0-3 3000 · 5000 (jump_landing_probe): 이륙 전 웅크림 골반 0.31 m · 무릎 굽힘 1.52 rad 인데 착지는 골반 2.6~4.0 cm 만 내려가고
    무릎 최대 0.78~0.83 (40~60 ms) → 120~140 ms 에 발 힘 0 (튀어 오름). 발 힘 = m(g + v²/2d) — 흡수 거리 d 4 → 13 cm 면 받치는 힘 ~1/2."""
    s = ep_time(env) - _touchdown(env)
    on = (s >= window[0]) & (s < window[1])
    return on.float() * torch.exp(-_pose_err(env, delta, joint_names) / (sigma * sigma)) * _jump_gate(env, air_gate)


def landed_upright(env, after_t: float, margin: float, sigma: float, sensor_cfg: SceneEntityCfg,
                   air_gate: tuple[float, float] | None = None, land_after: float | None = None) -> torch.Tensor:
    """다 돈 뒤 (rho ≥ 2π − margin, t ≥ after_t) 두 발 딛고 바로 섬 — exp(−(기울기/σ)²). land_after: 착지 뒤 그만큼 지나서부터 (V0-10)"""
    ok = _flipped(env, margin) & (ep_time(env) >= after_t) & _feet_contact(env, sensor_cfg) & _since_land(env, land_after)
    return ok.float() * torch.exp(-torch.square(_tilt(env) / sigma)) * _jump_gate(env, air_gate)


def landed_height(env, after_t: float, margin: float, target: float, sigma: float,
                  air_gate: tuple[float, float] | None = None, land_after: float | None = None) -> torch.Tensor:
    z = env.scene["robot"].data.root_pos_w[:, 2]
    ok = _flipped(env, margin) & (ep_time(env) >= after_t) & _since_land(env, land_after)
    return ok.float() * torch.exp(-torch.square((z - target) / sigma)) * _jump_gate(env, air_gate)


def feet_stance(env, after_t: float, margin: float, sigma: float, width0: float, x0: float, asset_cfg: SceneEntityCfg,
                air_gate: tuple[float, float] | None = None) -> torch.Tensor:
    """Jump-V0-4 (11-09): 착지 뒤 (t ≥ after_t) 발 위치가 기본자세 발 위치에 가까울수록 exp(−(Δ폭² + Δ앞²)/σ²) × 공중 시간 문.
    골반 방위 (yaw 만) 좌표: 폭 = 왼발 − 오른발 옆 거리, 앞 = 두 발 평균 앞 거리 (발 링크 원점, 기본자세 폭 0.23 · 앞 +0.072 m — MuJoCo 같은 USD).
    V0-2 2500 · V0-3 3000 은 발 폭 33~46 cm · 앞 +2.6~5.3 cm 로 착지 → 반납 뒤 브리지가 관절만 기본자세로 당기면 발이 붙어 있어 몸이 끌려 넘어짐
    (MuJoCo 실기 맞춤 b_real 6/30 · 겹침 28/30). 발은 공중에서 옮길 수 있으니 착지 전에 기본자리로 (사용자 '최종 자세는 걷기랑 같아야')."""
    r = env.scene[asset_cfg.name]
    d = r.data.body_pos_w[:, asset_cfg.body_ids] - r.data.root_pos_w[:, None, :]
    w, x, y, z = r.data.root_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s_ = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    fx = d[..., 0] * c + d[..., 1] * s_
    fy = -d[..., 0] * s_ + d[..., 1] * c
    e2 = torch.square(fy[:, 0] - fy[:, 1] - width0) + torch.square(fx.mean(dim=1) - x0)
    ok = _flipped(env, margin) & (ep_time(env) >= after_t)
    return ok.float() * torch.exp(-e2 / (sigma * sigma)) * _jump_gate(env, air_gate)


def landed_still(env, after_t: float, margin: float) -> torch.Tensor:
    """다 돈 뒤 몸 각속도 제곱 (음수 weight) — 착지 뒤 가라앉히기"""
    ok = _flipped(env, margin) & (ep_time(env) >= after_t)
    w = env.scene["robot"].data.root_ang_vel_b
    return ok.float() * torch.sum(torch.square(w), dim=1)


def landed_linvel(env, after_t: float, margin: float) -> torch.Tensor:
    """Jump-V0-9 (11-09): 착지 뒤 골반 속도 제곱 (음수 weight) — V0-8 은 5 s 가 끝나도 0.1~0.3 m/s 로 계속 움직여 (MuJoCo) 조용히 반납을 못 함"""
    ok = _flipped(env, margin) & (ep_time(env) >= after_t)
    return ok.float() * torch.sum(torch.square(env.scene["robot"].data.root_lin_vel_w), dim=1)


def _pose_rms(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """전 관절이 걷기 기본자세에서 벗어난 RMS [rad]"""
    d = env.scene[asset_cfg.name].data
    return torch.sqrt(torch.mean(torch.square(d.joint_pos - d.default_joint_pos), dim=1))


def flip_success_bonus(env, after_t: float, margin: float, max_tilt: float, min_height: float, hold_s: float,
                       sensor_cfg: SceneEntityCfg, min_peak: float | None = None, pose_tol: float | None = None,
                       lat_tol: float | None = None, max_angvel: float | None = None, max_linvel: float | None = None) -> torch.Tensor:
    """다 돌고 (rho ≥ 2π − margin) 두 발로 바로 (기울기 < max_tilt, 골반 > min_height) hold_s 이어 서면 그 스텝 한 번 1.0.
    공중 시작 (RSI) 에피소드도 성공은 표시하지만 커리큘럼 성공률은 서서 시작한 에피소드만 (assist_curriculum)."""
    _ensure(env)
    z = env.scene["robot"].data.root_pos_w[:, 2]
    good = (_flipped(env, margin) & (ep_time(env) >= after_t) & _feet_contact(env, sensor_cfg)
            & (_tilt(env) < max_tilt) & (z > min_height))
    if min_peak is not None:                                         # V0-3: 높이 뛰었어야 (골반 최고 ≥ min_peak)
        good &= env.flip_maxz >= min_peak
    if pose_tol is not None:                                         # V0-4: 걷기 기본자세로 섬 (사용자 '최종 자세는 걷기랑 같아야')
        good &= _pose_rms(env) < pose_tol
    if lat_tol is not None:                                          # V0-4: 처음 방위 그대로 (비틀어 돌기 막기)
        good &= lateral_axis_dev(env) < lat_tol
    if max_angvel is not None:                                       # Jump-V0-4: 조용히 섬 (반납 뒤 브리지 기본자세로 넘길 수 있게)
        good &= env.scene["robot"].data.root_ang_vel_b.norm(dim=1) < max_angvel
    if max_linvel is not None:
        good &= env.scene["robot"].data.root_lin_vel_w.norm(dim=1) < max_linvel
    env.flip_ok_run[:] = torch.where(good, env.flip_ok_run + env.step_dt, torch.zeros_like(env.flip_ok_run))   # 제자리 (추론 모드 텐서 섞임 방지)
    new = (env.flip_ok_run >= hold_s) & ~env.flip_success
    env.flip_success |= new
    return new.float()


def over_rotation(env, margin: float) -> torch.Tensor:
    """2π + margin 넘게 돈 만큼 (음수 weight) — 한 바퀴 반 · 두 바퀴 막기"""
    _ensure(env)
    return (env.flip_rho - (TWO_PI + margin)).clamp(min=0.0)


# ============================================================================
# 종료
# ============================================================================
def no_flip_deadline(env, deadline_s: float, min_rho: float) -> torch.Tensor:
    """deadline_s 까지 min_rho 도 못 돌았으면 끝 (서 있기만 하는 국소해 끊기)"""
    _ensure(env)
    return (ep_time(env) >= deadline_s) & (env.flip_rho < min_rho)


# ============================================================================
# 커리큘럼
# ============================================================================
def assist_curriculum(env, env_ids: torch.Tensor, up_rate: float, step: float, min_samples: int = 2048,
                      ema_window: float = 4096.0, start: float = 1.0, criterion: str = "bonus", margin: float = 0.6) -> float:
    """끝나는 '서서 시작한' 에피소드의 성공률 EMA → up_rate 넘으면 보조를 step 만큼 줄임 (0 까지). 반환값 = 보조 배율 (로그 Curriculum/assist).
    criterion: "bonus" (V0) = 성공 보너스 (두 발로 바로 0.5 s) 받음 / "survive" (V0-1) = 다 돌고 (rho ≥ 2π − margin) 끝까지 (time_out) 버팀.
      V0 은 보조 1 에서 정책 행동이 상관없어 탐험 잡음 std 가 1.0 → 1.6 으로 커지고, 그 잡음에 '두 발 0.5 s' 가 끊겨 성공 0 → 보조가 안 줄었다."""
    _ensure(env)
    if not hasattr(env, "_flip_assist_init"):
        env.flip_assist = float(start)
        env._flip_assist_init = True
    if len(env_ids) == 0:
        return float(env.flip_assist)
    done = (env.episode_length_buf[env_ids] > 0) & ~env.flip_rsi[env_ids]       # 시작 리셋 · 공중 시작 제외
    if bool(done.any()):
        if criterion == "survive":
            ok = env.termination_manager.time_outs[env_ids] & (env.flip_rho[env_ids] >= TWO_PI - margin)
        else:
            ok = env.flip_success[env_ids]
        rate = ok[done].float().mean().item()
        kk = int(done.sum())
        w = ema_window / (ema_window + kk)
        env.flip_succ_ema = w * env.flip_succ_ema + (1.0 - w) * rate if env.flip_succ_count else rate
        env.flip_succ_count += kk
        if env.flip_succ_count >= min_samples and env.flip_succ_ema > up_rate and env.flip_assist > 0.0:
            prev = env.flip_assist
            env.flip_assist = max(0.0, env.flip_assist - step)
            env.flip_succ_count = 0
            print(f"[backflip] 보조 {prev:.2f} → {env.flip_assist:.2f} (서서 시작 성공률 EMA {env.flip_succ_ema:.2f})", flush=True)
    return float(env.flip_assist)


def assist_curriculum_v2(env, env_ids: torch.Tensor, up_rate: float, down_rate: float, step: float, window: int = 2048,
                         start: float = 1.0, margin: float = 0.6, min_peak: float | None = None,
                         pose_tol: float | None = None, lat_tol: float | None = None) -> float:
    """V0-2 (11-08 17:25): 보조 커리큘럼 — 마지막으로 바꾼 뒤 끝난 '서서 시작' 에피소드 window 개의 성공률 (단순 평균) 로
    > up_rate 면 보조 −step, < down_rate 면 +step (0~1). 성공 = 다 돌고 (rho ≥ 2π − margin) 끝까지 (time_out) 버팀.
    V0-1 의 EMA 는 바꾼 직후 첫 작은 묶음 (몇 env) 의 성공률로 다시 시작해, 운 좋게 1 이면 창 4096 으로는 2048 개 뒤에도
    ~0.61 이상 남아 실제 성공 ~0 인데도 0.5 를 넘겼다 (500 iter: 보조 1 → 0.3, 결정적 정책은 보조 0.3 에서 0.13 바퀴).
    반환값 = 보조 배율 (로그 Curriculum/assist). 성공률은 env.flip_succ_rate (로그 success_rate)."""
    _ensure(env)
    if not hasattr(env, "_flip_assist_init"):
        env.flip_assist = float(start)
        env._flip_assist_init = True
        env.flip_win_ok, env.flip_win_n, env.flip_succ_rate = 0, 0, 0.0
    if len(env_ids) == 0:
        return float(env.flip_assist)
    done = (env.episode_length_buf[env_ids] > 0) & ~env.flip_rsi[env_ids]
    if bool(done.any()):
        ok = env.termination_manager.time_outs[env_ids] & (env.flip_rho[env_ids] >= TWO_PI - margin)
        if min_peak is not None:
            ok &= env.flip_maxz[env_ids] >= min_peak
        if pose_tol is not None:                                     # V0-4: 끝 상태가 걷기 기본자세
            ok &= _pose_rms(env)[env_ids] < pose_tol
        if lat_tol is not None:                                      # V0-4: 끝 방위가 처음 그대로
            ok &= lateral_axis_dev(env)[env_ids] < lat_tol
        env.flip_win_ok += int(ok[done].sum())
        env.flip_win_n += int(done.sum())
        if env.flip_win_n >= window:
            rate = env.flip_win_ok / env.flip_win_n
            env.flip_succ_rate = rate
            env.flip_succ_ema = rate
            prev = env.flip_assist
            if rate > up_rate:
                env.flip_assist = max(0.0, env.flip_assist - step)
            elif rate < down_rate:
                env.flip_assist = min(1.0, env.flip_assist + step)
            if env.flip_assist != prev:
                print(f"[backflip] 보조 {prev:.2f} → {env.flip_assist:.2f} (서서 시작 성공률 {rate:.2f}, {env.flip_win_n} 판)", flush=True)
            env.flip_win_ok, env.flip_win_n = 0, 0
    return float(env.flip_assist)


def success_rate_metric(env) -> torch.Tensor:
    """로그용 (weight 0 이면 호출 안 됨 — memory isaaclab-weight0-reward-skipped): 아주 작은 weight 로"""
    _ensure(env)
    return torch.full((env.num_envs,), float(env.flip_succ_ema), device=env.device)


# ============================================================================
# 환경 (V0-3): 액션을 환경 안에서 ±learner_action_clip 으로 자르고 초과분을 raw_action_excess 로 남김
#   (kick_switch_mdp.KickSwitchEnv 와 같은 방식). 러너 clip_actions 는 크게 (100) — 자르기가 러너에 있으면 환경은 초과를 못 본다.
#   관측 last_action = 자른 값 (실기 노드 action_clip 8 · 이력 자르기와 같음).
# ============================================================================
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402


class BackflipEnv(ManagerBasedRLEnv):
    def step(self, action: torch.Tensor):
        clip = getattr(self.cfg, "learner_action_clip", None)
        if clip:
            self.raw_action_excess = (action.abs() - clip).clamp(min=0.0)
            action = action.clamp(-clip, clip)
        ho = getattr(self.cfg, "handover_t", None)
        if ho is not None:
            # Jump-V0-5 (11-09): 넘겨주기 연습 — t ≥ handover_t 면 액션 0 (= 관절 목표 기본자세, 실기 브리지가 반납 뒤 잡는 자세).
            #   정책이 '기본자세로 붙잡아도 서 있는' 상태를 만들어 두게. 직전 액션 관측도 0 (노드 jump_core 와 같게).
            t = self.episode_length_buf.float() * self.step_dt + (self.flip_t0 if hasattr(self, "flip_t0") else 0.0)
            action = torch.where((t >= ho - 1e-6)[:, None], torch.zeros_like(action), action)
        return super().step(action)

