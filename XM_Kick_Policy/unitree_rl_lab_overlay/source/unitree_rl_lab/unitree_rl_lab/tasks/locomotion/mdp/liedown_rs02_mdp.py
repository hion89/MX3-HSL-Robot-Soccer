"""RS02 뒤로 눕기 (Unitree-RS02-LieDown-V0) 전용 — 서 있는 상태에서 쪼그려 앉기 → 엉덩이 → 등 순서로 천천히 반듯이 눕기.

2026-11-09 사용자: "서있는상태에서 안정적으로 눕기로 하자. 일단 뒤로 눕기". 결정 (사용자 선택):
  끝 자세 = 반듯이, 다리 펴고 팔 옆 (기상 은행 flat 반듯이와 같은 자세 → 기상 정책이 바로 이어받기 좋음)
  내려가는 방식 = 쪼그려 앉기 → 엉덩이 먼저 바닥 → 등을 천천히 (엉덩이·등 접촉 순서를 보상으로 유도, 충격 최소)

관측 · 액션 · 출력 자르기 (±100) · 관절 목표 속도 제한 (2.5 rad/s) 은 기상 (getup_rs02_mdp) 그대로 → 실기 getup_core 를 그대로 쓴다.
골반 = 몸통 (한 몸체) 이라 '엉덩이가 먼저' 는 접촉 위치가 아니라 '골반이 바닥에 처음 닿을 때 몸통이 거의 서 있었나' 로 가른다.

에피소드마다 상태 (env._ld_*) 를 reset_liedown_state 가 지우고, liedown_tracker (보상 항 맨 앞, weight 1e-6) 가 매 스텝 갱신한다.
  _ld_squat    : 쪼그림 래치 — 몸통 pitch −back ~ +fwd 안인 채 골반 < deep_z 를 한 번이라도
  _ld_sat      : 앉음 래치 — 쪼그림 래치 뒤 골반이 처음 바닥에 닿음
  _ld_fellback : 쪼그림 없이 골반이 바닥에 닿음 (= 그냥 뒤로 넘어짐)
  접촉은 전부 바닥 필터 센서 (ground_contact, filter = 바닥 충돌체) — 5 차 (11-09 18:1x): contact_forces 의 골반 '접촉' 은 팔이 몸통에
  닿는 자기 접촉까지 세서 (서 있을 때 7~10 N) 1~4 차 앉음 판정이 엉터리였다. 쪼그림 시험 (squat_probe) 의 '서 있는 채 엉덩이 닿음' 도
  같은 착시 — 실제 바닥 접촉 땐 몸통이 뒤로 ~60° (기구상 깊게 쪼그리면 균형이 뒤로 넘어간다: 엉덩이 굽힘 90° · 발목 50° 한계).
  _ld_sat_new  : 이번 스텝에 처음 앉음 (일회 보상용) · _ld_fell_new : 이번 스텝에 처음 뒤로 넘어짐 (일회 벌점용)

2 차 (11-09 17:2x): 1 차 500 iter 가 '그냥 뒤로 넘어지기' (뒤로 넘어짐 100 % · 골반 2265 N · 하강 1.38 m/s) 로 빠짐 — 누움 ×0.3 이
  lying 에만 걸려 lie_pose · lie_still 은 넘어져도 다 받고, 충격 벌점은 −0.3 / 에피소드로 무의미. → 누움 3 항 모두 ×0.3 ·
  squat_progress (앉기 전 몸통 세운 채 골반 내리기) · fellback_event (뒤로 넘어진 순간 일회 벌점) · 충격 가중치 ×2.5.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from .getup_v1_mdp import hold_masked


def _tilt_cos(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 직립 코사인 = −g_z (1 서 있음 · 0 누움)."""
    return -env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]


def _supine_frac(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """등이 바닥 쪽인 정도 = −g_x (1 반듯이 누움 · 0 서 있음 · 음수 엎드림). RS02 몸 x 축 = 앞."""
    return -env.scene[asset_cfg.name].data.projected_gravity_b[:, 0]


def _lie_frac(env, face_down: bool) -> torch.Tensor:
    """눕기 방향 쪽으로 누운 정도 — 뒤로 눕기 −g_x (반듯이 1), 앞으로 눕기 (11-10, face_down) +g_x (엎드림 1)."""
    gx = env.scene["robot"].data.projected_gravity_b[:, 0]
    return gx if face_down else -gx


def _base_z(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _peak_force(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """접촉 힘 크기의 최근 history 최대 [N] (몸체별). 센서 update_period = sim dt, history ≥ decimation 이면 서브스텝 충격까지 잡힌다.
    바닥 필터 센서 (filter_prim_paths_expr 있음) 면 바닥과의 힘만 (force_matrix_w_history), 아니면 net (자기 접촉 포함)."""
    s = env.scene.sensors[sensor_cfg.name]
    if s.data.force_matrix_w_history is not None:
        return s.data.force_matrix_w_history[:, :, sensor_cfg.body_ids].sum(dim=3).norm(dim=-1).max(dim=1).values
    return s.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1).values


def _ensure_state(env):
    if not hasattr(env, "_ld_sat"):
        n, dev = env.num_envs, env.device
        env._ld_squat = torch.zeros(n, dtype=torch.bool, device=dev)
        env._ld_squat_new = torch.zeros(n, dtype=torch.bool, device=dev)
        env._ld_sat = torch.zeros(n, dtype=torch.bool, device=dev)
        env._ld_fellback = torch.zeros(n, dtype=torch.bool, device=dev)
        env._ld_sat_new = torch.zeros(n, dtype=torch.bool, device=dev)
        env._ld_fell_new = torch.zeros(n, dtype=torch.bool, device=dev)
        env._ld_lay = torch.zeros(n, dtype=torch.bool, device=dev)


# ═══════════════════════════════════════════════════════════════
#  리셋 · 상태 갱신
# ═══════════════════════════════════════════════════════════════
def reset_liedown_state(env, env_ids):
    _ensure_state(env)
    env._ld_squat[env_ids] = False
    env._ld_squat_new[env_ids] = False
    env._ld_sat[env_ids] = False
    env._ld_fellback[env_ids] = False
    env._ld_sat_new[env_ids] = False
    env._ld_fell_new[env_ids] = False
    env._ld_lay[env_ids] = False
    if hasattr(env, "_ld_push_t"):
        env._ld_push_t[env_ids] = -1.0e3
    if hasattr(env, "_ld_jerk_init"):
        env._ld_jerk_init[env_ids] = True


def _pitch_deg(env) -> torch.Tensor:
    """몸통 pitch [deg], 앞으로 숙임 +, 뒤로 젖힘 −."""
    g = env.scene["robot"].data.projected_gravity_b
    return torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))


def liedown_tracker(
    env,
    deep_z: float,
    back_deg: float,
    fwd_deg: float,
    lie_z: float,
    lie_supine: float,
    pelvis_cfg: SceneEntityCfg,
    contact_threshold: float = 5.0,
) -> torch.Tensor:
    """매 스텝 쪼그림 · 앉음 · 뒤로 넘어짐 · 누움 상태를 갱신 (보상 항 맨 앞에 두어 다른 항보다 먼저 돈다). 반환 0 (기록용 w 1e-6).
    pelvis_cfg 는 바닥 필터 센서 (ground_contact) 여야 한다."""
    _ensure_state(env)
    hold = getattr(env, "getup_hold_active", None)
    live = torch.ones(env.num_envs, dtype=torch.bool, device=env.device) if hold is None else ~hold
    z = _base_z(env)
    pitch = _pitch_deg(env)
    pelvis_on = _peak_force(env, pelvis_cfg).max(dim=1).values > contact_threshold
    squat_now = live & (z < deep_z) & (pitch > -back_deg) & (pitch < fwd_deg) & ~pelvis_on & ~env._ld_fellback
    env._ld_squat_new = squat_now & ~env._ld_squat
    env._ld_squat |= squat_now
    sit_now = live & pelvis_on & env._ld_squat
    fell_now = live & pelvis_on & ~env._ld_squat & ~env._ld_sat
    env._ld_sat_new = sit_now & ~env._ld_sat & ~env._ld_fellback
    env._ld_sat |= sit_now & ~env._ld_fellback
    env._ld_fell_new = fell_now & ~env._ld_fellback
    env._ld_fellback |= fell_now
    env._ld_lay = (z < lie_z) & (_supine_frac(env) > lie_supine)
    return torch.zeros(env.num_envs, device=env.device)


# ═══════════════════════════════════════════════════════════════
#  과제 보상
# ═══════════════════════════════════════════════════════════════
@hold_masked
def sit_bonus(env) -> torch.Tensor:
    """처음 앉은 스텝에 1/dt (가중치 w 면 에피소드에 w 한 번). 뒤로 넘어진 뒤엔 앉아도 없음."""
    _ensure_state(env)
    return env._ld_sat_new.float() / env.step_dt


@hold_masked
def lying_after(
    env,
    t_from: float,
    supine_height: float,
    height_std: float,
    unsat_scale: float,
    face_down: bool = False,
) -> torch.Tensor:
    """t_from [s] 뒤부터 '반듯이 누워 있음' 보상: clamp((−g_x − 0.5)/0.5) × exp(−(z − 누운 높이)²/std²).
    t_from 전엔 0 → 빨리 눕는 이득이 없다 (서두를 이유를 없애고 '어떻게' 는 다른 항이 정함).
    앉기 전에 그냥 뒤로 넘어졌으면 (_ld_fellback) × unsat_scale — 쪼그려 앉기 → 엉덩이 → 등 순서를 강제."""
    _ensure_state(env)
    t = env.episode_length_buf * env.step_dt
    sup = torch.clamp((_lie_frac(env, face_down) - 0.5) / 0.5, 0.0, 1.0)   # face_down (앞으로 눕기): 엎드림 쪽
    low = torch.exp(-torch.square(torch.clamp(_base_z(env) - supine_height, min=0.0)) / height_std**2)
    scale = torch.where(env._ld_fellback, torch.full_like(sup, unsat_scale), torch.ones_like(sup))
    return (t >= t_from).float() * sup * low * scale


@hold_masked
def squat_bonus(env) -> torch.Tensor:
    """처음 쪼그림 래치가 선 스텝에 1/dt (일회). 6 차: 쪼그린 뒤 매초 주던 squat (keep_after_squat) 은 '쪼그린 채 버티기' 로 쌓여서 일회로."""
    _ensure_state(env)
    return env._ld_squat_new.float() / env.step_dt


@hold_masked
def recline_after_squat(env) -> torch.Tensor:
    """쪼그림 래치 뒤 (뒤로 넘어짐 아님) 등이 바닥 쪽으로 젖혀진 정도 clamp(−g_x, 0, 1) — 7 차: 쪼그린 채 버티는 결정적 정책에
    '뒤로 주저앉아 눕기' 쪽 길잡이 (6 차 2400 명목: 쪼그림 100 % · 누움 0 %, 탐험 중에만 ~10 %). 누우면 1 이 최대라 쌓을 수 없음."""
    _ensure_state(env)
    return (env._ld_squat & ~env._ld_fellback).float() * torch.clamp(_supine_frac(env), 0.0, 1.0)


@hold_masked
def fellback_event(env) -> torch.Tensor:
    """앉기 전에 뒤로 넘어진 스텝에 1/dt (가중치 w 면 에피소드에 w 한 번)."""
    _ensure_state(env)
    return env._ld_fell_new.float() / env.step_dt


@hold_masked
def squat_progress(env, stand_height: float, sit_z: float, back_deg: float, fwd_deg: float, keep_after_sit: bool = False,
                   keep_after_squat: bool = False) -> torch.Tensor:
    """앉기 전 (래치 전 · 뒤로 넘어지기 전) 몸통 pitch 가 −back_deg ~ +fwd_deg (앞 +) 안인 채 골반을 내린 비율 (0 서 있음 · 1 앉음 높이).
    '그냥 뒤로 넘어지기' 와 '쪼그려 앉기' 를 가르는 탐색 길잡이 — 앉고 나면 (sit_bonus 받은 뒤) 0.
    3 차: 깊게 쪼그리면 몸통이 앞으로 25~42° 숙는다 (squat_probe) → 2 차의 대칭 30° 는 깊은 쪼그림에서 보상을 끊었다. 앞은 넉넉히.
    4 차: keep_after_sit — 앉은 뒤에도 1 (3 차는 앉는 순간 0 이 돼 '앉음 높이 위에 떠서 쪼그린 채 버티기' 가 국소해). sit_z = 실제 엉덩이 닿는 높이.
    5 차: keep_after_squat — 쪼그림 래치 (_ld_squat) 뒤엔 1 (깊게 쪼그린 뒤 엉덩이로 뒤로 주저앉는 동안 몸통이 넘어가도 안 잃음). sit_z = 래치 높이."""
    _ensure_state(env)
    g = env.scene["robot"].data.projected_gravity_b
    pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
    upright = (pitch > -back_deg) & (pitch < fwd_deg)
    frac = torch.clamp((stand_height - _base_z(env)) / (stand_height - sit_z), 0.0, 1.0)
    v = (upright & ~env._ld_sat & ~env._ld_fellback).float() * frac
    if keep_after_sit:
        v = torch.where(env._ld_sat & ~env._ld_fellback, torch.ones_like(v), v)
    if keep_after_squat:
        v = torch.where(env._ld_squat & ~env._ld_fellback, torch.ones_like(v), v)
    return v


def _unsat(env, unsat_scale: float) -> torch.Tensor:
    return torch.where(env._ld_fellback, torch.full((env.num_envs,), unsat_scale, device=env.device),
                       torch.ones(env.num_envs, device=env.device))


@hold_masked
def descend_progress(env, stand_height: float, supine_height: float) -> torch.Tensor:
    """몸통이 내려간 비율 (0 서 있음 · 1 누운 높이) — 처음에 '내려가야 한다' 는 방향만 주는 약한 항."""
    return torch.clamp((stand_height - _base_z(env)) / (stand_height - supine_height), 0.0, 1.0)


@hold_masked
def lie_pose(env, target_pose: dict, std: float, asset_cfg: SceneEntityCfg, unsat_scale: float = 1.0) -> torch.Tensor:
    """누워 있는 동안 (_ld_lay) 관절 ↔ 누운 기본 자세 (다리 펴고 팔 옆) RMS 가우시안. 뒤로 넘어졌으면 × unsat_scale (2 차)."""
    _ensure_state(env)
    asset = env.scene[asset_cfg.name]
    names = [asset.joint_names[i] for i in asset_cfg.joint_ids] if not isinstance(asset_cfg.joint_ids, slice) else asset.joint_names
    tgt = torch.tensor([target_pose[n] for n in names], device=env.device)
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - tgt
    rms2 = torch.square(err).mean(dim=1)
    return env._ld_lay.float() * torch.exp(-rms2 / std**2) * _unsat(env, unsat_scale)


@hold_masked
def lie_still(env, lin_std: float = 0.1, ang_std: float = 0.3, unsat_scale: float = 1.0,
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """누워 있는 동안 몸이 가만히 (선속도 · 각속도 가우시안 곱). 뒤로 넘어졌으면 × unsat_scale (2 차)."""
    _ensure_state(env)
    d = env.scene[asset_cfg.name].data
    v = d.root_lin_vel_w.norm(dim=1)
    w = d.root_ang_vel_w.norm(dim=1)
    return env._ld_lay.float() * torch.exp(-torch.square(v / lin_std)) * torch.exp(-torch.square(w / ang_std)) * _unsat(env, unsat_scale)


# ═══════════════════════════════════════════════════════════════
#  벌점 — 충격 · 무릎 꿇기 · 엎드림/옆 · 빠르게 내려감
# ═══════════════════════════════════════════════════════════════
@hold_masked
def contact_impact(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """몸체별 접촉 힘 최고 (서브스텝 포함) 가 threshold [N] 를 넘은 만큼의 합 [N] — 쿵 하고 닿기 막기."""
    return torch.clamp(_peak_force(env, sensor_cfg) - threshold, min=0.0).sum(dim=1)


@hold_masked
def kneel_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 5.0, supine_ok: float = 0.5) -> torch.Tensor:
    """몸통이 뒤로 누운 상태 (−g_x > supine_ok) 가 아닐 때 무릎·정강이가 바닥에 닿음 = 무릎 꿇기 (누운 뒤 종아리가 닿는 건 괜찮음)."""
    on = (_peak_force(env, sensor_cfg) > threshold).any(dim=1)
    return (on & (_supine_frac(env) < supine_ok)).float()


@hold_masked
def wrong_side(env, low_z: float = 0.25, face_down: bool = False) -> torch.Tensor:
    """낮은데 (z < low_z) 엎드림 (g_x > 0) 이나 옆 (|g_y| 큼) 쪽 — 앞·옆으로 쓰러지기 막기. 값 = max(g_x, |g_y| − 0.3, 0).
    face_down (앞으로 눕기): 반대로 반듯이 (g_x < 0) 쪽이 잘못."""
    g = env.scene["robot"].data.projected_gravity_b
    gx = -g[:, 0] if face_down else g[:, 0]
    bad = torch.maximum(torch.clamp(gx, min=0.0), torch.clamp(g[:, 1].abs() - 0.3, min=0.0))
    return (_base_z(env) < low_z).float() * bad


@hold_masked
def base_descent_speed_over(env, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """몸통 하강 속도가 limit [m/s] 를 넘은 만큼의 제곱 — 털썩 주저앉기·쓰러지기 막기 (올라가는 건 안 봄)."""
    vz = env.scene[asset_cfg.name].data.root_lin_vel_w[:, 2]
    return torch.square(torch.clamp(-vz - limit, min=0.0))


@hold_masked
def fellback_flag(env) -> torch.Tensor:
    """앉기 전에 뒤로 넘어진 env 표시 (기록용 w 1e-6 · 평가 스크립트가 읽음)."""
    _ensure_state(env)
    return env._ld_fellback.float()


@hold_masked
def squat_flag(env) -> torch.Tensor:
    """쪼그림 래치 (기록용)."""
    _ensure_state(env)
    return env._ld_squat.float()


@hold_masked
def sat_flag(env) -> torch.Tensor:
    """앉음 래치 (기록용)."""
    _ensure_state(env)
    return env._ld_sat.float()


# ═══════════════════════════════════════════════════════════════
#  V1 (11-09 18:4x) — 키프레임 추종
#  사용자: "손끝이 땅에 닿고 눕게해야될거같은데, 지금 그리고 너무 빠르게 주저앉는 느낌이야" (V0 7 판: 쪼그리고 버티기에서 못 벗어남).
#  관절 목표 궤적 (키프레임 cos 보간, 오른쪽 값 · 왼쪽 = 부호 반대) 과 몸통 pitch 궤적을 시간으로 따라가게 하고 phase 를 관측에 준다.
#  PD 만으로 재생하면 서 있기부터 균형을 못 잡아 0.8 s 에 뒤로 넘어짐 (keyframe_play.py) → 균형은 정책 몫.
# ═══════════════════════════════════════════════════════════════
import math as _math


def _kf_tables(env, keyframes: list, pitch_keyframes: list):
    """키프레임 → (시각 [K], 관절 목표 [K, J], pitch 시각, pitch 값) 텐서. env 에 캐시 (cfg 는 hydra 왕복해도 같은 값)."""
    key = "_ld_kf_cache"
    if not hasattr(env, key):
        robot = env.scene["robot"]
        names = list(robot.data.joint_names)
        dev = env.device
        cur, ts, qs = {}, [], []
        for t, kf in keyframes:
            cur = {**cur, **kf}
            ts.append(float(t))
            qs.append([(1.0 if n.startswith("right_") else -1.0) * float(cur[n.split("_", 1)[1]]) for n in names])
        pt = [float(t) for t, _ in pitch_keyframes]
        pv = [float(v) for _, v in pitch_keyframes]
        setattr(env, key, (torch.tensor(ts, device=dev), torch.tensor(qs, device=dev),
                           torch.tensor(pt, device=dev), torch.tensor(pv, device=dev)))
    return getattr(env, key)


def _interp_cos(t: torch.Tensor, ts: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
    """env 별 시각 t [N] → 키프레임 사이 cos 보간 값 [N, ...]. 처음 전 · 끝 뒤는 끝값."""
    k = torch.searchsorted(ts, t.contiguous(), right=True).clamp(1, len(ts) - 1)
    t0, t1 = ts[k - 1], ts[k]
    u = ((t - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0.0, 1.0)
    u = 0.5 - 0.5 * torch.cos(_math.pi * u)
    v0, v1 = vs[k - 1], vs[k]
    if vs.dim() == 2:
        u = u.unsqueeze(1)
    return v0 + (v1 - v0) * u


def _ref_time(env) -> torch.Tensor:
    return env.episode_length_buf.float() * env.step_dt


def ref_phase(env, t_ref: float) -> torch.Tensor:
    """관측: 기준 동작 진행도 t / t_ref (0 → 1, 끝 뒤 1.2 에서 멈춤). 실기 노드 = 눕기 시작 뒤 시각 / t_ref."""
    return (_ref_time(env) / t_ref).clamp(0.0, 1.2).unsqueeze(1)


@hold_masked
def ref_joint_tracking(env, keyframes: list, pitch_keyframes: list, std: float) -> torch.Tensor:
    """관절 ↔ 기준 궤적 RMS 가우시안 (전 18 관절)."""
    ts, qs, _, _ = _kf_tables(env, keyframes, pitch_keyframes)
    q_ref = _interp_cos(_ref_time(env), ts, qs)
    rms2 = torch.square(env.scene["robot"].data.joint_pos - q_ref).mean(dim=1)
    return torch.exp(-rms2 / std**2)


@hold_masked
def ref_pitch_tracking(env, keyframes: list, pitch_keyframes: list, std_deg: float) -> torch.Tensor:
    """몸통 pitch [deg] ↔ 기준 pitch 궤적 가우시안 (앞 + · 뒤 −, 반듯이 누움 −90)."""
    _, _, pt, pv = _kf_tables(env, keyframes, pitch_keyframes)
    ref = _interp_cos(_ref_time(env), pt, pv)
    return torch.exp(-torch.square(_pitch_deg(env) - ref) / std_deg**2)


def liedown_tracker_hands(
    env,
    hands_cfg: SceneEntityCfg,
    pelvis_cfg: SceneEntityCfg,
    lie_z: float,
    lie_supine: float,
    contact_threshold: float = 5.0,
    face_down: bool = False,
    sit_pitch: float | None = None,
    sit_pitch_tol: float = 25.0,
) -> torch.Tensor:
    """V1 상태 갱신: 손 (팔꿈치 링크 = 아래팔 끝) 바닥 접촉 래치 (_ld_squat 재사용 = '손 짚음') → 그 뒤 골반 바닥 접촉 = 앉음 (_ld_sat),
    손 없이 골반이 먼저 = 뒤로 넘어짐 (_ld_fellback). 둘 다 바닥 필터 센서.
    face_down (11-10 앞으로 눕기): 골반 = 몸통 한 몸체라 '손 없이 가슴이 먼저' = 앞으로 엎어짐 (같은 _ld_fellback), 누움 = 엎드림 (+g_x).
    sit_pitch (11-11 앉기): 주면 _ld_lay = '앉아 있음' (골반 바닥 접촉 · z < lie_z · |pitch − sit_pitch| < sit_pitch_tol) — lie_pose · lie_still ·
    lie_relax 가 앉은 자세에 그대로 걸린다."""
    _ensure_state(env)
    hold = getattr(env, "getup_hold_active", None)
    live = torch.ones(env.num_envs, dtype=torch.bool, device=env.device) if hold is None else ~hold
    hands_on = _peak_force(env, hands_cfg).max(dim=1).values > contact_threshold
    pelvis_on = _peak_force(env, pelvis_cfg).max(dim=1).values > contact_threshold
    hand_now = live & hands_on & ~env._ld_fellback
    env._ld_squat_new = hand_now & ~env._ld_squat
    env._ld_squat |= hand_now
    sit_now = live & pelvis_on & env._ld_squat
    fell_now = live & pelvis_on & ~env._ld_squat & ~env._ld_sat
    env._ld_sat_new = sit_now & ~env._ld_sat & ~env._ld_fellback
    env._ld_sat |= sit_now & ~env._ld_fellback
    env._ld_fell_new = fell_now & ~env._ld_fellback
    env._ld_fellback |= fell_now
    if sit_pitch is None:
        env._ld_lay = (_base_z(env) < lie_z) & (_lie_frac(env, face_down) > lie_supine)
    else:
        env._ld_lay = pelvis_on & (_base_z(env) < lie_z) & ((_pitch_deg(env) - sit_pitch).abs() < sit_pitch_tol)
    return torch.zeros(env.num_envs, device=env.device)


@hold_masked
def seated_after(env, t_from: float, sit_pitch: float, pitch_std: float, sit_height: float, height_std: float,
                 unsat_scale: float) -> torch.Tensor:
    """11-11 앉기: t_from [s] 뒤 '앉아 있음' (_ld_lay, liedown_tracker_hands 의 sit_pitch) × 몸통 pitch · 골반 높이 가우시안.
    lying_after 의 앉기 판 — t_from 전엔 0 (서두를 이유 없음), 손 없이 엉덩이부터 (_ld_fellback) 면 × unsat_scale."""
    _ensure_state(env)
    t = env.episode_length_buf * env.step_dt
    v = torch.exp(-torch.square(_pitch_deg(env) - sit_pitch) / pitch_std**2) * torch.exp(-torch.square(_base_z(env) - sit_height) / height_std**2)
    return (t >= t_from).float() * env._ld_lay.float() * v * _unsat(env, unsat_scale)


@hold_masked
def recline_after(env, t_from: float, max_back: float) -> torch.Tensor:
    """11-11 앉기: t_from [s] 뒤 몸통이 뒤로 젖혀진 정도 −g_x 가 max_back 을 넘은 만큼 — 앉았다가 뒤로 눕기 (뒤로 눕기 정책에서 이어받아 가장 쉬운 길) 막기."""
    t = env.episode_length_buf * env.step_dt
    return (t >= t_from).float() * torch.clamp(_supine_frac(env) - max_back, min=0.0)


def _kf_scalar(env, key: str, keyframes: list):
    """(시각, 값) 키프레임 → (시각 [K], 값 [K]) 텐서 캐시."""
    attr = "_ld_kf_" + key
    if not hasattr(env, attr):
        setattr(env, attr, (torch.tensor([float(t) for t, _ in keyframes], device=env.device),
                            torch.tensor([float(v) for _, v in keyframes], device=env.device)))
    return getattr(env, attr)


@hold_masked
def ref_height_tracking(env, height_keyframes: list, std: float) -> torch.Tensor:
    """V1 2 차: 골반 높이 ↔ 기준 높이 궤적 가우시안 — 1 차는 관절 · pitch 만이라 '엉거주춤 서 있기' 가 거의 손해 없었다."""
    ts, vs = _kf_scalar(env, "height", height_keyframes)
    ref = _interp_cos(_ref_time(env), ts, vs)
    return torch.exp(-torch.square(_base_z(env) - ref) / std**2)


def ref_deviation_termination(env, pitch_keyframes: list, height_keyframes: list, pitch_tol_deg: float, height_tol: float,
                              after_s: float = 0.5, push_grace_s: float = 0.0) -> torch.Tensor:
    """V1 2 차 조기 종료 (mimic 식): after_s 뒤 몸통 pitch 가 기준과 pitch_tol_deg 넘게 · 또는 골반 높이가 height_tol 넘게 벗어나면 끝.
    추종 보상이 양수일 때만 의미 (끝나면 앞으로 받을 보상을 잃음) — 벌점은 램프인으로 처음엔 작게.
    V1-1: push_grace_s — 밀린 뒤 (env._ld_push_t, push_in_window) 이 시간 동안은 종료 안 함 (정책이 못 막는 벗어남에 종료 벌점 −20 을 주지 않게)."""
    t = _ref_time(env)
    pts, pvs = _kf_scalar(env, "pitch_t", pitch_keyframes)
    hts, hvs = _kf_scalar(env, "height", height_keyframes)
    dp = (_pitch_deg(env) - _interp_cos(t, pts, pvs)).abs()
    dz = (_base_z(env) - _interp_cos(t, hts, hvs)).abs()
    off = (t > after_s) & ((dp > pitch_tol_deg) | (dz > height_tol))
    if push_grace_s > 0.0 and hasattr(env, "_ld_push_t"):
        off &= (t - env._ld_push_t) >= push_grace_s
    return off


@hold_masked
def lie_relax(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V1 3 차: 누워 있는 동안 (_ld_lay) 관절 토크가 작을수록 — exp(−mean(τ²)/std²). 2 차 1500 은 누운 뒤에도 팔이 바닥을 눌러
    RS00 11~14 N·m (정격 5) 를 계속 쓰고 몸통이 −81° 로 들려 있었다."""
    _ensure_state(env)
    tau = env.scene[asset_cfg.name].data.applied_torque
    return env._ld_lay.float() * torch.exp(-torch.square(tau).mean(dim=1) / std**2)


@hold_masked
def tip_back_before_hands(env, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """V1 4 차: 손이 바닥에 닿기 전 (_ld_squat = 손 짚음 래치 전) 몸통이 뒤로 젖혀지는 각속도가 limit [rad/s] 를 넘은 만큼의 제곱.
    3 차 2700: 쪼그림 0.30 m 에서 몸통 +17 → −32° 로 넘어가며 손 첫 접촉 최대 400~630 N. 몸 y 축 각속도 − = 뒤로 젖힘 (x 앞 · z 위)."""
    _ensure_state(env)
    wy = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 1]
    return (~env._ld_squat).float() * torch.square(torch.clamp(-wy - limit, min=0.0))


def push_in_window(env, env_ids, velocity_range: dict, t_min: float, t_max: float):
    """V1-1 (11-09 22:5x, 사용자 "밀기 넣어서 학습 진행해줘"): 에피소드 시각 t_min~t_max (서기 · 쪼그림 구간) 안의 env 만
    몸통 속도에 무작위 변화를 더한다 (push_by_setting_velocity 와 같은 방식). interval 이벤트로 걸고, 손 짚은 뒤엔 안 민다.
    밀린 시각을 env._ld_push_t 에 남김 → ref_deviation_termination 의 push_grace_s."""
    if not hasattr(env, "_ld_push_t"):
        env._ld_push_t = torch.full((env.num_envs,), -1.0e3, device=env.device)
    if env_ids is None or len(env_ids) == 0:
        return
    t = _ref_time(env)[env_ids]
    hold = getattr(env, "getup_hold_active", None)
    ok = (t >= t_min) & (t <= t_max)
    if hold is not None:
        ok &= ~hold[env_ids]
    _ensure_state(env)
    ok &= ~env._ld_squat[env_ids]
    ids = env_ids[ok]
    if len(ids) == 0:
        return
    asset = env.scene["robot"]
    vel = asset.data.root_vel_w[ids].clone()
    rng = [velocity_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
    lo = torch.tensor([r[0] for r in rng], device=env.device)
    hi = torch.tensor([r[1] for r in rng], device=env.device)
    vel += lo + (hi - lo) * torch.rand_like(vel)
    asset.write_root_velocity_to_sim(vel, env_ids=ids)
    env._ld_push_t[ids] = _ref_time(env)[ids]


@hold_masked
def target_jerk(env, action_name: str = "JointPositionAction") -> torch.Tensor:
    """V1-2 (11-10, 사용자 "왜 쪼그려앉기 내려갈때 부들부들 떨면서 내려가? 부드럽게 안돼?"): 실제로 모터에 가는 관절 목표
    (목표 속도 제한 뒤 _processed_actions) 의 2 차 차분² 관절 평균 [rad²] — 지그재그만 잡는다.
    쪼그림 구간 실측 (5200): Isaac 1.9e-4 · MuJoCo 9.6e-4, 기준 궤적 (cos 보간) 은 3e-7 이라 매끈한 동작엔 거의 0."""
    tgt = env.action_manager.get_term(action_name)._processed_actions
    if not hasattr(env, "_ld_jerk_init"):
        env._ld_jerk_init = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env._ld_tgt1 = tgt.clone()
        env._ld_tgt2 = tgt.clone()
    init = env._ld_jerk_init
    if bool(init.any()):
        env._ld_tgt1[init] = tgt[init]
        env._ld_tgt2[init] = tgt[init]
        env._ld_jerk_init[:] = False
    j = torch.square(tgt - 2.0 * env._ld_tgt1 + env._ld_tgt2).mean(dim=1)
    env._ld_tgt2 = env._ld_tgt1.clone()
    env._ld_tgt1 = tgt.clone()
    return j



@hold_masked
def knees_down_window(env, knees_cfg: SceneEntityCfg, t_from: float, t_to: float, threshold: float = 5.0) -> torch.Tensor:
    """앞으로 눕기 V0-1 (11-10): t_from~t_to [s] 동안 가슴이 닿기 전 (_ld_sat 전) 무릎이 바닥에 닿은 정도 — 왼 · 오 무릎 (허벅지 끝 또는
    정강이) 각각 닿으면 0.5. V0 1500 은 무릎 꿇기를 건너뛰고 발끝 · 손으로 엎드려뻗친 자세에서 팔굽혀 내려감 (팔 RS00 지속 7.7 N·m, 한도 8)."""
    _ensure_state(env)
    if not hasattr(env, "_ld_knee_left"):   # body_names 는 부를 때마다 PhysX prim 경로 (env 수 × 몸체) 를 다시 읽어 매 스텝 부르면 iter 2.7 → 11 s
        s = env.scene.sensors[knees_cfg.name]
        names = s.body_names
        ids = knees_cfg.body_ids if not isinstance(knees_cfg.body_ids, slice) else list(range(len(names)))
        env._ld_knee_left = torch.tensor(["left" in names[i] for i in ids], device=env.device)
    left = env._ld_knee_left
    on = _peak_force(env, knees_cfg) > threshold
    v = 0.5 * (on[:, left].any(dim=1).float() + on[:, ~left].any(dim=1).float())
    t = _ref_time(env)
    return ((t >= t_from) & (t <= t_to) & ~env._ld_sat).float() * v


def _shape_slices(asset) -> list:
    """몸체별 형상 구간 (재질 버퍼 [N, S, 3] 의 S 축) — IsaacLab randomize_rigid_body_material 와 같은 방법. 몸체 순서 = asset.body_names."""
    n = [asset._physics_sim_view.create_rigid_body_view(p).max_shapes for p in asset.root_physx_view.link_paths[0]]
    if sum(n) != asset.root_physx_view.max_shapes:
        raise RuntimeError(f"형상 수 {sum(n)} ≠ {asset.root_physx_view.max_shapes}")
    s = [0]
    for k in n:
        s.append(s[-1] + k)
    return [(s[i], s[i + 1]) for i in range(len(n))]


def set_friction_split(asset, env_ids, feet_names, body_st, body_dy, feet_st, feet_dy):
    """env 마다 몸 (발 빼고 전부) · 발 마찰 [len(env_ids)] 를 넣는다 (cpu 텐서). 바닥은 multiply · 1.0 이라 이 값이 곧 접촉 마찰."""
    view = asset.root_physx_view
    ids = torch.as_tensor(env_ids, dtype=torch.long).cpu()
    mats = view.get_material_properties().clone()
    mats[ids, :, 0] = body_st[:, None]
    mats[ids, :, 1] = body_dy[:, None]
    sl = _shape_slices(asset)
    for b in asset.find_bodies(feet_names)[0]:
        a, e = sl[b]
        mats[ids, a:e, 0] = feet_st[:, None]
        mats[ids, a:e, 1] = feet_dy[:, None]
    view.set_material_properties(mats, ids.to(torch.int32))


def randomize_friction_feet_split(env, env_ids, body_static_range: tuple, body_dynamic_range: tuple, feet_static_range: tuple,
                                  feet_dynamic_range: tuple, feet_names: str | list = ".*_ankle_roll_link",
                                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """앉기 V0-1 (11-11, 사용자: 실기 발바닥 고무가 다리 펼 때 바닥을 잡아 로봇이 뒤로 쭉 밀린다): 마찰을 발 / 나머지 따로 뽑는다.
    getup_v1_mdp.randomize_friction_per_env 는 전 형상 한 값이라 발 ≥ 엉덩이 · 손 이 재현 안 됐다 (MuJoCo 발 1.2 · 나머지 0.4 → 골반 −14 cm, 같은 마찰 −2~6)."""
    asset = env.scene[asset_cfg.name]
    n = asset.root_physx_view.count
    ids = torch.arange(n) if env_ids is None else torch.as_tensor(env_ids).cpu()

    def pair(sr, dr):
        st = torch.empty(len(ids)).uniform_(*sr)
        return st, torch.minimum(torch.empty(len(ids)).uniform_(*dr), st)   # 동마찰 ≤ 정마찰

    set_friction_split(asset, ids, feet_names, *pair(body_static_range, body_dynamic_range), *pair(feet_static_range, feet_dynamic_range))


def support_slip_per_body(env, sensor_cfg: SceneEntityCfg, threshold: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """앉기 V0-1: 발이 아닌 받침 (골반 · 손) 이 바닥에 닿아 있는 동안 닿은 점의 수평 미끄럼 속도 [N, B] [m/s] — 몸체마다
    |(v_무게중심 + ω × (접촉점 − 무게중심))_xy|. 몸체 원점 속도는 엉덩이 위로 구를 때도 움직여서 (몸통 −56 → −8° 세우면 +6 cm)
    미끄럼과 섞인다 → 접촉점에서의 속도. 가중치 × dt 로 쌓이니 에피소드 합 = 가중치 × 미끄러진 거리 [m].
    6300 Isaac: 골반은 다리 다 편 뒤 (5.0 s) 에야 닿고 그 전 4.4~5.0 s 손 · 발로 버틴 채 다리를 펴 손이 10~13 cm 뒤로 미끄러짐 → 손도 넣음.
    sensor = 골반 · 손 (팔꿈치 링크) · 바닥 필터 · track_contact_points. 발은 미끄러져도 됨 (다리 펼 때 발이 앞으로 가야 정상)."""
    s = env.scene.sensors[sensor_cfg.name]
    d = env.scene[asset_cfg.name].data
    if not hasattr(env, "_ld_slip_ids"):    # 센서 몸체 → 관절체 몸체 번호 (body_names 는 매 스텝 부르면 느림 — 처음 한 번)
        art = env.scene[asset_cfg.name]
        env._ld_slip_ids = torch.tensor([art.body_names.index(n) for n in s.body_names], device=env.device)
    ids = env._ld_slip_ids
    f = s.data.force_matrix_w.sum(dim=2).norm(dim=-1)                 # [N, B]
    c = s.data.contact_pos_w[:, :, 0]                                 # [N, B, 3], 접촉 없으면 nan
    on = (f > threshold) & torch.isfinite(c).all(dim=-1)
    r = torch.nan_to_num(c, nan=0.0) - d.body_com_pos_w[:, ids]
    v = d.body_com_lin_vel_w[:, ids] + torch.cross(d.body_com_ang_vel_w[:, ids], r, dim=-1)
    return on.float() * v[..., :2].norm(dim=-1)


@hold_masked
def support_slip(env, sensor_cfg: SceneEntityCfg, threshold: float = 5.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """support_slip_per_body 의 몸체 합 [m/s] (보상 항)."""
    return support_slip_per_body(env, sensor_cfg, threshold, asset_cfg).sum(dim=1)
