"""보행·킥 통합 (RS02 WalkKickUnified) — 한 정책이 명령대로 걷다가, 킥 요청이 오면 그 자리에서 공을 차러 간다.

에피소드마다 모드를 정한다 (reset_mode):
  보행 에피소드  킥 요청 없음. 공은 6~10 m 밖에 치워 두고 킥 관측은 0. 보행 보상·명령·밀기 그대로.
  킥 에피소드    처음엔 똑같이 걷다가 start_range_s 안 임의 시각에 kick_mode_switch 가 킥을 켠다:
                 속도 명령 → 다가가기 명령 (vx, 0, 0) 으로 바꾸고 재샘플 멈춤, 걸음 시계 재샘플 멈춤 (위상 리셋 없음),
                 지속 밀기 끔, 공·목표·킥 상태·공 믿음 오차를 킥 태스크 리셋 이벤트 그대로 (지금 자세 기준) 다시 놓음,
                 env.kick_t0_steps = 지금 episode_length_buf → kick_*.py 의 킥 시계가 0 부터 (mdp/kick_clock.py).
                 이후는 킥 태스크와 같다 (찬 뒤 3 s 버티면 kick_resolved 로 끝).

게이트 (gate_reward / gate_obs / gate_done / gate_event): 원래 함수를 그대로 부르고 모드 마스크를 곱한다.
  functools.wraps 로 서명을 그대로 보여 줘서 IsaacLab 매니저의 인자 검사를 통과한다.
  원래 함수는 모드와 상관없이 매 스텝 불린다 (래치 갱신 같은 부수 효과 유지 — 킥 시작 때 리셋되므로 무해).
  ⚠ train.py (hydra_task_config) 는 cfg 를 dict 로 바꿨다 되돌리며 함수를 "모듈:이름" 문자열로 다시 찾는다. 그래서 게이트 함수는
    이 모듈에 고유 이름 (예: kick_only__kick_mdp__ball_approach_reward) 으로 등록한다 — wraps 가 붙인 원래 이름을 그대로 두면
    원래 (게이트 없는) 함수로 바뀌어 버린다 (2026-11-05 첫 학습 시험: 보행 env 가 첫 스텝에 no_kick_deadline 으로 전부 끝남).
"""
from __future__ import annotations

import functools
import math

import torch
from isaaclab.managers import SceneEntityCfg

from . import kick_handoff_mdp as kh
from . import kick_mdp as km


def _ensure(env) -> None:
    if not hasattr(env, "kick_mode"):
        n, dev = env.num_envs, env.device
        env.kick_mode = torch.zeros(n, dtype=torch.bool, device=dev)
        env.kick_t0_steps = torch.zeros(n, dtype=torch.long, device=dev)
        env.kick_start_s = torch.full((n,), math.inf, device=dev)


def kick_mask(env) -> torch.Tensor:
    _ensure(env)
    return env.kick_mode


# ============================================================================
# 게이트
# ============================================================================
def _register(gated, fn, tag: str):
    """게이트 함수를 이 모듈에 고유 이름으로 올린다 (hydra 문자열 왕복에서 살아남게). 같은 이름이면 덮어씀 (동작 같음)."""
    name = f"{tag}__{fn.__module__.rsplit('.', 1)[-1]}__{fn.__name__}"
    gated.__name__ = gated.__qualname__ = name
    gated.__module__ = __name__
    globals()[name] = gated
    return gated


def gate_reward(fn, kick: bool):
    @functools.wraps(fn)
    def gated(env, *args, **kwargs):
        m = kick_mask(env) if kick else ~kick_mask(env)
        return fn(env, *args, **kwargs) * m.float()
    return _register(gated, fn, "kick_only" if kick else "walk_only")


def gate_obs(fn):
    """킥 관측: 보행 모드면 0 (kick_strength 가 0/1 킥 요청 신호가 된다)."""
    @functools.wraps(fn)
    def gated(env, *args, **kwargs):
        return fn(env, *args, **kwargs) * kick_mask(env).float().unsqueeze(-1)
    return _register(gated, fn, "kick_obs")


def gate_done(fn):
    @functools.wraps(fn)
    def gated(env, *args, **kwargs):
        return fn(env, *args, **kwargs) & kick_mask(env)
    return _register(gated, fn, "kick_done")


def gate_event(fn, kick: bool):
    """env_ids 를 그 모드인 env 만 남겨 원래 이벤트에 넘긴다."""
    @functools.wraps(fn)
    def gated(env, env_ids, *args, **kwargs):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        m = kick_mask(env)[env_ids]
        ids = env_ids[m if kick else ~m]
        if len(ids) > 0:
            fn(env, ids, *args, **kwargs)
    return _register(gated, fn, "kick_event" if kick else "walk_event")


# ============================================================================
# 이벤트
# ============================================================================
def _park_ball(env, env_ids, dist_range, height, ball_cfg, robot_cfg) -> None:
    """공을 로봇에서 dist_range 떨어진 임의 방향에 둔다 (평지 지형 안쪽으로 자름)."""
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    n = len(env_ids)
    dev = env.device
    r = torch.empty(n, device=dev).uniform_(*dist_range)
    a = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
    pose = ball.data.root_pose_w[env_ids].clone()
    xy = robot.data.root_pos_w[env_ids, :2] + torch.stack([r * torch.cos(a), r * torch.sin(a)], dim=-1)
    org = getattr(env.scene.terrain, "terrain_origins", None)
    if org is not None:
        o = org.reshape(-1, 3)[:, :2]
        size = env.scene.terrain.cfg.terrain_generator.size if env.scene.terrain.cfg.terrain_generator is not None else (0.0, 0.0)
        lo = o.min(dim=0).values - torch.tensor(size, device=dev) / 2 + 1.0
        hi = o.max(dim=0).values + torch.tensor(size, device=dev) / 2 - 1.0
        xy = torch.maximum(torch.minimum(xy, hi), lo)
    pose[:, :2] = xy
    pose[:, 2] = height
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


def reset_mode(
    env,
    env_ids: torch.Tensor,
    kick_prob: float,
    start_range_s: tuple[float, float],
    park_dist_range: tuple[float, float],
    ball_height: float,
    start_zero_prob: float = 0.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """리셋: 킥 에피소드 여부·킥 시작 시각을 뽑고, 보행 모드로 시작 (공 치움, 킥 상태 비움).

    start_zero_prob: 킥 에피소드 중 이 비율은 리셋 직후 (0 s) 시작 = 킥 태스크와 같은 조건 (지금 정책이 할 줄 아는 킥을 붙잡는 닻).
    """
    _ensure(env)
    n = len(env_ids)
    dev = env.device
    is_kick = torch.rand(n, device=dev) < kick_prob
    start = torch.empty(n, device=dev).uniform_(*start_range_s)
    start = torch.where(torch.rand(n, device=dev) < start_zero_prob, torch.zeros_like(start), start)
    env.kick_start_s[env_ids] = torch.where(is_kick, start, torch.full_like(start, math.inf))
    env.kick_mode[env_ids] = False
    env.kick_t0_steps[env_ids] = 0
    km._ensure_kick_state(env)
    env.kick_time_s[env_ids] = -1.0
    env.peak_kick_v[env_ids] = 0.0
    if hasattr(env, "duel_knock_t"):
        env.duel_knock_t[env_ids] = -1.0
    _park_ball(env, env_ids, park_dist_range, ball_height, ball_cfg, robot_cfg)


def kick_mode_switch(
    env,
    env_ids: torch.Tensor,
    cmd_vx_range: tuple[float, float],
    ball_params: dict,
    target_params: dict,
    state_params: dict,
    belief_params: dict,
    push_asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    gait_command_name: str = "gait",
    fill_kick_obs_history: bool = False,
) -> None:
    """매 스텝 (interval 0.02 s): 킥 시작 시각이 된 보행 모드 env 를 킥 모드로 바꾼다.

    fill_kick_obs_history: 킥 관측 항의 이력 (5 스텝) 을 비워서, 바로 다음 관측 계산이 5 칸 모두 지금 값으로 채우게 한다
      (리셋 직후 킥과 같은 이력 모양). False 면 이력이 [0, 0, 0, 0, 값] 에서 4 스텝에 걸쳐 차오른다.
    """
    _ensure(env)
    t = env.episode_length_buf[env_ids].float() * env.step_dt
    go = (~env.kick_mode[env_ids]) & (t >= env.kick_start_s[env_ids])
    if not bool(go.any()):
        return
    ids = env_ids[go]
    k = len(ids)
    dev = env.device

    # 1) 명령: 다가가기 속도 (vx, 0, 0), 남은 에피소드 동안 재샘플 없음. 걸음 시계도 재샘플 (= 위상 리셋) 없음.
    vt = env.command_manager.get_term(command_name)
    vx = torch.empty(k, device=dev).uniform_(*cmd_vx_range)
    vt.vel_command_b[ids] = torch.stack([vx, torch.zeros_like(vx), torch.zeros_like(vx)], dim=-1)
    if hasattr(vt, "is_standing_env"):
        vt.is_standing_env[ids] = False
    if hasattr(vt, "is_heading_env"):
        vt.is_heading_env[ids] = False
    vt.time_left[ids] = 1.0e9
    env.command_manager.get_term(gait_command_name).time_left[ids] = 1.0e9

    # 2) 지속 밀기 끔 (킥 태스크엔 없던 외란). 이번 스텝에 이미 넣은 힘도 0 으로.
    if hasattr(env, "_sp_t"):
        env._sp_t[ids] = -1.0
        env._sp_wait[ids] = 1.0e9
        asset = env.scene[push_asset_cfg.name]
        z = torch.zeros(k, 1, 3, device=dev)
        asset.set_external_force_and_torque(z, z, body_ids=push_asset_cfg.body_ids, env_ids=ids)

    # 3) 공 · 목표 · 킥 상태 (frozen 관측·마감) · 공 믿음 오차 — 킥 태스크 리셋 이벤트와 같은 함수, 같은 순서
    km.reset_ball_around_robot(env, ids, **ball_params)
    km.reset_kick_target(env, ids, **target_params)
    km.reset_kick_state_from_cmd(env, ids, **state_params)
    kh.perturb_ball_belief(env, ids, **belief_params)

    if fill_kick_obs_history:
        for terms in env.observation_manager._group_obs_term_history_buffer.values():
            for name in ("ball_pos_b", "ball_vel_b", "target_dir_b", "kick_strength"):
                if name in terms:
                    terms[name].reset(batch_ids=ids)

    # 4) 모드 · 킥 시계 (다음 스텝 보상에서 킥 시계 = 1 스텝 → 래치 리셋, 킥 태스크의 리셋 직후와 같음)
    env.kick_mode[ids] = True
    env.kick_t0_steps[ids] = env.episode_length_buf[ids]


# ============================================================================
# 종료
# ============================================================================
def far_kick_termination(
    env,
    max_dist: float,
    speed_threshold: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """킥 모드에서 공이 처음 speed_threshold 를 넘은 스텝 (= 킥 래치) 에 몸 중심 ↔ 공 수평 거리가 max_dist 보다 멀면 종료.

    킥 태스크의 early_kick_done (킥 시계 1 s 전 킥 = 실패) 대신. 1 s 규칙은 MX3 v1_1 에서 '선 자리에서 바로 차기' 를 막고
    '다가가서 차기' 를 만들려던 것인데, 정책은 요청 뒤 시간을 못 봐서 걷다가 받은 요청엔 지킬 수 없었다 (6500: 리셋 뒤 ~1 s 에만 참).
    거리는 볼 수 있다 (공 위치 + 자기 걸음). 측정 (명목, 6500): 맞은 직후 몸 중심 ↔ 공 0.30 m (p10~p90 0.23~0.35).
    """
    _ensure(env)
    km._maybe_record_kick(env, ball_cfg, speed_threshold)
    from .kick_clock import kick_clock_s
    just = (env.kick_time_s >= 0) & ((kick_clock_s(env) - env.kick_time_s).abs() < 0.5 * env.step_dt)
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    d = (ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]).norm(dim=-1)
    return env.kick_mode & just & (d > max_dist)


def weak_kick_termination(
    env,
    check_s: float,
    min_speed: float,
    speed_threshold: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """킥 모드에서 킥 래치 (공이 처음 speed_threshold 를 넘은 스텝) 뒤 check_s 동안 목표 방향 공속 최고값이 min_speed 미만이면 그 시점에 종료.

    9 차 (2026-11-05): 걷다가 발이 공을 '툭' 친 것 = 실패. 래치·성공 보너스는 공이 0.8 m/s 만 넘으면 킥으로 쳐서 툭도 성공 (+8) 이었고,
    래치 뒤엔 제대로 찬 킥도 보상이 없다. 12200 명목: 전진 0.8 → 요청 툭 7 % · 학습 분포 걷기 6.6 % · 공 0.8 m 42 % (정상 킥 4.1~5.3 m/s).
    판정은 rs02_eval/kick_eval.py '판정' 줄과 같다 (킥 뒤 0.5 s 목표 방향 공속 < 2 m/s = 툭).
    """
    from . import kick_inside_mdp as ki
    from .kick_clock import kick_clock_s

    _ensure(env)
    km._maybe_record_kick(env, ball_cfg, speed_threshold)
    if not hasattr(env, "kick_peak_proj"):
        env.kick_peak_proj = torch.zeros(env.num_envs, device=env.device)
    latched = env.kick_time_s >= 0
    since = kick_clock_s(env) - env.kick_time_s
    proj = (env.scene[ball_cfg.name].data.root_lin_vel_w[:, :2] * ki._target_dir_xy(env)).sum(dim=-1)
    win = latched & (since <= check_s + 1e-6)
    env.kick_peak_proj = torch.where(win, torch.maximum(env.kick_peak_proj, proj), env.kick_peak_proj)
    env.kick_peak_proj = torch.where(latched, env.kick_peak_proj, torch.zeros_like(proj))
    at_check = latched & (since >= check_s - 1e-6) & (since < check_s + env.step_dt - 1e-6)
    return env.kick_mode & at_check & (env.kick_peak_proj < min_speed)


# ============================================================================
# 보상
# ============================================================================
def ball_offaxis_after_kick(
    env,
    post_kick_window_s: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """kick_inside_mdp.ball_offaxis_penalty (공 속도의 목표 수직 성분²) 를 킥 뒤 창에만 · 킥 모드에만.

    원 함수는 킥 전에도 켜져 있어 (굴러오는 공 · 다툼으로 튕긴 공) 가중치를 키우면 정책이 못 고치는 벌점이 커진다.
    """
    from . import kick_inside_mdp as ki

    pen = ki.ball_offaxis_penalty(env, post_kick_window_s=post_kick_window_s, ball_cfg=ball_cfg)
    km._ensure_kick_state(env)
    return pen * ((env.kick_time_s >= 0) & kick_mask(env)).float()


def stance_foot_yaw_rate_l2(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """땅에 붙은 발 (접촉 z 힘 > force_threshold) 의 월드 z 각속도² 합 (N,) — 디딤발을 바닥에서 비틀어 도는 것 벌점 (음수 weight).

    sensor_cfg · asset_cfg 는 같은 발 이름을 preserve_order 로 — 접촉 센서와 articulation 의 몸체 순서가 달라서 각자 자기 인덱스로 고른다.
    5 차 (2026-11-05): 통합 4 차 9800 은 회전 때 디딤 한 번에 발 yaw 4.3° (1.0 rad/s) · 7.4° (1.5) 돌아감 (3400 2.4 · 3.6°),
    회전은 명령의 85 % · 요 흔들림 2.7 배. RS01 실기 1 순위 문제 (디딤발이 바닥에서 돎) 와 같은 동작.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    on = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > force_threshold
    wz = env.scene[asset_cfg.name].data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    return (wz.square() * on.float()).sum(dim=1)


def stance_foot_yaw_rate_unpushed(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 90.0,
    hold_s: float = 1.0,
) -> torch.Tensor:
    """stance_foot_yaw_rate_l2 를 체중 실린 발 (접촉 z 힘 > force_threshold, RS02 체중 ≈ 126 N) · 외란 회복 구간 밖에서만.

    8 차 (2026-11-05): 5 차 실패 (접촉 > 1 N, 무작위 명령 + 밀기 분포에선 3400 · 9800 이 같은 값 → 정상 걸음·밀림 회복을 벌함) 를 고친 판.
    명령 고정·밀기 없음으로 재면 > 90 N 에서 3400 0.005~0.022 · 11000 0.061~0.090 (직진 0.5 도 0.009 vs 0.090) — 디딤발을 비트는 건 늘 그렇고
    밀릴 땐 둘 다 비튼다. → 밀기 중 · 밀린 뒤 hold_s 는 뺀다 (push_recovery._recovering).
    """
    from . import push_recovery as pr

    pen = stance_foot_yaw_rate_l2(env, sensor_cfg, asset_cfg, force_threshold)
    return pen * (~pr._recovering(env, hold_s)).float()

