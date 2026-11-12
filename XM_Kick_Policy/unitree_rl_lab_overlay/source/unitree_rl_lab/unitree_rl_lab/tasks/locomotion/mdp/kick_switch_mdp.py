"""걷기·킥 정책 스위칭용 킥 학습 (RS02 KickSwitch, 2026-11-07) — 고정된 걷기 정책이 환경 안에서 걷고, 킥 요청 동안만 학습 정책이 조종.

실기 노드 (khuber_walk_policy, walk.yaml kick_policy_dir) 와 같은 흐름을 한 에피소드에:
  ① 걷기: 고정 걷기 정책 (V0-2 model_4600, 관측 앞 330 칸) 이 학습 분포 명령으로 걷는다 — KickSwitchEnv.step 이 그 env 의 액션을 바꿔 넣음
  ② 킥 요청 (kick_unified_mdp.kick_mode_switch 그대로): 이 순간부터 학습 정책 (관측 355 전부, 킥 세기 s 포함)
  ③ 돌려주기 (handback_switch): 요청 뒤 handback_s (에피소드마다 무작위 2.5~4 s, 실기 kick_duration 3.5 s 를 덮음) 에
     kick_mode 를 끄고 (킥 관측 0) 걷기 정책으로. 명령은 0 또는 느린 걷기
  ④ 돌려준 뒤 after_s 동안 계속 걷기 — 넘어지면 학습 정책에 벌점 (fall_after_handback), 끝나면 handback_done (time_out)
액션 이력 (관측 last_action) = 실제로 쓴 액션 (바꿔 넣은 걷기 액션 포함) — 노드의 공유 이력 버퍼와 같다.
걷기 스텝의 정책 표본은 rsl_rl PPO sample_mask (킥 세기 관측 최신 프레임 > 0) 로 정책 손실에서 뺀다.

게이트 함수는 kick_unified_mdp 처럼 이 모듈에 고유 이름으로 등록 (train.py hydra 왕복에서 원래 함수로 바뀌지 않게).
"""
from __future__ import annotations

import functools
import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from . import kick_handoff_mdp as kh
from . import kick_mdp as km
from .kick_clock import kick_clock_s


# ============================================================================
# 환경: 걷기 모드 스텝은 고정 걷기 정책 액션으로 바꿔 넣음
# ============================================================================
def load_actor_mlp(path: str, device) -> torch.nn.Sequential:
    sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    layers = []
    for i, li in enumerate(lin):
        w = sd[f"actor.{li}.weight"]
        m = torch.nn.Linear(w.shape[1], w.shape[0])
        m.weight.data.copy_(w)
        m.bias.data.copy_(sd[f"actor.{li}.bias"])
        layers += [m] + ([torch.nn.ELU()] if i < len(lin) - 1 else [])
    net = torch.nn.Sequential(*layers).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class KickSwitchEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv + 고정 걷기 정책. cfg.walk_policy_ckpt (rsl_rl .pt) · cfg.walk_policy_obs_dims (관측 앞 몇 칸)."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.walk_policy = load_actor_mlp(cfg.walk_policy_ckpt, self.device)
        self.walk_obs_dims = int(cfg.walk_policy_obs_dims)
        if self.walk_policy[0].in_features != self.walk_obs_dims:
            raise ValueError(f"걷기 정책 입력 {self.walk_policy[0].in_features} ≠ walk_policy_obs_dims {self.walk_obs_dims}")
        self.walk_ctrl_frac = 0.0
        print(f"[KickSwitchEnv] 고정 걷기 정책 {cfg.walk_policy_ckpt} (입력 {self.walk_obs_dims}) — 킥 모드 아닌 스텝은 이 정책 액션", flush=True)

    def step(self, action: torch.Tensor):
        # cfg.learner_action_clip (V0-1, 11-07): 쓰는 액션을 ±c 로 자름 → 관측 last_action 이력도 ±c 안 (실기 노드 action_clip 과 같게).
        #   자르기 없으면 직전 액션 관측을 통한 되먹임이 스텝마다 ×1.9 로 불어날 수 있음 (memory last-action-feedback-runaway).
        #   자르기 전 초과분은 raw_action_excess 로 남겨 보상 raw_action_over_clip 이 벌점 (평균이 한계 밖으로 새지 않게).
        clip = getattr(self.cfg, "learner_action_clip", None)
        if clip:
            self.raw_action_excess = (action.abs() - clip).clamp(min=0.0)
        # V0-4b (11-10): 원출력 폭주 (넘어질 때 직전 액션 이력 되먹임, |a| 2000~4900) 감지 — 킥 모드 env 에서 |원출력| 최대가 문턱을 넘으면
        #   종료 항 raw_action_runaway · 보상 runaway_penalty 가 이 값을 씀 (노드 포화 감지 → 반납과 같은 성격)
        rth = getattr(self.cfg, "learner_runaway_threshold", None)
        if rth and hasattr(self, "kick_mode"):
            self.raw_runaway = self.kick_mode & (action.abs().max(dim=1).values > rth)
        obs = getattr(self, "obs_buf", None)
        if obs is not None and hasattr(self, "kick_mode"):
            walk = ~self.kick_mode
            self.walk_ctrl_frac = float(walk.float().mean())
            if bool(walk.any()):
                with torch.no_grad():
                    a_w = self.walk_policy(obs["policy"][:, : self.walk_obs_dims])
                action = torch.where(walk[:, None], a_w.to(action.dtype), action)
        if clip:
            action = action.clamp(-clip, clip)      # 걷기 정책 출력도 (노드는 어느 정책이든 자름)
        action = self._rate_limit(action)
        return super().step(action)

    def _rate_limit(self, action: torch.Tensor) -> torch.Tensor:
        """cfg.learner_rate_limit (V0-4, 11-10): 학습 정책 (킥 모드) 스텝에서 cfg.learner_rate_joints 관절의 액션 변화를
        직전에 실제로 쓴 액션 (걷기 정책 것 포함) 대비 ±limit/scale (관절 목표 ±limit rad/스텝) 으로 자름 — 실기 노드 kick_rate_limit 과 같은 식
        (노드 policy_core: action_clip 다음, 관측 last_action 이력에도 자른 값). 자르기 전 초과분은 raw_rate_excess 로 남겨 벌점.
        실기 11-09: 찬 발이 내려올 때 디딤 무릎 목표가 한 스텝 0.8~1.3 rad (MCU 50 rad/s 가 1.0 으로 자름) → 무릎 30 rad/s → 버스 80 A."""
        lim = getattr(self.cfg, "learner_rate_limit", None)
        if not lim or not hasattr(self, "kick_mode"):
            self._applied_prev = action.clone()
            return action
        if getattr(self, "_rate_lim", None) is None:
            import re
            name = next(iter(self.action_manager._terms))
            term = self.action_manager.get_term(name)
            scale = float(getattr(self.cfg.actions, name).scale)
            pats = [re.compile(x) for x in self.cfg.learner_rate_joints]
            m = [any(p.fullmatch(n) for p in pats) for n in term._joint_names]
            self._rate_mask = torch.tensor(m, device=self.device)
            self._rate_lim = lim / scale
            print(f"[KickSwitchEnv] 킥 정책 관절 목표 변화 제한 ±{lim:g} rad/스텝 (액션 ±{self._rate_lim:g}): "
                  f"{[n for n, k in zip(term._joint_names, m) if k]}", flush=True)
        prev = getattr(self, "_applied_prev", None)
        if prev is None or prev.shape != action.shape:
            prev = action
        on = self.kick_mode[:, None] & self._rate_mask[None, :]
        d = action - prev
        self.raw_rate_excess = torch.where(on, (d.abs() - self._rate_lim).clamp(min=0.0), torch.zeros_like(d))
        action = torch.where(on, prev + d.clamp(-self._rate_lim, self._rate_lim), action)
        self._applied_prev = action.clone()
        return action


# ============================================================================
# 게이트 (kick_unified_mdp 와 같은 방식, 이 모듈에 등록)
# ============================================================================
def _register(gated, fn, tag: str):
    name = f"{tag}__{fn.__module__.rsplit('.', 1)[-1]}__{fn.__name__}"
    gated.__name__ = gated.__qualname__ = name
    gated.__module__ = __name__
    globals()[name] = gated
    return gated


def _ensure(env) -> None:
    if not hasattr(env, "handed_back"):
        n, dev = env.num_envs, env.device
        env.handed_back = torch.zeros(n, dtype=torch.bool, device=dev)
        env.handback_s = torch.full((n,), 3.5, device=dev)
        env.handback_t0 = torch.full((n,), math.inf, device=dev)
        env.nokick_pending = torch.zeros(n, dtype=torch.bool, device=dev)
        env.handback_post = torch.full((n,), math.inf, device=dev)   # V0-3: 찬 뒤 이만큼이면 돌려줌 (inf = 요청 기준만)
        env.bonus_pending = torch.zeros(n, dtype=torch.bool, device=dev)
        env.handback_posture = torch.zeros(n, device=dev)            # V0-3b: 돌려준 순간 자세 점수 (다음 스텝에 지급)


def gate_event_prekick(fn):
    """걷기 모드 이벤트 (밀기) 를 킥 요청 전에만 — 돌려준 뒤엔 안 민다 (그 넘어짐은 킥 정책 벌점이 되므로 외란 잡음을 빼려고)."""
    @functools.wraps(fn)
    def gated(env, env_ids, *args, **kwargs):
        _ensure(env)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        ok = ~env.kick_mode[env_ids] & ~env.handed_back[env_ids]
        ids = env_ids[ok]
        if len(ids) > 0:
            fn(env, ids, *args, **kwargs)
    return _register(gated, fn, "prekick_event")


# ============================================================================
# 이벤트
# ============================================================================
def reset_handback(env, env_ids: torch.Tensor, handback_range_s: tuple[float, float],
                   post_kick_ranges: tuple | None = None) -> None:
    """리셋: 돌려주는 시각 (요청 뒤) 을 뽑고 돌려주기 상태를 비움. reset_mode 뒤에 둘 것.
    post_kick_ranges (V0-3, 11-08 '킥 하고 멈추기'): ((a, b), ...) 중 하나를 같은 확률로 골라 '찬 뒤 몇 초에 돌려줄지' 를 뽑음 —
    handback_switch 가 그 시각 (또는 handback_range_s 로 뽑은 요청 뒤 시각, 먼저 오는 쪽) 에 돌려줌. None 이면 요청 뒤 시각만 (V0~V0-2)."""
    _ensure(env)
    n = len(env_ids)
    dev = env.device
    env.handback_s[env_ids] = torch.empty(n, device=dev).uniform_(*handback_range_s)
    if post_kick_ranges:
        rg = torch.tensor(post_kick_ranges, device=dev, dtype=torch.float)
        pick = torch.randint(len(post_kick_ranges), (n,), device=dev)
        lo, hi = rg[pick, 0], rg[pick, 1]
        env.handback_post[env_ids] = lo + (hi - lo) * torch.rand(n, device=dev)
    else:
        env.handback_post[env_ids] = math.inf
    env.handed_back[env_ids] = False
    env.handback_t0[env_ids] = math.inf
    env.nokick_pending[env_ids] = False
    env.bonus_pending[env_ids] = False
    env.handback_posture[env_ids] = 0.0


def handback_switch(
    env,
    env_ids: torch.Tensor,
    cmd_zero_prob: float,
    vx_range: tuple[float, float],
    wz_range: tuple[float, float],
    command_name: str = "base_velocity",
    min_s: float = 0.0,
    bonus_hold_s: float | None = None,
    posture_sigma: tuple[float, float] | None = None,
) -> None:
    """매 스텝: 킥 모드에서 요청 뒤 handback_s 가 된 env 를 걷기 모드로 (킥 관측 0, 걷기 정책). 명령 = 0 (cmd_zero_prob) 또는 느린 걷기.
    V0-3: 찬 뒤 handback_post (reset_handback post_kick_ranges) 가 지났고 요청 뒤 min_s 이상이어도 돌려줌.
    bonus_hold_s: 킥 보너스 (찬 뒤 그만큼 킥 모드로 버티면 지급) 를 못 받은 채 돌려주면 kick_bonus_at_handback 이 대신 지급하게 표시.
    posture_sigma (V0-3b): (기울기 °, 각속도 rad/s) — 찬 env 를 돌려주는 순간 exp(−(기울기/σ)²)·exp(−(|ω|/σ)²) 를 handback_posture 로 남김."""
    _ensure(env)
    if not hasattr(env, "kick_mode"):
        return
    km._ensure_kick_state(env)
    t = kick_clock_s(env)[env_ids]
    kt = env.kick_time_s[env_ids]
    early = (kt >= 0) & (t >= min_s - 1e-6) & (t - kt >= env.handback_post[env_ids] - 1e-6)
    go = env.kick_mode[env_ids] & ((t >= env.handback_s[env_ids] - 1e-6) | early)
    if not bool(go.any()):
        return
    ids = env_ids[go]
    k = len(ids)
    dev = env.device
    env.nokick_pending[ids] = env.kick_time_s[ids] < 0
    if bonus_hold_s is not None:
        since = t[go] - kt[go]
        env.bonus_pending[ids] = (kt[go] >= 0) & (since < bonus_hold_s - 1e-6)
    if posture_sigma is not None:
        rb = env.scene["robot"].data
        tilt = torch.rad2deg(torch.acos(torch.clamp(-rb.projected_gravity_b[ids, 2], -1.0, 1.0)))
        w = rb.root_ang_vel_b[ids].norm(dim=-1)
        q = torch.exp(-torch.square(tilt / posture_sigma[0])) * torch.exp(-torch.square(w / posture_sigma[1]))
        env.handback_posture[ids] = q * (kt[go] >= 0).float()
    env.kick_mode[ids] = False
    env.kick_start_s[ids] = math.inf          # kick_mode_switch 가 다시 켜지 않게
    env.handed_back[ids] = True
    env.handback_t0[ids] = env.episode_length_buf[ids].float() * env.step_dt
    vt = env.command_manager.get_term(command_name)
    zero = torch.rand(k, device=dev) < cmd_zero_prob
    vx = torch.empty(k, device=dev).uniform_(*vx_range)
    wz = torch.empty(k, device=dev).uniform_(*wz_range)
    cmd = torch.stack([vx, torch.zeros_like(vx), wz], dim=-1)
    vt.vel_command_b[ids] = torch.where(zero[:, None], torch.zeros_like(cmd), cmd)
    if hasattr(vt, "is_standing_env"):
        vt.is_standing_env[ids] = zero
    vt.time_left[ids] = 1.0e9


def sample_strength(
    env,
    env_ids: torch.Tensor,
    full_prob: float,
    s_min: float,
    lo: float,
    v_full: float,
    strike_full: float,
    lo_start: float | None = None,
    lo_step: float = 0.1,
    err_threshold: float = 0.20,
    min_samples: int = 30000,
    ema_window: float = 20000.0,
    eval_levels: tuple | None = None,
) -> None:
    """에피소드마다 킥 세기 s: full_prob 는 1, 나머지 U(s_min, 1) (s_min > 0 — 관측 0 = 걷기 모드와 구별되게, 노드도 같은 범위로 자름).
    목표 공속 v*(s) = v_full · (lo + (1 − lo)·s). 강타 래치 문턱 · 하한 커리큘럼은 kick_handoff_mdp.sample_kick_strength_v15 와 같다
    (MX3 v1_5b 시도 5 — 하한 lo_start → lo, s < 1 이고 찬 에피소드의 상대 오차 EMA < err_threshold 마다 lo_step).
    """
    km._ensure_kick_strength(env)
    dev = env.device
    n = len(env_ids)
    env.str_v_full = v_full
    if not hasattr(env, "str_peak"):
        env.str_peak = torch.zeros(env.num_envs, device=dev)
    if lo_start is not None and not hasattr(env, "str_lo"):
        env.str_lo = float(lo_start)
        env.str_ema, env.str_count = 0.0, 0
    if lo_start is not None:
        s_old = env.kick_strength_cmd[env_ids]
        pk = env.str_peak[env_ids]
        ok = (s_old < 0.999) & (pk > 0.0)
        if bool(ok.any()):
            tgt = v_full * (env.str_lo + (1.0 - env.str_lo) * s_old[ok])
            err = ((pk[ok] - tgt).abs() / tgt).mean().item()
            kk = int(ok.sum())
            w = ema_window / (ema_window + kk)
            env.str_ema = w * env.str_ema + (1.0 - w) * err if env.str_count else err
            env.str_count += kk
            if env.str_count >= min_samples and env.str_ema < err_threshold and env.str_lo > lo + 1e-6:
                prev = env.str_lo
                env.str_lo = max(lo, env.str_lo - lo_step)
                print(f"[kick_strength] 커리큘럼: 하한 {prev:.2f} → {env.str_lo:.2f} (오차 EMA {env.str_ema:.3f}, 표본 {env.str_count})",
                      flush=True)
                env.str_count = 0
    env.str_peak[env_ids] = 0.0
    if eval_levels is not None:
        lv = torch.tensor(eval_levels, device=dev, dtype=torch.float)
        s = lv[env_ids % len(eval_levels)]
    else:
        s = torch.where(torch.rand(n, device=dev) < full_prob, torch.ones(n, device=dev),
                        s_min + (1.0 - s_min) * torch.rand(n, device=dev))
    env.kick_strength_cmd[env_ids] = s
    cur_lo = getattr(env, "str_lo", lo)
    if not hasattr(env, "ins_min_strike_override"):
        env.ins_min_strike_override = torch.full((env.num_envs,), strike_full, device=dev)
    env.ins_min_strike_override[env_ids] = strike_full * (cur_lo + (1.0 - cur_lo) * s)


# ============================================================================
# 관측
# ============================================================================
def kick_strength_obs(env) -> torch.Tensor:
    """킥 세기 s (kick_unified_mdp.gate_obs 로 감싸 걷기 모드 0). 통합 정책의 ones_obs 자리 그대로."""
    km._ensure_kick_strength(env)
    return env.kick_strength_cmd.unsqueeze(-1)


# ============================================================================
# 종료
# ============================================================================
def handback_done(env, after_s: float) -> torch.Tensor:
    """돌려준 뒤 after_s 를 걸었으면 끝 (time_out 으로 — 실패 아님)."""
    _ensure(env)
    t = env.episode_length_buf.float() * env.step_dt
    return env.handed_back & (t - env.handback_t0 >= after_s - 1e-6)


def weak_kick_strength_termination(
    env,
    check_s: float,
    min_frac: float,
    v_full: float,
    lo: float,
    speed_threshold: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """kick_unified_mdp.weak_kick_termination 의 세기판: 래치 뒤 check_s 동안 목표 방향 공속 최고값 < min_frac · v*(s) 면 툭 = 실패.
    s = 1 에서 min_frac · v_full = 통합 9 차의 2.0 m/s 가 되게 min_frac 를 고른다."""
    from . import kick_inside_mdp as ki

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
    v_min = min_frac * v_full * kh.strength_frac(env, lo)
    return env.kick_mode & at_check & (env.kick_peak_proj < v_min)


# ============================================================================
# 보상
# ============================================================================
def fall_after_handback(env, limit_angle: float, minimum_height: float,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """돌려준 뒤 걷기 정책이 넘어지면 1.0 (종료와 같은 판정) — 큰 음수 weight. 킥 정책이 받아낼 수 있는 상태로 넘겨줘야 한다."""
    _ensure(env)
    robot = env.scene[asset_cfg.name]
    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)) > limit_angle
    fell = tilt | (robot.data.root_pos_w[:, 2] < minimum_height)
    return (fell & env.handed_back).float()


def nokick_at_handback(env) -> torch.Tensor:
    """공을 안 찬 채 돌려준 스텝에 한 번 1.0 — 음수 weight (돌려주기 이벤트가 표시, 다음 보상 계산에서 지움)."""
    _ensure(env)
    out = env.nokick_pending.float()
    env.nokick_pending[:] = False
    return out


def kick_bonus_at_handback(env) -> torch.Tensor:
    """V0-3: 찬 뒤 보너스 버팀 시간 (kick_success_bonus hold_time_s) 전에 돌려준 env 에 그 다음 스텝 한 번 1.0 — kick_success_bonus 와 같은 weight.
    보너스가 킥 모드 안에서만 나가므로, 일찍 돌려주면 보너스가 사라져 '찬 직후 돌려주기' 가 손해가 되는 것을 막음.
    버팀 대신 돌려준 뒤 넘어짐 (fall_after_handback) 이 '차고 서기' 를 맡음. weak_kick 판정 (래치 뒤 0.5 s) 이 먼저 끝나게 post 하한 > 0.5."""
    _ensure(env)
    out = env.bonus_pending.float()
    env.bonus_pending[:] = False
    return out


def handback_posture(env) -> torch.Tensor:
    """V0-3b: 돌려준 다음 스텝에 한 번, 돌려준 순간의 자세 점수 (0~1, handback_switch posture_sigma) — 양수 weight.
    V0-3 은 킥 모드가 일찍 끝나 킥 모드 보상 (자세 · capture point 등) 이 받쳐 주던 '찬 뒤 가라앉히기' 가 줄고, 돌려준 뒤는 넘어짐 −300 만
    봐서 (4600 은 거의 안 넘어짐) 바로 선 채 넘겨줄 압력이 약했다 (20800: 돌려줄 때 기울기 1.6 → 2.8°). 넘겨주는 순간을 직접 본다."""
    _ensure(env)
    out = env.handback_posture.clone()
    env.handback_posture[:] = 0.0
    return out


def raw_action_over_clip(env) -> torch.Tensor:
    """자르기 전 정책 출력이 learner_action_clip 을 넘은 만큼의 제곱합 (KickSwitchEnv.step 이 남김) — 음수 weight, 킥 모드 게이트로.
    자르기만 있으면 한계 밖 출력은 환경에서 똑같이 ±c 라 벌점이 없어 평균이 한계 밖에 머물 수 있다 → 그 방향만 막는다."""
    ex = getattr(env, "raw_action_excess", None)
    if ex is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(ex * ex, dim=1)


# ============================================================================
# V0-4 (11-10): 순간 전력 — 실기 20300 킥 버스 전류 80 A 표본 → 차단 (memory rs02-kick-efuse-trip). 킥 모드 게이트로 쓸 것.
# ============================================================================
PFIT = (0.75, 1.21, 0.396, 1.41, 50.0)   # 실기 버스 전력 적합 (robo01 11-09, NOTES §8): RS06 τω · RS00 τω · RS06 τ² · RS00 τ² · 상수 [W]
_RS06_KEYS = ("hip_pitch", "hip_roll", "knee")


def raw_action_rate_excess(env) -> torch.Tensor:
    """변화 제한 (learner_rate_limit) 전 액션 변화가 한계를 넘은 만큼의 제곱합 — raw_action_over_clip 과 같은 이유 (평균이 한계 밖에 머물지 않게)."""
    ex = getattr(env, "raw_rate_excess", None)
    if ex is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(ex * ex, dim=1)


def knee_speed_excess(env, soft: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_knee_joint"])) -> torch.Tensor:
    """Σ_무릎 relu(|dq| − soft)² [(rad/s)²]. 실기 버스 전류 스파이크 = 디딤 무릎 고속 (25~40 rad/s 에서 적합식 밖 +0.7 kW)."""
    dq = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return torch.relu(dq.abs() - soft).pow(2).sum(dim=1)


def bus_power_excess(env, soft_w: float, unit_w: float = 100.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """(relu(P − soft_w) / unit_w)², P = 실기 버스 전력 적합식 (PFIT, 전 관절 τ · ω). 동손 τ² 항이 대부분 (스파이크 창 479 W 중 τω 95 W)."""
    robot = env.scene[asset_cfg.name]
    big = getattr(env, "_pfit_big", None)
    if big is None:
        big = torch.tensor([any(k in n for k in _RS06_KEYS) for n in robot.joint_names], device=env.device)
        env._pfit_big = big
    tau, dq = robot.data.applied_torque, robot.data.joint_vel
    tw, t2 = tau * dq, tau * tau
    p = (PFIT[0] * tw[:, big].sum(1) + PFIT[1] * tw[:, ~big].sum(1) + PFIT[2] * t2[:, big].sum(1)
         + PFIT[3] * t2[:, ~big].sum(1) + PFIT[4])
    return (torch.relu(p - soft_w) / unit_w).pow(2)


# ============================================================================
# V0-4b (11-10): 원출력 폭주 대책 — V0-4 20750~ 넘어질 때 출력 ×2~3/스텝 → over_clip 제곱 벌점 −수천 → 비평가 붕괴 → 폭주 더 잦아짐.
# ============================================================================
def raw_action_over_clip_capped(env, cap: float = 4.0) -> torch.Tensor:
    """raw_action_over_clip 과 같되 관절당 초과분을 cap 으로 자름 — 폭주 한 번이 −수천 보상이 되어 비평가를 깨지 않게 (끌어당기는 기울기는 남김)."""
    ex = getattr(env, "raw_action_excess", None)
    if ex is None:
        return torch.zeros(env.num_envs, device=env.device)
    ex = ex.clamp(max=cap)
    return torch.sum(ex * ex, dim=1)


def raw_action_runaway(env) -> torch.Tensor:
    """종료: 킥 모드에서 |원출력| 최대 > cfg.learner_runaway_threshold (KickSwitchEnv.step 이 raw_runaway 로 남김)."""
    r = getattr(env, "raw_runaway", None)
    if r is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return r


def runaway_penalty(env) -> torch.Tensor:
    """폭주로 끝나는 스텝에 1 (음수 weight) — 넘어짐 벌점과 같은 크기로."""
    return raw_action_runaway(env).float()


def bus_power_excess_capped(env, soft_w: float, unit_w: float = 100.0, cap_w: float = 2000.0,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """bus_power_excess 와 같되 초과분을 cap_w 로 자름 (넘어질 때 토크 포화로 P 가 수 kW → 상한 없는 제곱이 비평가를 흔들지 않게)."""
    full = bus_power_excess(env, soft_w, unit_w, asset_cfg).sqrt() * unit_w
    return (full.clamp(max=cap_w) / unit_w).pow(2)
