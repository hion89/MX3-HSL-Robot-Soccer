# kick_inside_mdp.py
# ============================================================================
# 인사이드 킥 전용 보상 모듈 (v1_3 설계, 2026-08-16)
# ----------------------------------------------------------------------------
# 【왜 새로 만드는가】
#   walk_kick_v1_2 의 인사이드 3종(ball_offaxis / kick_push / kick_inside_align)은
#   "접촉 중이면 매 스텝 보상"하는 dense 구조였다. 이 구조의 최적해는 킥이 아니라
#   **접촉 시간을 최대화하는 드리블**이다. 실제로 인사이드 드리블이 나왔다.
#
#   수치로 보면(step_dt=0.02s, KICK_HOLD_TIME_S=3.0s → 킥 후 150 스텝):
#     · 강타:   접촉 게이트가 열려 있는 시간 ≈ 3~5 스텝
#               kick_push(w=3.0) ≈ 3.0 × 0.7(gate) × 2.5(push) × 4 ≈ 21
#     · 드리블: 접촉이 계속 유지 ≈ 150 스텝
#               kick_push(w=3.0) ≈ 3.0 × 0.9(gate) × 0.5(push) × 150 ≈ 200
#               kick_inside_align(w=0.3) ≈ 0.3 × 1.0 × 150 ≈ 45
#   → dense 항만 놓고 보면 드리블이 강타의 10 배다. 밤튜닝에서 kick_push 를
#     1.0 → 3.0 으로 올린 것이 드리블 basin 을 결정적으로 키웠다.
#
#   여기에 구조적 함정이 두 개 더 겹쳐 있었다.
#     ① ball_velocity 의 post_kick_window(0.4 s)는 kick_time_s 부터 열리는데,
#        kick_time_s 는 공 속도 > 0.01 m/s 인 **첫 프레임**에 래치된다.
#        즉 걸어가다 공을 살짝 스치는 순간 0.4 s 창이 시작되고, 그 안에 강타를
#        못 넣으면 킥 파워 보상(최대 2.0×3.0²×20 ≈ 360)이 통째로 사라진다.
#        창이 닫힌 뒤 남는 양의 보상은 dense 접촉항뿐 → 정책이 드리블로 흐른다.
#     ② kick_then_stand_bonus(+8) / kick_resolved_termination 도 같은 0.01 문턱을
#        쓴다. 공을 살짝 밀고 3 초 버티면 **드리블이 "성공"으로 판정**되어 +8 을
#        받는다. 보상 설계가 드리블을 성공이라고 가르치고 있었다.
#        → 이 둘은 이 모듈이 아니라 env cfg 의 speed_threshold 로 고친다(0.01→0.8).
#
# 【이 모듈의 설계 원칙】
#   원칙 1. 인사이드 품질은 **독립 보상이 아니라 킥 파워의 배율**이다.
#           보상 채널을 하나로 합치면(파워 × 품질) "품질만 따먹고 킥은 안 하는"
#           대안 경로 자체가 사라진다. 항을 더하면 정책은 항상 제일 싼 항을 고른다.
#   원칙 2. 모든 인사이드 항은 **에피소드당 1 회 래치**한다.
#           강한 타격 순간에만 한 번 잠기고, 이후 접촉으로는 갱신되지 않는다.
#           → 접촉 시간을 늘려서 얻을 이득이 0 이 되어 드리블 farm 이 불가능하다.
#   원칙 3. 접촉 시간 자체에 **예산(budget)**을 둔다. 예산을 넘긴 접촉은 벌점.
#           드리블은 정의상 접촉이 길므로 직접적으로 손해가 된다.
#
# 【인사이드 킥의 기하 — 왜 y축 정렬인가】
#   공이 날아가는 방향은 임팩트 순간의 **접촉면 법선** 방향으로 결정된다.
#   발 안쪽면(medial face)의 법선은 발 로컬 ±y 축이다(실측 probe_foot_frame:
#   발 로컬 +x = 발끝/전방, +y = 좌/측방. 안쪽 = 우발 +y, 좌발 -y).
#   따라서 "발 링크의 y축 ∥ 목표방향" 이 곧 "공이 목표로 간다" 이다.
#   여기에 하나 더 필요하다: **발의 진행 속도 방향도 같은 방향**이어야 한다.
#     · 법선만 맞고 속도가 옆이면 → 면으로 공을 깎는 slice (공이 회전만 먹음)
#     · 속도만 맞고 법선이 틀면  → 발끝/발등으로 찌르는 poke (접촉점 오차에 취약)
#   그래서 품질 q 는 (법선·목표) × (속도·목표) 의 곱으로 정의한다.
#   면 킥의 장점은 접촉점이 몇 cm 어긋나도 법선이 같아 방향 오차가 안 커지는 것
#   이므로, sim2real 갭(발 위치 오차)에 강건해진다.
#
# 【2026-08-16 리뷰 반영 — 2 가지 결함 수정】
#   결함 A. _update_state 의 "스텝당 1 회" 가드 때문에, 보상 항마다 다른
#           min_strike_speed / window_steps / asset_cfg 를 넣으면 **먼저 평가된
#           항의 값이 이기고 나머지는 조용히 무시**된다. kick_mdp 의
#           _maybe_record_kick(공유 래치) 과 정확히 같은 종류의 함정이다.
#           → _assert_params_consistent() 로 첫 스텝에 즉시 예외를 던진다.
#             조용히 틀리느니 시끄럽게 죽는 편이 낫다.
#   결함 B. metric_* 함수까지 파라미터를 받게 두면 위 함정의 표면적이 넓어진다.
#           → metric_* 는 파라미터를 아예 받지 않고 **버퍼를 읽기만** 한다.
#             선언 순서에 따라 1 스텝 늦을 수 있으나 로깅에는 무해하다.
# ============================================================================

from __future__ import annotations

import torch
from .kick_clock import kick_clock_s, kick_clock_steps  # 통합 태스크의 킥 시작 기준 시각 (기존 태스크는 에피소드 시각 그대로)

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

# ----------------------------------------------------------------------------
# 상수 — kick_mdp_v0 와 동일 값을 쓴다(접촉 판정 기준을 통일해야 항끼리 안 싸운다).
# ----------------------------------------------------------------------------
BALL_RADIUS = 0.065

# 발 링크 중심 ↔ 공 중심 거리가 이 값 이하이면 "접촉"으로 본다.
# 발 반길이(~0.09) + 공 반지름(0.065) + 여유.
CONTACT_RADIUS = 0.17

# 접촉 게이트의 감쇠 폭. 하드 0/1 게이트는 보상이 스파스해져 학습이 느리므로
# 접촉 반경 밖으로 이 정도 벗어나면 부드럽게 꺼지도록 한다.
CONTACT_FALLOFF = 0.05

# 좌우 발의 "안쪽(medial)" 부호. body 순서는 반드시 [우발, 좌발] (MX_FEET).
#   오른발: 몸 안쪽 = 왼쪽 = 발 로컬 +y  →  +1
#   왼발  : 몸 안쪽 = 오른쪽 = 발 로컬 -y →  -1
# 이 부호를 발마다 제대로 안 주면 보상이 좌우 비대칭이 되어 미러 손실과 싸운다.
_INSIDE_SIGN = (1.0, -1.0)

_EPS = 1e-6


# ============================================================================
# 내부 상태 버퍼 — 스텝당 1 회만 갱신
# ----------------------------------------------------------------------------
# 여러 보상 항이 같은 래치 상태를 읽어야 하는데, IsaacLab RewardManager 는 항을
# "선언 순서"대로 평가한다. 특정 항에서만 상태를 갱신하면 선언 순서가 바뀌는
# 순간 조용히 깨진다(과거 kick_dist_valid 에서 실제로 겪은 버그와 같은 종류).
# 그래서 갱신 함수를 별도로 두고, common_step_counter 로 "이번 스텝에 이미
# 갱신했는지"를 확인해 순서와 무관하게 정확히 1 회만 돌게 만든다.
# ============================================================================
def _ensure_buffers(env: ManagerBasedRLEnv) -> None:
    """인사이드 킥 상태 버퍼를 없으면 만든다."""
    n, dev = env.num_envs, env.device
    if not hasattr(env, "ins_latched"):
        # 이번 에피소드에서 "강타"가 이미 한 번 래치되었는지 (bool)
        env.ins_latched = torch.zeros(n, dtype=torch.bool, device=dev)
    if not hasattr(env, "ins_quality"):
        # 래치된 인사이드 품질 q ∈ [0, 1]
        env.ins_quality = torch.zeros(n, device=dev)
    if not hasattr(env, "ins_strike_speed"):
        # 래치 순간 발의 목표방향 속도 성분 [m/s]
        env.ins_strike_speed = torch.zeros(n, device=dev)
    if not hasattr(env, "ins_best_score"):
        # 래치된 강타의 점수 (v_along × q). 더 좋은 강타가 오면 값만 갱신한다.
        env.ins_best_score = torch.zeros(n, device=dev)
    if not hasattr(env, "ins_window"):
        # 래치 후 남은 보상 윈도우 스텝 수
        env.ins_window = torch.zeros(n, device=dev)
    if not hasattr(env, "ins_touch_steps"):
        # 이번 에피소드 누적 접촉 스텝 수 (드리블 탐지용)
        env.ins_touch_steps = torch.zeros(n, device=dev)
    if not hasattr(env, "ins_in_contact"):
        # 이번 스텝에 접촉 중인지 (bool)
        env.ins_in_contact = torch.zeros(n, dtype=torch.bool, device=dev)
    if not hasattr(env, "ins_run"):
        # 현재 연속 접촉 런 길이 [스텝]
        env.ins_run = torch.zeros(n, device=dev)
    if not hasattr(env, "ins_max_run"):
        # 이번 에피소드 최장 연속 접촉 런 [스텝] — touch_budget 튜닝 근거
        env.ins_max_run = torch.zeros(n, device=dev)
    if not hasattr(env, "ins_max_vlong"):
        # 이번 에피소드 접촉 중 관측된 최대 목표방향 발속도 [m/s]
        # ── min_strike_speed 튜닝의 핵심 근거 ──
        # 래치가 한 번도 안 걸리면(ins_latch_rate≈0) 이 값이 실제로 몇까지
        # 올라갔는지 봐야 문턱을 얼마나 내릴지 데이터로 정할 수 있다.
        env.ins_max_vlong = torch.zeros(n, device=dev)
    if not hasattr(env, "ins_last_step"):
        env.ins_last_step = -1


def _assert_params_consistent(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_strike_speed: float,
    window_steps: int,
) -> None:
    """이 모듈을 쓰는 모든 보상 항이 같은 파라미터를 쓰는지 첫 스텝에 검증한다.

    _update_state 는 스텝당 1 회만 돌기 때문에, 항마다 파라미터가 다르면 먼저
    평가된 항의 값만 반영되고 나머지는 **조용히 무시**된다. 학습은 그냥 돌아가고
    로그도 정상으로 보여서 며칠을 날릴 수 있는 종류의 버그다.
    Dynamixel 의 final_rpm > Velocity Limit 이 SyncWrite 무응답으로 조용히
    실패하던 것과 같은 실패 모드라, 여기서는 즉시 예외로 죽인다.
    """
    key = (float(min_strike_speed), int(window_steps), str(asset_cfg.body_ids))
    prev = getattr(env, "ins_param_key", None)
    if prev is None:
        env.ins_param_key = key
        return
    if prev != key:
        raise ValueError(
            "[kick_inside_mdp] 보상 항마다 파라미터가 다릅니다. "
            "_update_state 는 스텝당 1 회만 돌기 때문에 먼저 평가된 항의 값이 "
            "이기고 나머지는 무시됩니다.\n"
            f"  먼저 등록된 값: (min_strike_speed, window_steps, body_ids) = {prev}\n"
            f"  지금 들어온 값: {key}\n"
            "  → RewardsCfg 의 모든 kick_inside_mdp 항에서 min_strike_speed, "
            "window_steps, asset_cfg(body_names/preserve_order) 를 동일하게 맞추세요."
        )


def _target_dir_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """목표방향 world xy 단위벡터 (N,2). 버퍼 미설정이면 정면(+x)을 기본값으로."""
    if not hasattr(env, "kick_target_dir_w"):
        d = torch.zeros(env.num_envs, 2, device=env.device)
        d[:, 0] = 1.0
        return d
    tdir = env.kick_target_dir_w[..., :2]
    return tdir / (torch.norm(tdir, dim=-1, keepdim=True) + _EPS)


def _update_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    min_strike_speed: float,
    window_steps: int,
) -> None:
    """스텝당 1 회 — 접촉 판정 / 강타 래치 / 접촉 스텝 누적을 모두 처리한다.

    Args:
        asset_cfg:        발 2 개. body 순서는 반드시 [우발, 좌발]
                          (MX_FEET, preserve_order=True).
        ball_cfg:         공 RigidObject.
        min_strike_speed: 이 속도(목표방향 성분, m/s) 이상으로 접촉해야 "강타"로
                          인정해 래치한다. 걷는 중 스치는 속도(~0.5~1.2)보다
                          충분히 높게 잡아야 드리블이 래치를 훔치지 못한다.
                          ★ 이 모듈의 최우선 튜너블. metric_max_vlong /
                            metric_latch_rate 를 보고 조정한다.
        window_steps:     래치 후 보상을 유지할 스텝 수.
    """
    _ensure_buffers(env)
    _assert_params_consistent(env, asset_cfg, min_strike_speed, window_steps)

    step = int(env.common_step_counter)
    if env.ins_last_step == step:
        return  # 이번 스텝은 이미 갱신됨 (다른 보상 항이 먼저 호출)
    env.ins_last_step = step

    dev = env.device
    n = env.num_envs

    # ── 에피소드 리셋 처리 ───────────────────────────────────────────────
    # 별도 EventTerm 을 등록하지 않고 episode_length_buf 로 감지한다.
    # IsaacLab step() 은 episode_length_buf 를 먼저 증가시키고 보상을 계산하므로
    # 리셋 직후 첫 보상 계산 시점의 값은 1 이다. (휴리스틱이지만 현행 순서에선 정확)
    just_reset = kick_clock_steps(env) <= 1
    if bool(just_reset.any()):
        env.ins_latched = torch.where(
            just_reset, torch.zeros_like(env.ins_latched), env.ins_latched
        )
        z = torch.zeros_like(env.ins_quality)
        env.ins_quality = torch.where(just_reset, z, env.ins_quality)
        env.ins_strike_speed = torch.where(just_reset, z, env.ins_strike_speed)
        env.ins_best_score = torch.where(just_reset, z, env.ins_best_score)
        env.ins_window = torch.where(just_reset, z, env.ins_window)
        env.ins_touch_steps = torch.where(just_reset, z, env.ins_touch_steps)
        env.ins_run = torch.where(just_reset, z, env.ins_run)
        env.ins_max_run = torch.where(just_reset, z, env.ins_max_run)
        env.ins_max_vlong = torch.where(just_reset, z, env.ins_max_vlong)

    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    fids = asset_cfg.body_ids

    foot_pos = robot.data.body_pos_w[:, fids, :]              # (N,2,3)
    foot_vel = robot.data.body_lin_vel_w[:, fids, :]          # (N,2,3)
    foot_quat = robot.data.body_quat_w[:, fids, :]            # (N,2,4)
    ball_pos = ball.data.root_pos_w.unsqueeze(1)              # (N,1,3)

    # ── 접촉 게이트 ─────────────────────────────────────────────────────
    d = torch.norm(foot_pos - ball_pos, dim=-1)               # (N,2)
    excess = (d - CONTACT_RADIUS).clamp(min=0.0)
    gate = torch.exp(-torch.square(excess) / (CONTACT_FALLOFF ** 2))   # (N,2)
    contact = gate > 0.5                                       # (N,2)

    # ── 발 안쪽면 법선 (world) ──────────────────────────────────────────
    # medial = inside_sign · (발 로컬 +y) 를 world 로 회전.
    inside_sign = torch.tensor(_INSIDE_SIGN, device=dev)       # [우, 좌]
    y_local = torch.zeros(n, 2, 3, device=dev)
    y_local[:, :, 1] = inside_sign                             # (0, ±1, 0)
    medial_w = quat_apply(
        foot_quat.reshape(-1, 4), y_local.reshape(-1, 3)
    ).reshape(n, 2, 3)                                         # (N,2,3)

    # 법선의 수평(xy) 성분만 쓴다. 발이 롤 되어 안쪽면이 하늘/땅을 보면
    # xy 성분의 크기가 줄어드는데, 이 크기를 그대로 품질에 곱해(planar)
    # "안쪽면이 지면에 수직으로 서 있어야 한다"는 조건을 자동으로 넣는다.
    medial_xy = medial_w[..., :2]                              # (N,2,2)
    medial_norm = torch.norm(medial_xy, dim=-1)                # (N,2)  = planar
    medial_hat = medial_xy / (medial_norm.unsqueeze(-1) + _EPS)

    # ── 목표방향 정렬 ───────────────────────────────────────────────────
    tdir = _target_dir_xy(env)                                 # (N,2)
    t_exp = tdir.unsqueeze(1)                                  # (N,1,2)

    # (a) 면 정렬: 안쪽면 법선이 목표를 향하는 정도. +1 인사이드 / 0 발등 / -1 아웃
    align_face = (medial_hat * t_exp).sum(dim=-1).clamp(min=0.0)       # (N,2)

    # (b) 속도 정렬: 발이 목표방향으로 실제로 밀고 나가는 정도
    foot_v_xy = foot_vel[..., :2]                              # (N,2,2)
    v_along = (foot_v_xy * t_exp).sum(dim=-1)                  # (N,2) [m/s]
    v_speed = torch.norm(foot_v_xy, dim=-1)                    # (N,2)
    align_vel = (v_along / (v_speed + _EPS)).clamp(min=0.0)    # (N,2) 방향만

    # (c) 품질 q = 면정렬 × 속도정렬 × 수평성.
    #     곱이라서 셋 중 하나만 어긋나도 크게 깎인다 → poke / slice 모두 저평가.
    quality = align_face * align_vel * medial_norm             # (N,2) ∈ [0,1]

    # ── 강타 판정 ───────────────────────────────────────────────────────
    # "접촉 중" + "목표방향 속도 성분이 min_strike_speed 초과" 인 발만 강타.
    # walk_kick v1_5b (킥 세기): env 별 문턱 env.ins_min_strike_override (N,) 가 있으면 그걸 쓴다.
    #   약한 패스는 발 속도가 1.8 m/s 에 못 미쳐 래치가 안 걸린다. 다른 환경은 이 속성을 만들지 않으므로 그대로.
    thr = getattr(env, "ins_min_strike_override", None)
    thr = min_strike_speed if thr is None else thr.unsqueeze(-1)
    strong = contact & (v_along > thr)                         # (N,2)

    # 두 발이 동시에 조건을 만족하면 더 빠른 쪽을 강타 발로 본다.
    score = torch.where(strong, v_along, torch.full_like(v_along, -1.0))
    best_v, best_i = score.max(dim=-1)                         # (N,), (N,)
    struck_now = strong.any(dim=-1)                            # (N,)
    q_now = quality.gather(1, best_i.unsqueeze(-1)).squeeze(-1)   # (N,)

    # ── 래치: 윈도우는 에피소드당 1 회, 값은 "더 나은 강타" 에만 갱신 ──────
    # 【왜 값만 갱신을 허용하는가】
    #   윈도우까지 완전히 1 회로 못박으면, 걷다가 스윙발이 공 근처를 빠르게
    #   지나가며 발등 자세로 먼저 래치되는 순간(q≈0) 그 에피소드는 배율이
    #   floor 로 고정되어 버린다. 뒤이어 제대로 된 인사이드 킥을 넣어도 보상이
    #   안 오르니, 정책 입장에선 "인사이드로 차 봐야 소용없다" 는 신호가 된다.
    # 【그런데 왜 farm 이 안 되는가】
    #   갱신 조건이 score = v_along × q 의 **단조 증가**다. 드리블은 v_along 이
    #   min_strike_speed 미만이라 애초에 strong 이 아니고, 설령 문턱을 낮춰도
    #   점수를 올리려면 "더 빠르게 + 더 인사이드로" 라는 원하는 방향뿐이다.
    #   결정적으로 **윈도우는 재개방되지 않으므로** 접촉을 끌어서 보상 총량을
    #   늘리는 것은 여전히 불가능하다(inside_swing 총량 상한 = window_steps).
    score_now = best_v * q_now                                 # (N,)
    first_latch = struck_now & (~env.ins_latched)
    improve = struck_now & env.ins_latched & (score_now > env.ins_best_score)
    do_update = first_latch | improve

    env.ins_quality = torch.where(do_update, q_now, env.ins_quality)
    env.ins_strike_speed = torch.where(do_update, best_v, env.ins_strike_speed)
    env.ins_best_score = torch.where(do_update, score_now, env.ins_best_score)
    # 윈도우는 최초 래치에서만 열린다 (재개방 없음 → 총량 상한 고정)
    env.ins_window = torch.where(
        first_latch,
        torch.full_like(env.ins_window, float(window_steps)),
        (env.ins_window - 1.0).clamp(min=0.0),
    )
    env.ins_latched = env.ins_latched | first_latch

    # ── 접촉 통계 (드리블 탐지 + 문턱 튜닝 근거) ────────────────────────
    in_contact = contact.any(dim=-1)                           # (N,)
    env.ins_in_contact = in_contact
    env.ins_touch_steps = env.ins_touch_steps + in_contact.float()

    # 연속 접촉 런: 접촉이 끊기면 0 으로 리셋, 최댓값은 따로 보관
    env.ins_run = torch.where(
        in_contact, env.ins_run + 1.0, torch.zeros_like(env.ins_run)
    )
    env.ins_max_run = torch.maximum(env.ins_max_run, env.ins_run)

    # 접촉 중 관측된 최대 목표방향 발속도 (래치 여부와 무관)
    v_long_contact = torch.where(contact, v_along, torch.zeros_like(v_along))
    env.ins_max_vlong = torch.maximum(
        env.ins_max_vlong, v_long_contact.max(dim=-1).values
    )


# ============================================================================
# ★ 주 보상 ① — 킥 파워 × 인사이드 품질 (기존 ball_velocity 를 대체)
# ============================================================================
def ball_velocity_inside_gated(
    env: ManagerBasedRLEnv,
    max_speed: float = 3.0,
    post_kick_window_s: float = 0.4,
    quality_floor: float = 0.3,
    min_strike_speed: float = 1.8,
    window_steps: int = 20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공의 목표방향 속도² 에 인사이드 품질을 **곱한** 보상 (N,). 양수 weight.

    reward = clamp(v_proj, 0, max_speed)² × (floor + (1 - floor) · q_latched)

    · 이 항 하나가 "세게" 와 "인사이드로" 를 동시에 담당한다. 항을 나누면 정책이
      싼 쪽(=품질만 따먹기)으로 새는데, 곱으로 묶으면 품질을 올리는 유일한 방법이
      실제로 세게 차는 것뿐이다.
    · quality_floor 는 발등킥에도 남겨두는 최소 배율이다. 0 으로 두면 학습 초기에
      "아직 인사이드가 아니니 아예 차지 말자"로 붕괴할 수 있다(과거 one-shot
      실험에서 킥률 1% 로 무너진 것과 같은 실패 모드). 0.3 → 발등킥은 30%,
      완전한 인사이드 킥은 100% 를 받는다. 학습이 붙으면 0.2 까지 낮춰도 된다.
      ★ quality_floor=1.0 으로 두면 kick_mdp.ball_velocity_toward_target 과
        수학적으로 완전히 동일해진다 → 인사이드 항만 끄는 A/B 스위치로 쓸 것.
    · post_kick_window_s: kick_time_s 래치 후 이 시간 동안만 보상(임펄스 구간만).
      굴러가는 공을 쫓아가 재킥하는 유인을 없앤다.

    Args:
        max_speed:        공 속도 보상 포화 지점 [m/s].
        post_kick_window_s: 킥 후 보상 유지 시간 [s].
        quality_floor:    인사이드가 아닐 때 남는 최소 배율 ∈ [0,1].
        min_strike_speed: 강타 래치 문턱 [m/s]. 이 모듈의 모든 항에서 동일해야 함.
        window_steps:     래치 윈도우. 이 모듈의 모든 항에서 동일해야 함.
    """
    _update_state(env, asset_cfg, ball_cfg, min_strike_speed, window_steps)

    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    tdir = _target_dir_xy(env)
    proj = (v_xy * tdir).sum(dim=-1)
    base = torch.clamp(proj, min=0.0, max=max_speed) ** 2

    # 인사이드 품질 배율. 아직 래치 전이면 q=0 → floor 배율만 적용된다.
    mult = quality_floor + (1.0 - quality_floor) * env.ins_quality
    reward = base * mult

    # 킥 임펄스 구간만 게이팅 (kick_mdp.ball_velocity_toward_target 과 동일 규칙)
    if hasattr(env, "kick_time_s"):
        current_t = kick_clock_s(env)
        elapsed = current_t - env.kick_time_s
        past = (env.kick_time_s >= 0) & (elapsed > post_kick_window_s)
        reward = torch.where(past, torch.zeros_like(reward), reward)

    return reward


# ============================================================================
# ★ 주 보상 ② — 인사이드 스윙 shaping (탐색용, 1 회 래치)
# ============================================================================
def inside_swing_shaping(
    env: ManagerBasedRLEnv,
    ref_speed: float = 2.5,
    min_strike_speed: float = 1.8,
    window_steps: int = 20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """래치된 (인사이드 품질 × 정규화 타격속도) 를 윈도우 동안 dense 로 (N,). +weight.

    ①만으로는 학습 초기에 "안쪽면으로 정렬해서 스윙" 이라는 동작을 찾기 어렵다
    (공 속도라는 결과에만 보상이 걸려 있어 신호가 늦다). 이 항은 **발의 동작
    자체**에 보상을 줘서 탐색을 돕는다.

    farm 방지 장치:
      · 에피소드당 1 회만 래치 → 접촉을 오래 유지해도 추가 이득 없음
      · min_strike_speed 이상의 강타에서만 래치 → 느린 드리블 터치로는 못 엶
      · dense 로 보이지만 총량은 window_steps 로 상한이 고정 (≤ 1.0 × window)

    Args:
        ref_speed:        타격속도 정규화 기준 [m/s]. 이 속도면 1.0.
        min_strike_speed: 강타 래치 문턱 [m/s]. 이 모듈의 모든 항에서 동일해야 함.
        window_steps:     래치 후 보상 스텝 수. 이 모듈의 모든 항에서 동일해야 함.
    """
    _update_state(env, asset_cfg, ball_cfg, min_strike_speed, window_steps)

    speed_norm = (env.ins_strike_speed / max(ref_speed, _EPS)).clamp(0.0, 1.0)
    active = (env.ins_window > 0.0).float()
    return active * env.ins_quality * speed_norm


# ============================================================================
# ★ 주 보상 ③ — 접촉 시간 예산 벌점 (드리블 직접 차단)
# ============================================================================
def touch_budget_penalty(
    env: ManagerBasedRLEnv,
    free_steps: int = 10,
    min_strike_speed: float = 1.8,
    window_steps: int = 20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """에피소드 누적 접촉 스텝이 예산을 넘은 뒤의 접촉에 스텝당 1.0 벌점 (N,). 음수 weight.

    킥과 드리블을 가르는 가장 직접적인 물리량은 "접촉 지속 시간" 이다.
      · 제대로 된 강타: 발이 공을 스쳐 지나가므로 접촉 ≈ 3~5 스텝 (0.06~0.10 s)
      · 드리블:        발이 공을 계속 밀고 가므로 접촉 = 수십~150 스텝
    free_steps 를 10(=0.20 s)으로 두면 "셋업 터치 한 번 + 강타" 정도는 벌을 안
    받고, 드리블만 스텝당 벌을 받는다. weight -1.0 이면 드리블 100 스텝 = -100
    으로, 어떤 dense farm 으로 얻을 수 있는 이득보다도 크다.

    ※ 누적 방식이라 "셋업 터치가 길어져 과벌" 이 이론상 가능하다.
       metric_max_run(최장 연속 접촉) 이 free_steps 를 자주 넘는데 드리블은
       아니라면 free_steps 를 12~14 로 올린다. 데이터 보고 판단할 것.

    Args:
        free_steps: 벌점 없이 허용되는 누적 접촉 스텝 수.
    """
    _update_state(env, asset_cfg, ball_cfg, min_strike_speed, window_steps)
    over = env.ins_touch_steps > float(free_steps)
    return (env.ins_in_contact & over).float()


# ============================================================================
# ★ 보조 — 공이 목표에서 옆으로 새는 성분 벌점 (v1_2 ① 유지, 창 게이팅 동일)
# ============================================================================
def ball_offaxis_penalty(
    env: ManagerBasedRLEnv,
    post_kick_window_s: float = 0.4,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공 속도의 목표방향 수직성분을 제곱 벌점 (N,). 음수 weight.

    투영 기반 보상은 옆으로 새도 목표 성분만큼은 받으므로 부정확을 허용한다.
    이 항이 수직성분을 벌해 "정확히 목표로" 를 만든다.
    weight 는 작게(-0.2 ~ -0.3). 크게 주면 "안 차는 게 이득" 이 되어 킥이 죽는다
    (v1_2 밤튜닝에서 -1.0 이 하드킥을 억제했던 그 실패).

    ※ 이 함수는 _update_state 를 호출하지 않는다(공 속도만 보면 되므로).
      따라서 파라미터 일관성 검증 대상이 아니다.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    tdir = _target_dir_xy(env)
    along = (v_xy * tdir).sum(dim=-1, keepdim=True)
    v_perp = v_xy - along * tdir
    penalty = (v_perp * v_perp).sum(dim=-1)

    if hasattr(env, "kick_time_s"):
        current_t = kick_clock_s(env)
        elapsed = current_t - env.kick_time_s
        past = (env.kick_time_s >= 0) & (elapsed > post_kick_window_s)
        penalty = torch.where(past, torch.zeros_like(penalty), penalty)
    return penalty


# ============================================================================
# ★ 보조 — 공 근처에서는 골반 yaw 편차 벌점을 풀어 준다
# ----------------------------------------------------------------------------
# 인사이드 킥은 정의상 골반 yaw 를 돌려 발 안쪽면을 목표로 향하게 하는 동작이다.
# 그런데 joint_deviation_hip_yaw(weight -0.5)는 yaw 편차 자체를 항상 벌한다.
# 45° 회전 = 0.785 rad → 스텝당 -0.39. 킥 자세를 만들 때마다 벌을 받으니
# "골반은 안 돌리고 발목만 살짝 트는" poke 로 흐르기 쉽다 (지난 45° 트위스트 실패).
# 공이 가까울 때만 벌점을 꺼서, 걸을 때의 자세 정돈은 유지하고 킥 자세는 풀어 준다.
# ============================================================================
def joint_deviation_l1_ball_gated(
    env: ManagerBasedRLEnv,
    release_radius: float = 0.5,
    falloff: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절 기본자세 L1 편차 벌점을 공-로봇 거리로 게이팅 (N,). 음수 weight.

    d >= release_radius + falloff : 벌점 100% (평소 보행 자세 유지)
    d <= release_radius           : 벌점 ~0%  (킥 자세 자유)
    사이는 부드럽게 전환 → 값 불연속으로 value function 이 흔들리지 않는다.

    ★ release_radius=0.0 으로 두면 게이트가 사실상 항상 1 이라
      mdp.joint_deviation_l1 과 동일하게 동작한다 → A/B 스위치로 쓸 것.

    Args:
        release_radius: 이 거리 안쪽에서는 벌점을 해제 [m].
        falloff:        전환 폭 [m].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - \
        asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(err), dim=1)

    d = torch.norm(
        ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    # d 가 release_radius 를 넘어선 만큼을 falloff 로 나눠 0→1 로 부드럽게 켠다.
    gate = ((d - release_radius) / max(falloff, _EPS)).clamp(0.0, 1.0)
    return penalty * gate


# ============================================================================
# ★ 메트릭 (weight=0.0 으로 등록해 로그로만 확인) — "데이터로 검증" 용
# ----------------------------------------------------------------------------
# 학습 곡선만 보면 왜 드리블이 나왔는지 알 수 없다. 아래를 항상 같이 켜서
# TensorBoard 에서 직접 확인한다.
#
# ★ 설계 주의 ★ 메트릭 함수는 **파라미터를 받지 않는다.**
#   파라미터를 받게 하면 _update_state 의 "스텝당 1 회" 가드와 얽혀
#   min_strike_speed 불일치 함정의 표면적이 넓어진다. 메트릭은 버퍼를 읽기만
#   하므로, 선언 순서에 따라 1 스텝 늦을 수 있으나 로깅에는 전혀 문제없다.
# ============================================================================
def metric_inside_quality(env: ManagerBasedRLEnv) -> torch.Tensor:
    """래치된 인사이드 품질 q (N,). 목표 > 0.6. weight=0.0 로 등록."""
    _ensure_buffers(env)
    return env.ins_quality


def metric_latch_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """강타 래치 여부 0/1 (N,). 평균이 곧 래치율. weight=0.0 로 등록.

    이게 0 에 가까우면 min_strike_speed 가 너무 높은 것이다.
    metric_max_vlong 과 같이 보고 문턱을 내린다.
    """
    _ensure_buffers(env)
    return env.ins_latched.float()


def metric_max_vlong(env: ManagerBasedRLEnv) -> torch.Tensor:
    """접촉 중 관측된 최대 목표방향 발속도 [m/s] (N,). weight=0.0 로 등록.

    ★ min_strike_speed 튜닝의 직접 근거 ★
      래치율이 낮은데 이 값이 1.2 근처에서 정체 → 문턱을 1.2 아래로 내린다.
      래치율이 높은데 드리블이 남아 있음 → 문턱이 너무 낮아 느린 터치가
      래치를 훔치는 중. 문턱을 올린다.
    """
    _ensure_buffers(env)
    return env.ins_max_vlong


def metric_touch_steps(env: ManagerBasedRLEnv) -> torch.Tensor:
    """이번 에피소드 누적 접촉 스텝 수 (N,). 킥 3~6 / 드리블 40+. weight=0.0."""
    _ensure_buffers(env)
    return env.ins_touch_steps


def metric_max_run(env: ManagerBasedRLEnv) -> torch.Tensor:
    """최장 연속 접촉 런 [스텝] (N,). touch_budget free_steps 튜닝 근거. weight=0.0.

    누적(touch_steps)은 크지만 최장 런(max_run)이 짧으면 드리블이 아니라
    "여러 번 툭툭 건드림" 이다. 두 지표를 같이 봐야 구분된다.
    """
    _ensure_buffers(env)
    return env.ins_max_run


def metric_ball_dir_err_deg(
    env: ManagerBasedRLEnv,
    min_speed: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공 진행방향과 목표방향의 각도 오차 [deg] (N,). weight=0.0 로 등록.

    최종 목표 지표. 공이 min_speed 미만이면(=아직 안 찼거나 멈춤) 방향이
    무의미하므로 0 을 낸다. _update_state 를 안 쓰므로 일관성 검증 대상 아님.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    speed = torch.norm(v_xy, dim=-1)
    tdir = _target_dir_xy(env)
    v_hat = v_xy / (speed.unsqueeze(-1) + _EPS)
    cos = (v_hat * tdir).sum(dim=-1).clamp(-1.0, 1.0)
    err = torch.rad2deg(torch.acos(cos))
    return torch.where(speed > min_speed, err, torch.zeros_like(err))
