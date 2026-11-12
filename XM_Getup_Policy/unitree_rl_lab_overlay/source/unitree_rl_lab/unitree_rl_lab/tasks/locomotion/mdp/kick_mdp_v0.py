# kick_mdp_v0.py
# ============================================================================
# walk_kick_v0 전용 MDP 함수 모음
# ----------------------------------------------------------------------------
# 기존 kick_mdp.py 는 건드리지 않는다. 설계 원칙이 여러 군데 달라져서
# 같은 파일에 섞으면 어느 쪽 규약인지 헷갈린다.
#
# ── kick_mdp.py 대비 핵심 변경 ─────────────────────────────────────────
#   1. 공 스폰이 극좌표 + 넓은 범위
#      기존: x(0.3,0.5) y(-0.05,0.05) — 폭 10cm 상자. 사실상 상수라
#            네트워크가 입력이 아니라 바이어스로 학습했다. 실기에서
#            다른 값이 들어오면 첫 레이어부터 외삽 → 발산.
#      변경: 거리 0.3~1.0 m, 방위 ±60° (커리큘럼으로 확장)
#
#   2. 목표 각도 랜덤화
#      기존: TARGET_DIR_ANGLE_RANGE=(0,0) — 항상 정면. 관측이 상수.
#      변경: ±60° (커리큘럼으로 확장)
#
#   3. 시뮬 전용 상태가 관측에 새어나가지 않음
#      기존: kick_time_s 래치로 속도명령 관측을 0으로 떨어뜨렸다.
#            그 래치는 실기에 존재하지 않아 관측 분포가 어긋난다.
#      변경: 래치 없음. 게이팅은 전부 "정책이 보는 값"(공 거리)으로만.
#
#   4. 킥 후 종료 없음 — 공만 재배치하고 로봇 상태는 유지
#      킥 직후 "정지 모드"가 별도 상태가 되면 실기에서 그 진입 조건을
#      재현해야 한다. 연속 킥으로 두면 킥이 보행 중 일시적 변조로 남는다.
#
#   5. 리워드가 전부 dense + potential-based
#      기존: proj² (제곱이라 극단 스윙 유도), kick_success_bonus (스파스라
#            단일 critic 의 value 분산을 키움)
#      변경: Δ거리 shaping (Ng et al. 1999 policy invariance) + 속도 매칭
#
#   6. 인사이드킥 (arXiv 2511.03996 Table 3)
#      접촉 시 발의 "측면" 속도에 보상, "전방" 속도에 페널티.
#      정면 스윙킥은 시상면 각운동량이 크고 스윙 다리의 피크 가감속을
#      요구하는데, 액추에이터 모델 오차가 정확히 그 피크에서 최대가 된다.
#      측면킥은 필요 피크 속도가 낮아 sim2real 갭에 훨씬 둔감하다.
# ============================================================================

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

# 형제 모듈 직접 import — 패키지 __init__ 을 거치지 않아 순환 import 가 없다.
from . import htwk_gait_v01 as _gait
from . import rewards as _rewards
from . import reset_hold as _reset_hold

# ----------------------------------------------------------------------------
# 상수
# ----------------------------------------------------------------------------
BALL_RADIUS = 0.065

# 발 링크 중심에서 공 중심까지의 "접촉으로 볼" 거리.
# 발 반길이(~0.09) + 공 반지름(0.065) 에 여유를 조금 더한 값.
CONTACT_RADIUS = 0.17

# 접촉 게이트의 감쇠 폭. 접촉 반경 밖으로 이 정도 벗어나면 게이트가 꺼진다.
# 하드 게이트(0/1)로 두면 리워드가 스파스해져 학습이 느려지므로 완만하게.
CONTACT_FALLOFF = 0.05


# ============================================================================
# 내부 버퍼
# ============================================================================
def _ensure_buffers(env: ManagerBasedRLEnv) -> None:
    """킥 관련 상태 버퍼를 최초 1회 생성한다.

    전부 "리워드/이벤트 전용" 이거나 "실기에서도 같은 규칙으로 재현
    가능한 값" 이다. 시뮬레이터만 아는 상태가 관측으로 새어나가지 않도록
    주의할 것 — 그게 기존 walk_kick 의 발산 원인 중 하나였다.
    """
    if hasattr(env, "_kick_v0_ready"):
        return

    n = env.num_envs
    dev = env.device

    # 목표점(월드 XY) 과 공→목표 단위방향(월드). 리셋 시 고정.
    env.kick_target_pos_w = torch.zeros(n, 2, device=dev)
    env.kick_target_dir_w = torch.zeros(n, 2, device=dev)
    env.kick_target_dir_w[:, 0] = 1.0

    # Δ거리 shaping 용 직전 거리
    env.kick_prev_robot_ball_d = torch.zeros(n, device=dev)
    env.kick_prev_ball_goal_d = torch.zeros(n, device=dev)
    # 리셋/재배치 직후 1 스텝은 Δ 가 튀므로 무효 처리
    env.kick_dist_valid = torch.zeros(n, dtype=torch.bool, device=dev)

    # 관측 지연 링버퍼 (파이프라인 지연 모사). 크기는 최초 호출 시 확정.
    env.kick_obs_delay_buf = None
    env.kick_obs_last_step = -1

    # 접촉 직후 K 스텝 동안 속도 매칭 보상을 켜기 위한 카운터
    env.kick_contact_recent = torch.zeros(n, device=dev)

    # 정체 판정용 속도 EMA
    env.kick_speed_ema = torch.zeros(n, device=dev)

    # 커리큘럼 스케일 (0~1). 곱해져서 실제 샘플 범위를 만든다.
    env.kick_bearing_scale = torch.ones((), device=dev)
    env.kick_target_angle_scale = torch.ones((), device=dev)

    # 커리큘럼이 리워드 가중치를 복원할 수 있도록 목표값을 저장
    env.kick_reward_target_w = {}

    # ── discrete 킥 상태 (reward/termination 전용, 관측 아님) ──────────
    # 킥 후 "멈춤" 을 관측 래치 없이 리워드-사이드로 구현하기 위한 상태.
    # 이 값들은 절대 관측으로 새어나가면 안 된다 (sim2real 원칙).
    env.kick_v0_kicked = torch.zeros(n, dtype=torch.bool, device=dev)
    # 킥이 감지된 시점의 common_step_counter (kick_resolved hold 계산용)
    env.kick_v0_kick_step = torch.zeros(n, dtype=torch.long, device=dev)
    # 이 스텝까지 안 차면 실패 종료 (no_kick_deadline)
    env.kick_v0_deadline_step = torch.zeros(n, dtype=torch.long, device=dev)

    env._kick_v0_ready = True


def _yaw(robot: Articulation) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    return yaw


def _to_robot_frame(vec_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """월드 XY 벡터를 로봇 yaw 프레임으로 회전. vec_w (N,2) -> (N,2)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    x = c * vec_w[:, 0] + s * vec_w[:, 1]
    y = -s * vec_w[:, 0] + c * vec_w[:, 1]
    return torch.stack([x, y], dim=-1)


def _ball_robot_xy(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(공 월드 XY, 로봇 월드 XY, 로봇 yaw) 를 한 번에 반환."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    return (
        ball.data.root_pos_w[:, :2],
        robot.data.root_pos_w[:, :2],
        _yaw(robot),
    )


# ============================================================================
# 이벤트 — 공 스폰 / 목표 샘플링 / 재배치
# ============================================================================
def _sample_ball_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    dist_range: tuple[float, float],
    bearing_range_deg: tuple[float, float],
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """로봇 기준 극좌표로 공 월드 위치 (M,3) 를 샘플링한다.

    극좌표를 쓰는 이유: 직교 상자로 뽑으면 거리와 방위가 얽혀서
    "가까운데 옆쪽" 같은 조합이 잘 안 나온다. 정책이 ball_pos_b 를
    실제로 쓰려면 두 축이 독립적으로 변해야 한다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    dev = env.device
    m = len(env_ids)

    d = torch.empty(m, device=dev).uniform_(*dist_range)

    # 커리큘럼 스케일 적용. 방위 범위는 반드시 0 을 중심으로 좌우 대칭이어야
    # 한다. 비대칭이면 "한쪽으로만 차는" 분포가 되어 미러 손실과 싸운다.
    b_lo = math.radians(bearing_range_deg[0]) * float(env.kick_bearing_scale)
    b_hi = math.radians(bearing_range_deg[1]) * float(env.kick_bearing_scale)
    bearing = torch.empty(m, device=dev).uniform_(b_lo, b_hi)

    yaw = _yaw(robot)[env_ids]
    ang = yaw + bearing

    pos = torch.zeros(m, 3, device=dev)
    pos[:, 0] = robot.data.root_pos_w[env_ids, 0] + d * torch.cos(ang)
    pos[:, 1] = robot.data.root_pos_w[env_ids, 1] + d * torch.sin(ang)
    pos[:, 2] = env.scene.env_origins[env_ids, 2] + BALL_RADIUS
    return pos


def _write_ball(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pos: torch.Tensor,
    ball_cfg: SceneEntityCfg,
) -> None:
    """공의 위치를 쓰고 속도를 0 으로 초기화."""
    ball: RigidObject = env.scene[ball_cfg.name]
    root = ball.data.default_root_state[env_ids].clone()
    root[:, 0:3] = pos
    root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root[:, 7:13] = 0.0
    ball.write_root_state_to_sim(root, env_ids=env_ids)


def _sample_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_xy: torch.Tensor,
    dist_range: tuple[float, float],
    angle_range_deg: tuple[float, float],
    robot_cfg: SceneEntityCfg,
) -> None:
    """공 위치를 기준으로 목표점을 샘플링해 버퍼에 기록한다.

    방향 기준이 "로봇 heading + 오프셋" 이고 월드에 고정된다. 관측은
    이 고정 방향을 매 스텝 현재 몸통 프레임으로 재표현하므로, 로봇이
    돌면 관측값이 변한다. 즉 정책은 "고정된 월드 목표가 지금 내 기준
    어느 쪽인가" 를 본다. kick_mdp.py 의 설계를 그대로 유지했다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    dev = env.device
    m = len(env_ids)

    a_lo = math.radians(angle_range_deg[0]) * float(env.kick_target_angle_scale)
    a_hi = math.radians(angle_range_deg[1]) * float(env.kick_target_angle_scale)
    off = torch.empty(m, device=dev).uniform_(a_lo, a_hi)

    ang = _yaw(robot)[env_ids] + off
    dir_w = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

    d = torch.empty(m, device=dev).uniform_(*dist_range)

    env.kick_target_dir_w[env_ids] = dir_w
    env.kick_target_pos_w[env_ids] = ball_xy + dir_w * d.unsqueeze(-1)


def _refresh_kick_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_xy: torch.Tensor,
    robot_cfg: SceneEntityCfg,
) -> None:
    """공/목표가 새로 놓인 직후, Δ shaping 의 기준 거리를 새 값으로 채운다.

    ★ 이게 없으면 킥 성공이 페널티가 된다 ★
    공이 목표에 닿아 재배치되면 prev_ball_goal_d 는 옛 공 기준(~0.4 m),
    새 거리는 1.5~4.0 m 다. delta = 0.4 - 2.5 = -2.1 -> clamp -0.15
    -> weight 20 -> **-3.0**. 즉 골을 넣을 때마다 -3.0 을 맞는다.
    continuous kicking 신호를 정면으로 갉아먹는다.

    kick_dist_valid 플래그만으로는 못 막는다. RewardManager 가 term 을
    선언 순서대로 평가하는데, ball_approach 가 먼저 실행되면서 valid 를
    전역 True 로 바꾸면 뒤에 오는 ball_goal 은 한 번도 마스킹되지 않는다.
    플래그 순서에 의존하는 대신 기준값 자체를 맞추는 게 견고하다.

    ── 타이밍 근거 ────────────────────────────────────────────────
    IsaacLab step() 순서는
        물리 -> 리워드 계산 -> 종료 판정 -> reset 이벤트 -> interval 이벤트 -> 관측
    이다. 즉 이 함수가 도는 시점은 "이번 스텝 리워드 계산이 끝난 뒤" 이고,
    다음 스텝 리워드는 여기서 쓴 prev 를 기준으로 계산된다. 따라서
    delta 가 정확히 "한 제어스텝 분의 이동" 이 되어 불연속이 사라진다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    robot_xy = robot.data.root_pos_w[env_ids, :2]

    env.kick_prev_robot_ball_d[env_ids] = torch.norm(
        ball_xy - robot_xy, dim=-1
    )
    env.kick_prev_ball_goal_d[env_ids] = torch.norm(
        ball_xy - env.kick_target_pos_w[env_ids], dim=-1
    )
    env.kick_dist_valid[env_ids] = True
    env.kick_contact_recent[env_ids] = 0.0

    # 관측 지연 링버퍼도 새 공 위치로 채운다.
    # 안 채우면 재배치 후 latency_steps(2 스텝 = 40 ms) 동안 정책이
    # "이미 없어진 공" 을 본다. 공 텔레포트는 물리적 이동이 아니므로
    # 지연으로 모사할 대상이 아니다. 경미하지만 같은 종류의 OOD 다.
    if env.kick_obs_delay_buf is not None:
        yaw = _yaw(robot)[env_ids]
        rel = ball_xy - robot_xy
        c, s = torch.cos(yaw), torch.sin(yaw)
        rel_b = torch.stack(
            [c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]],
            dim=-1,
        )
        k = env.kick_obs_delay_buf.shape[1]
        env.kick_obs_delay_buf[env_ids] = (
            rel_b.unsqueeze(1).expand(-1, k, -1).clone()
        )


def reset_ball_and_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_dist_range: tuple[float, float] = (0.3, 1.0),
    ball_bearing_range_deg: tuple[float, float] = (-60.0, 60.0),
    target_dist_range: tuple[float, float] = (1.5, 4.0),
    target_angle_range_deg: tuple[float, float] = (-60.0, 60.0),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """에피소드 리셋 시 공과 목표를 함께 샘플링한다.

    ★ 이벤트 선언 순서 주의 ★
    반드시 reset_base(로봇 root 랜덤화) "다음" 에 선언해야 한다.
    IsaacLab EventManager 는 같은 mode 안에서 선언 순서대로 실행하므로,
    앞에 두면 랜덤화 전 로봇 자세를 기준으로 공을 놓게 된다.
    """
    _ensure_buffers(env)
    pos = _sample_ball_positions(
        env, env_ids, ball_dist_range, ball_bearing_range_deg, robot_cfg
    )
    _write_ball(env, env_ids, pos, ball_cfg)
    _sample_target(
        env, env_ids, pos[:, :2], target_dist_range,
        target_angle_range_deg, robot_cfg,
    )
    # Δ shaping 기준 거리와 관측 지연 버퍼를 새 공 기준으로 재설정
    _refresh_kick_state(env, env_ids, pos[:, :2], robot_cfg)
    # 정체 페널티가 리셋 직후 터지지 않도록 EMA 를 높은 값에서 시작
    env.kick_speed_ema[env_ids] = 1.0

    # ── discrete 킥 상태 리셋 + no-kick 데드라인 계산 ──────────────────
    # 데드라인 = 공까지 도달예상시간 + 여유. tight 하게 잡아야 "안 차고
    # 배회하며 보상만 쌓기"(킥 회피)가 이득이 안 된다.
    #   도달시간 ≈ 공거리 / 접근속도(nominal 0.25 m/s)
    env.kick_v0_kicked[env_ids] = False
    env.kick_v0_kick_step[env_ids] = 0
    dist = env.kick_prev_robot_ball_d[env_ids]        # 방금 _refresh 로 갱신됨
    # 여유 2.0 s: scratch 초반(걷기 미숙) 에 데드라인이 너무 빡빡해
    # 접근도 못 하고 종료되는 걸 막는다. reset_hold 0.5 s 도 흡수.
    deadline_s = dist / 0.25 + 2.0
    deadline_steps = (deadline_s / env.step_dt).long()
    env.kick_v0_deadline_step[env_ids] = (
        int(env.common_step_counter) + deadline_steps
    )


def respawn_ball_when_done(
    env: ManagerBasedRLEnv,
    # ★ 기본값(=None) 을 두면 안 된다 ★
    # IsaacLab manager_base._resolve_common_term_cfg 는 인자를 "기본값
    # 없음/있음" 으로 나눠 재정렬한 뒤 앞 2개(env, env_ids)를 건너뛴다.
    # env_ids 에 기본값이 있으면 "기본값 있음" 그룹으로 밀려 앞 2개에서
    # 빠지고, term 파라미터 집합 비교가 어긋나 환경 빌드 시 ValueError 가
    # 난다. interval 이벤트는 env_ids 를 항상 위치 인자로 넘기므로 불필요.
    env_ids: torch.Tensor | None,
    max_robot_dist: float = 2.5,
    goal_reach_dist: float = 0.4,
    ball_dist_range: tuple[float, float] = (0.3, 1.0),
    ball_bearing_range_deg: tuple[float, float] = (-60.0, 60.0),
    target_dist_range: tuple[float, float] = (1.5, 4.0),
    target_angle_range_deg: tuple[float, float] = (-60.0, 60.0),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """공이 목표에 닿았거나 너무 멀어지면 "공만" 재배치한다.

    ★ 로봇 상태는 건드리지 않는다 ★
    이게 이 설계의 핵심이다. 기존처럼 킥 성공 시 에피소드를 종료하면
    정책이 "킥 → 종료 → 정지" 를 하나의 패턴으로 학습한다. 그러면
    실기에서 킥 직후 얼어붙고, 그 상태를 재현할 조건도 없다.

    공만 갈아끼우면 킥은 보행 중 일시적 변조로 남고, 정책은 한 에피소드
    안에서 접근-킥-재접근을 반복하며 전환을 자연스럽게 배운다.
    (arXiv 2511.03996 4.1 절의 continuous kicking 과 같은 취지)

    interval 모드 + interval_range_s=(0.02, 0.02) 로 매 제어스텝 호출한다.
    """
    _ensure_buffers(env)
    ball_xy, robot_xy, _ = _ball_robot_xy(env, ball_cfg, robot_cfg)

    d_robot = torch.norm(ball_xy - robot_xy, dim=-1)
    d_goal = torch.norm(ball_xy - env.kick_target_pos_w, dim=-1)

    need = (d_robot > max_robot_dist) | (d_goal < goal_reach_dist)

    # interval 모드는 타이머가 만료된 env 만 넘겨준다. 교집합을 취한다.
    if env_ids is not None:
        mask = torch.zeros_like(need)
        mask[env_ids] = True
        need = need & mask

    ids = need.nonzero(as_tuple=False).flatten()
    if len(ids) == 0:
        return

    pos = _sample_ball_positions(
        env, ids, ball_dist_range, ball_bearing_range_deg, robot_cfg
    )
    _write_ball(env, ids, pos, ball_cfg)
    _sample_target(
        env, ids, pos[:, :2], target_dist_range,
        target_angle_range_deg, robot_cfg,
    )
    # ★ 이 호출이 "골 넣으면 -3.0" 버그를 막는다 ★ (_refresh_kick_state 참조)
    _refresh_kick_state(env, ids, pos[:, :2], robot_cfg)


def update_kick_state(
    env: ManagerBasedRLEnv,
    # ★ 기본값 금지 ★ interval 이벤트는 env_ids 를 위치 인자로 넘긴다.
    env_ids: torch.Tensor | None,
    kick_speed_threshold: float = 1.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """매 스텝 공 속도를 보고 "킥됨" 플래그를 세운다 (한 번 서면 유지).

    interval 이벤트(매 제어스텝)로 호출한다. env_ids 는 무시하고 전체
    env 를 갱신한다 — 어차피 매 스텝 전체가 fire 한다.

    ★ 이 플래그는 reward / termination 에서만 읽는다. 관측에 절대 넣지
      않는다 ★ (sim2real 원칙: 시뮬 전용 상태가 관측에 새면 안 됨)

    임계 1.0 m/s: 살짝 스치는 건 킥으로 안 치고, 실제로 찬 것만 잡는다.
    """
    _ensure_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
    newly = (~env.kick_v0_kicked) & (speed > kick_speed_threshold)
    step = int(env.common_step_counter)
    env.kick_v0_kick_step = torch.where(
        newly,
        torch.full_like(env.kick_v0_kick_step, step),
        env.kick_v0_kick_step,
    )
    env.kick_v0_kicked = env.kick_v0_kicked | newly


# ============================================================================
# 관측 — 킥 후 속도명령 0 (옛 walk_kick commands_zero_on_kick_or_hold 복원)
# ============================================================================
def velocity_commands_zero_on_kick(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """속도명령 관측. reset_hold 중 OR 킥 후(kicked) 에는 0 으로 만든다.

    ★ 투터치 방지의 핵심 ★
    킥이 감지되면(감지 임계 0.01 = 첫 접촉) 속도명령 관측을 0 으로 떨어뜨려
    정책이 "멈춰" 를 보게 한다. 정책은 리워드가 아니라 관측에 반응하므로,
    관측이 여전히 "전진" 이면(reward 만 kick_aware) 로봇은 계속 나아가 두
    번째 터치를 한다. 관측을 0 으로 해야 로봇이 즉시 정지 → 재접촉 불가.

    ※ 이건 우리가 sim2real 때문에 뺐던 그 관측 래치다. 하지만 이게
      단일 킥을 만드는 검증된 메커니즘이고, 배포 시엔 상위 FSM 이 킥
      감지 후 cmd=0 을 보내 그대로 재현한다. (옛 walk_kick 구조 복원)
    """
    _ensure_buffers(env)
    # reset_hold 처리는 기존 함수가 담당 (hold 중 cmd=0)
    cmd = _reset_hold.generated_commands_with_reset_hold(
        env, command_name=command_name
    )
    post_kick = env.kick_v0_kicked
    if torch.any(post_kick):
        cmd = cmd.clone()
        cmd[post_kick] = 0.0
    return cmd


# ============================================================================
# 종료 판정 — discrete 에피소드 (킥 후 종료 / 안 차면 데드라인)
# ============================================================================
def kick_resolved(
    env: ManagerBasedRLEnv,
    hold_time_s: float = 1.0,
) -> torch.Tensor:
    """킥이 감지되고 hold_time_s 가 지나면 True (N,). 에피소드 종료.

    킥 직후 바로 끝내지 않고 hold 를 두는 이유: 그 사이 kick_aware 리워드로
    로봇이 "멈추는" 걸 학습할 여지를 준다. hold 후 종료로 컨베이어벨트
    (연속 드리블)를 원천 차단한다.
    """
    _ensure_buffers(env)
    hold_steps = int(round(hold_time_s / env.step_dt))
    elapsed = int(env.common_step_counter) - env.kick_v0_kick_step
    return env.kick_v0_kicked & (elapsed >= hold_steps)


def no_kick_deadline(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """데드라인까지 안 차면 True (N,). 실패 종료.

    데드라인은 reset 시 공거리/접근속도+여유로 tight 하게 잡힌다.
    "안 차고 배회하며 보상만 쌓기"(킥 회피)를 막는 유일한 장치다.
    """
    _ensure_buffers(env)
    return (~env.kick_v0_kicked) & (
        int(env.common_step_counter) >= env.kick_v0_deadline_step
    )


# ============================================================================
# 관측
# ============================================================================
def ball_pos_b_noisy(
    env: ManagerBasedRLEnv,
    noise_a: float = 0.03,
    noise_b: float = 0.01,
    latency_steps: int = 2,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """로봇 프레임 공 위치 (N,2). 거리 비례 노이즈 + 파이프라인 지연.

    노이즈 모델 sigma = noise_a * d + noise_b 는 arXiv 2511.03996
    Appendix E 의 형식이다. 계수는 D405 + YOLO 실측으로 교체할 것.
    거리에 비례시키는 이유는 스테레오 깊이 오차가 거리 제곱에 가깝게
    커지기 때문이고, 선형 근사로도 근거리 구간에서는 충분하다.

    지연은 링버퍼로 latency_steps 만큼 과거 값을 읽어 모사한다.
    50 Hz 제어에서 2 스텝 = 40 ms.

    ※ 이 함수는 policy 관측 전용이다. critic 은 ball_pos_b_clean 을 쓴다.
    """
    _ensure_buffers(env)
    ball_xy, robot_xy, yaw = _ball_robot_xy(env, ball_cfg, robot_cfg)

    rel_b = _to_robot_frame(ball_xy - robot_xy, yaw)

    d = torch.norm(rel_b, dim=-1, keepdim=True)
    sigma = noise_a * d + noise_b
    noisy = rel_b + torch.randn_like(rel_b) * sigma

    if latency_steps <= 0:
        return noisy

    k = latency_steps + 1
    if env.kick_obs_delay_buf is None or env.kick_obs_delay_buf.shape[1] != k:
        # 최초 호출 시 현재값으로 전부 채운다. 0 으로 채우면 에피소드
        # 시작 직후 latency_steps 만큼 관측이 통째로 OOD 가 된다.
        env.kick_obs_delay_buf = noisy.unsqueeze(1).repeat(1, k, 1)

    # ObservationManager 가 한 스텝에 두 번 호출하는 경로가 있을 수 있으므로
    # 스텝 카운터가 바뀐 경우에만 버퍼를 전진시킨다.
    step = int(env.common_step_counter)
    if step != env.kick_obs_last_step:
        env.kick_obs_delay_buf = torch.roll(env.kick_obs_delay_buf, -1, dims=1)
        env.kick_obs_last_step = step
    env.kick_obs_delay_buf[:, -1] = noisy

    return env.kick_obs_delay_buf[:, 0]


def ball_pos_b_clean(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """로봇 프레임 공 위치 (N,2), 노이즈/지연 없음. critic 전용."""
    ball_xy, robot_xy, yaw = _ball_robot_xy(env, ball_cfg, robot_cfg)
    return _to_robot_frame(ball_xy - robot_xy, yaw)


def ball_vel_b(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """로봇 프레임 공 속도 (N,2). critic 전용 특권 정보.

    킥 결과의 즉각적 지표라 value 추정 분산을 크게 줄인다.
    실기에서는 안정적으로 얻기 어려우므로 policy 에는 주지 않는다.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    return _to_robot_frame(ball.data.root_lin_vel_w[:, :2], _yaw(robot))


def target_dir_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """목표 방향 단위벡터 (N,2), 로봇 프레임.

    단위벡터로 주는 이유는 조건수다. 목표점 원시좌표 (x,y) 는 4~8 같은
    큰 수가 되어 첫 레이어 스케일이 나빠지는데, 단위벡터는 항상 O(1) 이다.
    거리는 target_dist_norm 으로 분리해서 준다.
    """
    _ensure_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    return _to_robot_frame(env.kick_target_dir_w, _yaw(robot))


def target_dist_norm(
    env: ManagerBasedRLEnv,
    max_dist: float = 5.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공→목표 거리 정규화 (N,1). clip(d, 0, max) / max.

    킥 세기를 결정하는 신호다. kick_speed_match 의 v_req 와 짝을 이룬다.
    """
    _ensure_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    d = torch.norm(ball.data.root_pos_w[:, :2] - env.kick_target_pos_w, dim=-1)
    return (d.clamp(0.0, max_dist) / max_dist).unsqueeze(-1)


# ============================================================================
# 거리 게이트 — 보행 리워드를 공 근처에서 완화
# ============================================================================
def ball_prox_gate(
    env: ManagerBasedRLEnv,
    d_near: float = 0.35,
    d_far: float = 1.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """공 거리에 따른 게이트 (N,). 멀면 1.0, d_near 안쪽이면 0.0.

    킥 순간에는 지지발을 심고 차는 발을 크게 흔들어야 하는데, 속도 추종과
    위상 일치 보상이 살아 있으면 정면으로 싸운다. 이 게이트로 완화한다.

    ★ 게이팅 변수(공 거리)가 정책 관측이라는 점이 중요하다 ★
    시뮬레이터만 아는 래치로 게이팅하면 실기에서 재현할 수 없다.
    기존 walk_kick 의 kick_time_s 가 정확히 그 문제였다.
    """
    ball_xy, robot_xy, _ = _ball_robot_xy(env, ball_cfg, robot_cfg)
    d = torch.norm(ball_xy - robot_xy, dim=-1)
    return ((d - d_near) / (d_far - d_near)).clamp(0.0, 1.0)


def track_lin_vel_xy_kick_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """v0_3 선속도 추종 × (킥 전 1 / 킥 후 0).

    ★ 킥 후 "멈춤" 의 핵심 ★
    킥 전에는 전진명령 추종이 로봇을 공으로 밀어넣어 킥을 유도한다.
    킥 후에는 이 보상이 0 이 되어 "계속 걸어" 압력이 사라지고, 로봇은
    (ball_approach 잠금 + 안정성만 남아) 자연히 멈춘다.

    kicked 는 reward-side 플래그다. 정책은 이걸 관측 못 하지만, discrete
    구조상 "공 가까움=킥 전=걷기 / 공 멂=킥 후=정지" 가 공 거리(관측)로
    자연히 갈리므로 정책이 관측만으로 올바르게 학습한다.
    """
    base = _rewards.track_lin_vel_xy_hold_aware(
        env, command_name=command_name, std=std
    )
    _ensure_buffers(env)
    return base * (~env.kick_v0_kicked).float()


def track_ang_vel_z_kick_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """v0_3 각속도 추종 × (킥 전 1 / 킥 후 0). track_lin 과 동일 논리."""
    base = _rewards.track_ang_vel_z_hold_aware(
        env, command_name=command_name, std=std
    )
    _ensure_buffers(env)
    return base * (~env.kick_v0_kicked).float()


def track_lin_vel_xy_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    d_near: float = 0.35,
    d_far: float = 1.0,
) -> torch.Tensor:
    """v0_3 의 선속도 추종에 공 거리 게이트를 곱한 버전."""
    base = _rewards.track_lin_vel_xy_hold_aware(
        env, command_name=command_name, std=std
    )
    return base * ball_prox_gate(env, d_near, d_far)


def track_ang_vel_z_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    d_near: float = 0.35,
    d_far: float = 1.0,
) -> torch.Tensor:
    """v0_3 의 각속도 추종에 공 거리 게이트를 곱한 버전."""
    base = _rewards.track_ang_vel_z_hold_aware(
        env, command_name=command_name, std=std
    )
    return base * ball_prox_gate(env, d_near, d_far)


def feet_gait_gated(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_command_name: str = "gait",
    offset: tuple[float, ...] = (0.0, 0.5),
    threshold: float = 0.55,
    d_near: float = 0.35,
    d_far: float = 1.0,
) -> torch.Tensor:
    """위상 일치 보상에 공 거리 게이트를 곱한 버전.

    이 게이팅이 네 개 중 가장 중요하다. 인사이드킥은 한 발을 오래 지지에
    두고 다른 발을 크게 흔드는 동작이라 좌우 반주기 교대와 정면으로
    충돌한다. 게이트가 없으면 정책이 "차지 않는 것" 을 최적해로 찾는다.
    """
    base = _gait.feet_gait_v01(
        env,
        sensor_cfg=sensor_cfg,
        gait_command_name=gait_command_name,
        offset=offset,
        threshold=threshold,
    )
    return base * ball_prox_gate(env, d_near, d_far)


def stop_default_posture_gated(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    gait_command_name: str = "gait",
    include_stopping: bool = False,
    d_near: float = 0.35,
    d_far: float = 1.0,
) -> torch.Tensor:
    """정지 시 기본자세 유지 보상에 공 거리 게이트를 곱한 버전.

    공 바로 앞에서 정지 명령이 걸리면 "발을 모아라" 와 "차라" 가 충돌한다.
    """
    base = _gait.stop_default_posture_v01(
        env,
        asset_cfg=asset_cfg,
        gait_command_name=gait_command_name,
        include_stopping=include_stopping,
    )
    return base * ball_prox_gate(env, d_near, d_far)


# ============================================================================
# 킥 리워드
# ============================================================================
def ball_approach_progress(
    env: ManagerBasedRLEnv,
    max_step: float = 0.05,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """로봇→공 거리의 "감소량" (N,). potential-based shaping.

    절대거리 가우시안이 아니라 Δd 를 쓰는 이유는 Ng et al. (1999) 의
    policy invariance 다. 절대거리 보상은 "공 옆에 머무는" 어트랙터를
    만들어 최적 정책 자체를 왜곡한다. Δ 형은 최적 정책을 보존한다.

    max_step 으로 클립하는 이유: 공 재배치 순간 거리가 불연속으로 뛰는데
    그게 그대로 보상이 되면 정책이 "공을 멀리 보내 재배치를 유도" 하는
    익스플로잇을 찾는다. kick_dist_valid 로도 막지만 이중 안전장치다.
    """
    _ensure_buffers(env)
    ball_xy, robot_xy, _ = _ball_robot_xy(env, ball_cfg, robot_cfg)
    d = torch.norm(ball_xy - robot_xy, dim=-1)

    delta = (env.kick_prev_robot_ball_d - d).clamp(-max_step, max_step)
    delta = torch.where(env.kick_dist_valid, delta, torch.zeros_like(delta))

    env.kick_prev_robot_ball_d = d
    # ※ kick_dist_valid 를 여기서 쓰지 않는다.
    #   RewardManager 는 term 을 선언 순서대로 평가하므로, 여기서 전역
    #   True 로 바꾸면 뒤에 오는 ball_goal_progress 가 한 번도 마스킹되지
    #   않는다. valid 는 _refresh_kick_state(이벤트) 에서만 설정한다.
    return delta


def ball_goal_progress(
    env: ManagerBasedRLEnv,
    max_step: float = 0.15,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공→목표 거리의 "감소량" (N,). potential-based shaping.

    킥의 본래 목적이다. 공이 목표 쪽으로 굴러가는 동안 계속 지급되므로
    스파스 보너스 없이도 신호가 촘촘하다.

    max_step 이 approach 보다 큰 이유는 공이 로봇보다 훨씬 빠르기 때문.
    2 m/s * 0.02 s = 0.04 m 가 전형값이고 0.15 는 충분한 여유다.
    """
    _ensure_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    d = torch.norm(ball.data.root_pos_w[:, :2] - env.kick_target_pos_w, dim=-1)

    delta = (env.kick_prev_ball_goal_d - d).clamp(-max_step, max_step)
    delta = torch.where(env.kick_dist_valid, delta, torch.zeros_like(delta))

    env.kick_prev_ball_goal_d = d
    return delta


def _foot_contact_gate_and_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """발-공 접촉 게이트와 발 속도를 로봇 yaw 프레임에서 반환.

    Returns:
        gate   (N, 2) 발별 접촉 게이트 [우, 좌]
        v_fwd  (N, 2) 발별 전방 속도 성분
        v_in   (N, 2) 발별 "인사이드 방향" 속도 성분

    인사이드 방향 부호가 좌우로 다르다.
        오른발 : +y (몸 안쪽 = 왼쪽) 가 인사이드
        왼발   : -y (몸 안쪽 = 오른쪽) 가 인사이드
    이 부호를 발마다 제대로 처리하지 않으면 리워드 자체가 좌우 비대칭이
    되어 미러 손실과 정면으로 싸운다. 킥 리워드에서 가장 조심할 지점이다.

    asset_cfg.body_ids 는 [우발, 좌발] 순서여야 한다 (MX_FEET 순서).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    yaw = _yaw(robot)

    foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :]     # (N,2,3)
    foot_vel = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (N,2,3)
    ball_pos = ball.data.root_pos_w.unsqueeze(1)                    # (N,1,3)

    d = torch.norm(foot_pos - ball_pos, dim=-1)                     # (N,2)
    excess = (d - CONTACT_RADIUS).clamp(min=0.0)
    gate = torch.exp(-torch.square(excess) / (CONTACT_FALLOFF ** 2))

    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    v_fwd = c * foot_vel[..., 0] + s * foot_vel[..., 1]
    v_lat = -s * foot_vel[..., 0] + c * foot_vel[..., 1]

    # [우발, 좌발] 순서로 인사이드 부호 적용
    inside_sign = torch.tensor([1.0, -1.0], device=env.device)
    v_in = v_lat * inside_sign

    return gate, v_fwd, v_in


def kick_lateral_velocity(
    env: ManagerBasedRLEnv,
    v_cap: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """접촉 시 발의 인사이드 방향 속도 (N,). 양수 가중치로 사용.

    arXiv 2511.03996 Table 3 의 "Sideways kick" (+20) 에 해당한다.
    논문의 근거는 발 아치의 접촉 면적이 커서 미세 위치 오차가 방향
    오차로 덜 번역된다는 것이고, 이 로봇에는 이유가 하나 더 있다.

    정면 스윙킥은 스윙 다리의 피크 가감속을 요구하는데, 액추에이터 모델
    (armature / friction / effort) 의 오차가 정확히 그 피크에서 최대가
    된다. 측면킥은 필요 피크 속도가 낮아 sim2real 갭에 훨씬 둔감하다.
    """
    gate, _, v_in = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)
    return (gate * v_in.clamp(0.0, v_cap)).sum(dim=-1)


def kick_forward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """접촉 시 발의 전방 속도 (N,). 음수 가중치로 사용(페널티).

    Table 3 의 "Forward kick" (-20) 에 해당한다. 측면 보상만 주고
    전방을 벌하지 않으면 정책이 "전방 + 약간의 측면" 이라는 절충해로
    수렴한다. 명시적 페널티가 있어야 순수 인사이드킥이 나온다.
    """
    gate, v_fwd, _ = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)
    return (gate * v_fwd.clamp(min=0.0)).sum(dim=-1)


def kick_speed_match(
    env: ManagerBasedRLEnv,
    rolling_decel: float = 1.0,
    v_min: float = 0.5,
    v_max: float = 4.0,
    std: float = 0.8,
    window_steps: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """킥 세기가 목표 거리에 맞는지 (N,).

    필요 속도는 등감속 구름 모델에서 나온다.
        v_req = sqrt(2 * a * d)
    인조잔디 구름저항계수 0.1 이면 a ~ 1 m/s^2 이고, d=3 m 에서
    v_req ~ 2.4 m/s 다.

    기존 kick_mdp.py 의 proj**2 를 대체한다. 제곱형은 상한이 없어
    "무조건 세게" 를 유도하고, 그게 시상면 외란과 sim2real 갭의
    직접 원인이었다. 이 형태는 목표 속도 주변에서 최대라 과도한
    스윙을 오히려 벌한다.

    접촉 직후 window_steps 동안만 평가한다. 접촉 순간에는 공이 아직
    가속 중이라 속도가 낮고, 너무 늦으면 구름 감속이 섞이기 때문이다.
    """
    _ensure_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]

    gate, _, _ = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)
    touching = gate.max(dim=-1).values > 0.5
    env.kick_contact_recent = torch.where(
        touching,
        torch.full_like(env.kick_contact_recent, float(window_steps)),
        (env.kick_contact_recent - 1.0).clamp(min=0.0),
    )

    to_goal = env.kick_target_pos_w - ball.data.root_pos_w[:, :2]
    d_goal = torch.norm(to_goal, dim=-1)
    dir_goal = to_goal / (d_goal.unsqueeze(-1) + 1e-6)

    v_req = torch.sqrt(2.0 * rolling_decel * d_goal).clamp(v_min, v_max)
    v_proj = (ball.data.root_lin_vel_w[:, :2] * dir_goal).sum(dim=-1)

    active = (env.kick_contact_recent > 0.0).float()
    return active * torch.exp(-torch.square(v_proj - v_req) / (std ** 2))


def stagnation_penalty(
    env: ManagerBasedRLEnv,
    speed_threshold: float = 0.05,
    ema_alpha: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """거의 정지 상태가 지속되면 1.0 (N,). 음수 가중치로 사용.

    arXiv 2511.03996 Table 3 의 stagnation (-100) 에 해당한다.
    킥 리워드가 붙으면 "안 차고 가만히 서 있는 것" 이 국소 최적해가 되기
    쉽다. 넘어지지 않으면 alive 보상은 계속 받으면서 킥 실패 위험은
    피할 수 있기 때문이다. 이 항이 그 국소해를 깬다.

    ema_alpha=0.02 는 시상수 ~50 스텝 = 1 s 에 해당한다.
    별도 링버퍼 없이 EMA 하나로 같은 판정을 한다.

    ※ 정지 명령(rel_standing_envs) 인 env 는 벌하면 안 되므로
      env_cfg 에서 gait frequency 가 0 인 경우를 제외해야 한다.
      여기서는 속도 EMA 만 보고, 제외는 gait 커맨드가 담당한다.
    """
    _ensure_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    env.kick_speed_ema = (
        (1.0 - ema_alpha) * env.kick_speed_ema + ema_alpha * speed
    )
    return (env.kick_speed_ema < speed_threshold).float()


# ============================================================================
# 정면킥 리워드 — 옛 walk_kick_env_cfg.py 에서 이식 (검증된 구성)
# ----------------------------------------------------------------------------
# v0 의 게이팅 설계가 "공 근처 = 리워드 사막" 을 만들어 정책이 공을 일부러
# 피하는 문제(iter 3000~5500 정체)를 낳았다. 옛 walk_kick 은 게이팅이 없고
# 절대거리 어트랙터가 있어 공에 접근·킥에 성공했다. 그 구성을 가져오되
# sim2real 독소(commands_zero 래치 / 킥 종료)는 이식하지 않는다.
#   - 게이팅 제거는 env_cfg 에서 (보행 리워드를 v0_3 원본으로 되돌림)
#   - 공 재배치(respawn)는 v0 그대로 유지 → 옛 post-kick 잠금 불필요
# ============================================================================
def ball_approach_gaussian(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """로봇↔공 절대거리 가우시안 근접 보상 (N,). exp(-d/std).

    ★ 이게 "공으로 당기는" 어트랙터다 ★
    d=0 -> 1.0, d=std -> 0.37, d=1.0(std=0.3) -> 0.036.
    게이팅이 없으므로 공이 정면에 있으면 "앞으로 걷기(track_lin)" 와
    "공에 다가가기(이 항)" 가 같은 방향이라 서로 보강한다 — 옛 walk_kick
    이 성공한 핵심 이유다.

    ★ 킥 후 잠금(1.0) ★
    킥이 감지되면 이 값을 1.0 으로 고정한다. 두 가지 목적:
      1) 날아간 공을 쫓지 않게 함 (거리 그래디언트 제거 → 정지 유도)
      2) 킥해도 이 보상이 안 떨어지게 함 (킥 회피 유인 제거,
         value 함수 연속성 유지 — 옛 walk_kick 과 동일한 논리)
    잠금은 kicked 플래그(reward-side)로만 판단하고 관측엔 안 넣는다.
    """
    _ensure_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    d = torch.norm(
        ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    reward = torch.exp(-d / std)
    return torch.where(env.kick_v0_kicked, torch.ones_like(reward), reward)


def ball_velocity_toward_target(
    env: ManagerBasedRLEnv,
    v_min_kick: float = 0.5,
    max_speed: float = 2.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공 속도를 목표방향(월드, 리셋 시 고정)으로 투영한 값의 제곱 (N,).

    정면킥의 주 신호다. 뒤로 차는 건 보상 안 하고(clamp min 0), 강한 킥이
    약한 드리블을 지배하도록 제곱한다.

    ★ v_min_kick 문턱 (드리블 억제) ★
    clamp(proj - v_min_kick, 0, max)**2. proj 가 v_min_kick 이하인 "살살
    미는" 소프트 터치는 0 을 준다. 이게 없으면(예전) 접근 중 살짝살짝
    건드려 공을 몰고 가는 게(walk-through 드리블) 보상돼서, 로봇이 한 방에
    안 차고 두세 번 툭툭 건드린 뒤 찼다. 문턱 위만 보상하면 "한 방 세게"로
    수렴한다. 감지 임계(update_kick_state)와 같은 값(0.5)으로 정렬해,
    에피소드를 끝내는 그 접촉이 곧 보상받는 접촉이 되게 한다.

    kick_target_dir_w 는 이벤트에서 설정된다. 미러 대칭: 목표방향 y 반전 +
    공 속도 y 반전 → 투영값 보존 → 미러 손실과 충돌 없음.
    """
    _ensure_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    proj = (v_xy * env.kick_target_dir_w).sum(dim=-1)
    return torch.clamp(proj - v_min_kick, min=0.0, max=max_speed) ** 2


def ball_z_velocity_l2(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공 수직속도 제곱 (N,). 음수 가중치로 사용 — 위로 차는(칩샷) 억제."""
    ball: RigidObject = env.scene[ball_cfg.name]
    return torch.square(ball.data.root_lin_vel_w[:, 2])


# ============================================================================
# 옛 walk_kick 킥 레이어 복원 — "한 방 킥" 의 핵심
# ----------------------------------------------------------------------------
# 임계값 땜질이 아니라 이 두 항이 깨끗한 단일 킥을 만든다.
#   kick_success_bonus     : 결정적 킥+정지에 큰 sparse 보상 (farming 차단)
#   joint_pose_to_default  : 킥 후 다리 원위치 → 깔끔한 멈춤
# 둘 다 kicked 플래그/관절값만 쓰고 관측에 안 샘 (sim2real 안전).
# ============================================================================
def kick_success_bonus(
    env: ManagerBasedRLEnv,
    hold_time_s: float = 1.0,
) -> torch.Tensor:
    """킥 감지 후 hold_time_s 를 버티면 그 단일 스텝에 +1 (N,). weight 8.0.

    dense ball_velocity 만으로는 "여러 번 툭툭 밀어 누적 farming" 이
    최적해가 된다. 이 큰 보너스가 "한 번 제대로 차고 서기" 를 결정적으로
    이득으로 만들어 단일 킥을 유도한다 (옛 walk_kick kick_then_stand_bonus).

    kick_resolved 종료와 같은 조건(elapsed == hold_steps)에서 정확히 1회
    발화한다. 그 스텝에 에피소드가 끝나므로 중복 발화가 없다. 넘어지면
    (bad_orientation 이 먼저 종료) 발화 안 함 → "킥 + 생존" 만 보상.
    """
    _ensure_buffers(env)
    hold_steps = int(round(hold_time_s / env.step_dt))
    elapsed = int(env.common_step_counter) - env.kick_v0_kick_step
    return (env.kick_v0_kicked & (elapsed == hold_steps)).float()


def joint_pose_to_default(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """다리 관절이 기본자세에 가까울수록 가우시안 보상 (N,). 옛 walk_kick.

    exp(-mean((joint - default)²)/std²). std 0.5 로 넓어 킥 스윙의 큰 이탈은
    용서하고, 킥 후 다리를 원위치로 되돌려 "멈춤" 을 깔끔하게 만든다.
    보행 중엔 다리가 크게 흔들려 값이 ~0 이라 걷기를 방해하지 않는다.

    미러 대칭: 관절이 좌우 swap+부호반전 되어도 제곱합은 보존 → 무관.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.exp(-torch.mean(torch.square(err), dim=1) / (std * std))


# ============================================================================
# 진단 지표 (weight=0, 텐서보드 추적 전용)
# ============================================================================
def metric_ball_goal_dist(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공→목표 거리. 학습이 진행되면 내려가야 한다."""
    _ensure_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    return torch.norm(
        ball.data.root_pos_w[:, :2] - env.kick_target_pos_w, dim=-1
    )


def metric_kick_foot_balance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """좌우 킥 균형. +1 이면 오른발만, -1 이면 왼발만 쓰는 중.

    미러 손실이 실제로 작동하는지 보는 가장 직접적인 지표다.
    0 근처를 유지해야 한다. 한쪽으로 치우치면 미러 계수를 올린다.
    """
    gate, _, v_in = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)
    contrib = gate * v_in.clamp(min=0.0)
    total = contrib.sum(dim=-1)
    return torch.where(
        total > 1e-3,
        (contrib[:, 0] - contrib[:, 1]) / (total + 1e-6),
        torch.zeros_like(total),
    )


def kick_wrong_foot_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공 스폰 쪽과 '반대(먼) 발'로 차면 벌점 → 양발 사용 유도(오른발 편향 교정).

    공이 왼쪽(frozen +y)인데 오른발로 스트라이크하면 벌, 오른쪽인데 왼발이면 벌.
    벌점이 좌우 미러에 대해 불변(mirror-invariant)이라 미러손실과 안 싸운다.
    킥 접촉 순간(일회성)에만 걸려 feet_fb 처럼 평균에 희석되지 않는다.

    ⚠ asset_cfg 는 preserve_order=True 로 [우발, 좌발](MX_FEET) 순서를 보장할 것.
    """
    gate, v_fwd, _ = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)
    strike = gate * v_fwd.clamp(min=0.0)          # (N,2) [우발, 좌발] 전방 스트라이크
    ball_y = env.ball_pos_b_frozen[:, 1]          # frozen 로봇프레임 y (+y = 왼쪽)
    left_ball = ball_y.clamp(min=0.0)             # 공이 왼쪽인 정도
    right_ball = (-ball_y).clamp(min=0.0)         # 공이 오른쪽인 정도
    # 왼쪽 공을 오른발(strike[:,0])로 / 오른쪽 공을 왼발(strike[:,1])로 = 먼 발 → 벌
    return strike[:, 0] * left_ball + strike[:, 1] * right_ball


def metric_ball_dist(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """로봇→공 거리. 접근이 되고 있는지 확인."""
    ball_xy, robot_xy, _ = _ball_robot_xy(env, ball_cfg, robot_cfg)
    return torch.norm(ball_xy - robot_xy, dim=-1)


# ============================================================================
# 커리큘럼
# ============================================================================
def kick_reward_ramp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    term_names: list[str],
    start_iter: int = 1500,
    end_iter: int = 3000,
    num_steps_per_env: int = 24,
) -> torch.Tensor:
    """킥 리워드 가중치를 0 -> 목표값으로 선형 램프한다.

    scratch 학습에서 킥 리워드를 처음부터 켜면 걷기도 전에 공에 달려들다
    넘어지는 데 수렴한다. 걸음마를 먼저 배우게 하는 장치다.

    관측은 처음부터 전부 들어간다. 정책이 "공 관측은 있지만 무시해도
    되는 것" 부터 배우고 나중에 쓰임새를 익히는 순서다.

    Returns:
        현재 스케일 (텐서보드 Curriculum/ 에 기록됨)
    """
    _ensure_buffers(env)
    rm = env.reward_manager

    # 목표 가중치를 최초 1회 저장 (이후 램프의 기준값)
    if not env.kick_reward_target_w:
        for name in term_names:
            env.kick_reward_target_w[name] = rm.get_term_cfg(name).weight

    it = int(env.common_step_counter) / max(num_steps_per_env, 1)
    if end_iter <= start_iter:
        # play cfg 처럼 start=end=0 으로 램프를 끄는 경우.
        # it <= start_iter 조건이 먼저 걸리면 첫 스텝만 scale=0 이 되므로
        # 이 분기를 앞에 둔다.
        scale = 1.0
    elif it <= start_iter:
        scale = 0.0
    elif it >= end_iter:
        scale = 1.0
    else:
        scale = (it - start_iter) / (end_iter - start_iter)

    for name in term_names:
        cfg = rm.get_term_cfg(name)
        cfg.weight = env.kick_reward_target_w[name] * scale
        rm.set_term_cfg(name, cfg)

    return torch.tensor(scale, device=env.device)


def kick_angle_levels(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    stage_iters: tuple[int, int] = (1500, 3000),
    bearing_scales: tuple[float, float, float] = (0.333, 0.667, 1.0),
    target_scales: tuple[float, float, float] = (0.25, 0.5, 1.0),
    num_steps_per_env: int = 24,
) -> torch.Tensor:
    """공 방위 / 목표 각도 범위를 단계적으로 넓힌다.

    기본값이 만드는 실제 범위 (env_cfg 최종 범위 ±60° 기준):
        iter    0 ~ 1500 : 공 방위 ±20°, 목표 각도 ±15°
        iter 1500 ~ 3000 : 공 방위 ±40°, 목표 각도 ±30°
        iter 3000 ~      : 공 방위 ±60°, 목표 각도 ±60°

    거리 범위는 처음부터 전 구간을 쓴다. 각도와 달리 거리는 난이도 차이가
    작고, 좁게 시작하면 오히려 관측이 상수화되어 실기에서 외삽이 난다.
    (기존 walk_kick 의 x(0.3,0.5) 가 정확히 그 실패였다)

    Returns:
        현재 공 방위 스케일 (텐서보드 기록용)
    """
    _ensure_buffers(env)
    it = int(env.common_step_counter) / max(num_steps_per_env, 1)

    if it < stage_iters[0]:
        idx = 0
    elif it < stage_iters[1]:
        idx = 1
    else:
        idx = 2

    env.kick_bearing_scale = torch.tensor(
        bearing_scales[idx], device=env.device
    )
    env.kick_target_angle_scale = torch.tensor(
        target_scales[idx], device=env.device
    )
    return env.kick_bearing_scale.clone()


# ============================================================================
# 자유 내비게이션 — 전진 속도 상한 페널티 (walk_kick_v1 재설계)
# ============================================================================
def planar_speed_cap_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """평면 속도가 명령 상한(cmd vx)을 넘을 때만 벌점 (넘은 양의 제곱).

    설계 의도:
        base_velocity 를 "추종 목표" 가 아니라 "최대 속도 상한" 으로 쓴다.
        vx(=cmd[:,0]) 를 몸통 평면 속도의 상한으로 보고, 실제 평면 속도
        norm 이 상한을 넘는 만큼만 벌점을 준다. 방향(전/후/좌/우)은
        자유 — 정책이 공까지 알아서 최적 경로로 이동한다.

        속도 자체엔 보상이 없다(천장만 있음). 이동 유인은 ball_approach
        가 담당한다. 킥 후에는 KickAwareVelocityCommand 가 cmd 를 0 으로
        만들므로 상한도 0 → 잔여 이동에 벌점 → 정지를 강화한다.

    음수 weight 로 사용한다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    # 몸통 프레임 평면 속도 (forward, lateral)
    v_planar = robot.data.root_lin_vel_b[:, :2]
    speed = torch.norm(v_planar, dim=-1)

    cap = env.command_manager.get_command(command_name)[:, 0].abs()
    excess = (speed - cap).clamp(min=0.0)
    return torch.square(excess)


def track_forward_speed_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """전진 속도(body +x)를 명령 vx 에 "추종". vy/wz 는 건드리지 않음(자유).

    설계 의도 (walk_kick_v2 "일반 보행 접근"):
        기존 planar_speed_cap_penalty 는 상한(cap)만 걸어 그 아래로 급가속
        버스트를 허용했다 → 성큼성큼 불안정. 대신 전진 속도를 "목표 페이스"
        로 추종하면 그 값에 딱 맞춰 걷게 되어 일정하고 부드러운 걸음이 된다.
        오직 body +x(전진) 성분만 본다. 옆(vy)/회전(wz)은 벌점 없음 →
        정책이 알아서 몸을 틀어(ball_approach 유인) 공을 향한다.
        명령 vx 는 ForwardToBallSpeedCommand 가 거리 기반으로 주므로
        근거리/킥후엔 cmd=0 → 전진 0 추종 → 자연스러운 정지.

    exp(-(v_fwd - cmd_vx)² / std²). 양수 weight 로 사용.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    v_fwd = robot.data.root_lin_vel_b[:, 0]
    cmd_vx = env.command_manager.get_command(command_name)[:, 0]
    err = v_fwd - cmd_vx
    return torch.exp(-torch.square(err) / (std * std))


def feet_fb_separation_penalty(
    env: ManagerBasedRLEnv,
    deadband: float = 0.14,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """앞뒤 발간격(로봇 전방기준)이 deadband 초과 시 벌점(초과분 L1).

    킥 때 다리를 앞뒤로 과하게 벌리는(윈드업) 걸 억제. 단, 정상 보행도 한 발
    앞/한 발 뒤로 자연스럽게 벌어지므로(측정: 보행 최대 ~0.11m, 킥 ~0.22m),
    보행 범위(deadband=0.14) 이하는 벌하지 않고 그 초과분만 벌점 → 보행은
    안 건드리고 킥의 과다 벌림만 잡는다. asset_cfg.body_ids=[우발, 좌발].
    음수 weight 로 사용.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    fp = robot.data.body_pos_w[:, asset_cfg.body_ids, :]      # (N, 2, 3) world
    d_w = fp[:, 0, :] - fp[:, 1, :]                            # 우 - 좌 (N, 3)
    d_b = quat_apply_inverse(robot.data.root_quat_w, d_w)     # base 프레임
    fb = torch.abs(d_b[:, 0])                                 # 앞뒤(전방) 성분
    return (fb - deadband).clamp(min=0.0)                     # 초과분(L1)


# ============================================================================
# 인사이드 킥 — 세기 매칭 (자체 완결판, walk_kick_v1 인사이드 킥용)
# ----------------------------------------------------------------------------
# kick_speed_match 원본은 _ensure_buffers 를 호출해 env.kick_target_pos_w 를
# (N,2) 로 덮어쓴다. 현재 env 는 kick_mdp.reset_kick_target 로 (N,3) target 을
# 쓰므로 그 버전을 쓰면 target 이 오염된다. 이 버전은:
#   - _ensure_buffers 를 타지 않고 (kick_mdp 상태 무손상),
#   - target 을 [..., :2] 로 슬라이스해 (N,2)/(N,3) 양쪽 호환,
#   - 전용 접촉 윈도우 버퍼(_kick_smatch_recent, 다른 버퍼와 분리) 사용.
# 방향+세기를 동시에 본다: 공속도의 목표방향 투영 v_proj 를 거리기반 v_req 에
# 매칭. proj² (무조건 세게) 를 대체해 과도 스윙/불안정을 억제한다.
# ============================================================================
def kick_speed_match_v1(
    env: ManagerBasedRLEnv,
    rolling_decel: float = 1.0,
    v_min: float = 0.5,
    v_max: float = 4.0,
    std: float = 0.8,
    window_steps: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """킥 세기가 목표 거리에 맞는지 + 목표 방향인지 (N,). 양수 weight.

    v_req = sqrt(2*a*d_goal) (구름감속 모델, v_min~v_max clamp).
    v_proj = 공속도의 목표방향 투영 → 목표 쪽으로 갈 때만 커짐.
    exp(-(v_proj - v_req)^2 / std^2) → 목표방향으로 적정 세기일 때 최대.
    접촉 직후 window_steps 동안만 평가(접촉 순간 공이 아직 가속 중이라).
    """
    ball: RigidObject = env.scene[ball_cfg.name]

    if not hasattr(env, "_kick_smatch_recent"):
        env._kick_smatch_recent = torch.zeros(env.num_envs, device=env.device)

    gate, _, _ = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)
    touching = gate.max(dim=-1).values > 0.5
    env._kick_smatch_recent = torch.where(
        touching,
        torch.full_like(env._kick_smatch_recent, float(window_steps)),
        (env._kick_smatch_recent - 1.0).clamp(min=0.0),
    )

    target_xy = env.kick_target_pos_w[..., :2]
    to_goal = target_xy - ball.data.root_pos_w[:, :2]
    d_goal = torch.norm(to_goal, dim=-1)
    dir_goal = to_goal / (d_goal.unsqueeze(-1) + 1e-6)

    v_req = torch.sqrt(2.0 * rolling_decel * d_goal).clamp(v_min, v_max)
    v_proj = (ball.data.root_lin_vel_w[:, :2] * dir_goal).sum(dim=-1)

    active = (env._kick_smatch_recent > 0.0).float()
    return active * torch.exp(-torch.square(v_proj - v_req) / (std ** 2))


# ============================================================================
# 인사이드 킥 — one-shot (드래그/리워드 해킹 방지판)
# ----------------------------------------------------------------------------
# 기존 kick_lateral_velocity 는 접촉하는 매 스텝 dense 로 v_in 을 보상해서,
# 정책이 발을 공에 붙인 채 천천히 안쪽으로 쓸며 "질질 끄는"(드래그) 국소해로
# farm 했다. 이 함수는:
#   - 첫 강한 타격(발 타격속도 > min_strike_speed) 순간 "1회만" 채점,
#   - 그 순간의 v_in(인사이드, +) - forward_penalty*v_fwd(발등, -) 로 스타일 평가.
# 드래그는 발이 느려(strike speed 낮음) 절대 발화하지 않으므로 farm 불가.
# 게이팅을 공속도가 아니라 "발 타격속도" 로 하는 이유: 발속도가 곧 스타일
# 신호라 동시점이고, 공속도(임펄스보다 한두 스텝 늦게 피크)와의 타이밍
# 어긋남이 없다. 에피소드당 1회(_kick_tech_scored latch; kick_time_s<0 리셋).
# ============================================================================
def kick_inside_oneshot(
    env: ManagerBasedRLEnv,
    min_strike_speed: float = 1.0,
    forward_penalty: float = 0.5,
    v_cap: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """첫 강한 타격 1회에만 인사이드 킥 품질 (N,). 양수 weight.

    quality = v_in(안쪽, +) - forward_penalty * v_fwd(발등, +방향만).
    접촉 gate 가중 후 양발 합산. 드래그(느린 발)는 발화 안 함.
    """
    n = env.num_envs
    dev = env.device
    if not hasattr(env, "_kick_tech_scored"):
        env._kick_tech_scored = torch.zeros(n, dtype=torch.bool, device=dev)

    # 에피소드 리셋 감지: reset_kick_state 가 kick_time_s 를 -1 로 초기화한다.
    kt = getattr(env, "kick_time_s", None)
    if kt is not None:
        env._kick_tech_scored = env._kick_tech_scored & (kt >= 0)

    gate, v_fwd, v_in = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)

    # 발별 타격속도 (접촉 관련 성분). 드래그면 작다.
    foot_strike = torch.sqrt(v_in * v_in + v_fwd * v_fwd + 1e-9)      # (N,2)
    strong = (gate > 0.5) & (foot_strike > min_strike_speed)          # (N,2)
    strike_now = (~env._kick_tech_scored) & strong.any(dim=-1)        # (N,)

    lat = (gate * v_in.clamp(0.0, v_cap)).sum(dim=-1)
    fwd = (gate * v_fwd.clamp(min=0.0)).sum(dim=-1)
    quality = lat - forward_penalty * fwd

    env._kick_tech_scored = env._kick_tech_scored | strike_now
    return torch.where(strike_now, quality, torch.zeros_like(quality))


# ============================================================================
# 골 파워 — 명령 기반 킥 세기 (kick_strength 슬롯을 진짜 파워 명령으로)
# ----------------------------------------------------------------------------
# 설계: 추론 실측 최대 킥속도 V_MAX(=3.5 m/s)를 kick_strength=1.0 에 대응시키고,
# 에피소드마다 kick_strength 를 샘플(0.3~1.0)해 관측으로 준다. 보상은 공의
# 목표방향 킥 속도를 명령 파워(kick_strength * V_MAX)에 매칭한다.
# 기존 ball_velocity_toward_target(proj², "무조건 세게")를 대체 — 그건 파워
# 조절과 충돌(항상 최대만 유도)하기 때문.
# ============================================================================
def reset_kick_strength(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    strength_range: tuple[float, float] = (0.3, 1.0),
) -> None:
    """에피소드마다 kick_strength 명령(정규화 파워)을 샘플한다.

    ※ env_ids 에 기본값을 주지 말 것 (IsaacLab EventManager 가 인자 순서를
      기본값 유무로 재정렬 → 빌드 실패). 위치 인자로 둔다.
    """
    if not hasattr(env, "kick_strength_cmd"):
        env.kick_strength_cmd = torch.ones(env.num_envs, device=env.device)

    # 추론에서 --kick_strength 로 고정 지정한 경우 그 값을 유지 (play.py 가
    # env.kick_strength_fixed 를 세팅). 없으면(학습) 매 에피소드 랜덤 샘플.
    fixed = getattr(env, "kick_strength_fixed", None)
    if fixed is not None:
        env.kick_strength_cmd[env_ids] = float(fixed)
        return

    lo, hi = strength_range
    r = torch.rand(len(env_ids), device=env.device)
    env.kick_strength_cmd[env_ids] = lo + (hi - lo) * r


def kick_strength_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """kick_strength 명령을 관측으로 (N,1). 상수 1.0(ones_obs) 대체."""
    if not hasattr(env, "kick_strength_cmd"):
        env.kick_strength_cmd = torch.ones(env.num_envs, device=env.device)
    return env.kick_strength_cmd.unsqueeze(-1)


def kick_power_match(
    env: ManagerBasedRLEnv,
    v_max: float = 3.5,
    std: float = 0.6,
    window_steps: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공 킥 속도(목표방향 투영)를 명령 파워(kick_strength * v_max)에 매칭 (N,).

    exp(-(v_proj - v_cmd)^2 / std^2). 접촉 직후 window_steps 동안만 평가
    (접촉 순간엔 공이 아직 가속 중). 양수 weight.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    if not hasattr(env, "kick_strength_cmd"):
        env.kick_strength_cmd = torch.ones(env.num_envs, device=env.device)
    if not hasattr(env, "_kick_pmatch_recent"):
        env._kick_pmatch_recent = torch.zeros(env.num_envs, device=env.device)

    gate, _, _ = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)
    touching = gate.max(dim=-1).values > 0.5
    env._kick_pmatch_recent = torch.where(
        touching,
        torch.full_like(env._kick_pmatch_recent, float(window_steps)),
        (env._kick_pmatch_recent - 1.0).clamp(min=0.0),
    )

    target_xy = env.kick_target_pos_w[..., :2]
    to_goal = target_xy - ball.data.root_pos_w[:, :2]
    d_goal = torch.norm(to_goal, dim=-1)
    dir_goal = to_goal / (d_goal.unsqueeze(-1) + 1e-6)
    v_proj = (ball.data.root_lin_vel_w[:, :2] * dir_goal).sum(dim=-1)

    v_cmd = env.kick_strength_cmd * v_max

    active = (env._kick_pmatch_recent > 0.0).float()
    return active * torch.exp(-torch.square(v_proj - v_cmd) / (std ** 2))


# ============================================================================
# 인사이드 킥 — 접촉면(발 안쪽) 기반 (walk_kick_v2)
# ----------------------------------------------------------------------------
# 실측(probe_foot_frame): 발 로컬 +x=발끝(전방), +y=좌(측방). 안쪽면 법선은
# medial = inside_sign·(발 로컬 +y)  (우발 +y, 좌발 -y — 몸 중심을 향하는 면).
# 인사이드 킥 = 발을 바깥으로 틀어(eversion) 안쪽면 법선이 "목표 방향"을 향함.
#   alignment = medial_world · target_dir  →  인사이드 +1 / 발등 ~0 / 아웃 -1.
# dense(강타 순간 정렬값 래치 → post-kick window_steps 동안 매 스텝 보상):
#   - one-shot 은 총량이 dense 인 ball_velocity 를 못 이겨 스타일 전환 실패했다.
#     dense 로 총량을 맞춰 발등→인사이드 전환을 밀 수 있게 한다.
#   - window 는 "강한 타격" 순간에만 열리고, 느린 접촉엔 리프레시 안 됨 →
#     느린 인사이드 드리블은 window 를 못 열어 farm 불가.
#   - proj²(강타·목표방향)와 겹쳐야 하므로 "빠르게+목표로+안쪽면"만 남음.
# 지난 실패(속도프록시 드리블 / one-shot 약함 / 몸통기준)를 모두 회피한다.
# ============================================================================
def kick_inside_surface(
    env: ManagerBasedRLEnv,
    min_strike_speed: float = 1.0,
    window_steps: int = 20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """강타 순간 접촉발의 안쪽면 정렬값을 래치, post-kick window 동안 dense (N,).

    +1 인사이드 / ~0 발등 / -1 아웃사이드. 양수 weight.
    asset_cfg.body_ids 는 [우발, 좌발] 순서여야 함 (MX_FEET, preserve_order).
    """
    n = env.num_envs
    dev = env.device
    if not hasattr(env, "_kick_inside_win"):
        env._kick_inside_win = torch.zeros(n, device=dev)
    if not hasattr(env, "_kick_inside_align"):
        env._kick_inside_align = torch.zeros(n, device=dev)

    robot: Articulation = env.scene[asset_cfg.name]

    # 접촉 게이트 + 발 타격속도 (per foot, [우,좌])
    gate, v_fwd, v_in = _foot_contact_gate_and_vel(env, asset_cfg, ball_cfg)   # (N,2)
    foot_strike = torch.sqrt(v_in * v_in + v_fwd * v_fwd + 1e-9)               # (N,2)
    strong = (gate > 0.5) & (foot_strike > min_strike_speed)                  # (N,2)

    # 발 안쪽면 법선 (world): inside_sign·(발 로컬 +y)
    foot_quat = robot.data.body_quat_w[:, asset_cfg.body_ids, :]              # (N,2,4)
    inside_sign = torch.tensor([1.0, -1.0], device=dev)                       # [우,좌]
    y_local = torch.zeros(n, 2, 3, device=dev)
    y_local[:, :, 1] = inside_sign                                            # (0,±1,0)
    medial_w = quat_apply(
        foot_quat.reshape(-1, 4), y_local.reshape(-1, 3)
    ).reshape(n, 2, 3)                                                        # (N,2,3)

    # 목표 방향 (world xy, 단위벡터)
    tdir = env.kick_target_dir_w[..., :2]                                     # (N,2)
    tdir = tdir / (torch.norm(tdir, dim=-1, keepdim=True) + 1e-6)
    align = (medial_w[..., :2] * tdir.unsqueeze(1)).sum(dim=-1)              # (N,2)

    # 강타한 발의 정렬값
    align_strong = torch.where(strong, align, torch.zeros_like(align))
    n_strong = strong.float().sum(dim=-1).clamp(min=1.0)
    align_struck = align_strong.sum(dim=-1) / n_strong                       # (N,)
    struck = strong.any(dim=-1)                                              # (N,)

    # 강타 순간: 정렬값 래치 + window 열기. 그 외: 감쇠(느린 접촉엔 리프레시 X)
    env._kick_inside_align = torch.where(struck, align_struck, env._kick_inside_align)
    env._kick_inside_win = torch.where(
        struck,
        torch.full_like(env._kick_inside_win, float(window_steps)),
        (env._kick_inside_win - 1.0).clamp(min=0.0),
    )
    active = (env._kick_inside_win > 0.0).float()
    return active * env._kick_inside_align


# ============================================================================
# v1_2 인사이드 킥 보상 3종 (2026-08-16)
#   "진짜 인사이드 푸시" 강제 — 지난 실패(정면 뻗다 닿기직전 45도 트위스트 poke,
#   sim2real 갭에서 방향 부정확)를 결과+동작 양쪽으로 막는다.
#     ① ball_offaxis_penalty      : 공이 목표에서 옆으로 새는 성분 벌 (방향 정확도, -w)
#     ② kick_push_along_target    : 접촉발이 목표방향으로 밀고 지나감 (push=poke방지, +w)
#     ③ kick_inside_align_approach: 접촉발이 공 근처+활성일 때 안쪽면이 목표 향함
#                                   (접근 내내 정렬 → 마지막순간 트위스트 차단, +w)
#   ②는 목표각이 클수록 골반을 더 돌려야 먹으므로 "골반 yaw 확 돌리기"가 emergent.
#   ①②③ + 기존 ball_velocity(파워)/kick_success/data-aug/절대1초종료와 함께 사용.
# ============================================================================
def _target_dir_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """목표방향 world xy 단위벡터 (N,2). 버퍼 미설정(매니저 init 등)이면 정면 기본."""
    if not hasattr(env, "kick_target_dir_w"):
        d = torch.zeros(env.num_envs, 2, device=env.device)
        d[:, 0] = 1.0
        return d
    tdir = env.kick_target_dir_w[..., :2]
    return tdir / (torch.norm(tdir, dim=-1, keepdim=True) + 1e-6)


def ball_offaxis_penalty(
    env: ManagerBasedRLEnv,
    post_kick_window_s: float = 0.4,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """① 공 속도의 목표방향 '수직성분(옆으로 새는 양)'을 제곱 벌점 → 방향 정확도.

    투영기반 ball_velocity 는 옆으로 새도 목표성분만 보상해 부정확을 허용했음.
    이 항이 수직성분을 벌해 '정확히 목표로'를 강제한다. 음수 weight.
    ball_velocity 와 같은 post-kick window(임펄스 구간)만 게이팅.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    tdir = _target_dir_xy(env)
    along = (v_xy * tdir).sum(dim=-1, keepdim=True)
    v_perp = v_xy - along * tdir
    penalty = (v_perp * v_perp).sum(dim=-1)
    if hasattr(env, "kick_time_s"):
        current_t = env.episode_length_buf.float() * env.step_dt
        elapsed = current_t - env.kick_time_s
        past = (env.kick_time_s >= 0) & (elapsed > post_kick_window_s)
        penalty = torch.where(past, torch.zeros_like(penalty), penalty)
    return penalty


def kick_push_along_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """② 접촉 중 접촉발의 '목표방향 속도성분(밀기)'을 dense 보상 (poke 방지 핵심).

    진짜 인사이드 스윕은 발이 목표방향으로 쓸고감(push) → 높음.
    정면 poke 는 발이 로봇 x 로 움직이다 틀어 스침 → 목표방향 성분 작음 → 낮음.
    목표각이 클수록 골반을 더 돌려야 이 성분이 커짐 → 골반 회전 emergent. +weight.
    asset_cfg.body_ids 는 [우발, 좌발] 순서(MX_FEET, preserve_order).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    fids = asset_cfg.body_ids
    foot_pos = robot.data.body_pos_w[:, fids, :]                 # (N,2,3)
    foot_vel = robot.data.body_lin_vel_w[:, fids, :]            # (N,2,3)
    ball_pos = ball.data.root_pos_w.unsqueeze(1)               # (N,1,3)
    d = torch.norm(foot_pos - ball_pos, dim=-1)                 # (N,2)
    excess = (d - CONTACT_RADIUS).clamp(min=0.0)
    gate = torch.exp(-torch.square(excess) / (CONTACT_FALLOFF ** 2))  # (N,2)
    tdir = _target_dir_xy(env).unsqueeze(1)                     # (N,1,2)
    push = (foot_vel[..., :2] * tdir).sum(dim=-1).clamp(min=0.0)  # (N,2) 목표방향 전진성분
    return (gate * push).sum(dim=-1)                            # (N,)


def kick_inside_align_approach(
    env: ManagerBasedRLEnv,
    near_radius: float = 0.22,
    min_foot_speed: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """③ 접촉발이 공 근처(near_radius)+활성(속도>min)일 때 안쪽면 정렬을 dense 보상.

    접촉 '순간'만 보던 kick_inside_surface 는 정면 뻗다 닿기직전 트위스트로 hack 됐음.
    여기선 '접근 내내(공 근처+스윙중)' 정렬돼야 보상 → 미리 골반 돌려 안쪽면을 목표로
    향하게 함. medial = inside_sign·(발 로컬 +y). align = medial_world·target_dir.
    farm 방지: 정지(속도<min)면 미보상. +weight.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    fids = asset_cfg.body_ids
    dev = env.device
    n = env.num_envs
    foot_pos = robot.data.body_pos_w[:, fids, :]                 # (N,2,3)
    foot_vel = robot.data.body_lin_vel_w[:, fids, :]            # (N,2,3)
    foot_quat = robot.data.body_quat_w[:, fids, :]             # (N,2,4)
    ball_pos = ball.data.root_pos_w.unsqueeze(1)               # (N,1,3)
    d = torch.norm(foot_pos - ball_pos, dim=-1)                 # (N,2)
    foot_speed = torch.norm(foot_vel[..., :2], dim=-1)          # (N,2)
    near_active = (d < near_radius) & (foot_speed > min_foot_speed)   # (N,2)
    inside_sign = torch.tensor([1.0, -1.0], device=dev)         # [우,좌]
    y_local = torch.zeros(n, 2, 3, device=dev)
    y_local[:, :, 1] = inside_sign                             # (0,±1,0)
    medial_w = quat_apply(
        foot_quat.reshape(-1, 4), y_local.reshape(-1, 3)
    ).reshape(n, 2, 3)                                          # (N,2,3)
    tdir = _target_dir_xy(env)                                  # (N,2)
    align = (medial_w[..., :2] * tdir.unsqueeze(1)).sum(dim=-1)  # (N,2) +1 인사이드
    reward = torch.where(near_active, align.clamp(min=0.0), torch.zeros_like(align))
    return reward.sum(dim=-1)                                   # (N,)
