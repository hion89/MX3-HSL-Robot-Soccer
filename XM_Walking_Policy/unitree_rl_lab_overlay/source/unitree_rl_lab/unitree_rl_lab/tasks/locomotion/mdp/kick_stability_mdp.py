# kick_stability_mdp.py
# ============================================================================
# 킥 후 안정성 / sim-to-real 갭 축소 모듈 (v1_4 설계, 2026-08-19)
# ----------------------------------------------------------------------------
# 【문제】 sim 99% 안정 / 실기 5회 중 1회 킥 후 전도.
#
# 【진단의 출발점】
#   "sim 99%" 는 좋은 소식이 아니라 **경고**다. 실패율이 1% 면 정책이 안정 마진을
#   배울 gradient 가 사실상 없다. 넘어지는 env 가 거의 없으니 "더 안전하게" 라는
#   방향으로 밀 힘이 없고, 정책은 마진 없이 딱 되는 해에 수렴한다.
#   실기에서만 나는 실패는 보상 튜닝으로 못 고친다. **sim 을 어렵게 만들어
#   sim 실패율을 의미 있는 수준(10~15%)까지 올려야** 마진이 학습된다.
#
# 【가장 유력한 물리적 원인 — 속도-토크 곡선 미모델링】
#   현재 ImplicitActuatorCfg 는 effort_limit_sim=5.0 N·m 를 관절속도와 무관하게
#   평평하게 준다. 실제 XM540-W150(12V)은 선형 N-T 곡선을 따른다:
#       스톨 7.3 N·m / 무부하 53 rpm(=5.55 rad/s)
#       τ(ω) ≈ 7.3 · (1 − ω/5.55)
#         ω=2.0 → 4.67   ω=3.0 → 3.35   ω=4.0 → 2.04   ω=4.5 → 1.38 [N·m]
#   즉 킥 스윙처럼 "빠르면서 동시에 토크가 필요한" 구간에서 sim 은 실기의
#   2.5~3.6 배 토크를 준다. 걷기(관절속도 낮음)에서는 갭이 작아 잘 맞다가
#   킥에서만 어긋나는 현상과 정확히 일치한다.
#   ※ ROBOTIS 도 스톨토크는 순간 최대치이고 실제 성능은 N-T 곡선에 가깝다고
#     명시한다. 연속 정격은 1.46 N·m(스톨의 20%)에 불과하다.
#   ※ 배터리 새그까지 겹치면 더 나빠진다(11.1V 기준 스톨 6.9 N·m / 50 rpm).
#
# 【이 모듈의 접근】
#   액추에이터 모델 자체를 DCMotor 로 바꾸는 것이 정공법이지만, 그건 dynamics 가
#   변해 기존 정책을 처음부터 다시 학습해야 한다. 그래서 두 갈래를 모두 제공한다.
#     (A) 저위험 — dynamics 는 그대로 두고, **실제 모터 포락선을 벗어난 동작에
#         벌점**을 준다(torque_envelope_penalty). 파인튜닝만으로 정책을 실현
#         가능 영역 안으로 밀어 넣는다. 먼저 이걸 쓴다.
#     (B) 정공법 — mx_bot.py 에서 DCMotorCfg 전환(README 참조). (A)로 충분치
#         않을 때, 재학습 각오하고 간다.
#
#   여기에 "넘어짐" 의 직접 예측량인 capture point 마진, 지지발 미끄러짐,
#   비행상(양발 공중) 벌점을 더해 안정 마진을 명시적으로 학습시킨다.
# ============================================================================

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

_EPS = 1e-6
_G = 9.81


# ============================================================================
# 공통 헬퍼
# ============================================================================
def _kick_window(env: ManagerBasedRLEnv, post_kick_window_s: float) -> torch.Tensor:
    """킥 래치 후 post_kick_window_s 이내이면 True (N,).

    kick_mdp._maybe_record_kick 이 래치한 env.kick_time_s 를 그대로 읽는다.
    이 모듈은 래치를 하지 않는다(공유 래치를 건드리면 조용히 깨진다).
    """
    if not hasattr(env, "kick_time_s"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    current_t = env.episode_length_buf.float() * env.step_dt
    elapsed = current_t - env.kick_time_s
    return (env.kick_time_s >= 0) & (elapsed <= post_kick_window_s)


def _boost(env: ManagerBasedRLEnv, post_kick_window_s: float, boost: float) -> torch.Tensor:
    """킥 윈도우 안에서는 boost 배, 밖에서는 1 배인 스케일 (N,).

    킥 직후가 전도가 나는 구간이므로 그 구간만 압력을 키운다. 항상 boost 를 걸면
    보행 자체가 굳어져(과도한 자세 벌점) 오히려 접근이 나빠진다.
    """
    w = _kick_window(env, post_kick_window_s).float()
    return 1.0 + (boost - 1.0) * w


def _foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """발별 접지 여부 (N, 2). history 최대값으로 판정해 1 스텝 튐에 강건하게."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N,T,2,3)
    fmag = torch.norm(forces, dim=-1).max(dim=1).values                      # (N,2)
    return fmag > force_threshold


# ============================================================================
# ★ 1. 토크 포락선 위반 벌점 — 실제 모터가 못 내는 동작을 억제 (저위험 핵심)
# ============================================================================
def torque_envelope_penalty(
    env: ManagerBasedRLEnv,
    stall_torque: float = 7.3,
    no_load_speed: float = 5.55,
    effort_cap: float = 5.0,
    margin: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """실제 모터의 속도-토크 곡선을 넘는 토크 요구량을 제곱 벌점 (N,). 음수 weight.

    τ_max(ω) = clamp(stall · (1 − |ω| / ω_noload), 0, effort_cap) · margin
    penalty  = Σ_j ( |τ_j| − τ_max(ω_j) )₊²

    【왜 이게 핵심인가】
    sim 의 ImplicitActuator 는 관절속도와 무관하게 effort_limit 을 평평하게 준다.
    걷기는 관절속도가 낮아 갭이 작지만, 킥 스윙은 ω 가 3~4.5 rad/s 까지 올라가는데
    그 구간의 실제 가용 토크는 1.4~2.0 N·m 로 sim 의 1/3 수준이다.
    → sim 정책은 실기가 못 내는 궤적을 명령하고, 스윙발이 뒤처진 채 임팩트가
      일어나 반작용 타이밍이 어긋난다. 지지발은 그 오차를 못 받아내고 넘어진다.

    【왜 dynamics 를 안 바꾸고 벌점으로 하는가】
    DCMotorCfg 로 바꾸면 물리가 달라져 재학습이 필요하다. 이 항은 물리를 그대로
    두고 "실현 가능 영역" 밖의 해에만 비용을 매기므로 **기존 체크포인트에서
    파인튜닝**으로 넘어갈 수 있다. 증분 검증 원칙에도 맞다.

    Args:
        stall_torque:  스톨 토크 [N·m]. XM540-W150 @12V = 7.3 (11.1V 면 6.9).
        no_load_speed: 무부하 속도 [rad/s]. 53 rpm = 5.55 (11.1V 면 5.24).
        effort_cap:    저속 구간 상한 [N·m]. sim actuator 의 effort 와 맞출 것.
        margin:        안전 계수. 0.8 이면 실제 곡선의 80% 안에서 놀게 한다.
                       ROBOTIS 도 실제 성능은 스톨토크보다 N-T 곡선에 가깝다고
                       명시하므로 1.0 을 그대로 믿지 않는다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    jids = asset_cfg.joint_ids

    tau = robot.data.applied_torque[:, jids]              # (N,J)
    omega = robot.data.joint_vel[:, jids].abs()           # (N,J)

    tau_max = stall_torque * (1.0 - omega / max(no_load_speed, _EPS))
    tau_max = tau_max.clamp(min=0.0, max=effort_cap) * margin

    over = (tau.abs() - tau_max).clamp(min=0.0)
    return torch.sum(torch.square(over), dim=1)


def joint_vel_over_limit_penalty(
    env: ManagerBasedRLEnv,
    vel_limit: float = 3.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절속도가 vel_limit 을 넘는 초과분 제곱 벌점 (N,). 음수 weight.

    torque_envelope_penalty 의 단순·강건 버전. 토크 포락선은 applied_torque 에
    의존해 액추에이터 구현에 민감한데, 이 항은 관절속도만 보므로 항상 동작한다.
    3.5 rad/s 를 넘으면 실제 XM540 의 가용 토크가 2.6 N·m 아래로 떨어지므로,
    그 위로는 "명령해도 실기가 못 따라오는 영역" 이다.
    일반 보행은 관절속도가 2 rad/s 를 잘 안 넘으므로 보행을 해치지 않는다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    qd = robot.data.joint_vel[:, asset_cfg.joint_ids].abs()
    over = (qd - vel_limit).clamp(min=0.0)
    return torch.sum(torch.square(over), dim=1)


# ============================================================================
# ★ 2. Capture point 마진 — "넘어짐" 의 직접 예측량
# ============================================================================
def capture_point_margin_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.10,
    post_kick_window_s: float = 1.0,
    boost: float = 3.0,
    force_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Capture point 가 지지 발 중심에 가까울수록 높은 보상 (N,). 양수 weight.

    ξ = p_com + v_com · sqrt(h / g)
    reward = exp( −(‖ξ − p_support‖ / std)² ) · boost_scale

    【왜 base_roll/pitch 벌점으로는 부족한가】
    자세각 벌점은 "이미 기울어진 뒤" 에만 반응한다. 킥의 전도는 자세가 아직
    멀쩡한데 **수평 속도가 이미 지지면 밖을 향하고 있어서** 일어난다.
    capture point 는 그 속도를 위치로 환산해 주므로 한 박자 먼저 반응한다.
    선형 도립진자 근사라 정확한 값은 아니지만, 학습 신호로는 충분하다.

    CoM 은 base link 위치로 근사한다(키드사이즈는 몸통이 질량 대부분).
    정확한 전신 CoM 이 필요하면 body_pos_w × 질량 가중으로 바꾸면 되지만
    매 스텝 비용이 커지고, 근사로도 gradient 방향은 맞다.

    Args:
        std:                마진 가우시안 폭 [m]. 발 길이의 절반 정도(0.09)보다
                            약간 크게 잡아 완만한 gradient 를 준다.
        post_kick_window_s: 이 시간 동안 boost 배 강하게 건다 [s].
        boost:              킥 윈도우 내 배율.
        force_threshold:    접지 판정 힘 [N].
    """
    robot: Articulation = env.scene[asset_cfg.name]

    com_xy = robot.data.root_pos_w[:, :2]
    vel_xy = robot.data.root_lin_vel_w[:, :2]
    h = robot.data.root_pos_w[:, 2].clamp(min=0.15)
    omega0 = torch.sqrt(_G / h)                                   # (N,)
    xi = com_xy + vel_xy / omega0.unsqueeze(-1)                   # (N,2) capture point

    # 지지 다각형 근사: 접지한 발들의 중심
    contact = _foot_contact(env, sensor_cfg, force_threshold)     # (N,2)
    foot_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]    # (N,2,2)
    w = contact.float()                                           # (N,2)
    n_contact = w.sum(dim=-1, keepdim=True)                       # (N,1)
    # 양발 다 공중이면 가까운 발을 기준으로 (보상이 갑자기 0 이 되면 value 가 튄다)
    fallback = torch.zeros_like(w)
    d_all = torch.norm(foot_xy - xi.unsqueeze(1), dim=-1)         # (N,2)
    fallback.scatter_(1, d_all.argmin(dim=-1, keepdim=True), 1.0)
    w = torch.where(n_contact > 0.5, w, fallback)
    support_xy = (foot_xy * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=-1, keepdim=True)

    d = torch.norm(xi - support_xy, dim=-1)                       # (N,)
    reward = torch.exp(-torch.square(d / max(std, _EPS)))
    return reward * _boost(env, post_kick_window_s, boost)


def metric_capture_point_dist(
    env: ManagerBasedRLEnv,
    force_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Capture point ↔ 지지발 중심 거리 [m] (N,). weight=0.0 로그 전용.

    실기 전도의 선행지표. 0.10 m 를 넘기 시작하면 위험 구간이다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    com_xy = robot.data.root_pos_w[:, :2]
    vel_xy = robot.data.root_lin_vel_w[:, :2]
    h = robot.data.root_pos_w[:, 2].clamp(min=0.15)
    xi = com_xy + vel_xy / torch.sqrt(_G / h).unsqueeze(-1)
    contact = _foot_contact(env, sensor_cfg, force_threshold)
    foot_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    w = contact.float()
    n_contact = w.sum(dim=-1, keepdim=True)
    d_all = torch.norm(foot_xy - xi.unsqueeze(1), dim=-1)
    fallback = torch.zeros_like(w)
    fallback.scatter_(1, d_all.argmin(dim=-1, keepdim=True), 1.0)
    w = torch.where(n_contact > 0.5, w, fallback)
    support_xy = (foot_xy * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=-1, keepdim=True)
    return torch.norm(xi - support_xy, dim=-1)


# ============================================================================
# ★ 3. 지지발 미끄러짐 / 비행상 벌점
# ============================================================================
def support_foot_slip_penalty(
    env: ManagerBasedRLEnv,
    force_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """접지한 발의 수평 속도를 제곱 벌점 (N,). 음수 weight.

    인사이드 킥은 지지발에 큰 좌우 반작용을 건다. sim 마찰(0.7~1.2)이
    실제 필드(인조잔디/매트)보다 후하면 sim 에서는 안 미끄러지는데 실기에서만
    지지발이 밀리면서 넘어진다. 미끄러짐을 벌해 반작용을 더 수직에 가깝게
    처리하도록 유도하면, 마찰이 낮은 실기에서도 여유가 생긴다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    contact = _foot_contact(env, sensor_cfg, force_threshold)          # (N,2)
    foot_v_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]   # (N,2,2)
    slip = torch.sum(torch.square(foot_v_xy), dim=-1)                  # (N,2)
    return (slip * contact.float()).sum(dim=-1)


def flight_phase_penalty(
    env: ManagerBasedRLEnv,
    post_kick_window_s: float = 1.0,
    boost: float = 3.0,
    force_threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """양발이 동시에 공중이면 스텝당 1.0 벌점 (N,). 음수 weight.

    킥 도중 반작용으로 지지발까지 뜨는 순간이 생기면 그 시점부터는 어떤 제어도
    자세를 못 바꾼다. sim 에서는 착지 후 회복이 되지만 실기에서는 착지 충격 +
    센서 지연 + 토크 부족이 겹쳐 그대로 넘어간다.
    단각지지(한 발)는 정상이므로 벌하지 않는다. 양발 공중만 벌한다.
    """
    contact = _foot_contact(env, sensor_cfg, force_threshold)
    airborne = ~contact.any(dim=-1)
    return airborne.float() * _boost(env, post_kick_window_s, boost)


def base_ang_vel_kick_windowed(
    env: ManagerBasedRLEnv,
    post_kick_window_s: float = 1.0,
    boost: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """몸통 roll/pitch 각속도 제곱 벌점, 킥 윈도우에서만 강화 (N,). 음수 weight.

    기존 ang_vel_xy_l2(weight −0.05)는 보행 전체에 걸린 약한 항이라 킥 순간의
    각운동량 폭주를 못 잡는다. 다리를 휘두르면 반작용으로 몸통에 각운동량이
    실리는데, 팔이 고정(stiffness 0)이라 상쇄해 줄 자유도가 없다.
    킥 윈도우에서만 강하게 벌해 "스윙 각운동량을 지지발 토크로 미리 상쇄" 하도록
    유도한다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    w_xy = robot.data.root_ang_vel_b[:, :2]
    return torch.sum(torch.square(w_xy), dim=1) * _boost(env, post_kick_window_s, boost)


# ============================================================================
# ★ 4. IMU 관측 오차 모델 — i.i.d. 노이즈 → 에피소드 상수 바이어스
# ----------------------------------------------------------------------------
# 【왜 필요한가】
# 현재 projected_gravity 에 Unoise(±0.1)를 준다. 그런데 실기의 IMU 오차는
# i.i.d. 가 아니다:
#   · IMU 장착 각도 오차 / 캘리브레이션 잔차 → **에피소드 내내 일정한 바이어스**
#   · 자이로 바이어스 드리프트 → 느리게 변하는 상관 오차
#   · Madgwick 은 가속도로 중력을 추정하는데, 킥의 큰 선형가속도가 그대로
#     중력 추정 오차로 들어간다 → 하필 킥 순간에 오차가 최대
# i.i.d. 노이즈는 history_length=5 스택 안에서 평균이 상쇄돼 정책이 그냥 무시하는
# 법을 배운다. **상수 바이어스는 상쇄되지 않는다.** 그래서 i.i.d. 만으로 학습한
# 정책은 실기의 1~3° 장착 오차에 그대로 취약하다.
# ============================================================================
def reset_imu_bias(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    grav_bias_range: float = 0.05,
    gyro_bias_range: float = 0.05,
) -> None:
    """에피소드마다 IMU 상수 바이어스를 샘플한다.

    ※ env_ids 에 기본값을 주지 말 것 (IsaacLab EventManager 가 인자 순서를
      기본값 유무로 재정렬해 빌드가 깨진다). 위치 인자로 둔다.

    Args:
        grav_bias_range: projected_gravity 각 성분 바이어스 범위 [무차원].
                         0.05 ≈ 2.9° 장착/캘리브 오차.
        gyro_bias_range: base_ang_vel 각 성분 바이어스 범위 [rad/s].
    """
    n, dev = env.num_envs, env.device
    if not hasattr(env, "imu_grav_bias"):
        env.imu_grav_bias = torch.zeros(n, 3, device=dev)
    if not hasattr(env, "imu_gyro_bias"):
        env.imu_gyro_bias = torch.zeros(n, 3, device=dev)

    k = len(env_ids)
    env.imu_grav_bias[env_ids] = (
        (torch.rand(k, 3, device=dev) * 2.0 - 1.0) * grav_bias_range
    )
    env.imu_gyro_bias[env_ids] = (
        (torch.rand(k, 3, device=dev) * 2.0 - 1.0) * gyro_bias_range
    )


def projected_gravity_biased(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """projected_gravity + 에피소드 상수 바이어스 (N,3).

    기존 mdp.projected_gravity 를 그대로 대체한다. ObsTerm 의 Unoise 는 그대로
    두고 이 바이어스를 더한다(i.i.d. 센서잡음 + 상수 장착오차 = 실제 오차 구조).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    g = robot.data.projected_gravity_b
    if not hasattr(env, "imu_grav_bias"):
        return g
    return g + env.imu_grav_bias


def base_ang_vel_biased(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base_ang_vel + 에피소드 상수 자이로 바이어스 (N,3)."""
    robot: Articulation = env.scene[asset_cfg.name]
    w = robot.data.root_ang_vel_b
    if not hasattr(env, "imu_gyro_bias"):
        return w
    return w + env.imu_gyro_bias


# ============================================================================
# ★ 5. 킥 직후 표적 외란 — 실패가 나는 바로 그 구간을 훈련
# ============================================================================
def push_robot_on_kick(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    post_kick_window_s: float = 0.6,
    probability: float = 0.02,
    velocity_range: dict | None = None,
) -> None:
    """킥 윈도우 안의 env 에만 확률적으로 속도 외란을 준다 (mode="interval").

    기존 push_robot 은 1~3 초 간격으로 무작위 시점에 밀기 때문에, 정작 전도가
    나는 "킥 직후 0.6 초" 에 걸릴 확률이 낮다. 이 항은 그 구간만 노려서 민다.
    실기의 미지 외란(모델 오차·지면 요철·임팩트 반작용 오차)을 대표한다.

    Args:
        post_kick_window_s: 킥 후 이 시간 안에서만 민다 [s].
        probability:        스텝당 밀 확률. 0.02 면 0.6 s(30스텝) 안에 ~45%.
        velocity_range:     {"x": (lo,hi), "y": (lo,hi)} [m/s].
    """
    if velocity_range is None:
        velocity_range = {"x": (-0.35, 0.35), "y": (-0.35, 0.35)}

    robot: Articulation = env.scene["robot"]
    in_win = _kick_window(env, post_kick_window_s)
    roll = torch.rand(env.num_envs, device=env.device) < probability
    sel = torch.nonzero(in_win & roll, as_tuple=False).flatten()
    if sel.numel() == 0:
        return

    vel = robot.data.root_lin_vel_w[sel].clone()
    for axis, idx in (("x", 0), ("y", 1)):
        lo, hi = velocity_range.get(axis, (0.0, 0.0))
        vel[:, idx] += lo + (hi - lo) * torch.rand(sel.numel(), device=env.device)

    # root_vel_w 는 IsaacLab 이 push_by_setting_velocity 에서 쓰는 것과 동일한
    # 6-vector(lin+ang) 다. 버전에 따라 이름이 바뀔 수 있어 방어적으로 접근한다.
    root_vel_all = getattr(robot.data, "root_vel_w", None)
    if root_vel_all is None:
        root_vel_all = torch.cat(
            [robot.data.root_lin_vel_w, robot.data.root_ang_vel_w], dim=-1
        )
    root_vel = root_vel_all[sel].clone()
    root_vel[:, :3] = vel
    robot.write_root_velocity_to_sim(root_vel, env_ids=sel)


# ============================================================================
# ★ 버전 호환 — 없는 IsaacLab 함수를 안전하게 건너뛰기
# ----------------------------------------------------------------------------
# mdp.randomize_joint_parameters 는 IsaacLab 2.3.x 에는 있지만 버전에 따라
# 이름/존재가 다르다. EventTerm 은 클래스 정의 시점에 func 속성에 접근하므로,
# 없으면 import 단계에서 AttributeError 로 죽는다.
# → env cfg 에서 getattr 로 확인하고, 없으면 이 no-op 을 대신 꽂는다.
#   params 키는 그대로 두어도 되도록 **kwargs 를 받는다(스트레스 cfg 가
#   params["armature_distribution_params"] 를 수정해도 KeyError 가 안 난다).
# ============================================================================
_NOOP_WARNED = {"done": False}


def noop_event(env: ManagerBasedRLEnv, env_ids: torch.Tensor, **kwargs) -> None:
    """아무것도 하지 않는 이벤트. 버전 호환용 자리 채우기.

    한 번만 경고를 출력해 "조용히 무시" 를 막는다. DR 항목이 하나 빠진 채
    학습이 돌아가면 나중에 원인을 못 찾기 때문이다.
    """
    if not _NOOP_WARNED["done"]:
        _NOOP_WARNED["done"] = True
        print(
            "[kick_stability_mdp] 경고: randomize_joint_parameters 를 찾을 수 없어 "
            "armature/joint-friction 랜덤화를 건너뜁니다. "
            "IsaacLab 버전을 확인하거나 해당 EventTerm 을 직접 구현하세요."
        )


# ============================================================================
# ★ 메트릭 (weight=0.0) — 원인 특정용. metric_* 는 파라미터 최소화.
# ============================================================================
def metric_max_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """이번 스텝 관절속도의 최댓값 [rad/s] (N,). weight=0.0 로그 전용.

    3.5 를 넘으면 실제 XM540 가용 토크가 2.6 N·m 아래로 떨어지는 영역이다.
    이 값이 4 를 넘나들면 실기에서 스윙발이 궤적을 못 따라가고 있다는 뜻.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, asset_cfg.joint_ids].abs().max(dim=1).values


def metric_torque_envelope_violation(
    env: ManagerBasedRLEnv,
    stall_torque: float = 7.3,
    no_load_speed: float = 5.55,
    effort_cap: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """포락선 초과 토크의 최댓값 [N·m] (N,). weight=0.0 로그 전용.

    0 에 붙어 있으면 정책이 실제 모터로 실현 가능한 궤적만 쓰고 있다는 뜻이다.
    이 값이 크면 sim 에서만 되는 킥을 하고 있는 것.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    jids = asset_cfg.joint_ids
    tau = robot.data.applied_torque[:, jids].abs()
    omega = robot.data.joint_vel[:, jids].abs()
    tau_max = (stall_torque * (1.0 - omega / max(no_load_speed, _EPS))).clamp(
        min=0.0, max=effort_cap
    )
    return (tau - tau_max).clamp(min=0.0).max(dim=1).values


def metric_available_torque_at_speed(
    env: ManagerBasedRLEnv,
    stall_torque: float = 7.3,
    no_load_speed: float = 5.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """현재 관절속도에서 실제 모터가 낼 수 있는 최소 토크 [N·m] (N,). 로그 전용.

    가장 빠른 관절 기준. 킥 순간 이 값이 1.5 아래로 내려가면, 그 순간 sim 은
    5 N·m 를 쓰고 있는데 실기는 1.5 N·m 밖에 못 낸다는 뜻이다.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    omega = robot.data.joint_vel[:, asset_cfg.joint_ids].abs().max(dim=1).values
    return (stall_torque * (1.0 - omega / max(no_load_speed, _EPS))).clamp(min=0.0)


def motor_envelope_table(
    stall_torque: float = 7.3,
    no_load_speed_rpm: float = 53.0,
    speeds: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0),
) -> str:
    """속도-토크 포락선 표를 문자열로 (설정 검토용, 학습과 무관).

    파이썬에서 바로 호출해 sim effort 값과 비교해 보라는 용도다.
    """
    w_nl = no_load_speed_rpm * 2.0 * math.pi / 60.0
    lines = [f"no_load_speed = {w_nl:.2f} rad/s, stall = {stall_torque:.2f} N.m",
             "  omega[rad/s]   tau_max[N.m]"]
    for w in speeds:
        lines.append(f"  {w:8.2f}   {max(0.0, stall_torque * (1.0 - w / w_nl)):10.2f}")
    return "\n".join(lines)
