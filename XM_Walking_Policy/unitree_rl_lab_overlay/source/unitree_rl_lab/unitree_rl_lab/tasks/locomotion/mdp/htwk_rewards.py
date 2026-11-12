# htwk_rewards.py
# ============================================================================
# htwk-gym (booster_gym 계열) 에서 이식한 리워드 모음.
#
# 이식 원칙:
#   - 기존 rewards.py 는 건드리지 않는다 (v0 튜닝값 보존)
#   - 태스크 공간(발 위치/자세)에서 재는 항을 우선한다
#   - 보행 중 진동하는 양은 "창 평균 후 abs" 로 DC 편향만 잡는다
#
# 좌표 규약:
#   발 관련 항은 전부 "몸통 heading 프레임" 에서 잰다.
#   yaw_quat() 으로 roll/pitch 를 제거한 순수 yaw 쿼터니언을 만든 뒤
#   quat_rotate_inverse() 로 월드 차이벡터를 회전시킨다.
#   → 로봇이 어느 방향을 보든, 몸통이 기울어져 있든 값이 불변.
#
# 이 파일의 모든 리워드는 "양수 penalty" 를 반환한다.
# env_cfg 에서 음의 weight 를 곱해 쓸 것. (보상 성격인 feet_swing 만 예외)
# ============================================================================

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# 내부 헬퍼
# ============================================================================
def _hold_mask(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """reset hold 구간 마스크. reset_hold 미사용 환경이면 None."""
    hold = getattr(env, "reset_hold_steps", None)
    if hold is None:
        return None
    return hold > 0


def _feet_offset_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """두 발의 상대 위치를 몸통 heading 프레임에서 계산한다.

    HTWK get_feet_offset() 이식판.

    Args:
        asset_cfg: body_names 에 발 2 개가 지정된 SceneEntityCfg.
                   preserve_order=True 로 [왼발, 오른발] 순서를 고정할 것.

    Returns:
        (fore_aft, lateral)
            fore_aft: 몸통 전방(+x) 기준 두 발의 전후 간격 [m]
                      부호는 body_names 순서에 의존 (첫 발 - 둘째 발)
            lateral:  몸통 좌측(+y) 기준 두 발의 좌우 간격 [m]
                      부호 무관하게 쓰려면 abs() 를 취할 것
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 발 2 개의 월드 위치 [N, 2, 3]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    diff_w = foot_pos_w[:, 0, :] - foot_pos_w[:, 1, :]  # [N, 3]

    # 몸통의 yaw 성분만 남긴 쿼터니언 → roll/pitch 기울기의 영향 제거
    heading_q = yaw_quat(asset.data.root_quat_w)
    diff_b = quat_rotate_inverse(heading_q, diff_w)  # [N, 3]

    return diff_b[:, 0], diff_b[:, 1]


def _windowed_mean_abs(
    env: ManagerBasedRLEnv,
    key: str,
    value: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """값을 링버퍼에 누적하고 "창 평균 후 abs" 를 반환한다.

    보행 중 좌우로 진동하는 양(예: 발 전후 간격)의 순간값을 벌하면
    보행 자체를 벌하게 된다. 한 주기 이상을 덮는 창으로 평균 내면
    진동은 상쇄되고 DC 편향(= 한쪽으로 치우친 gait)만 남는다.

    기존 pelvis_pitch_symmetry_windowed 의 링버퍼 로직과 동일하되,
    측정량만 관절 공간 -> 태스크 공간으로 바꾼 것.
    """
    buf_attr = f"_hw_buf_{key}"
    idx_attr = f"_hw_idx_{key}"

    buf = getattr(env, buf_attr, None)
    if buf is None or buf.shape[0] != env.num_envs or buf.shape[1] != window_size:
        buf = torch.zeros(env.num_envs, window_size, device=env.device)
        setattr(env, buf_attr, buf)
        setattr(env, idx_attr, 0)

    # 리셋된 env 는 창을 비운다 (이전 에피소드 값이 섞이지 않도록)
    reset_envs = env.episode_length_buf == 0
    if torch.any(reset_envs):
        buf[reset_envs] = 0.0

    write_idx = getattr(env, idx_attr)
    buf[:, write_idx] = value
    setattr(env, idx_attr, (write_idx + 1) % window_size)

    return torch.abs(torch.mean(buf, dim=1))


def _ema(
    env: ManagerBasedRLEnv,
    key: str,
    value: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """1 차 저역통과 필터. HTWK filtered_lin_vel / filtered_ang_vel 이식판.

    발 착지 충격이 동체 속도에 만드는 고주파 성분을 제거한다.
    필터링하지 않으면 정책이 매 착지마다 속도 오차를 보정하려 들어
    액션이 떨린다.

    alpha=0.1, step_dt=0.02 기준 시상수 약 0.2 s.
    """
    attr = f"_hw_ema_{key}"
    filt = getattr(env, attr, None)
    if filt is None or filt.shape != value.shape:
        filt = torch.zeros_like(value)
        setattr(env, attr, filt)

    reset_envs = env.episode_length_buf == 0
    if torch.any(reset_envs):
        filt[reset_envs] = 0.0

    filt.mul_(1.0 - alpha).add_(value, alpha=alpha)
    return filt


# ============================================================================
# 1. EMA 필터 속도 추종
# ============================================================================
def track_lin_vel_xy_ema(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    alpha: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """heading 프레임 xy 속도 추종. 실측 속도에 EMA 를 적용해 채점한다.

    기존 track_lin_vel_xy_hold_aware 와 동일한 hold 게이팅을 유지한다.
    차이는 순간 속도 대신 필터 속도를 쓴다는 점뿐.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # heading 프레임 선속도 (isaaclab 표준 track_lin_vel_xy_yaw_frame_exp 와 동일)
    vel_b = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)
    vel_xy = _ema(env, f"linvel_{command_name}", vel_b[:, :2], alpha)

    cmd = env.command_manager.get_command(command_name)
    err = torch.sum(torch.square(cmd[:, :2] - vel_xy), dim=1)
    reward = torch.exp(-err / std**2)

    mask = _hold_mask(env)
    if mask is not None:
        reward = reward.clone()
        reward[mask] = 0.0
    return reward


def track_ang_vel_z_ema(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    alpha: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """월드 기준 yaw 각속도 추종. EMA 필터 적용판."""
    asset: Articulation = env.scene[asset_cfg.name]
    wz = _ema(env, f"angvel_{command_name}", asset.data.root_ang_vel_w[:, 2], alpha)

    cmd = env.command_manager.get_command(command_name)
    err = torch.square(cmd[:, 2] - wz)
    reward = torch.exp(-err / std**2)

    mask = _hold_mask(env)
    if mask is not None:
        reward = reward.clone()
        reward[mask] = 0.0
    return reward


# ============================================================================
# 2. 태스크 공간 발 배치
# ============================================================================
def feet_stance_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_distance: float,
    clip: float = 0.05,
) -> torch.Tensor:
    """두 발의 좌우 간격을 기준값으로 유지. HTWK feet_offset_y 이식판.

    ── 왜 필요한가 ──
    폭을 규제하지 않으면 RL 정책이 양 극단으로 간다.
      - 너무 좁음: 두 다리가 서로 부딪힘 (scissoring), self-collision
      - 너무 넓음: 좌우로 뒤뚱거림, 에너지 낭비, 인조잔디에서 발 바깥날 걸림
    v0 에는 이 항이 아예 없었다. HTWK 대비 유일하게 빠져 있던 규제.

    ── L1 + 포화 를 쓰는 이유 ──
    이차(L2)로 하면 오차가 클 때 폭주하고 작을 때 그래디언트가 사라진다.
    발 간격처럼 절대 정밀도가 중요한 양은 L1 이 낫고, clip 으로 포화시키면
    학습 초반 넘어진 상태의 거대한 오차가 전체 리워드를 지배하는 것을 막는다.

    Args:
        asset_cfg:       발 2 개 (순서 무관 — abs 로 처리)
        target_distance: 목표 좌우 간격 [m]. MX3 골반 폭 기준 0.12~0.14 권장.
        clip:            오차 포화 한계 [m]
    """
    _, lateral = _feet_offset_body(env, asset_cfg)
    err = torch.abs(torch.abs(lateral) - target_distance)
    return torch.clamp(err, max=clip)


def feet_fore_aft_symmetry_windowed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    window_size: int = 30,
    command_name: str | None = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """좌우 발의 전후 위치 편향을 창 평균으로 잡는다.

    pelvis_pitch_symmetry_windowed 의 태스크 공간 대체품.

    ── 왜 바꾸나 ──
    골반 pitch 합은 발 전후 위치의 "대리 변수" 일 뿐이다. 발 x 위치는
        x_foot ~ L_thigh*sin(q_hip) + L_shank*sin(q_hip - q_knee) + ...
    로 무릎이 개입한다. 따라서
      - 골반 pitch 에 DC 편향이 있어도 무릎이 상쇄하면 발은 대칭 (위양성)
      - 골반은 대칭인데 무릎이 비대칭이면 발은 어긋남 (위음성)
    발 위치를 직접 재면 둘 다 사라진다. 링크 길이가 바뀌어도 의미 불변이라
    v2 이식에도 그대로 쓸 수 있다.

    창 평균 로직은 기존 함수와 동일하다. 보행 중 두 발이 번갈아 앞뒤로
    놓이는 것은 정상이므로 순간값을 벌하면 안 되고, 한 주기(0.5 s) 이상을
    덮는 창으로 평균 내야 DC 편향만 남는다.

    Args:
        window_size: step_dt=0.02, 주기 0.5 s 기준 25~30 권장
    """
    fore_aft, _ = _feet_offset_body(env, asset_cfg)
    penalty = _windowed_mean_abs(env, "fore_aft", fore_aft, window_size)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm > command_threshold)
    return penalty


def feet_yaw_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    symmetric_weight: float = 1.0,
    antisymmetric_weight: float = 1.6,
) -> torch.Tensor:
    """몸통 기준 발 yaw 편차. 대칭/반대칭 모드로 분해해 가중한다.

    ── HTWK 의 4 개 항을 2 개 모드로 정리한 것 ──
    HTWK 는 foot_yaw_L(-1.0), foot_yaw_R(-1.0), feet_yaw_diff(-0.5),
    feet_yaw_mean(-0.5) 네 항을 쓰는데, 뒤 둘은 앞 둘의 선형 재조합이다.
    전개하면 이차형식
        1.625*(e_L^2 + e_R^2) - 0.75*e_L*e_R
    이고, 대칭/반대칭 모드로 대각화하면
        1.25 * s^2 + 2.0 * a^2     (s = (e_L+e_R)/sqrt2, a = (e_L-e_R)/sqrt2)
    즉 "두 발이 벌어지는 오차" 를 "같이 도는 오차" 보다 1.6 배 강하게 벌한다.
    팔자걸음(toe-out) 억제를 우선한다는 뜻이고 물리적으로 타당하다.
    이 함수는 그 비율을 파라미터로 직접 노출한다.

    ── 관절 공간 대비 장점 ──
    joint_deviation_hip_yaw 는 hip_yaw 관절각만 본다. 그런데 다리 체인이
    hip_roll -> hip_pitch 순서라 두 회전이 교환되지 않아, hip_yaw = 0 이어도
    발이 돌아간다. 발바닥을 지면과 평행하게 유지(ankle_roll = -hip_roll)해도
    합성 회전의 yaw 성분은 대략
        yaw ~ theta * sin(phi)      (theta = 시상면 굴곡 합, phi = hip_roll)
    이므로, theta=30 deg / phi=15 deg 에서 약 7.8 deg 가 남는다.
    관절각 페널티는 이걸 전혀 보지 못한다.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # [N, 2, 4]
    n_env = foot_quat.shape[0]

    # 몸통 heading 프레임 (roll/pitch 제거)
    heading_q = yaw_quat(asset.data.root_quat_w)  # [N, 4]

    # 발의 전방 단위벡터를 월드로 보낸 뒤, 몸통 heading 프레임으로 되돌린다.
    # 그 벡터의 xy 각도가 곧 "몸통 기준 발 yaw" 이다.
    fwd = torch.zeros(n_env, 3, device=env.device)
    fwd[:, 0] = 1.0

    yaws = []
    for i in range(foot_quat.shape[1]):
        f_w = quat_apply(foot_quat[:, i, :], fwd)      # 발 전방벡터 (월드)
        f_b = quat_rotate_inverse(heading_q, f_w)      # 몸통 heading 프레임
        yaws.append(torch.atan2(f_b[:, 1], f_b[:, 0]))

    e_a = yaws[0]
    e_b = yaws[1]

    s = (e_a + e_b) * 0.5   # 대칭 모드: 두 발이 같은 방향으로 회전
    a = (e_a - e_b) * 0.5   # 반대칭 모드: 두 발이 벌어짐/모임

    return symmetric_weight * torch.square(s) + antisymmetric_weight * torch.square(a)


# ============================================================================
# 3. 위상 기반 접촉 리워드
# ============================================================================
def _gait_term(env: ManagerBasedRLEnv, command_name: str):
    """GaitPhaseCommand term 을 가져온다."""
    return env.command_manager.get_term(command_name)


def feet_swing_phase(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_command_name: str = "gait",
    swing_period: float = 0.2,
) -> torch.Tensor:
    """스윙 구간에 발이 떠 있으면 보상. HTWK feet_swing 이식판.

    위상 0.25 부근을 첫 번째 발, 0.75 부근을 두 번째 발의 스윙 구간으로 본다.
    (두 발의 역할이 바뀌어도 반주기 이동일 뿐이라 gait 자체는 동일하다.)

    feet_gait 가 "stance 여야 할 때 접촉 / swing 이어야 할 때 비접촉" 을
    양방향으로 채점하는 반면, 이 항은 "발을 확실히 들어라" 만 본다.
    발을 끌고 다니는(foot dragging) 습관을 직접 억제하며, 인조잔디에서
    가장 흔한 실패 모드에 대응한다.

    Returns:
        0 ~ 2 범위의 양수 보상. env_cfg 에서 양의 weight 를 줄 것.
    """
    term = _gait_term(env, gait_command_name)
    phase = term.phase
    active = term.effective_frequency > 1.0e-8

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    half = 0.5 * swing_period
    reward = torch.zeros(env.num_envs, device=env.device)
    centers = [0.25, 0.75]
    for i in range(min(len(sensor_cfg.body_ids), len(centers))):
        swing = (torch.abs(phase - centers[i]) < half) & active
        reward += (swing & ~is_contact[:, i]).float()
    return reward


def feet_gait_phase_aware(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_command_name: str = "gait",
    offset: tuple[float, ...] = (0.0, 0.5),
    threshold: float = 0.55,
) -> torch.Tensor:
    """기존 feet_gait 를 GaitPhaseCommand 의 위상으로 재작성한 것.

    기존 구현은 위상을 episode_length_buf 로 매번 재계산했다. 그 위상은
    관측에 들어가지 않으므로 정책이 볼 수 없는 시계로 채점하는 셈이었다.
    이 버전은 관측에 들어가는 것과 "같은" 위상을 참조한다.

    정지 시(주파수 0)에는 자동으로 꺼진다. 기존의 command_name 게이팅과
    동일한 효과이며, 판정 기준이 한 곳(GaitPhaseCommand)으로 통일된다.
    """
    term = _gait_term(env, gait_command_name)
    phase = term.phase.unsqueeze(1)
    active = (term.effective_frequency > 1.0e-8).float()

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        leg_phase = torch.fmod(phase[:, 0] + offset[i], 1.0)
        is_stance = leg_phase < threshold
        reward += (~(is_stance ^ is_contact[:, i])).float()
    return reward * active


# ============================================================================
# 4. 에너지 / 액추에이터
# ============================================================================
def joint_torque_tiredness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    effort_limit: float,
) -> torch.Tensor:
    """정격 대비 토크 사용률의 제곱합. HTWK torque_tiredness 이식판.

    ── torques(절대 토크 제곱합) 와의 차이 ──
    관절별 토크 정격이 다를 때 의미가 갈린다. 지금 MX3 는 전 관절
    effort=6.0 N*m 로 동일하므로 torques 와 상수배 차이일 뿐이지만,
    v2 에서 RS06 / RS05 / RS00 이 섞이면 정격이 관절마다 달라진다.
    그때 절대 토크를 벌하면 큰 모터만 아끼고 작은 모터를 혹사시킨다.
    사용률로 벌하면 모든 관절이 여유율 기준으로 균형 있게 쓰인다.

    한 발 지지 시 무릎 발열이 문제가 되는 저감속비 액추에이터에서는
    이 항이 발열 억제에 직접 대응한다.

    Args:
        effort_limit: 관절 토크 정격 [N*m]. 관절별로 다르면 이 함수를
                      복사해 텐서 파라미터로 바꿔 쓸 것.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(tau / effort_limit), dim=1)


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """기계적 일률 |tau * qdot| 의 합. HTWK power 이식판.

    토크만 벌하면 "힘은 안 쓰되 빠르게 움직이는" 해가 남고,
    속도만 벌하면 "느리게 버티며 전류를 계속 흘리는" 해가 남는다.
    일률은 둘의 곱이라 양쪽을 동시에 억제한다.
    배터리 33 V 컷오프 여유 확보에도 직접 기여한다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(tau * qd), dim=1)


# ============================================================================
# 5. 진단 전용 지표 (weight=0 으로 두고 텐서보드에서 추적)
# ============================================================================
def metric_stance_width(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """현재 좌우 발 간격 [m]. 실기 튜닝 시 자로 잰 값과 직접 비교 가능."""
    _, lateral = _feet_offset_body(env, asset_cfg)
    return torch.abs(lateral)


def metric_fore_aft_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """현재 전후 발 간격 [m]. 부호 포함 (한쪽 치우침 방향 확인용)."""
    fore_aft, _ = _feet_offset_body(env, asset_cfg)
    return fore_aft


def metric_gait_frequency(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """실효 보행 주파수 [Hz]. 정지 env 는 0."""
    return _gait_term(env, gait_command_name).effective_frequency


# 라디안/도 변환이 필요한 곳에서 쓰는 상수
DEG = math.pi / 180.0
