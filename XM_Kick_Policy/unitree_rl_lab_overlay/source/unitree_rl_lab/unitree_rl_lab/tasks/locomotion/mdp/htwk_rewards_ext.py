# htwk_rewards_ext.py
# ============================================================================
# htwk-gym 이식 2 차분. v1(htwk_rewards.py)에 이어 나머지 리워드를 채운다.
#
# 구성:
#   [A] 접촉/발 거동     feet_slip, feet_vel_z, feet_roll, feet_pitch
#   [B] 동체 거동        root_acc, base_orientation_target
#   [C] 파라미터 추종    feet_offset_x/y_command, feet_yaw_command
#
# [C] 는 v3(파라미터 보행) 전용이다. v2 에서는 쓰지 않는다.
# 속도 게이팅(vel_scale) 구조를 원본 그대로 살렸다 — 이게 없으면
# 걷는 중에도 발 배치를 강제해서 보행이 굳는다.
# ============================================================================

from __future__ import annotations

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
def _foot_euler_rp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """각 발의 roll / pitch 를 월드 기준으로 계산한다. [N, B] 두 개 반환.

    HTWK 는 get_euler_xyz 로 직접 뽑지만, IsaacLab 에는 발 개수만큼
    반복하는 편이 안전하다. 발 프레임의 x/z 축을 월드로 보내
    기울기 성분을 직접 잰다.
      pitch: 발 전방축(x)이 월드 수평면에서 얼마나 들렸는가
      roll : 발 좌측축(y)이 월드 수평면에서 얼마나 들렸는가
    작은 각도에서 오일러각과 일치하고, 짐벌락이 없다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # [N, B, 4]
    n_env, n_body = quat.shape[0], quat.shape[1]

    ex = torch.zeros(n_env, 3, device=env.device)
    ex[:, 0] = 1.0
    ey = torch.zeros(n_env, 3, device=env.device)
    ey[:, 1] = 1.0

    rolls, pitches = [], []
    for i in range(n_body):
        q = quat[:, i, :]
        x_w = quat_apply(q, ex)
        y_w = quat_apply(q, ey)
        # z 성분이 곧 수평면 대비 기울기 sin 값
        pitches.append(torch.asin(torch.clamp(-x_w[:, 2], -1.0, 1.0)))
        rolls.append(torch.asin(torch.clamp(y_w[:, 2], -1.0, 1.0)))

    return torch.stack(rolls, dim=1), torch.stack(pitches, dim=1)


def _raw_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """커맨드의 "원단위" 값을 가져온다.

    ParameterCommand 는 normalize_observation=True 일 때 command 프로퍼티가
    정규화된 값을 반환한다. 관측에는 그게 맞지만 리워드는 원단위(rad, m)를
    봐야 하므로 raw_command 속성을 직접 읽는다.
    일반 CommandTerm 이면 command 를 그대로 쓴다.
    """
    term = env.command_manager.get_term(command_name)
    raw = getattr(term, "raw_command", None)
    return raw if raw is not None else term.command


def _last_buf(
    env: ManagerBasedRLEnv,
    key: str,
    value: torch.Tensor,
) -> torch.Tensor:
    """직전 스텝 값을 반환하고 현재 값을 저장한다.

    리셋된 env 는 현재 값으로 채워 차분이 0 이 되게 한다.
    (리셋 직후 위치가 순간이동하므로 차분을 그대로 쓰면 거대한 값이 나온다)
    """
    attr = f"_hwx_last_{key}"
    last = getattr(env, attr, None)
    if last is None or last.shape != value.shape:
        last = value.clone()
        setattr(env, attr, last)

    reset_envs = env.episode_length_buf <= 1
    if torch.any(reset_envs):
        last[reset_envs] = value[reset_envs]

    out = last.clone()
    last.copy_(value)
    return out


# ============================================================================
# [A] 접촉 / 발 거동
# ============================================================================
def feet_slip(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """접촉 중인 발의 수평 이동 속도 제곱합. HTWK feet_slip 이식판.

    ── 왜 중요한가 ──
    지지발이 미끄러지면 (1) 추진력이 손실되고 (2) 오도메트리가 어긋나며
    (3) 인조잔디에서는 발이 걸려 넘어진다. v0 에는 이 항이 없어서
    정책이 "발을 끌면서 몸을 밀어내는" 해를 배울 여지가 있었다.

    ── 구현 주의 ──
    IsaacLab 의 body_pos_w 는 환경 원점 기준이 아니라 월드 기준이므로,
    env_origins 가 서로 다른 병렬 환경에서도 차분은 올바르게 나온다.
    z 성분은 제외한다 (발을 드는 동작까지 벌하면 안 됨).

    asset_cfg 와 sensor_cfg 의 body 순서가 일치해야 한다.
    둘 다 preserve_order=True 로 같은 이름 목록을 줄 것.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]      # [N, B, 2]
    last = _last_buf(env, "feet_pos_xy", pos)
    vel = (pos - last) / env.step_dt

    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    penalty = torch.sum(torch.sum(torch.square(vel), dim=-1) * contact.float(), dim=-1)

    # 리셋 직후 1 스텝은 제외 (순간이동 차분 방지)
    return penalty * (env.episode_length_buf > 1).float()


def feet_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """발의 수직 속도 제곱합. 착지 충격을 줄인다.

    발을 세게 내리찍으면 (1) 기구에 충격이 가고 (2) IMU 에 고주파
    노이즈가 실리며 (3) 동체가 튀어 자세 추정이 흔들린다.
    부드러운 착지를 유도한다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    last = _last_buf(env, "feet_pos_z", pos_z)
    vel = (pos_z - last) / env.step_dt
    return torch.sum(torch.square(vel), dim=-1) * (env.episode_length_buf > 1).float()


def feet_roll_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """발 roll 각의 제곱합. 발 안쪽/바깥날로 딛는 것을 억제한다.

    v0 의 foot_flat_orientation 은 발바닥 법선과 중력의 정렬을
    하나의 스칼라로 재므로 roll 과 pitch 를 구분하지 않는다.
    이 함수는 roll 만 따로 벌해 가중치를 분리할 수 있게 한다.
    인조잔디 측면 경사 대응에서는 roll 을 pitch 보다 강하게 잡는 것이
    유리하다 (HTWK 도 roll -0.2 / pitch -0.1 로 2 배 차이).
    """
    roll, _ = _foot_euler_rp(env, asset_cfg)
    return torch.sum(torch.square(roll), dim=-1)


def feet_pitch_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """발 pitch 각의 제곱합. 발끝/뒤꿈치로 딛는 것을 억제한다.

    다만 보행에서 heel-strike / toe-off 는 정상 동작이므로
    roll 보다 약하게 주어야 한다. 너무 세게 주면 발을 평평하게 유지하려
    무릎을 과도하게 굽히는 어색한 보행이 나온다.
    """
    _, pitch = _foot_euler_rp(env, asset_cfg)
    return torch.sum(torch.square(pitch), dim=-1)


# ============================================================================
# [B] 동체 거동
# ============================================================================
def root_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """동체 선/각가속도 제곱합. HTWK root_acc 이식판.

    lin_vel_z_l2 / ang_vel_xy_l2 가 "속도" 를 벌하는 반면 이 항은
    "가속도" 를 벌한다. 즉 급격한 속도 변화를 억제해 동작을 매끄럽게
    만든다. IMU 에 실리는 충격 성분을 줄이는 효과가 크다.

    가중치는 아주 작아야 한다 (HTWK 는 -1e-4). 스케일이 큰 양이라
    가중치를 크게 주면 로봇이 아예 안 움직이려 든다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel = torch.cat(
        [asset.data.root_lin_vel_w, asset.data.root_ang_vel_w], dim=-1
    )  # [N, 6]
    last = _last_buf(env, "root_vel", vel)
    acc = (vel - last) / env.step_dt
    return torch.sum(torch.square(acc), dim=-1) * (env.episode_length_buf > 1).float()


def base_orientation_command_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    roll_index: int,
    pitch_index: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """동체 roll/pitch 를 명령값으로 추종. HTWK orientation 이식판.

    v1 의 base_roll_l2 (목표 0 고정) + base_pitch_target_l2 (목표 -2 deg 고정)
    를 하나로 합치고 목표를 커맨드로 뺀 것. v3 전용.

    ── 용도 ──
      body_pitch_target: 상체를 숙여 ZED 2i 시야를 발밑으로 내리거나,
                         킥 진입 자세를 만든다.
      body_roll_target : 측면 경사 지형에서 동체를 기울여 COM 을 옮긴다.

    작은 각도 근사 대신 회전행렬 성분에서 직접 뽑는다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w

    ex = torch.zeros(env.num_envs, 3, device=env.device)
    ex[:, 0] = 1.0
    ey = torch.zeros(env.num_envs, 3, device=env.device)
    ey[:, 1] = 1.0

    x_w = quat_apply(q, ex)
    y_w = quat_apply(q, ey)
    pitch = torch.asin(torch.clamp(-x_w[:, 2], -1.0, 1.0))
    roll = torch.asin(torch.clamp(y_w[:, 2], -1.0, 1.0))

    cmd = _raw_command(env, command_name)
    err_r = roll - cmd[:, roll_index]
    err_p = pitch - cmd[:, pitch_index]
    return torch.square(err_r) + torch.square(err_p)


# ============================================================================
# [C] 파라미터 커맨드 추종 (v3 전용)
# ============================================================================
def _velocity_gate(
    env: ManagerBasedRLEnv,
    velocity_command_name: str,
    max_speed: float,
) -> torch.Tensor:
    """명령 속도가 클수록 0 에 가까워지는 이차 게이트. HTWK vel_scale 이식판.

        gate = clamp(1 - |v_cmd| / v_max, 0, 1) ** 2

    ── 왜 필요한가 ──
    발 배치 커맨드는 "멈춰서 자세를 잡는" 용도다. 걷는 중에는 두 발이
    번갈아 앞뒤로 놓이는 것이 정상이므로 순간 발 위치를 강제하면
    보행 자체를 벌하게 된다. HTWK 는 이 게이트로 전속에서 발 배치
    페널티를 완전히 끈다 (v=v_max 에서 gate=0).

    실효 가중치 예시 (weight=-12, v_max=0.5 기준):
        v=0.0 -> -12.0
        v=0.15 -> -5.9
        v=0.25 -> -3.0
        v=0.5  ->  0.0
    """
    cmd = env.command_manager.get_command(velocity_command_name)
    speed = torch.norm(cmd[:, :2], dim=1)
    gate = torch.clamp(1.0 - speed / max_speed, min=0.0, max=1.0)
    return torch.square(gate)


def _feet_offset_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """htwk_rewards._feet_offset_body 와 동일. 순환 import 를 피해 중복 정의."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    diff_w = foot_pos_w[:, 0, :] - foot_pos_w[:, 1, :]
    heading_q = yaw_quat(asset.data.root_quat_w)
    diff_b = quat_rotate_inverse(heading_q, diff_w)
    return diff_b[:, 0], diff_b[:, 1]


def feet_offset_x_command(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_index: int,
    velocity_command_name: str = "base_velocity",
    max_speed: float = 0.5,
    clip: float = 0.1,
) -> torch.Tensor:
    """두 발의 전후 간격을 명령값으로 추종. HTWK feet_offset_x 이식판.

    스태거 스탠스(한 발을 앞으로) 를 만드는 커맨드. 킥 준비 자세와
    전후 방향 밀림 방어에 쓴다.

    L1 + clip 포화를 쓰는 이유는 htwk_rewards.feet_stance_width 와 동일하다.
    """
    fore_aft, _ = _feet_offset_body(env, asset_cfg)
    cmd = _raw_command(env, command_name)
    err = torch.clamp(torch.abs(fore_aft - cmd[:, command_index]), max=clip)
    return err * _velocity_gate(env, velocity_command_name, max_speed)


def feet_offset_y_command(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    command_index: int,
    target_distance: float,
    velocity_command_name: str = "base_velocity",
    max_speed: float = 0.5,
    clip: float = 0.1,
) -> torch.Tensor:
    """두 발의 좌우 간격을 (기준폭 + 명령 오프셋) 으로 추종.

    v2 의 feet_stance_width 를 커맨드화한 것. 명령이 0 이면 기준폭 유지로
    동일하게 동작한다.

    ── 용도 ──
      공에 접근할 때 지지발을 공 옆에 놓으려면 폭을 넓혀야 한다.
      상대와 접촉이 예상되면 폭을 넓혀 측면 안정성을 올린다.
      인조잔디 측면 경사에서도 폭 조절이 유효하다.
    """
    _, lateral = _feet_offset_body(env, asset_cfg)
    cmd = _raw_command(env, command_name)
    target = target_distance + cmd[:, command_index]
    err = torch.clamp(torch.abs(torch.abs(lateral) - target), max=clip)
    return err * _velocity_gate(env, velocity_command_name, max_speed)


def feet_yaw_command(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    symmetric_index: int,
    antisymmetric_index: int | None = None,
    symmetric_weight: float = 1.0,
    antisymmetric_weight: float = 1.6,
) -> torch.Tensor:
    """몸통 기준 발 yaw 를 명령값으로 추종.

    htwk_rewards.feet_yaw_deviation 의 커맨드 버전. 목표가 0 고정에서
    커맨드로 바뀐 것 외에는 동일한 대칭/반대칭 분해를 쓴다.

    ── 좌표 ──
    ParameterCommand 는 좌/우 대신 대칭/반대칭 모드로 명령을 준다.
        target_a = sym + anti      (첫 번째 발)
        target_b = sym - anti      (두 번째 발)
    antisymmetric_index=None 이면 anti=0, 즉 두 발이 항상 나란히 돈다.

    ── 용도 ──
      대칭 모드: 몸통 방향과 발 방향의 분리. 카메라는 공을 보고
                 발은 골대를 향하게 할 수 있다.
      반대칭 모드: 팔자 각도. 실전에서는 대개 0 으로 명령하므로
                   기본 비활성이다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    heading_q = yaw_quat(asset.data.root_quat_w)

    fwd = torch.zeros(env.num_envs, 3, device=env.device)
    fwd[:, 0] = 1.0

    yaws = []
    for i in range(foot_quat.shape[1]):
        f_w = quat_apply(foot_quat[:, i, :], fwd)
        f_b = quat_rotate_inverse(heading_q, f_w)
        yaws.append(torch.atan2(f_b[:, 1], f_b[:, 0]))

    cmd = _raw_command(env, command_name)
    sym_target = cmd[:, symmetric_index]
    if antisymmetric_index is not None:
        anti_target = cmd[:, antisymmetric_index]
    else:
        anti_target = torch.zeros_like(sym_target)

    # 실제값을 대칭/반대칭 모드로 분해한 뒤 목표와 비교
    s_actual = (yaws[0] + yaws[1]) * 0.5
    a_actual = (yaws[0] - yaws[1]) * 0.5

    e_s = s_actual - sym_target
    e_a = a_actual - anti_target
    return symmetric_weight * torch.square(e_s) + antisymmetric_weight * torch.square(e_a)


# ============================================================================
# 진단 지표
# ============================================================================
def metric_feet_slip_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """접촉 중 발의 평균 수평 속도 [m/s]. 실기 진단용."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    last = _last_buf(env, "metric_feet_pos_xy", pos)
    speed = torch.norm((pos - last) / env.step_dt, dim=-1)

    contact = (sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0).float()
    denom = torch.clamp(contact.sum(dim=-1), min=1.0)
    return (speed * contact).sum(dim=-1) / denom
