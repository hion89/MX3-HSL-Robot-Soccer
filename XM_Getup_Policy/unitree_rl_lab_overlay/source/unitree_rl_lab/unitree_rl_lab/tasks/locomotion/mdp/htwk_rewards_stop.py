# htwk_rewards_stop.py
# ============================================================================
# StoppableGaitPhaseCommand 에 대응하는 위상 기반 리워드.
#
# htwk_rewards.py 의 feet_gait_phase_aware / feet_swing_phase 는
# "effective_frequency > 0" 하나로만 게이팅한다. 정지 과도 구간
# (주파수 0, 위상은 0 으로 수렴 중) 이라는 상태가 없었기 때문이다.
#
# 이 모듈은 세 상태를 구분해 각각 다르게 채점한다.
#
#   is_walking   : 주파수 > 0. 보행 또는 제자리 걷기
#   is_stopping  : 주파수 0, 위상이 0 으로 수렴 중 (과도 구간)
#   full_stop    : 주파수 0, 위상 0. 완전 정지
#
# ── 왜 구분이 필요한가 ──
#
# feet_gait (stance/swing 일치):
#   과도 구간에도 켜야 한다. 위상이 0 으로 가면 offset (0.0, 0.5) /
#   threshold 0.55 기준 양발 모두 stance 구간이 되므로, 이 항이 켜져
#   있으면 "두 발을 모두 접지시켜라" 가 되어 정지 자세를 유도한다.
#   이게 위상 수렴 설계의 핵심이다.
#
# feet_swing (스윙 구간 발 들기):
#   과도 구간에는 꺼야 한다. 위상 0.3 에서 역방향으로 0 을 향해 갈 때
#   스윙 창(0.25 ± 0.1)을 지나가는데, 이때 켜져 있으면 "멈추는 중인데
#   발을 들어라" 가 되어 정지를 방해한다.
# ============================================================================

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gait_term(env: ManagerBasedRLEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def feet_gait_stop_aware(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_command_name: str = "gait",
    offset: tuple[float, ...] = (0.0, 0.5),
    threshold: float = 0.55,
) -> torch.Tensor:
    """stance/swing 일치 보상. 보행 + 정지 과도 구간 모두에서 작동.

    위상 0 근처에서는 두 다리 모두 stance 로 판정되므로
    (leg0 phase = 0 < 0.55, leg1 phase = 0.5 < 0.55),
    정지 과도 구간에 이 항이 켜져 있으면 자연스럽게 양발 접지를 유도한다.

    완전 정지(위상 0 도달) 후에는 꺼진다. 그 상태의 접지 유지는
    feet_contact_without_cmd 가 담당한다.
    """
    term = _gait_term(env, gait_command_name)
    phase = term.phase
    active = term.phase_active.float()

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        leg_phase = torch.fmod(phase + offset[i], 1.0)
        is_stance = leg_phase < threshold
        reward += (~(is_stance ^ is_contact[:, i])).float()
    return reward * active


def feet_swing_walking_only(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_command_name: str = "gait",
    swing_period: float = 0.2,
) -> torch.Tensor:
    """스윙 구간에 발이 떠 있으면 보상. 보행 중에만 작동.

    정지 과도 구간에서는 꺼진다. 멈추는 중에 스윙 창을 지나가더라도
    발을 들라고 요구하면 정지가 지연되기 때문이다.
    """
    term = _gait_term(env, gait_command_name)
    phase = term.phase
    active = term.is_walking

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    half = 0.5 * swing_period
    reward = torch.zeros(env.num_envs, device=env.device)
    centers = [0.25, 0.75]
    for i in range(min(len(sensor_cfg.body_ids), len(centers))):
        swing = (torch.abs(phase - centers[i]) < half) & active
        reward += (swing & ~is_contact[:, i]).float()
    return reward


def stop_posture(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """완전 정지 상태에서 양발이 접지해 있으면 보상.

    feet_contact_without_cmd 와 유사하지만 판정 기준이 다르다.
      feet_contact_without_cmd : 속도 명령이 0 인가
      stop_posture             : 주파수 명령이 0 이고 위상도 0 인가

    제자리 걷기(속도 0, 주파수 > 0) 상태에서는 발을 들어야 하므로
    속도 기준 판정만으로는 잘못된 신호가 된다. 이 함수는 주파수 기준이라
    제자리 걷기를 벌하지 않는다.

    ※ 제자리 걷기를 쓸 경우 feet_contact_without_cmd 는 반드시 꺼야 한다.
      (두 항이 정면으로 충돌한다)
    """
    term = _gait_term(env, gait_command_name)
    full_stop = (~term.is_walking) & (~term.is_stopping)

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0

    both_contact = torch.all(is_contact, dim=1)
    return (both_contact & full_stop).float()


def stop_transition_speed(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """완전 정지 상태에서 동체가 움직이면 페널티.

    "멈추라고 했는데 계속 흐르는" 현상을 직접 억제한다.
    정지 과도 구간에는 적용하지 않는다 (감속 중이므로 속도가 남아있는 것이
    정상이다).
    """
    from isaaclab.assets import Articulation

    term = _gait_term(env, gait_command_name)
    full_stop = (~term.is_walking) & (~term.is_stopping)

    asset: Articulation = env.scene[asset_cfg.name]
    speed = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)
    return speed * full_stop.float()


def stop_default_posture(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    gait_command_name: str = "gait",
    include_stopping: bool = True,
) -> torch.Tensor:
    """정지(+과도) 구간에서 다리 관절이 기본자세에서 벗어나면 페널티.

    ── 왜 필요한가 ──
    stop_posture 는 "양발 접지" 만 본다. 그런데 한 발 앞/한 발 뒤로 벌어진
    staggered 자세도 양발 접지를 만족하므로, 정지 명령이 스윙 중에 들어오면
    그 mid-stride 자세 그대로 얼어붙는다(프리징). 이 항은 관절을
    default_joint_pos(발 모음 중립 자세)로 직접 당겨 정지 시 기본 자세로
    복귀시킨다.

    include_stopping=True 면 정지 과도(is_stopping) 구간에도 적용해, 나간 발을
    회수하는 동안 목표를 준다. stop_transition_speed(정지 중 이동 억제)와
    균형이 맞아야 하므로 weight 는 작게 시작할 것.

    반환은 penalty(양수) → env_cfg 에서 음수 weight.
    """
    from isaaclab.assets import Articulation

    term = _gait_term(env, gait_command_name)
    # ~is_walking = full_stop | is_stopping (둘 다 주파수 0)
    active = ~term.is_walking
    if not include_stopping:
        active = active & (~term.is_stopping)

    asset: Articulation = env.scene[asset_cfg.name]
    dev = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(dev), dim=1) * active.float()


# ============================================================================
# 진단 지표 (weight=0)
# ============================================================================
def metric_phase(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """현재 위상 (0~1). 정지 시 0 으로 수렴하는지 확인용."""
    return _gait_term(env, gait_command_name).phase


def metric_is_stopping(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """정지 과도 구간 비율. 너무 크면 stop_rate 를 올릴 것."""
    return _gait_term(env, gait_command_name).is_stopping.float()


def metric_is_marching(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
    velocity_command_name: str = "base_velocity",
    standing_threshold: float = 0.1,
) -> torch.Tensor:
    """제자리 걷기 비율 (속도 0, 주파수 > 0)."""
    term = _gait_term(env, gait_command_name)
    cmd = env.command_manager.get_command(velocity_command_name)
    standing = torch.norm(cmd[:, :3], dim=1) < standing_threshold
    return (standing & term.is_walking).float()
