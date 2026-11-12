# /home/roboser/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.utils.math import yaw_quat
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward

def base_height_l2_deadzone(
    env: ManagerBasedRLEnv,
    target_height: float,
    deadzone: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height using L2 squared kernel with deadzone.

    target_height ± deadzone 범위 안에서는 패널티가 없고,
    그 범위를 벗어난 부분만 L2 패널티를 적용한다.
    """
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]

    base_height = asset.data.root_pos_w[:, 2]

    # deadzone 적용
    error = torch.abs(base_height - target_height) - deadzone
    error = torch.clamp(error, min=0.0)

    return torch.square(error)


def base_roll_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base roll motion.

    projected_gravity_b 기준으로 body roll 성분을 추정해서 L2 패널티를 준다.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # [N, 3]

    # upright일 때 대체로 g_b ~= [0, 0, -1]
    # roll은 y/z 성분으로 근사
    roll = torch.atan2(g_b[:, 1], -g_b[:, 2])
    return torch.square(roll)


def base_pitch_target_l2(
    env: ManagerBasedRLEnv,
    target_pitch: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation of base pitch from a desired target pitch.

    Args:
        target_pitch: desired pitch angle in radians.
            forward lean의 부호는 로봇 좌표계에 따라 + 또는 -일 수 있음.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b  # [N, 3]

    # pitch는 x 성분과 yz 평면 크기로 근사
    pitch = torch.atan2(-g_b[:, 0], torch.sqrt(g_b[:, 1] ** 2 + g_b[:, 2] ** 2))

    return torch.square(pitch - target_pitch)


def base_ang_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base yaw angular velocity (body-z axis), L2.

    Targets the kick task's "kick then 90° spin then fall" failure mode:
    a strong kick generates yaw reaction torque on the support leg, and
    nothing else in the reward set directly punishes the resulting base
    spin (ang_vel_xy_l2 only covers roll/pitch). Squared so brief
    transient rotation during the kick swing is tolerated, but sustained
    spinning is heavily penalized.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def foot_lateral_distance_min_penalty(
    env: ManagerBasedRLEnv,
    min_distance: float,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize feet getting too close in lateral direction (body-frame y distance).

    Args:
        min_distance: minimum allowed lateral distance between left/right foot [m]
        asset_cfg: must contain exactly two foot body names in order [right_foot, left_foot] or [left_foot, right_foot]
        command_name: if provided, apply only when command norm is above threshold
        command_threshold: command norm threshold
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if len(asset_cfg.body_ids) != 2:
        raise ValueError(
            f"foot_lateral_distance_min_penalty expects exactly 2 foot bodies, got {len(asset_cfg.body_ids)}"
        )

    # feet positions in world frame
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]   # [N, 2, 3]

    # translate by root position
    feet_pos_rel = feet_pos_w - asset.data.root_pos_w[:, None, :]  # [N, 2, 3]

    # rotate into robot base/body frame
    feet_pos_b = torch.zeros_like(feet_pos_rel)
    for i in range(2):
        feet_pos_b[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, feet_pos_rel[:, i, :])

    # body-frame lateral(y) distance between feet
    lateral_dist = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])  # [N]

    # penalty only if feet are too close
    penalty = torch.clamp(min_distance - lateral_dist, min=0.0)
    penalty = torch.square(penalty)

    # optionally ignore when standing command
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm > command_threshold)

    return penalty

def foot_gap_min(env: ManagerBasedRLEnv, min_gap: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """왼발 − 오른발 의 몸 yaw 좌표 y 간격이 min_gap [m] 보다 작으면 relu(min_gap − gap)² [m²]. 음수 weight.

    RS01 V0-6 4 차 (2026-10-30): 실기 옆걸음 0.4~0.5 에서 좌우 모두 발끼리 부딪힘. 시뮬 발 중심 간격 최소 ~10 cm, 발 폭 7 cm → 여유 3 cm 뿐.
    foot_lateral_distance_min_penalty 는 |간격| 이라 발이 꼬이면(왼발이 오른발 오른쪽) 오히려 넓다고 본다 → 부호 있는 간격으로 새로 만듦.
    yaw 만 뺀 좌표 (roll·pitch 무관) — 몸이 기울어도 발의 수평 간격만 본다. 게이트 없음 (서기·걷기·밀림 모두).
    asset_cfg.body_ids 순서 = [왼발, 오른발].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    p = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    rel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), p[:, 0] - p[:, 1])
    return torch.relu(min_gap - rel[:, 1]).pow(2)


def arm_swing_coupling(
    env: ManagerBasedRLEnv, coupling: float, std: float, shoulder_cfg: SceneEntityCfg, hip_cfg: SceneEntityCfg
) -> torch.Tensor:
    """팔 흔들기 — 어깨 pitch 를 같은 쪽 엉덩이 pitch 의 반대로: exp(−Σ (Δq_어깨 + coupling·Δq_엉덩이)² / std²). 양수 weight.

    RS02 V0 (2026-11-03): 사람처럼 다리가 앞으로 가면 같은 쪽 팔은 뒤로 (= 반대쪽 팔은 앞으로).
    Δq = 기본자세 대비. rs_bot_02 는 같은 쪽 hip_pitch·shoulder_pitch 축이 같다 (왼쪽 (0,−1,0), 오른쪽 (0,1,0)
    → 왼쪽 + = 앞, 오른쪽 + = 뒤) 라 좌우 모두 같은 식이고 미러 대칭이다.
    다리는 묶지 않는다 — 팔이 어떤 다리 움직임이든 따라갈 수 있어서 이 보상 때문에 걸음이 바뀔 이유가 없다.
    서 있으면 Δq_엉덩이 ≈ 0 → 팔도 기본자세. shoulder_cfg·hip_cfg 의 관절 순서 = [왼, 오른] (preserve_order).
    """
    asset: Articulation = env.scene[shoulder_cfg.name]
    q, q0 = asset.data.joint_pos, asset.data.default_joint_pos
    d_sh = q[:, shoulder_cfg.joint_ids] - q0[:, shoulder_cfg.joint_ids]
    d_hip = q[:, hip_cfg.joint_ids] - q0[:, hip_cfg.joint_ids]
    err = d_sh + coupling * d_hip
    return torch.exp(-torch.sum(err.square(), dim=1) / std**2)


def joint_usage_symmetry_sliding_window(
    env: ManagerBasedRLEnv,
    joint_pairs: list[list[str]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_size: int = 20,
    command_name: str | None = "base_velocity",
    command_threshold: float = 0.05,
    min_usage_sum: float = 1e-4,
) -> torch.Tensor:
    """Penalize asymmetric joint usage over a sliding window.

    개념:
        최근 window_size step 동안 각 좌/우 대응 관절의 |joint_vel| 누적합을 구한 뒤
        좌/우 사용량 차이를 패널티로 준다.

    장점:
        - 순간 posture 차이보다 gait-level 사용량 비대칭을 더 잘 잡음
        - sign convention(좌우 축 부호 반대) 문제 없음
        - 한쪽 무릎만 계속 쓰는 패턴 교정에 유리

    Args:
        joint_pairs:
            예:
            [
                ["r_knee_pitch_joint", "l_knee_pitch_joint"],
                ["r_ankle_pitch_joint", "l_ankle_pitch_joint"],
            ]
        window_size:
            최근 몇 step을 볼지. 현재 env step_dt=0.04s 이므로
            20이면 약 0.8초.
        command_name:
            명령이 거의 0일 때는 reward를 끄고 싶으면 사용.
        command_threshold:
            command norm이 이 값보다 클 때만 적용.
        min_usage_sum:
            좌우 둘 다 거의 안 움직이는 경우(정지 근처) divide/노이즈 방지용.

    Returns:
        [num_envs] shape penalty tensor
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")

    num_pairs = len(joint_pairs)
    if num_pairs == 0:
        return torch.zeros(env.num_envs, device=env.device)

    # ------------------------------------------------------------------
    # lazy init: env buffer 생성
    # shape: [num_envs, num_pairs, window_size]
    # ------------------------------------------------------------------
    if (
        not hasattr(env, "_joint_usage_sym_buf")
        or env._joint_usage_sym_buf.shape[0] != env.num_envs
        or env._joint_usage_sym_buf.shape[1] != num_pairs
        or env._joint_usage_sym_buf.shape[2] != window_size
    ):
        env._joint_usage_sym_buf = torch.zeros(
            env.num_envs, num_pairs, window_size, device=env.device
        )
        env._joint_usage_sym_write_idx = 0

        # joint id cache
        env._joint_usage_sym_joint_ids = []
        for right_name, left_name in joint_pairs:
            right_ids, _ = asset.find_joints(right_name)
            left_ids, _ = asset.find_joints(left_name)

            if len(right_ids) != 1 or len(left_ids) != 1:
                raise ValueError(
                    f"Each joint name must map to exactly one joint: "
                    f"{right_name} -> {right_ids}, {left_name} -> {left_ids}"
                )

            env._joint_usage_sym_joint_ids.append((right_ids[0], left_ids[0]))

    # ------------------------------------------------------------------
    # reset된 env는 해당 버퍼 전체를 초기화
    # ------------------------------------------------------------------
    reset_envs = env.episode_length_buf == 0
    if torch.any(reset_envs):
        env._joint_usage_sym_buf[reset_envs] = 0.0

    write_idx = env._joint_usage_sym_write_idx

    # 현재 slot 먼저 0으로 비움
    env._joint_usage_sym_buf[:, :, write_idx] = 0.0

    # ------------------------------------------------------------------
    # 현재 step의 |joint_vel| 기록
    # 각 pair마다:
    #   usage_diff = |dq_right| - |dq_left|
    # 를 따로 저장하는 대신,
    #   right_usage, left_usage를 둘 다 계산하기 위해
    #   현재 slot에는 abs vel 자체를 pair별 두 값으로 합산할 수 있게 사용
    # 여기서는 한 pair당 "difference 계산용 raw abs vel" 대신
    # buffer에는 right-left 각각 저장 공간이 없으므로,
    # pair별 right usage와 left usage를 즉시 window sum으로 계산하기 위해
    # 별도 임시 tensor를 사용.
    # ------------------------------------------------------------------
    current_right = torch.zeros(env.num_envs, num_pairs, device=env.device)
    current_left = torch.zeros(env.num_envs, num_pairs, device=env.device)

    for pair_idx, (right_id, left_id) in enumerate(env._joint_usage_sym_joint_ids):
        current_right[:, pair_idx] = torch.abs(asset.data.joint_vel[:, right_id])
        current_left[:, pair_idx] = torch.abs(asset.data.joint_vel[:, left_id])

    # buffer를 right-left 따로 저장하려면 차원이 하나 더 필요하므로,
    # 여기서는 기존 buffer를 [env, pair, window] 하나 더 복제해서 관리
    if (
        not hasattr(env, "_joint_usage_sym_buf_left")
        or env._joint_usage_sym_buf_left.shape[0] != env.num_envs
        or env._joint_usage_sym_buf_left.shape[1] != num_pairs
        or env._joint_usage_sym_buf_left.shape[2] != window_size
    ):
        env._joint_usage_sym_buf_left = torch.zeros(
            env.num_envs, num_pairs, window_size, device=env.device
        )

    if torch.any(reset_envs):
        env._joint_usage_sym_buf_left[reset_envs] = 0.0

    env._joint_usage_sym_buf[:, :, write_idx] = current_right
    env._joint_usage_sym_buf_left[:, :, write_idx] = current_left

    # ------------------------------------------------------------------
    # sliding window sum
    # ------------------------------------------------------------------
    right_usage_sum = torch.sum(env._joint_usage_sym_buf, dim=2)      # [N, P]
    left_usage_sum = torch.sum(env._joint_usage_sym_buf_left, dim=2)  # [N, P]

    # 정규화된 차이:
    #   (R-L)^2 / (R+L+eps)
    # 로 쓰면 pair마다 사용량 절대치가 달라도 너무 크게 튀는 걸 줄일 수 있음
    usage_total = right_usage_sum + left_usage_sum
    usage_diff_sq = torch.square(right_usage_sum - left_usage_sum)

    pair_penalty = usage_diff_sq / torch.clamp(usage_total, min=min_usage_sum)

    # pair 평균
    penalty = torch.mean(pair_penalty, dim=1)

    # command gating
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm > command_threshold)

    # 다음 write index
    env._joint_usage_sym_write_idx = (write_idx + 1) % window_size

    return penalty


def joint_target_l2(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    target_values: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    command_threshold: float = 0.0,
) -> torch.Tensor:
    """Penalize deviation of specific joints from desired target values.

    Args:
        joint_names: target을 줄 joint 이름 목록
        target_values: 각 joint의 목표 각도 [rad]
        asset_cfg: robot asset config
        command_name: 지정하면 command norm이 threshold보다 클 때만 적용
        command_threshold: gating threshold
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if len(joint_names) != len(target_values):
        raise ValueError(
            f"joint_names and target_values must have same length, "
            f"got {len(joint_names)} and {len(target_values)}"
        )

    # lazy cache
    cache_name = "_joint_target_l2_cache"
    cache_key = (tuple(joint_names), asset_cfg.name)

    if not hasattr(env, cache_name):
        setattr(env, cache_name, {})

    cache = getattr(env, cache_name)

    if cache_key not in cache:
        joint_ids = []
        for joint_name in joint_names:
            ids, _ = asset.find_joints(joint_name)
            if len(ids) != 1:
                raise ValueError(f"joint name must map to exactly one joint: {joint_name} -> {ids}")
            joint_ids.append(ids[0])

        cache[cache_key] = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

    joint_ids = cache[cache_key]
    target = torch.tensor(target_values, device=env.device, dtype=asset.data.joint_pos.dtype)

    joint_pos = asset.data.joint_pos[:, joint_ids]
    penalty = torch.sum(torch.square(joint_pos - target), dim=1)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm > command_threshold)

    return penalty

# =====================================================================
# Hold-aware reward wrappers
# =====================================================================
# reset hold 구간에서는 action=0, observation command=0 이지만
# 기존 tracking reward 함수들은 내부에서 env.command_manager.get_command()로
# 마스킹 안 된 원본 command를 읽어서 "command를 못 따라간다"고 패널티를 줌.
#
# 해결: hold 구간에서는 tracking reward를 0으로 마스킹하여
#       observation ↔ reward 간 신호 불일치를 제거한다.
# =====================================================================

def _get_hold_mask(env) -> torch.Tensor | None:
    """reset_hold_steps > 0 인 env를 찾아 bool mask를 반환.
    hold 시스템이 없으면 None을 반환."""
    if not hasattr(env, "reset_hold_steps"):
        return None
    mask = env.reset_hold_steps > 0
    if not torch.any(mask):
        return None
    return mask


def track_lin_vel_xy_hold_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """track_lin_vel_xy_yaw_frame_exp 래퍼.
    hold 구간에서는 reward=0 (서 있는 것 자체가 올바른 행동이므로 패널티 없음)."""
    try:
        from isaaclab_tasks.manager_based.locomotion.velocity.mdp import track_lin_vel_xy_yaw_frame_exp
    except ImportError:
        from isaaclab.envs.mdp.rewards import track_lin_vel_xy_yaw_frame_exp

    reward = track_lin_vel_xy_yaw_frame_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)

    hold_mask = _get_hold_mask(env)
    if hold_mask is not None:
        reward = reward.clone()
        reward[hold_mask] = 0.0

    return reward


def track_lin_vel_x_hold_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """전진(x)속도만 추종 (yaw 프레임). y(옆)속도는 자유 — wz처럼 풀어 옆공 접근 허용.
    hold 구간에서는 reward=0. (track_lin_vel_xy_hold_aware 의 x-only 버전, 2026-08-15)"""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_error = torch.square(cmd_x - vel_yaw[:, 0])
    reward = torch.exp(-lin_vel_error / std**2)

    hold_mask = _get_hold_mask(env)
    if hold_mask is not None:
        reward = reward.clone()
        reward[hold_mask] = 0.0

    return reward


def track_ang_vel_z_hold_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """track_ang_vel_z_exp 래퍼.
    hold 구간에서는 reward=0."""
    try:
        from isaaclab_tasks.manager_based.locomotion.velocity.mdp import track_ang_vel_z_exp
    except ImportError:
        from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp

    reward = track_ang_vel_z_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)

    hold_mask = _get_hold_mask(env)
    if hold_mask is not None:
        reward = reward.clone()
        reward[hold_mask] = 0.0

    return reward


def action_rate_l1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L1 action rate penalty. L2와 달리 큰 변화에도 선형으로만 커짐."""
    return torch.sum(
        torch.abs(env.action_manager.action - env.action_manager.prev_action),
        dim=-1,
    )


def max_foot_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.03,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """매 step에서 두 발 중 더 높은 발이 target_height에 못 미치면 1.0 페널티.
    
    셔플링: max foot z ≈ 0mm < target → 매 step 1.0
    정상 보행: swing foot 3cm+ → 0.0
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [N, 2]
    max_foot_z = torch.max(feet_z, dim=1).values  # [N]
    
    # target 미만이면 1.0, 이상이면 0.0
    penalty = (max_foot_z < target_height).float()  # [N]
    
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    penalty = penalty * (cmd_norm > command_threshold)
    
    return penalty

def foot_clearance_continuous(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """두 발 중 더 높은 발의 target height 미달분을 연속 패널티로 반환.
 
    ── 설계 의도 ──
    기존 foot_clearance_reward 는 velocity gating (foot_z_error × tanh(velocity))
    구조 때문에, 발을 낮게 끄는 shuffling 에서도 velocity ≈ 0 → gate ≈ 0 → reward ≈ 1.0
    으로 사실상 패널티를 주지 못했다.
 
    이 함수는 velocity 를 전혀 사용하지 않고,
    매 step 에서 "두 발 중 더 높은 발" 의 z 좌표만 보고 판단한다.
    정상 보행이라면 한쪽은 반드시 swing 중이므로, max(feet_z) 가 곧
    swing 발의 높이가 된다.
 
    ── 왜 max(feet_z) 를 쓰는가 ──
    • contact sensor 나 air_time 버퍼 없이, body_pos_w 하나로 동작
    • stance 발은 지면 높이(≈ 0) 이므로 max 에 거의 기여하지 않음
    • swing 발이 target 만큼 올라가면 shortfall = 0 → 페널티 없음
    • stance 중 양발이 모두 지면에 있으면 cmd_norm < threshold 로 gating
 
    ── 반환값 ──
    shortfall = clamp(target_height - max(feet_z), min=0)
    → target 초과 시 0.0 (패널티 없음)
    → target 미달 시 미달분에 비례한 양수 (연속 gradient)
 
    config 에서 음의 weight 를 곱해 패널티로 사용:
        weight=-3.0  →  shortfall 0.04 m 일 때 weighted = -0.12/step
 
    Args:
        env:               ManagerBasedRLEnv 인스턴스
        asset_cfg:         발 body 2 개가 포함된 SceneEntityCfg
                           (body_names = ["r_ankle_roll_link", "l_ankle_roll_link"])
        target_height:     swing 발이 도달해야 하는 최소 높이 [m]
        command_name:      속도 명령 이름 (standing 시 gating 용)
        command_threshold: 이 norm 미만이면 standing 으로 간주 → 패널티 0
    """
    # ── 발 높이 추출 ──
    # body_pos_w shape: [num_envs, num_bodies, 3]
    # asset_cfg.body_ids 에 두 발 index 가 들어 있으므로 [:, body_ids, 2] → [N, 2]
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]   # [N, 2]
 
    # ── 두 발 중 더 높은 쪽의 z 좌표 ──
    # 정상 보행: swing 발이 더 높음 → 이 값이 swing 발 높이
    # 양발 stance: 둘 다 ≈ 0 → 아래 command gating 에서 처리
    max_foot_z = torch.max(feet_z, dim=1).values                # [N]
 
    # ── target 미달분만 추출 ──
    # target 이상이면 0, 미만이면 (target - 실제높이) 의 양수 값
    shortfall = torch.clamp(target_height - max_foot_z, min=0.0)  # [N]
 
    # ── standing 시 gating ──
    # 명령 norm 이 threshold 미만이면 서 있는 상태 → 발 들 필요 없으므로 패널티 0
    cmd_norm = torch.norm(
        env.command_manager.get_command(command_name), dim=1
    )
    return shortfall * (cmd_norm > command_threshold)


def foot_flat_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.0,
) -> torch.Tensor:
    """발바닥이 좌우로 비틀리면(roll) 페널티.

    ── 설계 의도 ──
    이 항은 발바닥의 "좌우 비틀림(roll)" 만 억제한다.
    앞뒤 기울임(pitch)은 일부러 보지 않는다:
      - toe-off / heel-strike 시 발목 pitch 가 도는 것은 정상 보행이며
        추진력의 원천이다.
      - toe-off 는 발이 아직 땅에 닿아 있는 stance 후반에 일어나므로
        stance/swing 으로 분리해도 pitch 를 깔끔하게 풀어줄 수 없다.
      → pitch 는 이 보상에서 제외하고, roll 만 phase 무관하게 억제한다.

    roll 은 stance 든 swing(공중) 이든 0 에 가까워야 정상이므로
    접지 여부(sensor_cfg)를 구분할 필요가 없다.
    이 때문에 함수에서 contact sensor 분기를 제거했다.

    Args:
        asset_cfg:          발 body 2 개가 포함된 SceneEntityCfg
                            (body_names = MX_FEET)
        command_name:       속도 명령 이름 (standing gating 용)
        command_threshold:  command norm 이 이 값 미만이면 standing 으로 보고
                            패널티 0. 0.0 이면 항상 적용.

    Returns:
        [num_envs] shape 의 penalty tensor (양수, config 에서 음의 weight 를 곱함)
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # 각 발의 world-frame quaternion [N, 2, 4]
    foot_quats = asset.data.body_quat_w[:, asset_cfg.body_ids, :]

    penalty = torch.zeros(env.num_envs, device=env.device)

    for i in range(len(asset_cfg.body_ids)):
        # 월드 기준 "아래" 방향 단위벡터 (-z)
        world_down = torch.zeros(env.num_envs, 3, device=env.device)
        world_down[:, 2] = -1.0

        # 중력(아래) 벡터를 발 좌표계로 변환
        # → 발이 어떻게 기울었는지 발 기준에서 본 값
        gravity_in_foot = quat_apply_inverse(foot_quats[:, i, :], world_down)

        # roll 만 추정: 발 좌표계 y/z 성분 (좌우 비틀림)
        # pitch(발 좌표계 x 성분) 는 의도적으로 계산하지 않음
        roll = torch.atan2(gravity_in_foot[:, 1], -gravity_in_foot[:, 2])

        # roll 제곱을 패널티로 누적
        # square 라서 좌우 발 좌표계 부호가 반대여도 상쇄되어 문제 없음
        penalty += torch.square(roll)

    # command gating — 거의 정지 상태면 패널티 0
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm > command_threshold)

    return penalty



def pelvis_pitch_swing_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_deviation: float = 0.15,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """걸을 때 두 pelvis pitch 중 하나라도 크게 벗어나야 함.
    
    양쪽 다 default 근처면 → 발을 안 들고 있다 → 페널티
    한쪽이라도 크게 벗어나면 → 정상 swing → 페널티 없음
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]     # [N, 2]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # [N, 2]
    
    deviation = torch.abs(joint_pos - default)  # [N, 2]
    max_deviation = torch.max(deviation, dim=1).values  # [N]
    
    # max deviation이 threshold 미만이면 1.0, 이상이면 0.0
    penalty = (max_deviation < min_deviation).float()
    
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    penalty = penalty * (cmd_norm > command_threshold)
    
    return penalty

def randomize_actuator_stiffness(
    env,
    env_ids: torch.Tensor,
    stiffness_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """startup 시 env별로 actuator stiffness를 랜덤화."""
    asset = env.scene[asset_cfg.name]
    
    stiffnesses = asset.root_physx_view.get_dof_stiffnesses().clone()
    
    rand_val = torch.empty(env.num_envs, 1, device=env.device).uniform_(
        stiffness_range[0], stiffness_range[1]
    )
    stiffnesses[:] = rand_val.expand_as(stiffnesses)
    
    # PhysX는 CPU index를 요구
    indices = torch.arange(env.num_envs, device="cpu", dtype=torch.int32)
    asset.root_physx_view.set_dof_stiffnesses(stiffnesses.cpu(), indices)


def pelvis_pitch_symmetry(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """좌우 pelvis pitch가 default 기준으로 mirror symmetry가 되도록 유도."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    q_default = asset.data.default_joint_pos

    joint_names = list(asset.data.joint_names)
    r_idx = joint_names.index("r_pelvis_pitch_joint")
    l_idx = joint_names.index("l_pelvis_pitch_joint")

    r_dev = q[:, r_idx] - q_default[:, r_idx]
    l_dev = q[:, l_idx] - q_default[:, l_idx]

    return torch.abs(r_dev + l_dev)


def pelvis_pitch_symmetry_windowed(
    env,
    window_size: int = 30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = "base_velocity",
    command_threshold: float = 0.1,
    joint_names: tuple[str, str] = ("r_pelvis_pitch_joint", "l_pelvis_pitch_joint"),
):
    """좌우 pelvis pitch의 전후 위치 편향(fore-aft asymmetry)을 '창 평균'으로 penalize.

    joint_names: (오른쪽, 왼쪽) 엉덩이 pitch 관절. 기본값은 MX3. 좌우 관절 부호가 거울(대칭 자세에서
        r_dev = -l_dev)인 로봇이어야 의미가 맞다 (rs_bot 도 해당).

    기존 pelvis_pitch_symmetry = |r_dev + l_dev| 는 매 순간 abs 라서, anti-phase 보행에서
    (한쪽 앞·한쪽 뒤) z=r_dev+l_dev 가 크게 진동 → 대칭 gait인데도 페널티를 먹는 버그가 있었다.

    핵심 개선: abs 를 '창 평균 낸 뒤'에 취한다.
        z(t) = (q_r - q_r_default) + (q_l - q_l_default)
        - 대칭 anti-phase gait  : 창 안에서 진동 상쇄 → mean(z) ≈ 0      → 페널티 ~0
        - 한쪽으로 치우친 gait   : DC 편향 b 가 남음   → mean(z) ≈ 2b     → 페널티 발생

    window_size:
        gait 한 주기 이상을 덮어야 진동이 상쇄됨. step_dt≈0.02s, 주기≈0.5s 이면 25~30 권장.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")

    # lazy init: 링버퍼 [N, window_size] + 관절 인덱스 캐시
    if (
        not hasattr(env, "_pps_win_buf")
        or env._pps_win_buf.shape[0] != env.num_envs
        or env._pps_win_buf.shape[1] != window_size
    ):
        env._pps_win_buf = torch.zeros(env.num_envs, window_size, device=env.device)
        env._pps_win_write_idx = 0
        jn = list(asset.data.joint_names)
        env._pps_r_idx = jn.index(joint_names[0])
        env._pps_l_idx = jn.index(joint_names[1])

    # reset된 env 버퍼 초기화
    reset_envs = env.episode_length_buf == 0
    if torch.any(reset_envs):
        env._pps_win_buf[reset_envs] = 0.0

    # 현재 step z = r_dev + l_dev 기록
    q = asset.data.joint_pos
    q_def = asset.data.default_joint_pos
    r_dev = q[:, env._pps_r_idx] - q_def[:, env._pps_r_idx]
    l_dev = q[:, env._pps_l_idx] - q_def[:, env._pps_l_idx]
    z = r_dev + l_dev  # [N]

    write_idx = env._pps_win_write_idx
    env._pps_win_buf[:, write_idx] = z

    # 창 평균 후 abs (초기 fill-up 구간은 zero-padding 으로 약하게 시작)
    penalty = torch.abs(torch.mean(env._pps_win_buf, dim=1))  # [N]

    # command gating: 서 있을 땐(명령~0) 대칭 의미 적으니 끔
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm > command_threshold)

    env._pps_win_write_idx = (write_idx + 1) % window_size
    return penalty


# =====================================================================
# v2 rewards / metrics — 발 들기·골반 활용·횡회전 개선용
# =====================================================================
#
# 기존 foot_clearance_reward 는 velocity gating 으로 인해 학습 신호가
# 거의 0 인 채로 max(1.0) 근처에 포화되는 것이 73000 시점 TB에서 정량 확인됨
# (Episode_Reward/feet_clearance ≈ 0.988). 이를 대체/보완하기 위한 항들.
#
# 모든 진단(metric) 함수는 학습에도 약한 양/음의 신호를 같이 주도록 설계되어
# 있다. config 에서 weight 를 통해 강도 조절. weight 0 로 두면 신호는 사라지지만
# TB Episode_Reward/* 에서 평균값을 그대로 추적할 수 있다.
# =====================================================================


def _cmd_axis_ratios(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """현재 명령 벡터 [vx, vy, wz] 에서 (vx_ratio, lateral_yaw_ratio) 계산.

    vy/wz 가 dominant 할수록 lateral_yaw_ratio 가 1 에 가까워짐.
    wz 는 rad/s 라서 0.5 로 스케일해 m/s 와 같은 차원으로 맞춤(체격 기준 손가락 어림셈).
    """
    cmd = env.command_manager.get_command(command_name)
    eps = 1e-6
    vx_part = torch.abs(cmd[:, 0])
    vy_part = torch.abs(cmd[:, 1])
    wz_part = torch.abs(cmd[:, 2]) * 0.5
    total = vx_part + vy_part + wz_part + eps
    vx_ratio = vx_part / total
    lateral_yaw_ratio = (vy_part + wz_part) / total
    return vx_ratio, lateral_yaw_ratio


def _swing_mask_from_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """ContactSensor 로부터 [N, num_feet] swing(접지 안한) 마스크 반환.

    force_threshold (N) 보다 net force 가 작은 발만 True (swing).
    standing 시 발당 추정 GRF 25~50N 정도이므로 5N 은 안전한 임계값.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1
    )  # [N, num_feet]
    is_swing = forces < force_threshold  # contact 거의 없는 발 = swing
    return is_swing


def foot_clearance_cmd_axis(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.05,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    lateral_yaw_multiplier: float = 2.0,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Contact-aware foot clearance penalty.

    v2.1 (까치발 hack fix):
        이전 버전은 max(feet_z) 만 봤기 때문에 한 발 stance + 다른 발 toe-down
        (까치발) 자세로 ankle_roll_link origin 만 위로 올리는 reward hack 이
        가능했다. 이번 버전은 contact sensor 로 "실제 swing 중인 발" 만
        후보로 두고, 양 발 모두 접지 중이면 패널티 신호를 끈다.

    구조:
        is_swing = (contact_force < threshold)          # [N, num_feet]
        swing_z  = where(is_swing, feet_z, -inf)
        max_swing_z = max(swing_z, dim=feet)
        shortfall = clamp(target - max_swing_z, min=0) if any swing else 0
        penalty   = shortfall * multiplier * cmd_gate

    의도:
        - 두 발 다 stance → shortfall=0 → 까치발 hack 무력화
        - 한 발이라도 진짜로 떠있어야만 신호 발생
        - ankle origin 의 z 가 아닌 "swing 발의 origin z" 만 보므로
          정상 보행에서는 기존 의도와 동일하게 동작
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]                # [N, num_feet]
    is_swing = _swing_mask_from_contact(env, sensor_cfg, contact_force_threshold)

    # stance 발은 매우 작은 값으로 마스킹 (max 후보에서 사실상 제외)
    neg_inf = torch.full_like(feet_z, -1.0)
    swing_z = torch.where(is_swing, feet_z, neg_inf)
    max_swing_z = torch.max(swing_z, dim=1).values                          # [N]

    any_swing = torch.any(is_swing, dim=1)                                  # [N]
    shortfall = torch.clamp(target_height - max_swing_z, min=0.0)
    shortfall = shortfall * any_swing.float()   # 두 발 다 stance 면 0

    _, lateral_yaw_ratio = _cmd_axis_ratios(env, command_name)
    multiplier = 1.0 + (lateral_yaw_multiplier - 1.0) * lateral_yaw_ratio

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return shortfall * multiplier * (cmd_norm > command_threshold)


def swing_foot_height_metric(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Contact-aware swing height metric (v2.1).

    이전 버전 max(feet_z) 는 stance 발의 ankle origin 도 함께 카운트해서
    까치발 hack 시 큰 값을 반환했음. 이번 버전은 contact 가 거의 없는 발만
    카운트하여 metric 이 실제 "들린 발 높이" 와 일치한다.

    두 발 다 stance 면 0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    is_swing = _swing_mask_from_contact(env, sensor_cfg, contact_force_threshold)

    # swing 발만 카운트, stance 발은 0
    swing_z_masked = feet_z * is_swing.float()
    max_swing_z = torch.max(swing_z_masked, dim=1).values

    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return max_swing_z * (cmd_norm > command_threshold)


def swing_foot_height_lateral_yaw_metric(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Contact-aware swing height metric (v2.1) — lateral|yaw 명령 dominant 시 분리.

    까치발 hack 무력화 + lateral|yaw 명령 모드일 때의 평균 swing height 추적.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    is_swing = _swing_mask_from_contact(env, sensor_cfg, contact_force_threshold)

    swing_z_masked = feet_z * is_swing.float()
    max_swing_z = torch.max(swing_z_masked, dim=1).values

    _, lateral_yaw_ratio = _cmd_axis_ratios(env, command_name)
    mask = lateral_yaw_ratio > 0.5
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return max_swing_z * mask.float() * (cmd_norm > command_threshold)


def stance_foot_flat_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.0,
    contact_force_threshold: float = 5.0,
    swing_weight: float = 0.0,
) -> torch.Tensor:
    """발의 pitch (앞뒤 기울임) 패널티 — 까치발/발목 꺾임 억제.

    설계:
        - foot_flat 은 roll 만 보고 pitch 는 풀어둠 (정상 toe-off 허용 위해).
        - 까치발 hack 은 stance 중 발이 toe-down 자세로 회전 → ankle origin 만 위로.
        - 이 항은 stance 발 pitch² 를 full weight 로 패널티.
        - swing_weight > 0 이면 swing 발 pitch² 도 그 비율로 약하게 패널티
          → 발 드는 중 발목 꺾여 잔디에 걸리는 것을 억제 (절충: toe-off 일부 보존).

    구현:
        per-foot weight = stance 발 1.0, swing 발 swing_weight.
        penalty = Σ pitch² · per_foot_weight.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_quats = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    is_swing = _swing_mask_from_contact(env, sensor_cfg, contact_force_threshold)

    penalty = torch.zeros(env.num_envs, device=env.device)
    world_down = torch.zeros(env.num_envs, 3, device=env.device)
    world_down[:, 2] = -1.0

    for i in range(len(asset_cfg.body_ids)):
        # 발 좌표계로 변환된 down vector
        gravity_in_foot = quat_apply_inverse(foot_quats[:, i, :], world_down)
        # pitch ≈ atan2(-g_x, sqrt(g_y² + g_z²)). 정상 stance 면 ≈ 0
        pitch = torch.atan2(
            -gravity_in_foot[:, 0],
            torch.sqrt(gravity_in_foot[:, 1] ** 2 + gravity_in_foot[:, 2] ** 2),
        )
        # stance 발: weight 1.0, swing 발: weight swing_weight
        is_swing_i = is_swing[:, i].float()
        per_foot_weight = (1.0 - is_swing_i) + is_swing_i * swing_weight
        penalty = penalty + torch.square(pitch) * per_foot_weight

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        penalty = penalty * (cmd_norm > command_threshold)

    return penalty


def hip_pitch_swing_amplitude(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """진단/약 incentive: 양쪽 hip(pelvis) pitch 의 default 대비 |편차| 합.

    "골반을 쭉쭉 쓰는 정도" 의 직접 지표. 보행 중 swing 다리의 hip flex 가
    클수록 이 값이 증가. weight 양수로 등록하면 step-length 증대 incentive.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(q - q_default), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return deviation * (cmd_norm > command_threshold)


def joint_deviation_l1_cmd_axis_aware(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    lateral_yaw_relax: float = 0.6,
) -> torch.Tensor:
    """기존 joint_deviation_l1 의 cmd-axis-aware 버전.

    lateral_yaw_ratio 가 클수록 패널티를 lateral_yaw_relax 비율만큼 감쇠.
    예: lateral_yaw_relax=0.6 → 횡/회전 dominant 시 최대 60% 감쇠.

    의도:
        joint_deviation_pelvis_roll/yaw 패널티는 전진 보행에서는 유효하지만
        옆걸음/제자리 회전 때는 골반 roll/yaw 사용이 필요하므로
        명령 축에 따라 자동으로 풀어준다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.sum(torch.abs(q - q_default), dim=1)

    _, lateral_yaw_ratio = _cmd_axis_ratios(env, command_name)
    multiplier = 1.0 - lateral_yaw_relax * lateral_yaw_ratio
    return deviation * multiplier


def feet_z_baseline_metric(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """진단 전용: 명령 ≈ 0 (standing) 일 때 max(feet_z).

    이게 곧 ankle_roll_link standing baseline 추정치다.
    target_height 튜닝의 기준값을 학습 중 자동으로 얻기 위함.
    weight 0 으로 등록해도 TB Episode_Reward 에 평균 sum 으로 기록됨.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    max_foot_z = torch.max(feet_z, dim=1).values
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    # standing 인 env 만 카운트
    return max_foot_z * (cmd_norm < command_threshold).float()