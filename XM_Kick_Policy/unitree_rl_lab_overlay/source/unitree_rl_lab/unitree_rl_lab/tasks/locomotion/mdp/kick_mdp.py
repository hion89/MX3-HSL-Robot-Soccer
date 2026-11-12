"""Ball-kick MDP terms: observations, rewards, and target sampling events.

Lives in its own module so the kick task can be tuned independently of
the velocity-tracking locomotion task.

Per-env state attached to `env`:
    env.kick_target_pos_w : (N, 3)  world-frame target the ball must reach
    env.kick_target_dir_w : (N, 3)  unit vector from current ball pos → target (z=0)
"""

from __future__ import annotations

import math
import torch
from .kick_clock import kick_clock_s, kick_clock_steps  # 통합 태스크의 킥 시작 기준 시각 (기존 태스크는 에피소드 시각 그대로)
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _ensure_target_buffers(env: ManagerBasedRLEnv) -> None:
    """Lazily create per-env target buffers on first access."""
    if not hasattr(env, "kick_target_pos_w"):
        env.kick_target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "kick_target_dir_w"):
        env.kick_target_dir_w = torch.zeros(env.num_envs, 3, device=env.device)
        env.kick_target_dir_w[:, 0] = 1.0  # default: +x


def _robot_forward_w(robot: Articulation) -> torch.Tensor:
    """Robot's body +x axis expressed in world frame, flattened to the xy plane
    and re-normalised. Shape (N, 3) with z = 0.

    "Kick forward" mode: the target direction is always the robot's current
    heading, so target_dir_b is a constant (1, 0, 0).
    """
    n = robot.data.root_quat_w.shape[0]
    fwd_b = torch.zeros(n, 3, device=robot.data.root_quat_w.device)
    fwd_b[:, 0] = 1.0
    fwd_w = quat_apply(robot.data.root_quat_w, fwd_b)
    fwd_w[:, 2] = 0.0
    return fwd_w / torch.norm(fwd_w, dim=-1, keepdim=True).clamp(min=1e-6)


# ----------------------------------------------------------------------
# Events  (reset-mode)
# ----------------------------------------------------------------------
def reset_ball_around_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    radius_range: tuple[float, float] = (0.4, 1.2),
    front_angle_range: tuple[float, float] = (-torch.pi, torch.pi),
    height: float = 0.065,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Spawn the ball at a random angle/distance around the robot's reset position.

    Args:
        radius_range: ball spawn distance from robot (xy plane), uniform.
        front_angle_range: angle relative to robot's CURRENT front direction,
            radians. Default (-π, π) spawns 360° around. Use e.g. (-π/3, π/3)
            to restrict to a ±60° cone in front of the robot. This is sampled
            AFTER the robot reset event, so the robot's actual yaw at reset is
            honoured (i.e. "front" tracks the randomized robot yaw).
        height: ball z position (typically the ball radius).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    n = len(env_ids)
    device = env.device

    # Polar sample, with theta relative to robot's current yaw
    r = torch.empty(n, device=device).uniform_(radius_range[0], radius_range[1])
    rel_theta = torch.empty(n, device=device).uniform_(
        front_angle_range[0], front_angle_range[1]
    )
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
    theta = robot_yaw + rel_theta  # world-frame angle

    robot_xy = robot.data.root_pos_w[env_ids, :2]
    ball_x = robot_xy[:, 0] + r * torch.cos(theta)
    ball_y = robot_xy[:, 1] + r * torch.sin(theta)
    ball_z = torch.full((n,), height, device=device)

    pos = torch.stack([ball_x, ball_y, ball_z], dim=-1)
    quat = torch.zeros(n, 4, device=device)
    quat[:, 0] = 1.0
    vel = torch.zeros(n, 6, device=device)

    ball.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)


def reset_ball_xy_in_robot_frame(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float] = (0.4, 1.2),
    y_range: tuple[float, float] = (-0.1, 0.1),
    height: float = 0.065,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Spawn the ball in a rectangular xy region in the robot's body frame.

    Unlike `reset_ball_around_robot` (polar), this samples (x_b, y_b) uniformly
    in a rectangle relative to the robot's CURRENT body frame at reset, then
    rotates by the robot's yaw to get the world-frame ball position.

    Args:
        x_range: forward (body x) range in meters. (0.4, 1.2) = 40~120 cm ahead.
        y_range: lateral (body y) range in meters. (-0.1, 0.1) = ±10 cm sideways.
        height: ball z position (typically the ball radius).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    n = len(env_ids)
    device = env.device

    x_b = torch.empty(n, device=device).uniform_(x_range[0], x_range[1])
    y_b = torch.empty(n, device=device).uniform_(y_range[0], y_range[1])

    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
    cos_y = torch.cos(robot_yaw)
    sin_y = torch.sin(robot_yaw)

    robot_xy = robot.data.root_pos_w[env_ids, :2]
    ball_x = robot_xy[:, 0] + cos_y * x_b - sin_y * y_b
    ball_y = robot_xy[:, 1] + sin_y * x_b + cos_y * y_b
    ball_z = torch.full((n,), height, device=device)

    pos = torch.stack([ball_x, ball_y, ball_z], dim=-1)
    quat = torch.zeros(n, 4, device=device)
    quat[:, 0] = 1.0
    vel = torch.zeros(n, 6, device=device)

    ball.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)


def reset_kick_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    distance_range: tuple[float, float] = (2.0, 4.0),
    target_angle_range: tuple[float, float] = (-0.5236, 0.5236),  # ±30°
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Fix an ABSOLUTE world kick direction at reset, offset ±range from the
    robot's reset facing.

    The target direction is `robot_reset_yaw + uniform(target_angle_range)`,
    frozen in the WORLD frame for the whole episode. The observation
    (`target_dir_in_robot_frame`) re-expresses it in the robot's CURRENT body
    frame, so it stays world-fixed as the robot moves. The target POSITION
    (for `ball_to_target_closeness` shaping) is the ball spawn projected along
    this fixed direction.
    """
    _ensure_target_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    device = env.device
    n = len(env_ids)

    r = torch.empty(n, device=device).uniform_(distance_range[0], distance_range[1])
    offset = torch.empty(n, device=device).uniform_(
        target_angle_range[0], target_angle_range[1]
    )
    _, _, robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
    target_yaw = robot_yaw + offset  # world-frame absolute kick direction

    dir_w = torch.zeros(n, 3, device=device)
    dir_w[:, 0] = torch.cos(target_yaw)
    dir_w[:, 1] = torch.sin(target_yaw)
    env.kick_target_dir_w[env_ids] = dir_w

    ball_xy = ball.data.root_pos_w[env_ids, :2]
    target_xy = ball_xy + r.unsqueeze(-1) * dir_w[:, :2]
    tz = torch.zeros(n, device=device)
    env.kick_target_pos_w[env_ids] = torch.cat([target_xy, tz.unsqueeze(-1)], dim=-1)


# ----------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------
def ball_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball position expressed in the robot torso frame (x_fwd, y_left, z_up).

    LIVE value — updates every step. Used by the critic (privileged obs).
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rel_w = ball.data.root_pos_w - robot.data.root_pos_w  # (N, 3)
    return quat_apply_inverse(robot.data.root_quat_w, rel_w)


def ball_pos_in_robot_frame_frozen(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball position in the robot frame, FROZEN at the reset value.

    Returns the spawn-time reading captured in `reset_kick_state`. As the robot
    walks toward the ball the value does NOT change — it models a one-shot ball
    detection (detect once, then kick open-loop). Used by the policy obs.
    """
    _ensure_kick_state(env)
    return env.ball_pos_b_frozen


def reset_foot_z_window(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_size: int = 50,
) -> None:
    """Reset the per-foot rolling-window buffer for `min_window_apex_clearance`."""
    n_feet = len(asset_cfg.body_ids)
    if not hasattr(env, "foot_z_window"):
        env.foot_z_window = torch.zeros(env.num_envs, n_feet, window_size, device=env.device)
    env.foot_z_window[env_ids] = 0.0


def min_window_apex_clearance(
    env: ManagerBasedRLEnv,
    target_height: float = 0.08,
    window_size: int = 50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward = min over feet of (max foot z over the last `window_size` steps).

    Per env, maintains a ring buffer of each foot's world-frame z value for the
    last `window_size` reward calls (≈ 1 s at step_dt=0.02s with default 50).
    Each step the buffer entry at the current ring index is overwritten with
    the current foot z; the per-foot max over the window is the "recent apex".
    The reward takes the MIN across feet so both legs must have lifted recently
    or the lower one drags reward to ~0.

    Foils the "single early lift then stand" hack of the previous swing-apex
    reward: old apex values age out of the window and the foot must keep
    lifting to keep the apex high. Returns shape (N,), clipped to [0, 1].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    n_feet = len(asset_cfg.body_ids)

    if not hasattr(env, "foot_z_window"):
        env.foot_z_window = torch.zeros(env.num_envs, n_feet, window_size, device=env.device)
    if not hasattr(env, "_foot_z_window_step"):
        env._foot_z_window_step = 0

    idx = env._foot_z_window_step % window_size
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    env.foot_z_window[:, :, idx] = foot_z
    env._foot_z_window_step += 1

    max_per_foot = env.foot_z_window.max(dim=-1)[0]
    min_apex = max_per_foot.min(dim=-1)[0]
    return torch.clamp(min_apex / target_height, max=1.0)


def reset_foot_swing_apex(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the per-foot swing-apex tracker buffers."""
    n_feet = len(asset_cfg.body_ids)
    if not hasattr(env, "foot_swing_cur_max"):
        env.foot_swing_cur_max = torch.zeros(env.num_envs, n_feet, device=env.device)
        env.foot_swing_last_apex = torch.zeros(env.num_envs, n_feet, device=env.device)
        env.foot_was_swing = torch.zeros(env.num_envs, n_feet, dtype=torch.bool, device=env.device)
    env.foot_swing_cur_max[env_ids] = 0.0
    env.foot_swing_last_apex[env_ids] = 0.0
    env.foot_was_swing[env_ids] = False


def min_swing_apex_clearance(
    env: ManagerBasedRLEnv,
    target_height: float = 0.07,
    contact_force_threshold: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward when BOTH feet's last-swing apex heights exceed `target_height`.

    Per foot, tracks the max world-z reached during the current swing phase
    (contact force < threshold). On swing→contact transition the max is
    "frozen" into a per-foot last-apex buffer. Reward = clip(min(L_apex,
    R_apex) / target_height, 0, 1) — the LOWER of the two feet's apexes drags
    the reward down, so both legs must lift properly.

    Foils the per-step `foot_clearance_reward` hack where a static foot
    (xy_vel≈0) gives full reward regardless of height: here a foot that never
    swings has apex = 0, so the min is 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact: ContactSensor = env.scene.sensors[sensor_cfg.name]

    n_feet = len(asset_cfg.body_ids)
    if not hasattr(env, "foot_swing_cur_max"):
        env.foot_swing_cur_max = torch.zeros(env.num_envs, n_feet, device=env.device)
        env.foot_swing_last_apex = torch.zeros(env.num_envs, n_feet, device=env.device)
        env.foot_was_swing = torch.zeros(env.num_envs, n_feet, dtype=torch.bool, device=env.device)

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    forces = contact.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    is_contact = forces > contact_force_threshold
    is_swing = ~is_contact

    # Update per-foot running max during swing
    env.foot_swing_cur_max = torch.where(
        is_swing,
        torch.maximum(env.foot_swing_cur_max, foot_z),
        env.foot_swing_cur_max,
    )

    # On swing→contact (landing), freeze the apex and reset the running max
    landed = env.foot_was_swing & is_contact
    env.foot_swing_last_apex = torch.where(landed, env.foot_swing_cur_max, env.foot_swing_last_apex)
    env.foot_swing_cur_max = torch.where(
        landed,
        torch.zeros_like(env.foot_swing_cur_max),
        env.foot_swing_cur_max,
    )

    env.foot_was_swing = is_swing.clone()

    min_apex = env.foot_swing_last_apex.min(dim=-1)[0]
    return torch.clamp(min_apex / target_height, max=1.0)


def reset_walk_phantom(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the phantom walking target to the robot's current position.

    The phantom target represents where the robot SHOULD be if it walks at the
    commanded velocity. Each step, the phantom advances by `cmd_vx * dt` in the
    direction of the robot's RESET heading. If the robot stands still, the
    phantom drifts away — used by `phantom_target_distance_l2` for an unbounded
    position-deficit penalty (foils the "stand and farm bounded rewards" hack).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "walk_phantom_pos_w"):
        env.walk_phantom_pos_w = torch.zeros(env.num_envs, 2, device=env.device)
        env.walk_phantom_yaw_w = torch.zeros(env.num_envs, device=env.device)
    env.walk_phantom_pos_w[env_ids] = robot.data.root_pos_w[env_ids, :2]
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
    env.walk_phantom_yaw_w[env_ids] = yaw


def phantom_target_distance_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 distance² between robot and the cmd-driven phantom target.

    Advances the phantom each step by `cmd_vx * dt` in the reset heading, then
    returns squared distance to the robot. Use with NEGATIVE weight for penalty.
    Penalty grows unbounded with episode time if the robot lags — so a static
    policy gets crushed proportional to how long it stands still.
    """
    if not hasattr(env, "walk_phantom_pos_w"):
        return torch.zeros(env.num_envs, device=env.device)
    robot: Articulation = env.scene[robot_cfg.name]

    cmd_vx = env.command_manager.get_command(command_name)[:, 0]
    cos_y = torch.cos(env.walk_phantom_yaw_w)
    sin_y = torch.sin(env.walk_phantom_yaw_w)

    dt = env.step_dt
    delta = cmd_vx * dt
    env.walk_phantom_pos_w[:, 0] += cos_y * delta
    env.walk_phantom_pos_w[:, 1] += sin_y * delta

    diff = robot.data.root_pos_w[:, :2] - env.walk_phantom_pos_w
    return torch.sum(diff * diff, dim=1)


def ones_obs(env: ManagerBasedRLEnv, dim: int = 1) -> torch.Tensor:
    """Placeholder all-ones observation of `dim` columns."""
    return torch.ones(env.num_envs, dim, device=env.device)


def foot_corner_distance_min_penalty(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.16,
    toe_offset: float = 0.132,
    heel_offset: float = -0.028,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize close approach between feet using toe/heel points (not just center).

    For each foot, compute world-frame positions of:
      - toe  = ankle_roll_link_center + R(quat) · (toe_offset, 0, 0)
      - heel = ankle_roll_link_center + R(quat) · (heel_offset, 0, 0)
    Then take the minimum xy distance across all 4 R↔L point pairs
    (toe↔toe, toe↔heel, heel↔toe, heel↔heel). This catches heel-to-heel and
    other rotated-foot collisions that the center-only check misses.

    Returns (N,) penalty (positive, used with negative weight).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # body_ids: expected [r_ankle_roll, l_ankle_roll]
    r_id, l_id = asset_cfg.body_ids[0], asset_cfg.body_ids[1]
    r_pos = asset.data.body_pos_w[:, r_id, :]
    r_quat = asset.data.body_quat_w[:, r_id, :]
    l_pos = asset.data.body_pos_w[:, l_id, :]
    l_quat = asset.data.body_quat_w[:, l_id, :]

    device = env.device
    N = env.num_envs
    toe_local = torch.zeros(N, 3, device=device)
    toe_local[:, 0] = toe_offset
    heel_local = torch.zeros(N, 3, device=device)
    heel_local[:, 0] = heel_offset

    r_toe_w  = r_pos + quat_apply(r_quat, toe_local)
    r_heel_w = r_pos + quat_apply(r_quat, heel_local)
    l_toe_w  = l_pos + quat_apply(l_quat, toe_local)
    l_heel_w = l_pos + quat_apply(l_quat, heel_local)

    # xy distances for all 4 pairs (z dropped — feet may be at different lift heights)
    def xy_dist(a, b):
        return torch.norm(a[:, :2] - b[:, :2], dim=-1)

    d_tt = xy_dist(r_toe_w,  l_toe_w)
    d_th = xy_dist(r_toe_w,  l_heel_w)
    d_ht = xy_dist(r_heel_w, l_toe_w)
    d_hh = xy_dist(r_heel_w, l_heel_w)
    min_d = torch.stack([d_tt, d_th, d_ht, d_hh], dim=-1).min(dim=-1)[0]

    gap = (min_distance - min_d).clamp(min=0.0)
    return (gap / min_distance) ** 2


def zeros_obs(env: ManagerBasedRLEnv, dim: int = 1) -> torch.Tensor:
    """Placeholder all-zero observation of `dim` columns.

    Used to reserve obs slots for kick info (ball_pos_b, target_dir_b,
    kick_strength) while training walking-only — keeps the policy obs
    dimension stable so a later kick-enabled env can drop in without weight
    surgery. Returns shape (N, dim).
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


def ball_pos_xy_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball XY position in robot torso frame (z dropped). Returns (N, 2)."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rel_w = ball.data.root_pos_w - robot.data.root_pos_w
    rel_b = quat_apply_inverse(robot.data.root_quat_w, rel_w)
    return rel_b[:, :2]


def ball_pos_xy_in_robot_frame_frozen(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball XY position in robot frame, FROZEN at reset. Returns (N, 2)."""
    _ensure_kick_state(env)
    return env.ball_pos_b_frozen[:, :2]


def target_dir_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Kick direction in robot torso frame — ABSOLUTE world target (2D).

    The target direction is frozen in the WORLD frame at reset (±30° from the
    robot's reset facing — see `reset_kick_target`). This observation
    re-expresses that fixed world direction in the robot's CURRENT body frame
    via a yaw-only rotation, so as the robot turns the 2D value changes —
    telling the policy where the fixed world target lies relative to its
    current heading. z is dropped (ground-plane direction). Returns (N, 2).
    """
    _ensure_target_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    _, _, base_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    tw = env.kick_target_dir_w
    cos_y = torch.cos(base_yaw)
    sin_y = torch.sin(base_yaw)
    dir_x_b = cos_y * tw[:, 0] + sin_y * tw[:, 1]
    dir_y_b = -sin_y * tw[:, 0] + cos_y * tw[:, 1]
    return torch.stack([dir_x_b, dir_y_b], dim=-1)


def ball_lin_vel_in_robot_frame(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Optional richer obs — ball linear velocity in robot torso frame."""
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, ball.data.root_lin_vel_w)


# ----------------------------------------------------------------------
# Rewards
# ----------------------------------------------------------------------
def robot_to_ball_closeness(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    motion_threshold: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward that peaks (1.0) when the robot is right next to the ball.

    When the ball is moving (speed > motion_threshold) the reward is frozen
    at its max (1.0) so the act of kicking the ball away does NOT punish the
    robot via a dropping approach reward. Once the ball stops, the regular
    distance-based reward resumes — giving the robot an incentive to walk
    back toward the (now distant) ball for the next kick.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    d = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    closeness = torch.exp(-d / std)

    ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
    is_moving = ball_speed > motion_threshold
    return torch.where(is_moving, torch.ones_like(closeness), closeness)


def face_ball(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward (max 1.0) for the base heading pointing at the ball.

    Encourages the robot to square up to the ball during approach so it kicks
    from a forward-facing stance instead of twisting the torso sideways (which
    crosses the legs — infeasible on the real robot where self-collision is on).

    Gated to the PRE-KICK phase only: once the ball is kicked it flies away,
    and we must NOT reward chasing the moving ball's bearing. After the kick
    the reward is 0; stance recovery in the hold window is handled by
    joint_pose_default. std is in radians (std=0.5 ≈ 29° at 1-sigma).
    """
    _ensure_kick_state(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    rel_w = ball.data.root_pos_w - robot.data.root_pos_w
    rel_b = quat_apply_inverse(robot.data.root_quat_w, rel_w)
    bearing = torch.atan2(rel_b[:, 1], rel_b[:, 0])  # 0 when ball dead ahead
    reward = torch.exp(-torch.square(bearing) / (std * std))
    pre_kick = (env.kick_time_s < 0).float()
    return reward * pre_kick


def ball_to_target_closeness(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Gaussian reward that peaks when the ball reaches the target (in xy)."""
    _ensure_target_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    d = torch.norm(
        env.kick_target_pos_w[:, :2] - ball.data.root_pos_w[:, :2], dim=-1
    )
    return torch.exp(-d / std)


def ball_z_velocity_l2(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    # L2 페널티: 공 수직 속도 제곱. 위로 차는 reward hacking 억제.
    ball: RigidObject = env.scene[ball_cfg.name]
    vz = ball.data.root_lin_vel_w[:, 2]
    return torch.square(vz)


def _ensure_kick_strength(env: ManagerBasedRLEnv) -> None:
    if not hasattr(env, "kick_strength_cmd"):
        env.kick_strength_cmd = torch.ones(env.num_envs, device=env.device)
    if not hasattr(env, "kick_strength_fixed"):
        env.kick_strength_fixed = None  # set to a float in [0,1] by play CLI to skip sampling


def _ensure_curriculum_state(env: ManagerBasedRLEnv) -> None:
    if not hasattr(env, "kick_curriculum_stage"):
        env.kick_curriculum_stage = 0
    if not hasattr(env, "kick_rate_ema"):
        env.kick_rate_ema = 0.0
    if not hasattr(env, "kick_curriculum_samples"):
        env.kick_curriculum_samples = 0


def sample_kick_strength(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    cmd_max: float = 1.0,
    stage_cmd_mins: tuple = (0.9, 0.5, 0.1),
    kick_rate_threshold: float = 0.7,
    min_samples_before_advance: int = 5000,
    ema_window: float = 5000.0,
) -> None:
    _ensure_kick_strength(env)
    _ensure_curriculum_state(env)

    # ── update kick-rate EMA from has_kicked BEFORE reset_kick_state clears it ──
    if hasattr(env, "has_kicked"):
        batch_kicks = env.has_kicked[env_ids].float()
        new_rate = batch_kicks.mean().item()
        n_new = len(env_ids)
        old_weight = ema_window / (ema_window + n_new)
        env.kick_rate_ema = old_weight * env.kick_rate_ema + (1.0 - old_weight) * new_rate
        env.kick_curriculum_samples += n_new

    # ── check stage transition ──
    if (env.kick_curriculum_stage < len(stage_cmd_mins) - 1
            and env.kick_rate_ema >= kick_rate_threshold
            and env.kick_curriculum_samples >= min_samples_before_advance):
        prev_rate = env.kick_rate_ema
        env.kick_curriculum_stage += 1
        env.kick_rate_ema = 0.0
        env.kick_curriculum_samples = 0
        cur_iter = env.common_step_counter // (env.num_envs * 24)
        new_min = stage_cmd_mins[env.kick_curriculum_stage]
        print(f"[Curriculum] iter {cur_iter}: Stage {env.kick_curriculum_stage} "
              f"-> cmd_min={new_min} (kick_rate={prev_rate:.1%})")

    cmd_min = stage_cmd_mins[env.kick_curriculum_stage]

    if env.kick_strength_fixed is not None:
        env.kick_strength_cmd[env_ids] = float(env.kick_strength_fixed)
        return
    rand = torch.rand(len(env_ids), device=env.device)
    env.kick_strength_cmd[env_ids] = cmd_min + (cmd_max - cmd_min) * rand


def kick_strength_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    _ensure_kick_strength(env)
    return env.kick_strength_cmd.unsqueeze(-1)


def _ensure_motion_state(env: ManagerBasedRLEnv) -> None:
    if not hasattr(env, "ball_motion_rewarded"):
        env.ball_motion_rewarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def reset_ball_motion(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    _ensure_motion_state(env)
    env.ball_motion_rewarded[env_ids] = False


def ball_motion_reward(
    env: ManagerBasedRLEnv,
    target_max: float = 4.5,
    threshold: float = 0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    # One-shot reward at first ball forward-motion above threshold.
    # After firing, latched per env until episode reset. Prevents dribble incentive.
    _ensure_target_buffers(env)
    _ensure_motion_state(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    dir_xy = env.kick_target_dir_w[:, :2]
    proj = (v_xy * dir_xy).sum(dim=-1).clamp(min=0.0)

    not_given = ~env.ball_motion_rewarded
    should_fire = (proj > threshold) & not_given

    reward = should_fire.float() * (proj / target_max).clamp(max=1.0)

    env.ball_motion_rewarded |= should_fire
    return reward


def ball_velocity_toward_target_cmd(
    env: ManagerBasedRLEnv,
    target_max: float = 4.5,
    stage_stds: tuple = (2.5, 2.0, 1.0),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    # Gaussian per-step reward peaking when ball forward-speed matches the
    # commanded target: target = kick_strength_cmd * target_max.
    # std varies per curriculum stage (wide early for weak-kick signal, narrow late for precision).
    _ensure_target_buffers(env)
    _ensure_kick_strength(env)
    _ensure_curriculum_state(env)
    stage = min(env.kick_curriculum_stage, len(stage_stds) - 1)
    std = stage_stds[stage]
    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    dir_xy = env.kick_target_dir_w[:, :2]
    proj = (v_xy * dir_xy).sum(dim=-1).clamp(min=0.0)
    target = env.kick_strength_cmd * target_max
    err = proj - target
    return torch.exp(-err.square() / (std * std))


def _kick_time_gate(
    env: ManagerBasedRLEnv,
    target_offset_s: float,
    ramp_s: float = 0.5,
    gate_min: float = 0.2,
) -> torch.Tensor:
    """킥 '시각'에 따라 킥보상을 스케일하는 게이트 [gate_min, 1.0] (A안, 2026-08-15).

    target = kick_deadline_s - target_offset_s (= 도착 + (margin-offset)).
    gate = clamp((kick_time - (target - ramp)) / ramp, gate_min, 1.0):
      kick_time <= target-ramp → gate_min (일찍 참 = 보상 gate_min배로 깎임),
      kick_time >= target       → 1.0     (목표시각 이후 = 풀보상).
    킥보상(ball_velocity, kick_success)에 곱해 '일찍 차는 유인' 자체를 제거 →
    걸어가 붙어서(도착+0.5s) 차야 풀보상. 스파스 가우시안(실패)과 달리 지배적
    dense 보상을 직접 게이팅해 강함. 버퍼 없으면 1.0(무게이트, v1 등 영향 없음).
    """
    if not hasattr(env, "kick_time_s") or not hasattr(env, "kick_deadline_s"):
        return torch.ones(env.num_envs, device=env.device)
    target = (env.kick_deadline_s - target_offset_s).clamp(min=0.0)
    ramp = max(ramp_s, 1e-3)
    g = (env.kick_time_s - (target - ramp)) / ramp
    return g.clamp(min=gate_min, max=1.0)


def ball_velocity_toward_target(
    env: ManagerBasedRLEnv,
    max_speed: float = 3.0,
    post_kick_window_s: float | None = None,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    kick_time_gate_offset_s: float | None = None,
    kick_time_gate_ramp_s: float = 0.5,
    kick_time_gate_min: float = 0.2,
) -> torch.Tensor:
    """Quadratic per-step reward on ball xy velocity projected onto the FIXED
    world target direction (frozen at reset in `reset_kick_target`).

    Squared so strong kicks dominate over gentle dribbles. Clamped to
    [0, max_speed] before squaring so a kick backward is not rewarded.

    History: a peak-fire-once variant was tried (reward only at
    hold_time_s after latch, proportional to peak velocity squared) to
    eliminate dribble entirely — but the reward was too sparse for the
    value function to learn, and the policy eventually collapsed to
    "don't kick at all" (~1% kick rate by iter 15000, 692 rollbacks).
    Reverted to cumulative quadratic — pair with `base_ang_vel_z_l2`
    penalty + `max_speed=5` cap to limit dribble incentive instead.

    ★ post_kick_window_s (walk_kick_v1 추가) ★
    None 이면 옛 동작(누적, 킥 후에도 계속 보상 → 공 쫓아가 재킥 유인).
    값을 주면 킥(kick_time_s 래치) 후 그 시간(초)까지만 보상하고 이후 0.
    킥 임펄스는 잡되(짧은 window), 굴러가는 공을 쫓을 유인은 없앤다.
    이게 없으면 commands_zero(cmd 0) 를 받고도 ball_velocity 따먹으려고
    쫓아가 재킥하는 충돌이 생긴다.
    """
    _ensure_target_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    dir_xy = env.kick_target_dir_w[:, :2]
    proj = (v_xy * dir_xy).sum(dim=-1)
    reward = torch.clamp(proj, min=0.0, max=max_speed) ** 2

    if post_kick_window_s is not None and hasattr(env, "kick_time_s"):
        current_t = kick_clock_s(env)
        elapsed = current_t - env.kick_time_s
        past_window = (env.kick_time_s >= 0) & (elapsed > post_kick_window_s)
        reward = torch.where(past_window, torch.zeros_like(reward), reward)

    # A안: 목표시각 전 킥은 보상 깎기(일찍 차는 유인 제거)
    if kick_time_gate_offset_s is not None:
        reward = reward * _kick_time_gate(
            env, kick_time_gate_offset_s, kick_time_gate_ramp_s, kick_time_gate_min
        )

    return reward


def ball_overspeed_penalty(
    env: ManagerBasedRLEnv,
    target_speed: float = 3.0,
    post_kick_window_s: float | None = 0.4,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """공 속도(목표방향 투영)가 target_speed 를 넘으면 초과분 제곱 벌점.

    ball_velocity_toward_target 은 proj 를 max_speed 에서 clamp 해 보상만
    포화시킬 뿐 초과 속도를 벌하지 않는다 → 정책이 실제로는 4+ m/s 로 참.
    이 항이 target 초과분을 벌해 실제 킥 속도를 target 으로 캡한다.
    킥 파워가 곧 윈드업(다리 앞뒤 벌림)이므로, 파워를 캡하면 벌림도 준다.
    target 이하는 벌하지 않아 드리블(너무 약한 킥)로는 안 흐른다.
    ball_velocity 와 동일 window 게이팅. 음수 weight 로 사용.
    """
    _ensure_target_buffers(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    v_xy = ball.data.root_lin_vel_w[:, :2]
    dir_xy = env.kick_target_dir_w[:, :2]
    proj = (v_xy * dir_xy).sum(dim=-1)
    over = (proj - target_speed).clamp(min=0.0)
    penalty = torch.square(over)

    if post_kick_window_s is not None and hasattr(env, "kick_time_s"):
        current_t = kick_clock_s(env)
        elapsed = current_t - env.kick_time_s
        past_window = (env.kick_time_s >= 0) & (elapsed > post_kick_window_s)
        penalty = torch.where(past_window, torch.zeros_like(penalty), penalty)

    return penalty


# ----------------------------------------------------------------------
# Kick-then-stand success bonus + termination
# ----------------------------------------------------------------------
def _ensure_kick_state(env: ManagerBasedRLEnv) -> None:
    if not hasattr(env, "kick_time_s"):
        env.kick_time_s = torch.full(
            (env.num_envs,), -1.0, device=env.device, dtype=torch.float
        )
    # Ball position in the robot frame, captured ONCE at reset (see
    # ball_pos_in_robot_frame_frozen). Models a one-shot ball detection:
    # the policy reads the spawn-time ball position and does not get a live
    # update as it walks closer.
    if not hasattr(env, "ball_pos_b_frozen"):
        env.ball_pos_b_frozen = torch.zeros(env.num_envs, 3, device=env.device)
    # Peak ball velocity (toward target, clamped) seen this episode. Used
    # by `ball_velocity_toward_target` (peak-fire-once design) to reward
    # only the strongest impulse, not cumulative pushing (no dribble farm).
    if not hasattr(env, "peak_kick_v"):
        env.peak_kick_v = torch.zeros(env.num_envs, device=env.device)


def _maybe_record_kick(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    speed_threshold: float,
) -> None:
    """Latch the time of the first frame where ball speed exceeds threshold."""
    _ensure_kick_state(env)
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_speed = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
    not_yet = env.kick_time_s < 0
    just_kicked = (ball_speed > speed_threshold) & not_yet
    current_t = kick_clock_s(env)
    env.kick_time_s = torch.where(just_kicked, current_t, env.kick_time_s)


def reset_kick_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    walk_speed: float = 0.3,
    margin_s: float = 1.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset event: clear kick_time and compute per-env no-kick deadline.

    The deadline is distance-aware:
        deadline = (initial robot↔ball xy distance) / walk_speed + margin_s

    so close-spawn balls don't get an unfairly long farming window and
    far-spawn balls get enough time for the robot to actually walk over.
    Must run AFTER the robot and ball reset events (declaration order).
    """
    _ensure_kick_state(env)
    env.kick_time_s[env_ids] = -1.0
    env.peak_kick_v[env_ids] = 0.0

    if not hasattr(env, "kick_deadline_s"):
        env.kick_deadline_s = torch.zeros(env.num_envs, device=env.device)

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    d = torch.norm(
        ball.data.root_pos_w[env_ids, :2] - robot.data.root_pos_w[env_ids, :2],
        dim=-1,
    )
    env.kick_deadline_s[env_ids] = d / max(walk_speed, 1e-3) + margin_s

    # Freeze the ball position in the robot frame at this reset.
    rel_w = ball.data.root_pos_w[env_ids] - robot.data.root_pos_w[env_ids]
    env.ball_pos_b_frozen[env_ids] = quat_apply_inverse(
        robot.data.root_quat_w[env_ids], rel_w
    )


def reset_kick_state_from_cmd(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "base_velocity",
    margin_s: float = 1.0,
    min_walk_speed: float = 0.1,
    max_deadline_s: float = 10.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Per-env no-kick deadline derived from each env's commanded velocity.

    deadline = init_ball_distance / max(cmd_vx, min_walk_speed) + margin_s
    (clamped at max_deadline_s).

    Standing envs (cmd_vx ≈ 0) fall back to `min_walk_speed` which gives a
    reasonable upper bound; the clamp caps the deadline so they don't get an
    excessively long farming window.
    """
    _ensure_kick_state(env)
    env.kick_time_s[env_ids] = -1.0
    env.peak_kick_v[env_ids] = 0.0

    if not hasattr(env, "kick_deadline_s"):
        env.kick_deadline_s = torch.zeros(env.num_envs, device=env.device)

    ball: RigidObject = env.scene[ball_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    d = torch.norm(
        ball.data.root_pos_w[env_ids, :2] - robot.data.root_pos_w[env_ids, :2],
        dim=-1,
    )

    cmd_vx = env.command_manager.get_command(command_name)[env_ids, 0]
    cmd_safe = torch.clamp(torch.abs(cmd_vx), min=min_walk_speed)
    deadline = d / cmd_safe + margin_s
    env.kick_deadline_s[env_ids] = torch.clamp(deadline, max=max_deadline_s)

    rel_w = ball.data.root_pos_w[env_ids] - robot.data.root_pos_w[env_ids]
    env.ball_pos_b_frozen[env_ids] = quat_apply_inverse(
        robot.data.root_quat_w[env_ids], rel_w
    )


def early_kick_penalty(
    env: ManagerBasedRLEnv,
    early_margin_s: float = 2.0,
    max_earliness_s: float = 2.0,
) -> torch.Tensor:
    """도착 전에 차면(조기/런지킥) 벌점 → 걸어 들어가서 붙어 차게 유도.

    earliest_allowed = kick_deadline_s - early_margin_s
      (kick_deadline_s = 초기거리/cmd_vx + margin. earliest ≈ 순수 이동시각=도착시각)
    kick_time_s < earliest 면 "도착 전에 참" → (earliest - kick_time) 비례 벌점.
    킥 순간(latch 직후 ~1스텝)에만 1회 적용. early_margin_s 로 "얼마나 늦게(붙어서)
    차야 하나" 튜닝: 크게=데드라인에 더 가까이, 작게=여유.
    """
    if not hasattr(env, "kick_time_s") or not hasattr(env, "kick_deadline_s"):
        return torch.zeros(env.num_envs, device=env.device)
    kicked = env.kick_time_s >= 0
    earliest = (env.kick_deadline_s - early_margin_s).clamp(min=0.0)
    earliness = (earliest - env.kick_time_s).clamp(min=0.0, max=max_earliness_s)
    # 킥 latch 직후 ~1스텝에만 (한 번만 벌)
    current_t = kick_clock_s(env)
    elapsed = current_t - env.kick_time_s
    at_kick = kicked & (elapsed >= 0.0) & (elapsed < 1.5 * env.step_dt)
    return torch.where(at_kick, earliness, torch.zeros_like(earliness))


def kick_time_gauss_reward(
    env: ManagerBasedRLEnv,
    target_offset_s: float = 1.0,
    std_s: float = 0.4,
) -> torch.Tensor:
    """킥 시각을 목표시각에 맞추는 양(+)의 가우시안 보상 → '붙어서 늦게' 차게 유도.

    target_time = kick_deadline_s - target_offset_s
      (kick_deadline_s = 도착시각 + margin(=2.0). offset=1.0 이면 target ≈ 도착 후
       1.0초 = 킥창 [도착+0.5, 도착+2.0]의 중앙.)
    reward = exp(-((kick_time - target)/std)²), 킥 순간(latch 직후 ~1스텝)에 1회.
    일찍 참=왼쪽꼬리(낮음)/목표=최대/늦음=오른쪽꼬리(낮음). 벌점(early_kick)과 달리
    양의 보상이라 킥 자체를 회피하지 않음. 감가율(gamma)이 '빨리 차라'를 밀기 때문에
    weight 를 그 손해(+8 보너스의 ~22%≈1.8)를 이길 만큼(≈3) 줘야 실제로 늦게 참.
    """
    if not hasattr(env, "kick_time_s") or not hasattr(env, "kick_deadline_s"):
        return torch.zeros(env.num_envs, device=env.device)
    kicked = env.kick_time_s >= 0
    target = (env.kick_deadline_s - target_offset_s).clamp(min=0.0)
    err = (env.kick_time_s - target) / max(std_s, 1e-3)
    gauss = torch.exp(-(err * err))
    # 킥 latch 직후 ~1스텝에만 (한 번만 보상)
    current_t = kick_clock_s(env)
    elapsed = current_t - env.kick_time_s
    at_kick = kicked & (elapsed >= 0.0) & (elapsed < 1.5 * env.step_dt)
    return torch.where(at_kick, gauss, torch.zeros_like(gauss))


def kick_then_stand_bonus(
    env: ManagerBasedRLEnv,
    hold_time_s: float = 1.0,
    speed_threshold: float = 1.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    kick_time_gate_offset_s: float | None = None,
    kick_time_gate_ramp_s: float = 0.5,
    kick_time_gate_min: float = 0.2,
) -> torch.Tensor:
    """Sparse +1 at the single step that completes (kick + hold_time_s).

    Combined with the existing fall terminations (base_height, bad_orientation),
    this fires only if the robot survives the post-kick window. The matching
    `kick_resolved_termination` ends the episode at that same step so the
    policy can't accumulate further reward by kicking endlessly.

    A안: kick_time_gate_offset_s 주면 목표시각 전 킥은 보너스도 gate_min배로 깎음.
    """
    _maybe_record_kick(env, ball_cfg, speed_threshold)
    current_t = kick_clock_s(env)
    elapsed = current_t - env.kick_time_s
    # fire exactly once: in the step where elapsed first crosses hold_time
    just_reached = (
        (env.kick_time_s >= 0)
        & (elapsed >= hold_time_s)
        & (elapsed < hold_time_s + env.step_dt + 1e-6)
    )
    reward = just_reached.float()
    if kick_time_gate_offset_s is not None:
        reward = reward * _kick_time_gate(
            env, kick_time_gate_offset_s, kick_time_gate_ramp_s, kick_time_gate_min
        )
    return reward


def kick_resolved_termination(
    env: ManagerBasedRLEnv,
    hold_time_s: float = 1.0,
    speed_threshold: float = 1.0,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Terminate the episode once the (kick + hold_time_s) window has elapsed."""
    _maybe_record_kick(env, ball_cfg, speed_threshold)
    current_t = kick_clock_s(env)
    return (env.kick_time_s >= 0) & ((current_t - env.kick_time_s) >= hold_time_s)


def joint_pose_to_default(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward (max +1.0) for joint positions near their default pose.

    Encourages return-to-neutral after a kick — without this, the policy is
    rewarded for the kick itself but has no pressure to bring the leg back to
    a standing pose, so it freezes mid-swing. The Gaussian shape forgives the
    large deviations during the kick contact and rewards recovery in the post-
    kick window before `kick_resolved` ends the episode.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.mean(torch.square(err), dim=1) / (std * std))


def hip_pitch_split_penalty(
    env: ManagerBasedRLEnv,
    threshold_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=["r_pelvis_pitch_joint", "l_pelvis_pitch_joint"],
        preserve_order=True,
    ),
) -> torch.Tensor:
    # Mirrored joint convention: r positive = right leg forward, l negative = left leg forward.
    # Physical split angle between legs = |r - (-l)| = |r + l|.
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    split_angle = torch.abs(pos[:, 0] + pos[:, 1])
    threshold = math.radians(threshold_deg)
    excess = torch.clamp(split_angle - threshold, min=0.0)
    return excess ** 2


def track_forward_velocity_exp(
    env: ManagerBasedRLEnv,
    target_vx: float = 0.3,
    std: float = 0.387,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward for tracking a fixed forward (body-frame x) velocity.

    Peaks at 1.0 when vx_body == target_vx; falls off with std. Replaces the
    command-driven `track_lin_vel_xy_hold_aware` (which needs a velocity
    command — kick task has none).

    Default std=0.387 ≈ sqrt(0.15) to match the velocity task's std² of 0.15.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vx_b = asset.data.root_lin_vel_b[:, 0]
    err = vx_b - target_vx
    return torch.exp(-torch.square(err) / (std * std))


def ball_approach_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian closeness reward — robot↔ball xy distance.

    Pre-kick:  exp(-d / std)        — closer is better, smooth gradient
                                       so the policy learns "walk to ball".
    Post-kick: locked at **1.0**    — once the kick is recorded the robot
                                       no longer needs to chase the ball;
                                       combined with `commands_zero_on_kick_or_hold`
                                       (cmd→0) and `track_lin_vel_xy_kick_aware`
                                       (track reward→0) the robot is asked to
                                       stand still in default pose.

    The post-kick lock means the policy doesn't suddenly "lose" this
    reward when the ball flies away after being kicked — preserving the
    value function continuity that distance-based rewards otherwise break.
    """
    ball = env.scene[ball_cfg.name]
    robot = env.scene[robot_cfg.name]
    d = torch.norm(
        ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    reward = torch.exp(-d / std)

    if hasattr(env, "kick_time_s"):
        post_kick = env.kick_time_s >= 0
        if torch.any(post_kick):
            reward = reward.clone()
            reward[post_kick] = 1.0

    return reward


def apply_constant_forward_force(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    force_x_range: tuple[float, float] = (5.0, 15.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply a per-episode constant forward (+x body frame) external force.

    At each reset, samples a forward force magnitude per env from `force_x_range`
    and applies it to the specified body via the permanent wrench mechanism —
    i.e. the SAME force is applied EVERY step until the next reset. y and z
    components are 0 (no lateral / vertical force). torque is 0.

    Use case: train the policy to lean back / brace against a steady forward
    disturbance, so it doesn't tip forward from the kick's momentum reaction.

    The force is applied in the **world frame** (asset.set_external_force is
    world-frame). For an MX3-sized humanoid starting facing world +x, this
    coincides with the robot's body +x at reset. If the robot rotates during
    the episode the force stays world-aligned (acceptable for short kick
    episodes 2~3 s).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    n = len(env_ids)

    if isinstance(asset_cfg.body_ids, list):
        num_bodies = len(asset_cfg.body_ids)
        body_ids = asset_cfg.body_ids
    else:
        num_bodies = asset.num_bodies
        body_ids = asset_cfg.body_ids  # slice(None)

    # Sample forward force per env, broadcast across bodies
    fx = torch.empty(n, num_bodies, device=asset.device).uniform_(
        force_x_range[0], force_x_range[1]
    )
    forces = torch.zeros(n, num_bodies, 3, device=asset.device)
    forces[..., 0] = fx
    torques = torch.zeros(n, num_bodies, 3, device=asset.device)

    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids,
    )


def commands_zero_on_kick_or_hold(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Velocity command, zeroed for envs in reset_hold OR post-kick.

    Same role as `generated_commands_with_reset_hold` but also zeros the cmd
    after the kick has been recorded (env.kick_time_s >= 0). The post-kick
    zero tells the policy "stop walking now" so the robot returns to a
    default standing pose during the hold_time window.
    """
    cmd = env.command_manager.get_command(command_name)

    needs_clone = False
    if hasattr(env, "reset_hold_steps"):
        hold_mask = env.reset_hold_steps > 0
        if torch.any(hold_mask):
            cmd = cmd.clone()
            needs_clone = True
            cmd[hold_mask] = 0.0

    if hasattr(env, "kick_time_s"):
        post_kick = env.kick_time_s >= 0
        if torch.any(post_kick):
            if not needs_clone:
                cmd = cmd.clone()
            cmd[post_kick] = 0.0

    return cmd


def track_lin_vel_xy_kick_aware(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Wrap `track_lin_vel_xy_hold_aware` and additionally zero the reward
    in envs that have already kicked. Combined with the obs-side
    `commands_zero_on_kick_or_hold`, this makes the post-kick window a
    "stand still" task, so the robot doesn't keep walking forward.
    """
    from unitree_rl_lab.tasks.locomotion.mdp.rewards import (
        track_lin_vel_xy_hold_aware,
    )

    reward = track_lin_vel_xy_hold_aware(env, command_name, std, asset_cfg)

    if hasattr(env, "kick_time_s"):
        post_kick = env.kick_time_s >= 0
        if torch.any(post_kick):
            reward = reward.clone()
            reward[post_kick] = 0.0

    return reward


def no_kick_timeout_termination(
    env: ManagerBasedRLEnv,
    fallback_s: float = 5.0,
) -> torch.Tensor:
    """Failure termination: end the episode if no kick was recorded by the
    per-env deadline computed in `reset_kick_state`.

    The deadline lives on `env.kick_deadline_s` (distance-aware: longer for
    far-spawn balls). If the buffer is missing (e.g. reset event misordered),
    falls back to a flat `fallback_s` so training never gets stuck.
    """
    _ensure_kick_state(env)
    current_t = kick_clock_s(env)
    no_kick = env.kick_time_s < 0
    if hasattr(env, "kick_deadline_s"):
        deadline_passed = current_t >= env.kick_deadline_s
    else:
        deadline_passed = current_t >= fallback_s
    return no_kick & deadline_passed


def early_kick_termination(
    env: ManagerBasedRLEnv,
    min_kick_time_s: float = 1.0,
    speed_threshold: float = 0.01,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """실패 종료: 킥이 에피소드 시작 후 min_kick_time_s(초) 이내에 기록되면 종료.

    (2026-08-15 개정) deadline 상대 earliest → **절대 시간 문턱**. deadline 기반은
    타이밍이 부적절했고 강한 킥 탐색까지 눌렀음(약한 킥 평탄화). 이제 초반 즉시킥/
    런지(시작 1초 이내)만 억제하고, 그 이후엔 언제 차든 허용 → 킥 파워는 안 누름.
    """
    _maybe_record_kick(env, ball_cfg, speed_threshold)
    kicked = env.kick_time_s >= 0
    return kicked & (env.kick_time_s < min_kick_time_s)
