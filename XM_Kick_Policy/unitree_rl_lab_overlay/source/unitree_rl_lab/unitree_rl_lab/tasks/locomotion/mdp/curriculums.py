from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    threshold: float = 0.8,            # ← 추가: curriculum level 올리는 reward 임계값
    delta: float = 0.1,               # ← 추가: 레벨업 1회당 범위 확장폭(반씩 상/하한)
) -> torch.Tensor:
    """vx, vy command 범위를 점진적으로 limit_ranges로 확장.
    
    Args:
        threshold: 이 비율(× weight)을 넘기는 평균 reward여야 level 상승.
            예) weight=1.0, threshold=0.8 → reward > 0.8 일 때 상승.
            기본값 0.8은 G1용. 작은 로봇은 0.5~0.6 권장.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * threshold:    # ← 0.8 → threshold 변수
            delta_command = torch.tensor([-delta, delta], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
    threshold: float = 0.8,            # ← 추가
    delta: float = 0.1,               # ← 추가: 레벨업 1회당 각속도 범위 확장폭
) -> torch.Tensor:
    """wz command 범위를 점진적으로 limit_ranges로 확장.
    
    Args:
        threshold: 이 비율(× weight)을 넘기는 평균 reward여야 level 상승.
            yaw 추적은 lin보다 본질적으로 어려우므로 lin보다 약간 낮게 권장.
            예) lin은 0.6, ang은 0.55.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * threshold:    # ← 0.8 → threshold 변수
            delta_command = torch.tensor([-delta, delta], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)