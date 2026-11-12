# mdp/.../arm_pose.py (혹은 네가 넣은 파일)
from __future__ import annotations
from typing import Mapping, Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def set_joint_pos_and_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    joint_pos: Mapping[str, float] | Sequence[float],
    joint_vel: Mapping[str, float] | Sequence[float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set joint state AND joint targets.

    - If joint_pos is dict: maps by joint name (order-safe)
    - If joint_pos is list/tuple: uses resolved joint order (order-dependent)
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    joint_ids = asset_cfg.joint_ids
    joint_names = getattr(asset_cfg, "joint_names", None)  # usually present when you pass joint_names=...

    # --- build pos/vel in resolved joint order ---
    if isinstance(joint_pos, dict):
        if not joint_names:
            raise RuntimeError("asset_cfg.joint_names is required when joint_pos is a dict.")
        pos_list = [float(joint_pos[n]) for n in joint_names]
    else:
        pos_list = list(joint_pos)

    pos = torch.tensor(pos_list, device=asset.device, dtype=torch.float32).view(1, -1).repeat(len(env_ids), 1)

    if joint_vel is None:
        vel = torch.zeros_like(pos)
    else:
        if isinstance(joint_vel, dict):
            if not joint_names:
                raise RuntimeError("asset_cfg.joint_names is required when joint_vel is a dict.")
            vel_list = [float(joint_vel.get(n, 0.0)) for n in joint_names]
        else:
            vel_list = list(joint_vel)
        vel = torch.tensor(vel_list, device=asset.device, dtype=torch.float32).view(1, -1).repeat(len(env_ids), 1)

    # 1) state
    asset.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=env_ids)
    # 2) targets
    asset.set_joint_position_target(pos, joint_ids=joint_ids, env_ids=env_ids)
    asset.set_joint_velocity_target(vel, joint_ids=joint_ids, env_ids=env_ids)
    