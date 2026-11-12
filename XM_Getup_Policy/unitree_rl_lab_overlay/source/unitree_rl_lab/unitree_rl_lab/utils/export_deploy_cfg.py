# /home/ros103/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py

import os
import re
import yaml
import numpy as np

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def _try_safe_match(asset_joint_names: list[str], joint_sdk_names: list[str]):
    """
    Returns:
        joint_ids_map: list[int]  (asset joint indices, in SDK order)
        effective_joint_sdk_names: list[str]  (sdk names that matched)
        missing: list[str]
    Matching priority:
      1) exact name match
      2) regex search (pattern contains meta-chars or exact still ok)
    """
    joint_ids_map = []
    effective = []
    missing = []

    # quick map for exact matching
    name_to_idx = {n: i for i, n in enumerate(asset_joint_names)}

    for pat in joint_sdk_names:
        # 1) exact
        if pat in name_to_idx:
            joint_ids_map.append(name_to_idx[pat])
            effective.append(pat)
            continue

        # 2) regex search
        try:
            rx = re.compile(pat)
        except re.error:
            rx = None

        found = None
        if rx is not None:
            for i, n in enumerate(asset_joint_names):
                if rx.search(n):
                    found = i
                    break

        if found is None:
            missing.append(pat)
        else:
            joint_ids_map.append(found)
            effective.append(pat)

    if len(joint_ids_map) == 0:
        raise ValueError(
            f"No joint_sdk_names matched any asset joints. "
            f"joint_sdk_names={joint_sdk_names}, asset_joints={asset_joint_names}"
        )

    return joint_ids_map, effective, missing


def _get_joint_sdk_names(env: ManagerBasedRLEnv, asset: Articulation) -> list[str]:
    """
    Priority:
      A) env.cfg.scene.robot.joint_sdk_names (if exists)
      B) action term joint names (JointPositionAction / JointVelocityAction)
      C) fallback to all asset joints
    """
    # A) robot cfg metadata
    if hasattr(env.cfg.scene, "robot") and hasattr(env.cfg.scene.robot, "joint_sdk_names"):
        jsn = env.cfg.scene.robot.joint_sdk_names
        if jsn is not None and len(jsn) > 0:
            return list(jsn)

    # B) try from action term (most correct for deployment)
    # (We avoid importing action classes; just introspect)
    for action_name, action_term in zip(env.action_manager.active_terms, env.action_manager._terms.values()):
        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            # cfg.joint_names is often list[str] already
            if hasattr(action_term.cfg, "joint_names") and action_term.cfg.joint_names is not None:
                return list(action_term.cfg.joint_names)

    # C) fallback
    return list(asset.data.joint_names)


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    asset: Articulation = env.scene["robot"]
    asset_joint_names = list(asset.data.joint_names)

    joint_sdk_names = _get_joint_sdk_names(env, asset)

    # ★ 2026-09-21 수정: 방향이 뒤집히던 버그 ★
    #
    #   기존: resolve_matching_names(asset_joint_names, joint_sdk_names, preserve_order=True)
    #   인자 순서상 asset 이름이 **질의**, sdk 이름이 **대상**이라 반환값은
    #   "asset 관절 i 의 SDK 인덱스" 다. 그런데 아래 gather 는 정반대인
    #   "SDK 관절 k 의 asset 인덱스" 를 가정한다.
    #
    #   실측 (XM_robot04, sdk 20 관절):
    #       반환값             [6, 15, 18, 0, 12, 7, ...]
    #       asset의 SDK 인덱스  [6, 15, 18, 0, 12, 7, ...]   <- 일치
    #       SDK의 asset 인덱스  [3,  8, 12, 15, 17, 19, ...]  <- 필요한 값
    #
    #   ⚠ SDK 리스트가 **전 관절을 덮을 때만** strict 가 성공한다. 보행(12)·
    #     기상(18)은 예외가 나서 _try_safe_match 로 빠지는데, 그쪽은 올바른
    #     방향("SDK 의 asset 인덱스")을 만든다. 그래서 지금껏 문제가 없었고,
    #     mimic(20)처럼 전 관절을 덮는 태스크에서만 터진다.
    #
    #   처방: 경로를 하나로 통일한다. 이름 기준 단방향 매핑(_try_safe_match)만
    #   쓴다. 부분 커버든 전체 커버든 같은 규약이 나온다.
    joint_ids_map, effective_joint_sdk_names, missing = _try_safe_match(asset_joint_names, joint_sdk_names)

    # Convert to python list[int]
    joint_ids_map = list(map(int, joint_ids_map))

    # Build cfg
    cfg = {}
    cfg["joint_ids_map"] = joint_ids_map
    cfg["joint_sdk_names"] = effective_joint_sdk_names
    if len(missing) > 0:
        cfg["joint_sdk_names_missing"] = missing

    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation

    # IMPORTANT:
    # Save stiffness/damping/default_joint_pos IN SDK ORDER.
    # joint_ids_map is "asset joint indices in SDK order".
    joint_ids_t = np.array(joint_ids_map, dtype=np.int64)

    default_stiffness = asset.data.default_joint_stiffness[0].detach().cpu().numpy()
    default_damping = asset.data.default_joint_damping[0].detach().cpu().numpy()
    default_joint_pos = asset.data.default_joint_pos[0].detach().cpu().numpy()

    cfg["stiffness"] = default_stiffness[joint_ids_t].tolist()
    cfg["damping"] = default_damping[joint_ids_t].tolist()
    cfg["default_joint_pos"] = default_joint_pos[joint_ids_t].tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()

        # scale
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        # clip
        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        # offset
        if action_name in ["JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _k in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            if _k in term_cfg:
                del term_cfg[_k]
        cfg["actions"][action_name] = term_cfg

        if action_term._joint_ids == slice(None):
            cfg["actions"][action_name]["joint_ids"] = None
        else:
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()

        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]

        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)

        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        term_cfg = term_cfg.to_dict()
        for _k in ["func", "modifiers", "noise", "flatten_history_dim"]:
            if _k in term_cfg:
                del term_cfg[_k]
        cfg["observations"][obs_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)