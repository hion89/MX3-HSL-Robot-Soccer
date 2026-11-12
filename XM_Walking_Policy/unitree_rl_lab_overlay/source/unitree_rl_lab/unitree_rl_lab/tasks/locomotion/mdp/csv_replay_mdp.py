"""Residual learning utilities: CSV-driven reference + policy correction.

Use case: sim-to-real gap correction. Record desired motion from sim (CSV log),
replay it on the real robot via a residual policy that prevents falls without
deviating much from the recorded keyframes.

Action composition:
    target = default + action_scale * (reference[tick] + clip(policy_output, ±residual_clip))

CSV columns expected (per row, comma-separated):
    - "a_clip_<joint_name>" : reference action (policy output space, post-clip)
    - "q_<joint_name>"      : recorded joint position at that tick (used for reset)
"""

from __future__ import annotations

import csv as _csv_module
import os
from typing import Optional

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .reset_hold import DelayedJointPositionAction, DelayedJointPositionActionCfg


# --------------------------------------------------------------------
# Internal CSV cache (one per file path) so multiple envs share data
# --------------------------------------------------------------------
_CSV_CACHE: dict[str, dict] = {}


def _load_csv_once(csv_path: str, joint_names: list[str], device: str | torch.device) -> dict:
    """Load and cache reference trajectory + first frame joint state from CSV.

    Returns dict with keys:
      - ref_actions: (N, 12) tensor — a_clip_* values per tick
      - first_q:     (12,)   tensor — q_* values at row 0 (used for reset)
      - n_ticks:     int     — number of rows
    """
    abs_path = os.path.abspath(os.path.expanduser(csv_path))
    if abs_path in _CSV_CACHE:
        return _CSV_CACHE[abs_path]

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"CSV not found: {abs_path}")

    with open(abs_path, "r") as f:
        reader = _csv_module.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"CSV has no data rows: {abs_path}")

    ref_cols = [f"a_clip_{jn}" for jn in joint_names]
    q_cols = [f"q_{jn}" for jn in joint_names]

    missing = [c for c in (ref_cols + q_cols) if c not in rows[0]]
    if missing:
        raise KeyError(f"CSV missing columns: {missing[:5]}... (in {abs_path})")

    n = len(rows)
    ref_np = np.zeros((n, len(joint_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, c in enumerate(ref_cols):
            ref_np[i, j] = float(row[c])

    first_q_np = np.array(
        [float(rows[0][c]) for c in q_cols], dtype=np.float32
    )

    entry = {
        "ref_actions": torch.tensor(ref_np, device=device),
        "first_q":     torch.tensor(first_q_np, device=device),
        "n_ticks":     n,
        "joint_names": list(joint_names),
    }
    _CSV_CACHE[abs_path] = entry
    print(f"[csv_replay_mdp] cached '{abs_path}': {n} ticks, 12 joints")
    return entry


# --------------------------------------------------------------------
# Residual Action — reference (CSV) + clipped policy output
# --------------------------------------------------------------------
class ResidualJointPositionAction(DelayedJointPositionAction):
    """target = default + scale * (ref[tick] + clip(policy_output, ±residual_clip))

    - Inherits hold-mask / action-delay / target clamping from DelayedJointPositionAction.
    - Maintains a per-env tick counter advanced each `process_actions` call.
    - Reference is loaded from CSV at init (cached across envs).
    """

    cfg: "ResidualJointPositionActionCfg"

    def __init__(self, cfg: "ResidualJointPositionActionCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if not cfg.reference_csv_path:
            raise ValueError("ResidualJointPositionActionCfg.reference_csv_path is empty")

        cache = _load_csv_once(cfg.reference_csv_path, list(cfg.joint_names), env.device)
        self._ref_actions: torch.Tensor = cache["ref_actions"]  # (N, 12)
        self._ref_first_q: torch.Tensor = cache["first_q"]      # (12,)
        self._ref_len: int = cache["n_ticks"]

        self._residual_clip: float = float(cfg.residual_clip)
        self._tick: torch.Tensor = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

        # expose for reward / termination terms
        env._residual_action_term = self  # back-reference; convenience

        print(
            f"[ResidualAction] ref_len={self._ref_len}, residual_clip=±{self._residual_clip}"
        )

    @property
    def tick(self) -> torch.Tensor:
        return self._tick

    @property
    def ref_len(self) -> int:
        return self._ref_len

    @property
    def first_frame_q(self) -> torch.Tensor:
        return self._ref_first_q

    def process_actions(self, actions: torch.Tensor):
        # Clip residual to symmetric range
        residual = torch.clamp(actions, -self._residual_clip, self._residual_clip)

        # Save residual separately — `_raw_actions` will be overwritten with
        # the composed (reference + residual) action by the parent, so reward
        # terms that want the pure residual read `_policy_residual`.
        if not hasattr(self, "_policy_residual"):
            self._policy_residual = torch.zeros_like(residual)
        self._policy_residual[:] = residual

        # Reference for this tick (clamp at end to avoid OOB)
        idx = torch.clamp(self._tick, max=self._ref_len - 1)
        reference = self._ref_actions[idx]  # (num_envs, 12)

        full = reference + residual
        super().process_actions(full)

        # Advance tick — but NOT for envs currently in reset_hold (the parent
        # zeros their action this step, so we shouldn't burn a CSV frame).
        hold = getattr(self._env, "reset_hold_steps", None)
        if hold is not None:
            self._tick = torch.where(hold > 0, self._tick, self._tick + 1)
        else:
            self._tick = self._tick + 1

    def reset(self, env_ids):
        super().reset(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            self._tick[env_ids] = 0


@configclass
class ResidualJointPositionActionCfg(DelayedJointPositionActionCfg):
    class_type = ResidualJointPositionAction

    # CSV file with a_clip_* and q_* columns
    reference_csv_path: str = ""

    # Residual clip range (policy output space).
    # With scale=0.25, residual_clip=0.4 → ±0.10 rad joint offset.
    residual_clip: float = 0.4


# --------------------------------------------------------------------
# Reset event — set joint state to CSV[0]
# --------------------------------------------------------------------
def reset_robot_to_csv_first_frame(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    csv_path: str,
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the named leg joints to CSV row 0 values, velocities to 0.

    Non-leg joints are left untouched.
    """
    cache = _load_csv_once(csv_path, joint_names, env.device)
    first_q = cache["first_q"]  # (12,)

    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)

    # Resolve joint ids (ordered to match joint_names because preserve_order=True
    # is expected on the asset_cfg). joint_ids can be a slice OR a list.
    asset_cfg.resolve(env.scene)
    joint_ids = asset_cfg.joint_ids

    q = first_q.unsqueeze(0).expand(n, -1).contiguous()  # (n, 12)
    qd = torch.zeros_like(q)

    asset.write_joint_state_to_sim(q, qd, joint_ids=joint_ids, env_ids=env_ids)


# --------------------------------------------------------------------
# Termination — CSV exhausted
# --------------------------------------------------------------------
def csv_finished_termination(
    env: ManagerBasedRLEnv,
    action_name: str = "JointPositionAction",
) -> torch.Tensor:
    """End episode when the reference tick has reached the end of the CSV."""
    term = env.action_manager.get_term(action_name)
    return term.tick >= term.ref_len


# --------------------------------------------------------------------
# Reward — penalize large residual (encourage minimal correction)
# --------------------------------------------------------------------
def residual_l2_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "JointPositionAction",
) -> torch.Tensor:
    """Sum of squared residual values (post-clip, pre-reference-addition).

    Reads `_policy_residual` stored by ResidualJointPositionAction.process_actions —
    `_raw_actions` is overwritten with the composed (reference + residual) action
    by the parent and therefore cannot be used directly here.
    """
    term = env.action_manager.get_term(action_name)
    residual = getattr(term, "_policy_residual", None)
    if residual is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(residual * residual, dim=-1)
