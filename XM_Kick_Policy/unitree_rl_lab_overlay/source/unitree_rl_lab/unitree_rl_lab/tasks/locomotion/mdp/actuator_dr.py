# actuator_dr.py — 구동기 게인 DR (kp·kd 같은 배율)
# ============================================================================
# 배경 (2026-10-28, RS01 V0-5): 실기에서 발목(RS00)이 움직일 때 PD 명령보다 토크를 덜 낸다 (느릴 때 0.8~0.9, 2~7 rad/s 0.15~0.6).
#   정지 강성은 설정대로 (홀드에서 토크 센서 = kp × 처짐). 시뮬에서 RS00 kp·kd 를 ×0.65 로 낮추면 실기 '쾅쾅' 걸음
#   (착지 0.37 vs 실기 0.38 m/s, 무릎 범위, 몸통 각, 엉덩이·무릎 지연 40 ms)이 재현됨. 토크 상한·armature·출력 지연은 재현 안 됨.
# 설계: kp 와 kd 를 같은 배율 s ~ U(lo, hi) 로 (토크 부족은 둘 다에 걸리므로). env 별로 리셋마다 다시 뽑는다.
#   IsaacLab randomize_actuator_gains 는 kp·kd 를 따로 뽑아 비율이 흐트러지므로 따로 만든다.
#   명목값은 처음 호출 때 액추에이터에서 복사해 둔다 (리셋마다 명목 × s, 누적 안 됨).
# ============================================================================
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["scale_actuator_gains"]


def scale_actuator_gains(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg,
                         scale_range: tuple[float, float]):
    """asset_cfg.joint_names 관절의 kp·kd 를 명목 × s (s ~ U(scale_range), env 별 하나) 로 설정."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids = env_ids.to(asset.device)
    key = "_actdr_nominal"
    if not hasattr(env, key):
        setattr(env, key, {})
    nominal = getattr(env, key)
    lo, hi = scale_range
    s = torch.rand(len(env_ids), device=asset.device) * (hi - lo) + lo
    target = set(asset_cfg.joint_ids if not isinstance(asset_cfg.joint_ids, slice) else range(asset.num_joints))
    for name, act in asset.actuators.items():
        idx = act.joint_indices
        idx = list(range(asset.num_joints)) if isinstance(idx, slice) else [int(i) for i in idx]
        cols = [k for k, j in enumerate(idx) if j in target]
        if not cols:
            continue
        if name not in nominal:
            nominal[name] = (act.stiffness.clone(), act.damping.clone())
        kp0, kd0 = nominal[name]
        c = torch.tensor(cols, device=asset.device)
        act.stiffness[env_ids.unsqueeze(1), c.unsqueeze(0)] = kp0[env_ids.unsqueeze(1), c.unsqueeze(0)] * s.unsqueeze(1)
        act.damping[env_ids.unsqueeze(1), c.unsqueeze(0)] = kd0[env_ids.unsqueeze(1), c.unsqueeze(0)] * s.unsqueeze(1)
        if isinstance(act, ImplicitActuator):
            jids = [idx[k] for k in cols]
            asset.write_joint_stiffness_to_sim(act.stiffness[env_ids][:, c], joint_ids=jids, env_ids=env_ids)
            asset.write_joint_damping_to_sim(act.damping[env_ids][:, c], joint_ids=jids, env_ids=env_ids)
