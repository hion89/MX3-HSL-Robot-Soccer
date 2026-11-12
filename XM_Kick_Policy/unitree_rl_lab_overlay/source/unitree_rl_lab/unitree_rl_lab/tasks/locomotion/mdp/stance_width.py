# stance_width.py
# ============================================================================
# 서 있을 때 두 발 간격 벌점 (rs_bot RS02 보행 V0-3 에서 도입, 로봇 무관)
#
# 배경 (2026-11-12, RS02 걷기 V0-2 model_4600 실기 robo01 · MuJoCo)
#   오른쪽 옆걸음 뒤 멈추면 두 발이 모여 다리가 V 자 (발목 간격 실기 20.5 · MuJoCo 18.9 cm, 기본자세 23.0 = 발이 hip roll 바로 아래).
#   정지가 늘 같은 위상에서 끝나 마지막 디딤이 늘 같은 발이고, 지금 보상엔 발 간격 항이 없다
#   (hip roll L1 · 정지 관절 기본자세 제곱뿐 — hip roll 3° 면 0.003). Isaac 옆 밀기에서도 16.4 cm 까지 모임.
#   사용자: "서 있는 상태에서만 적용" — 걷는 중 한 발 지지의 안쪽 기울기 (실기 걷는 시간 40 %) 는 균형에 필요해 건드리지 않는다.
#
# 벌점 = max(0, min_width − 발 간격) [m], 완전 정지 (걸음 시계 멈춤 · 위상 수렴 끝) 이고 밀림 회복 중이 아닐 때만.
#   발 간격 = 왼발 − 오른발 (ankle_roll_link 원점) 의 좌우 거리, 골반 heading (yaw 만) 좌표 — 발이 바닥에 붙은 채로는
#   골반을 옮기거나 기울여도 안 바뀌므로 피할 길은 ① 멈추기 전 마지막 디딤을 넓게 ② 멈춘 뒤 한 걸음 고쳐 딛기 둘뿐.
#   게이트는 정책이 조작할 수 없는 양 (명령에서 온 걸음 시계 · 외란 타이머) 으로만 건다.
#   측정 도구 scripts/rsl_rl/rs02_eval/real/stand_width.py 와 같은 정의 (기본자세 23.0 cm).
# ============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .push_recovery import _recovering

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["stand_feet_width_short"]


def stand_feet_width_short(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_width: float = 0.23,
    gait_command_name: str = "gait",
    hold_s: float = 1.5,
) -> torch.Tensor:
    """완전 정지 중 두 발 좌우 간격이 min_width 보다 좁은 만큼 [m] (asset_cfg.body_ids = 왼발, 오른발 순서). 음수 weight."""
    term = env.command_manager.get_term(gait_command_name)
    active = (~term.is_walking) & (~term.is_stopping) & (~_recovering(env, hold_s))
    asset = env.scene[asset_cfg.name]
    rel = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, None, :]
    q = yaw_quat(asset.data.root_quat_w)[:, None, :].expand(-1, rel.shape[1], -1)
    rel_h = quat_apply_inverse(q.reshape(-1, 4), rel.reshape(-1, 3)).reshape(rel.shape)
    width = rel_h[:, 0, 1] - rel_h[:, 1, 1]
    return (min_width - width).clamp(min=0.0) * active.float()
