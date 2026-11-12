# htwk_wrench.py
# ============================================================================
# 지속형 외란(persistent wrench) DR — "옆에서 툭 밀치는" 힘/토크를 동체에
# 주기적으로 걸었다가 일정 시간 뒤 뗀다. 상대 로봇과의 몸싸움 모사용.
#
# 세 함수가 한 세트로 동작한다.
#   apply_persistent_wrench  : 밀치기 시작 (interval, env별)
#   clear_expired_wrench     : 만료된 밀치기 해제 (interval, 전역)
#   clear_wrench_on_reset    : 리셋 시 즉시 해제 (reset)  ← 에피소드 경계 오염 방지
#   wrench_state_observation : 현재 힘/토크를 critic 에 노출 (privileged, 6차원)
#
# ── 설계 메모 ──
# * IsaacLab 의 set_external_force_and_torque 는 버퍼에 값을 채우고,
#   그 버퍼가 매 스텝 write_data_to_sim 으로 적용된다. 즉 한 번 set 하면
#   다시 바꾸기 전까지 지속(persistent). 그래서 apply 로 걸고 clear 로 0 을
#   써서 뗀다.
# * 힘은 body(local) frame 기준이다 (set_external_force_and_torque 기본).
#   로봇이 회전하면 힘 방향도 같이 돈다 — "막대로 미는" 상황에 가깝다.
#   월드 고정 외란을 원하면 apply 에서 is_global=True 로 바꿀 것.
# * 만료 판정은 common_step_counter(전역 단조증가) 기준이라 에피소드 리셋과
#   무관하게 흐른다. 그래서 clear_wrench_on_reset 이 반드시 필요하다.
# * 수직(z) 성분은 기본 제거한다. 6N 수직이면 소형 MX3 를 들거나 눌러버린다.
#   몸싸움은 수평이 대부분이라 z 를 빼는 게 자연스럽다. 토크는 3축 다 둔다.
# ============================================================================

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_buffers(env: ManagerBasedRLEnv, n_body: int) -> None:
    """외란 버퍼를 지연 초기화한다. [N, n_body, 3] + [N] 만료스텝."""
    if not hasattr(env, "_wrench_force"):
        env._wrench_force = torch.zeros(env.num_envs, n_body, 3, device=env.device)
        env._wrench_torque = torch.zeros(env.num_envs, n_body, 3, device=env.device)
        env._wrench_expire_step = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )


def apply_persistent_wrench(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
    force_std: float = 5.0,
    torque_std: float = 0.5,
    duration_s: float = 0.5,
    zero_vertical: bool = True,
) -> None:
    """지정 env 에 랜덤 힘/토크를 걸고 만료 시각을 기록한다 (interval, env별).

    Args:
        force_std:  힘 표준편차 [N]. |force| 가 대략 이 값 수준이 된다.
        torque_std: 토크 표준편차 [N*m].
        duration_s: 힘 지속 시간 [s]. clear_expired_wrench 가 이 뒤에 뗀다.
        zero_vertical: True 면 수직(z) 힘 제거 (fix #3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    n_body = asset.num_bodies if isinstance(body_ids, slice) else len(body_ids)
    _ensure_buffers(env, n_body)

    n = len(env_ids)
    if n == 0:
        return

    f = torch.randn(n, n_body, 3, device=env.device) * force_std
    if zero_vertical:
        f[:, :, 2] = 0.0
    t = torch.randn(n, n_body, 3, device=env.device) * torque_std

    env._wrench_force[env_ids] = f
    env._wrench_torque[env_ids] = t

    dur_steps = max(1, int(round(duration_s / env.step_dt)))
    env._wrench_expire_step[env_ids] = env.common_step_counter + dur_steps

    # 해당 env 서브셋만 버퍼에 기록 (성능 — 전체 텐서 write 회피)
    asset.set_external_force_and_torque(f, t, body_ids=body_ids, env_ids=env_ids)


def clear_expired_wrench(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
) -> None:
    """만료된 env 의 외란을 0 으로 되돌린다 (interval, is_global_time=True 권장).

    is_global_time=True 로 두면 전역 타이머로 0.05초에 한 번만 전체를 훑어
    성능을 지킨다 (fix #1). env별 타이머(기본)로 두면 매 스텝 발화해 느려진다.
    """
    if not hasattr(env, "_wrench_expire_step"):
        return

    asset: Articulation = env.scene[asset_cfg.name]
    expired = env._wrench_expire_step <= env.common_step_counter
    if not torch.any(expired):
        return

    env._wrench_force[expired] = 0.0
    env._wrench_torque[expired] = 0.0
    # 전역 1회 호출이므로 전체 버퍼를 다시 적용 (활성분은 유지, 만료분은 0)
    asset.set_external_force_and_torque(
        env._wrench_force, env._wrench_torque, body_ids=asset_cfg.body_ids
    )


def clear_wrench_on_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
) -> None:
    """리셋된 env 의 외란을 즉시 해제한다 (mode="reset", fix #2).

    만료 시각이 common_step_counter 기준이라 에피소드 경계와 무관하게 흐른다.
    이 term 이 없으면 리셋 직전에 걸린 힘이 새 에피소드 시작 구간까지 이어져,
    정책이 원인 모를 외란을 리셋 직후마다 겪는다.
    """
    if not hasattr(env, "_wrench_force"):
        return

    env._wrench_expire_step[env_ids] = 0
    env._wrench_force[env_ids] = 0.0
    env._wrench_torque[env_ids] = 0.0

    asset: Articulation = env.scene[asset_cfg.name]
    asset.set_external_force_and_torque(
        env._wrench_force[env_ids],
        env._wrench_torque[env_ids],
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
    )


def wrench_state_observation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
    scale: float = 0.1,
) -> torch.Tensor:
    """현재 걸린 힘(3) + 토크(3) = 6차원. critic 전용 privileged 관측.

    apply 는 base_link 하나에만 걸므로 버퍼 인덱스 0 이 base 다.
    scale 은 힘(N)을 대략 O(1) 로 정규화하기 위한 계수.

    ※ policy 관측에는 넣지 말 것 — 실기에서 관측 불가능한 양이다.
    """
    if not hasattr(env, "_wrench_force"):
        return torch.zeros(env.num_envs, 6, device=env.device)
    f = env._wrench_force[:, 0, :]
    t = env._wrench_torque[:, 0, :]
    return torch.cat([f, t], dim=-1) * scale


def metric_wrench_active(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """외란 활성 비율 진단용 (weight=0). interval(6,10)s + duration 0.5s 면
    6~8% 근처가 정상. 0 이면 이벤트 미발화, 0.5 초과면 clear 미작동."""
    if not hasattr(env, "_wrench_force"):
        return torch.zeros(env.num_envs, device=env.device)
    return (env._wrench_force[:, 0, :2].norm(dim=-1) > 0.1).float()
