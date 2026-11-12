# htwk_observations.py
# ============================================================================
# 크리틱 전용 privileged 관측.
#
# ── 왜 필요한가 ──
# HTWK 의 privileged_obs 14 차원 중 핵심은 base_mass_scaled(4) 인데,
# 이건 단순한 상태가 아니라 "그 환경에 실제로 적용된 도메인 랜덤화 실현값"
# 이다. 크리틱이 "이 env 는 COM 이 +2 cm 밀렸고 질량이 1.13 배" 라는 걸
# 알면, 리턴 차이 중 DR 때문에 생긴 부분을 설명해낼 수 있다.
# 모르면 그게 전부 환원 불가능한 value 분산으로 남아 학습을 느리게 한다.
#
# 비대칭 actor-critic 이 실제로 값어치를 하는 지점이 여기다.
# 질량/COM 랜덤화를 추가하면서 이 관측을 크리틱에 같이 넣지 않으면
# 이득이 절반으로 줄어든다. 반드시 세트로 쓸 것.
#
# ── 캐싱 ──
# 질량/COM 랜덤화는 mode="startup" 이라 에피소드 중에는 변하지 않는다.
# 첫 호출 때 한 번만 읽어 env 에 캐시한다.
#
# ── 안전장치 ──
# root_physx_view API 는 IsaacLab 버전에 따라 세부가 다를 수 있다.
# 실패하면 예외를 던지는 대신 0 텐서를 반환해 학습이 죽지 않게 한다.
# (크리틱 입력 일부가 0 이 되는 것뿐이므로 학습은 계속된다)
# 로그에 경고가 찍히면 IsaacLab 버전에 맞게 _read_body_props 를 고칠 것.
# ============================================================================

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_WARNED = {"props": False}


def _read_body_props(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """지정 body 의 (질량비, COM x, COM y, COM z) 를 읽는다. [N, 4*len(bodies)]"""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids

    view = asset.root_physx_view
    masses = view.get_masses().to(env.device)          # [N, num_bodies]
    coms = view.get_coms().to(env.device)              # [N, num_bodies, 7] (pos+quat)

    default_masses = asset.data.default_mass.to(env.device)

    m = masses[:, ids]
    m0 = torch.clamp(default_masses[:, ids], min=1.0e-6)
    mass_ratio = m / m0                                 # [N, B]

    com_xyz = coms[:, ids, :3]                          # [N, B, 3]

    n_env = mass_ratio.shape[0]
    out = torch.cat([mass_ratio.unsqueeze(-1), com_xyz], dim=-1)  # [N, B, 4]
    return out.reshape(n_env, -1)


def randomized_body_properties(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
) -> torch.Tensor:
    """DR 실현값 (질량비 + COM 위치). 크리틱 관측 전용.

    출력 차원 = 4 * (지정 body 개수). 기본은 base_link 하나 -> 4 차원.

    주의: policy 관측에는 절대 넣지 말 것. 실기에서 관측 불가능한 양이다.
    """
    cache_key = f"_dr_props_{asset_cfg.name}"
    cached = getattr(env, cache_key, None)
    if cached is not None and cached.shape[0] == env.num_envs:
        return cached

    try:
        props = _read_body_props(env, asset_cfg)
    except (AttributeError, RuntimeError, IndexError, TypeError) as exc:
        if not _WARNED["props"]:
            print(
                "[htwk_observations] randomized_body_properties 읽기 실패 -> 0 으로 대체. "
                f"IsaacLab 버전 확인 필요: {exc}"
            )
            _WARNED["props"] = True
        n_bodies = len(asset_cfg.body_ids) if asset_cfg.body_ids is not None else 1
        props = torch.zeros(env.num_envs, 4 * n_bodies, device=env.device)

    setattr(env, cache_key, props)
    return props


def gait_phase_from_command(
    env: ManagerBasedRLEnv,
    gait_command_name: str = "gait",
) -> torch.Tensor:
    """GaitPhaseCommand 의 command 텐서를 그대로 관측으로 노출.

    사실 generated_commands(command_name="gait") 로도 동일한 결과를 얻지만,
    이름이 명시적이라 env_cfg 가독성이 좋아 별도 래퍼를 둔다.

    반환 차원은 GaitPhaseCommandCfg.include_frequency 에 따라 2 또는 3.
    """
    return env.command_manager.get_command(gait_command_name)
