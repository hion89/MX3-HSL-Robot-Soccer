"""구동기 관련 이벤트 (2026-11-06) — 실기 구동기 차이를 학습 시뮬에 넣되, 내보내는 deploy.yaml 게인은 설정값 그대로 두기.

scale_joint_gains_startup: 시작할 때 한 번, 지정 관절의 PhysX PD 게인만 kp_scale · kd_scale 배로 바꾼다.
  export_deploy_cfg 는 asset.data.default_joint_stiffness/damping (설정값) 을 내보내므로 실기 노드는 설정 게인 (팔 20 / 0.5) 을 그대로 보낸다
  — 실기 팔은 같은 명령에 힘이 절반쯤만 나오므로 (robo01 11-06 실측), 학습 시뮬만 그만큼 약하게 해서 정책이 그 팔에 맞춰 목표를 내게 한다.
  암시 액추에이터는 게인을 매 스텝 다시 쓰지 않아 한 번이면 된다 (게인 랜덤화 이벤트가 있으면 그 뒤에 덮어쓰니 같이 쓰지 말 것).
"""
from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


def scale_joint_gains_startup(env, env_ids, asset_cfg: SceneEntityCfg, kp_scale: float = 1.0, kd_scale: float = 1.0) -> None:
    asset = env.scene[asset_cfg.name]
    jid = asset_cfg.joint_ids
    if isinstance(jid, slice):
        jid = torch.arange(asset.num_joints, device=env.device)[jid]
    jid = torch.as_tensor(jid, device=env.device, dtype=torch.long)
    kp = asset.data.default_joint_stiffness[:, jid] * kp_scale
    kd = asset.data.default_joint_damping[:, jid] * kd_scale
    asset.write_joint_stiffness_to_sim(kp, joint_ids=jid)
    asset.write_joint_damping_to_sim(kd, joint_ids=jid)
    names = [asset.joint_names[int(i)] for i in jid]
    print(f"[scale_joint_gains_startup] {names}: kp ×{kp_scale:g} (→ {float(kp[0, 0]):.2f}), kd ×{kd_scale:g} (→ {float(kd[0, 0]):.3f}) — "
          f"deploy 게인은 설정값 그대로", flush=True)
