"""v1_5 공 믿음 종류별 넘어짐 — Isaac 에서 한 종류만 강제로 돌려 본다 (MuJoCo kick_match_cond.py 와 대조용).

  conda activate isaac_env && cd ~/unitree_rl_lab
  python scripts/kick_v1_5/eval_belief_isaac.py --headless --checkpoint <model.pt> --kind whiff
한 프로세스에 환경 하나만 만들 수 있다 (두 번째 gym.make 는 출력 없이 멈춘다) → 종류마다 따로 실행.
서서 시작 (standing_prob 1.0), 외란 끔, 종료 조건 끔 (time_out 만) 으로 4 s. 넘어짐 = 기울기 45° (MuJoCo 와 같음).
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-MX3-WalkKick-V1-5")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--kind", type=str, default="whiff", help="nominal / bias / roll / whiff / duel")
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--from_bank", action="store_true", help="넘겨받기 상태에서 시작 (기본은 서서)")
parser.add_argument("--friction", type=float, default=None, help="로봇 마찰 고정 (정지·동 같게). 기본 = 학습 DR 0.45~1.35")
parser.add_argument("--torsion_radius", type=float, default=None,
                    help="로봇 충돌체 torsional_patch_radius [m] — 발 회전(비틀림) 저항. 기본 Isaac 은 발이 쉽게 돈다")
parser.add_argument("--friction_corr", type=float, default=None,
                    help="PhysX friction_correlation_distance [m] (기본 0.025). 키우면 발바닥 마찰 고정점이 합쳐져 비틀림 저항이 준다")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402


def _obs(env):
    out = env.get_observations()
    return out[0] if version("rsl-rl-lib").startswith("2.3.") else out


def run(kind):
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    probs = {k: 0.0 for k in ("nominal", "bias", "roll", "whiff", "duel")}
    probs[kind] = 1.0
    env_cfg.events.ball_belief.params["probs"] = probs
    if args_cli.friction is not None:
        env_cfg.events.physics_material.params.update(static_friction_range=(args_cli.friction, args_cli.friction),
                                                      dynamic_friction_range=(args_cli.friction, args_cli.friction))
    if args_cli.torsion_radius is not None:
        from isaaclab.sim.schemas import CollisionPropertiesCfg
        env_cfg.scene.robot.spawn.collision_props = CollisionPropertiesCfg(
            torsional_patch_radius=args_cli.torsion_radius, min_torsional_patch_radius=args_cli.torsion_radius)
    if args_cli.friction_corr is not None:
        env_cfg.sim.physx.friction_correlation_distance = args_cli.friction_corr
    env_cfg.events.reset_robot_joints.params["standing_prob"] = 0.0 if args_cli.from_bank else 1.0
    for k in ("push_robot", "push_on_kick", "base_external_force_torque"):
        setattr(env_cfg.events, k, None)
    t = env_cfg.terminations
    for k in list(vars(t)):
        if k != "time_out":
            setattr(t, k, None)
    env_cfg.episode_length_s = args_cli.seconds + 1.0
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg, render_mode=None))
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    robot, ball = u.scene["robot"], u.scene["ball"]
    obs = _obs(env)
    d0 = torch.norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    frozen = u.ball_pos_b_frozen[:, :2].norm(dim=-1)
    max_tilt = torch.zeros(u.num_envs, device=u.device)
    kicked = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    max_wz = torch.zeros(u.num_envs, device=u.device)
    peak_sp = torch.zeros(u.num_envs, device=u.device)
    peak_al = torch.zeros(u.num_envs, device=u.device)
    feet = [robot.body_names.index(n) for n in ("r_ankle_roll_link", "l_ankle_roll_link")]
    from isaaclab.utils.math import euler_xyz_from_quat
    yaw_lo = torch.full((u.num_envs, 2), 1e9, device=u.device)
    yaw_hi = torch.full((u.num_envs, 2), -1e9, device=u.device)
    yaw_prev = None
    yaw_acc = torch.zeros(u.num_envs, 2, device=u.device)
    for _ in range(int(args_cli.seconds / u.step_dt)):
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]
        g = robot.data.projected_gravity_b[:, 2].clamp(-1, 1)
        max_tilt = torch.maximum(max_tilt, torch.rad2deg(torch.acos(-g)))
        bsp = torch.norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
        kicked |= bsp > 1.0
        peak_sp = torch.maximum(peak_sp, bsp)
        along = (ball.data.root_lin_vel_w[:, :2] * u.kick_target_dir_w[:, :2]).sum(-1)
        peak_al = torch.maximum(peak_al, along)
        alive = max_tilt <= 45.0
        max_wz = torch.where(alive, torch.maximum(max_wz, robot.data.root_ang_vel_b[:, 2].abs()), max_wz)
        fy = torch.stack([euler_xyz_from_quat(robot.data.body_quat_w[:, f])[2] for f in feet], -1)
        if yaw_prev is not None:     # 발 yaw 누적 (감김 보정)
            yaw_acc += torch.atan2(torch.sin(fy - yaw_prev), torch.cos(fy - yaw_prev))
        yaw_prev = fy
        yaw_lo = torch.minimum(yaw_lo, yaw_acc)
        yaw_hi = torch.maximum(yaw_hi, yaw_acc)
    fell = max_tilt > 45.0
    print(f"[belief] mu={args_cli.friction} tr={args_cli.torsion_radius} fc={env_cfg.sim.physx.friction_correlation_distance} {kind:8s} 넘어짐 {100 * fell.float().mean():5.1f}%  공 찬 비율 {100 * kicked.float().mean():5.1f}%  "
          f"max_tilt 중앙 {max_tilt.median():5.1f}°  몸통 yaw 각속도 최대 중앙 {max_wz.median():.2f} rad/s  "
          f"공속 최대 (찬 판) 평면 {peak_sp[kicked].mean() if kicked.any() else 0:.2f} / 목표방향 {peak_al[kicked].mean() if kicked.any() else 0:.2f} m/s  "
          f"덜 돈 발의 yaw 범위 중앙 {torch.rad2deg((yaw_hi - yaw_lo).min(-1).values).median():.1f}°  실제 공 거리 중앙 {d0.median():.2f} m  frozen 거리 중앙 {frozen.median():.2f} m",
          flush=True)
    env.close()


if __name__ == "__main__":
    run(args_cli.kind)
    simulation_app.close()
