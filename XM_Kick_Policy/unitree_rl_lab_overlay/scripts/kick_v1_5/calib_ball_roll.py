"""v1_5 공 구름 저항 보정 — Isaac 공의 굴림 거리를 MuJoCo 경기 공 (convexhull, rolling 0.030, 감쇠 0.2) 에 맞춘다.

MuJoCo 기준 (scripts/mujoco_sim/ball_roll_calib.py roll 0.030):
  회전 없이 2 / 3 / 4 / 5 m/s → 0.63 / 1.32 / 2.24 / 3.34 m,  구르며 4 m/s → 3.11 m
Isaac: v1_5 환경 그대로 (공 회전 상한 해제 + ball_rolling_resistance), 공 물성은 기본값 고정, 로봇은 서 있기만.

  conda activate isaac_env && cd ~/unitree_rl_lab
  python scripts/kick_v1_5/calib_ball_roll.py --headless --decels 1.6,2.0,2.4,2.8
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-MX3-WalkKick-V1-5")
parser.add_argument("--decels", type=str, default="1.6,2.0,2.4,2.8")
parser.add_argument("--seconds", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402

CASES = [(2.0, False), (3.0, False), (4.0, False), (5.0, False), (4.0, True)]
MUJOCO = [0.63, 1.32, 2.24, 3.34, 3.11]
REPS = 8


def main():
    n = len(CASES) * REPS
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    t = env_cfg.terminations
    for k in list(vars(t)):
        if k != "time_out":
            setattr(t, k, None)
    env_cfg.episode_length_s = 1.0e6
    ev = env_cfg.events
    ev.ball_belief = None
    ev.duel_knock = None
    ev.push_robot = None
    ev.push_on_kick = None
    ev.randomize_ball_material.params.update(static_friction_range=(0.8, 0.8), dynamic_friction_range=(0.6, 0.6),
                                             restitution_range=(0.8, 0.8))
    ev.randomize_ball_mass.params["mass_distribution_params"] = (1.0, 1.0)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    u = env.unwrapped
    env.reset()
    dev = u.device
    ball = u.scene["ball"]
    r = 0.065
    ids = torch.arange(n, device=dev)
    v0 = torch.tensor([c[0] for c in CASES], device=dev).repeat_interleave(REPS)
    spin = torch.tensor([c[1] for c in CASES], device=dev).repeat_interleave(REPS)
    act = torch.zeros(n, u.action_manager.total_action_dim, device=dev)
    print(f"{'decel':>6} | " + " | ".join(f"{c[0]:.0f} m/s{' 굴림' if c[1] else ''}" for c in CASES) + "   (MuJoCo "
          + " / ".join(f"{m:.2f}" for m in MUJOCO) + ")", flush=True)
    for dcl in [float(x) for x in args_cli.decels.split(",")]:
        u.event_manager.get_term_cfg("ball_rolling_resistance").params["decel"] = dcl
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :3] = u.scene.env_origins + torch.tensor([0.0, 1.0, r], device=dev)
        pose[:, 3] = 1.0
        vel = torch.zeros(n, 6, device=dev)
        vel[:, 1] = v0
        vel[:, 3] = torch.where(spin, -v0 / r, torch.zeros_like(v0))     # +y 로 구름: ω_x = −v / r
        ball.write_root_pose_to_sim(pose, env_ids=ids)
        ball.write_root_velocity_to_sim(vel, env_ids=ids)
        p0 = pose[:, :2].clone()
        for _ in range(int(args_cli.seconds / u.step_dt)):
            env.step(act)
        dist = torch.norm(ball.data.root_pos_w[:, :2] - p0, dim=-1).view(len(CASES), REPS).mean(-1)
        print(f"{dcl:6.2f} | " + " | ".join(f"{d:7.2f}" for d in dist.tolist()), flush=True)
    env.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    simulation_app.close()
