"""v1_5 한 장면 궤적 (Isaac) — MuJoCo kick_match_cond.py trace 와 같은 열로 npz 저장, 두 시뮬 비교용.

무작위화·관측 잡음 전부 끔, 지연 1 스텝, Profile Velocity 45 rpm, 팔 토크 0.5 N·m, 마찰 1.0 (MuJoCo 브리지 물리와 같게).
서서 시작, 홀드 0.3 s, 명령 vx 0.3 (러너와 같음).

  conda activate isaac_env && cd ~/unitree_rl_lab
  python scripts/kick_v1_5/trace_isaac.py --headless --checkpoint <model.pt> --x 0.41 --y 0.17 --target -9 --kind whiff --out t.npz
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-MX3-WalkKick-V1-5-Play")
parser.add_argument("--x", type=float, default=0.41)
parser.add_argument("--y", type=float, default=0.17)
parser.add_argument("--target", type=float, default=-9.0, help="[deg]")
parser.add_argument("--kind", type=str, default="whiff")
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--out", type=str, required=True)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.assets.robots.mx_bot import MX_ARM_JOINTS, MX_LEG_JOINTS  # noqa: E402


def _obs(env):
    out = env.get_observations()
    return out[0] if version("rsl-rl-lib").startswith("2.3.") else out


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    cfg.observations.policy.enable_corruption = False
    ev = cfg.events
    ev.physics_material.params.update(static_friction_range=(1.0, 1.0), dynamic_friction_range=(1.0, 1.0),
                                      restitution_range=(0.0, 0.0))
    for k in ("push_robot", "push_on_kick", "base_external_force_torque", "randomize_base_com", "add_base_mass",
              "randomize_actuator_gains", "randomize_joint_params", "duel_knock"):
        setattr(ev, k, None)
    ev.reset_imu_bias.params.update(grav_bias_range=0.0, gyro_bias_range=0.0)
    ev.arm_effort_limit.params["effort_range"] = (0.5, 0.5)
    ev.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ev.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    ev.reset_robot_joints.params.update(standing_prob=1.0, position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    ev.reset_joints_noise.params["noise_range"] = (0.0, 0.0)
    d = math.hypot(args_cli.x, args_cli.y)
    b = math.atan2(args_cli.y, args_cli.x)
    ev.reset_ball.params.update(radius_range=(d, d), front_angle_range=(b, b))
    tg = math.radians(args_cli.target)
    ev.reset_target.params["target_angle_range"] = (tg, tg)
    probs = {k: 0.0 for k in ("nominal", "bias", "roll", "whiff", "duel")}
    probs[args_cli.kind] = 1.0
    ev.ball_belief.params["probs"] = probs
    a = cfg.actions.JointPositionAction
    a.action_delay_range = (1, 1)
    a.profile_rpm_range = (45.0, 45.0)
    t = cfg.terminations
    for k in list(vars(t)):
        if k != "time_out":
            setattr(t, k, None)
    cfg.episode_length_s = args_cli.seconds + 2.0

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg, render_mode=None))
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    robot = u.scene["robot"]
    sensor = u.scene["contact_forces"]
    leg, _ = robot.find_joints(MX_LEG_JOINTS, preserve_order=True)
    arm, _ = robot.find_joints(MX_ARM_JOINTS, preserve_order=True)
    feet_b, _ = robot.find_bodies(["r_ankle_roll_link", "l_ankle_roll_link"], preserve_order=True)
    feet_s, _ = sensor.find_bodies(["r_ankle_roll_link", "l_ankle_roll_link"], preserve_order=True)   # 센서 순서는 따로
    obs = _obs(env)
    print(f"[trace] frozen ball {u.ball_pos_b_frozen[0, :2].tolist()}  "
          f"실제 공 {(u.scene['ball'].data.root_pos_w[0, :2] - robot.data.root_pos_w[0, :2]).tolist()}", flush=True)
    rec = {k: [] for k in ("t", "base_pos", "base_quat", "lin_vel_w", "ang_vel_b", "grav_b", "q", "qd", "target",
                           "action", "arm_q", "feet_pos", "feet_yaw", "feet_fz")}
    for k in range(int(args_cli.seconds / u.step_dt)):
        with torch.inference_mode():
            act = policy(obs)
            obs = env.step(act)[0]
        qw = robot.data.body_quat_w[0, feet_b]
        fyaw = torch.atan2(2 * (qw[:, 0] * qw[:, 3] + qw[:, 1] * qw[:, 2]), 1 - 2 * (qw[:, 2] ** 2 + qw[:, 3] ** 2))
        vals = dict(t=(k + 1) * u.step_dt, base_pos=robot.data.root_pos_w[0], base_quat=robot.data.root_quat_w[0],
                    lin_vel_w=robot.data.root_lin_vel_w[0], ang_vel_b=robot.data.root_ang_vel_b[0],
                    grav_b=robot.data.projected_gravity_b[0], q=robot.data.joint_pos[0, leg],
                    qd=robot.data.joint_vel[0, leg], target=robot.data.joint_pos_target[0, leg], action=act[0],
                    arm_q=robot.data.joint_pos[0, arm], feet_pos=robot.data.body_pos_w[0, feet_b], feet_yaw=fyaw,
                    feet_fz=sensor.data.net_forces_w[0, feet_s, 2].abs())
        for kk, v in vals.items():
            rec[kk].append(v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(robot.data.projected_gravity_b[0, 2])))))
        if tilt > 70.0:
            break
    np.savez(args_cli.out, **{k: np.array(v) for k, v in rec.items()})
    print(f"[trace] {len(rec['t'])} 스텝 -> {args_cli.out}  마지막 기울기 {tilt:.1f}°", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
