"""v1_5b 킥 세기 평가 (Isaac) — 세기 s 별 목표방향 공속·킥 성공·넘어짐·방향 오차.

env 번호 순서대로 s = 0 / 0.25 / 0.5 / 0.75 / 1.0, 서서 시작, 공 정상 (믿음 오차 없음), 외란 끔, 종료 조건 끔.
공속 = 처음 1.0 m/s 를 넘은 뒤 0.5 s 안의 목표방향 최고 속도. 목표 v*(s) = STR_V_FULL · (STR_LO + (1 − STR_LO)·s).

  conda activate isaac_env && cd ~/unitree_rl_lab
  python scripts/kick_v1_5/eval_strength_isaac.py --headless --checkpoint <model.pt> [--num_envs 500] [--bank]
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-MX3-WalkKick-V1-5B-Play")
parser.add_argument("--num_envs", type=int, default=500)
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--bank", action="store_true", help="넘겨받기 상태에서 시작 (기본은 서서)")
parser.add_argument("--levels", type=str, default="0,0.25,0.5,0.75,1")
parser.add_argument("--lo", type=float, default=None, help="목표 계산 하한 (커리큘럼 중이면 학습 로그의 현재 하한). 기본 STR_LO")
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
from unitree_rl_lab.tasks.locomotion.robots.MX3 import walk_kick_v1_5_env as v15  # noqa: E402


def _obs(env):
    out = env.get_observations()
    return out[0] if version("rsl-rl-lib").startswith("2.3.") else out


def main():
    levels = tuple(float(x) for x in args_cli.levels.split(","))
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    ev = cfg.events
    ev.kick_strength.params["eval_levels"] = levels
    ev.ball_belief.params["probs"] = {"nominal": 1.0, "bias": 0.0, "roll": 0.0, "whiff": 0.0, "duel": 0.0}
    ev.reset_robot_joints.params["standing_prob"] = 0.0 if args_cli.bank else 1.0
    for k in ("push_robot", "push_on_kick", "base_external_force_torque", "duel_knock"):
        setattr(ev, k, None)
    t = cfg.terminations
    for k in list(vars(t)):
        if k != "time_out":
            setattr(t, k, None)
    cfg.episode_length_s = args_cli.seconds + 1.0
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg, render_mode=None))
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    robot, ball = u.scene["robot"], u.scene["ball"]
    n, dev = u.num_envs, u.device
    obs = _obs(env)
    s = u.kick_strength_cmd.clone()
    t_kick = torch.full((n,), -1.0, device=dev)
    peak = torch.zeros(n, device=dev)
    dir_err = torch.zeros(n, device=dev)
    max_tilt = torch.zeros(n, device=dev)
    for k in range(int(args_cli.seconds / u.step_dt)):
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]
        tnow = (k + 1) * u.step_dt
        v = ball.data.root_lin_vel_w[:, :2]
        sp = torch.norm(v, dim=-1)
        tdir = u.kick_target_dir_w[:, :2]
        along = (v * tdir).sum(-1)
        start = (t_kick < 0) & (sp > 1.0)
        t_kick = torch.where(start, torch.full_like(t_kick, tnow), t_kick)
        win = (t_kick >= 0) & (tnow - t_kick <= 0.5)
        better = win & (along > peak)
        peak = torch.where(better, along, peak)
        ang = torch.rad2deg(torch.acos(torch.clamp(along / (sp + 1e-9), -1, 1)))
        dir_err = torch.where(better, ang, dir_err)
        g = robot.data.projected_gravity_b[:, 2].clamp(-1, 1)
        max_tilt = torch.maximum(max_tilt, torch.rad2deg(torch.acos(-g)))
    print(f"[strength] {args_cli.checkpoint}  {'넘겨받기' if args_cli.bank else '서서'} 시작, env {n}, "
          f"목표 하한 {v15.STR_LO if args_cli.lo is None else args_cli.lo}", flush=True)
    strike = getattr(u, "ins_strike_speed", torch.zeros(n, device=dev))
    print(f"[strength] {'s':>5} {'목표':>6} {'공속 평균':>8} {'표준편차':>7} {'목표대비':>7} {'킥 성공':>6} {'넘어짐':>6} {'방향오차':>6} {'타격 발속도':>8}", flush=True)
    for lv in levels:
        m = (s - lv).abs() < 1e-4
        kk = m & (t_kick >= 0)
        lo_ = v15.STR_LO if args_cli.lo is None else args_cli.lo
        tgt = v15.STR_V_FULL * (lo_ + (1 - lo_) * lv)
        pk = peak[kk]
        mean = pk.mean().item() if kk.any() else float("nan")
        std = pk.std().item() if kk.sum() > 1 else float("nan")
        print(f"[strength] {lv:5.2f} {tgt:6.2f} {mean:8.2f} {std:7.2f} {100 * (mean / tgt - 1):+6.0f}% "
              f"{100 * kk.float().sum().item() / max(m.sum().item(), 1):5.0f}% "
              f"{100 * (max_tilt[m] > 45).float().mean().item():5.1f}% {dir_err[kk].mean().item() if kk.any() else float('nan'):6.1f}° "
              f"{strike[kk & (strike > 0)].mean().item() if (kk & (strike > 0)).any() else float('nan'):7.2f}",
              flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
