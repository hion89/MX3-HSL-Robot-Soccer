"""v1_5b 킥 세기 눈으로 보기 (Isaac GUI) — 세기 s 별로 로봇을 나란히 세워 계속 차게 한다.

env 번호 순서대로 s = 0 / 0.25 / 0.5 / 0.75 / 1.0. 로봇 머리 위 공 = 세기 표시 (작은 파랑 = 0 → 큰 빨강 = 1).
공 정상 (믿음 오차 없음), 외란 끔, 넘어져도 리셋 안 함, --seconds 마다 다시 시작. 평지 (env 간격 --spacing).
킥마다 콘솔에 목표방향 최고 공속·목표·방향 오차를 찍는다 (eval_strength_isaac.py 와 같은 정의).

  conda activate isaac_env && cd ~/unitree_rl_lab
  python scripts/kick_v1_5/view_strength.py --checkpoint <model.pt> [--bank] [--levels 0,0.5,1]
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-MX3-WalkKick-V1-5B-Play")
parser.add_argument("--levels", type=str, default="0,0.25,0.5,0.75,1")
parser.add_argument("--bank", action="store_true", help="넘겨받기 상태(걷던 자세)에서 시작 (기본은 서서)")
parser.add_argument("--seconds", type=float, default=5.0, help="한 판 길이 [s]")
parser.add_argument("--spacing", type=float, default=3.0, help="로봇 간격 [m]")
parser.add_argument("--lo", type=float, default=None, help="목표 계산 하한 (기본 STR_LO)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.locomotion.robots.MX3 import walk_kick_v1_5_env as v15  # noqa: E402


def _obs(env):
    out = env.get_observations()
    return out[0] if version("rsl-rl-lib").startswith("2.3.") else out


def _strength_markers(levels):
    """세기마다 크기·색이 다른 공 (파랑 → 빨강)."""
    protos = {}
    for i, lv in enumerate(levels):
        c = (0.15 + 0.85 * lv, 0.35 * (1 - abs(2 * lv - 1)) + 0.15, 1.0 - 0.85 * lv)
        protos[f"s{i}"] = sim_utils.SphereCfg(radius=0.03 + 0.05 * lv,
                                              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c))
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/KickStrength", markers=protos))


def main():
    levels = tuple(float(x) for x in args_cli.levels.split(","))
    n = len(levels)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    # 평지 격자 — 지형 생성기는 env 를 타일 원점에 흩어 놓아 한 화면에 안 들어온다
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.env_spacing = args_cli.spacing
    if getattr(cfg, "curriculum", None) is not None:
        for k in list(vars(cfg.curriculum)):
            setattr(cfg.curriculum, k, None)
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
    cfg.episode_length_s = args_cli.seconds
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg, render_mode=None))
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    robot, ball = u.scene["robot"], u.scene["ball"]
    dev = u.device

    org = u.scene.env_origins
    c = org.mean(0).tolist()
    span = float((org[:, :2].max(0).values - org[:, :2].min(0).values).norm()) + 2.0
    u.sim.set_camera_view(eye=[c[0] - 0.35 * span, c[1] - 0.9 * span, 0.7 * span], target=[c[0], c[1], 0.2])
    markers = _strength_markers(levels)
    idx = torch.arange(n, device=dev)
    lo = v15.STR_LO if args_cli.lo is None else args_cli.lo
    print("[view] 로봇 위치 (월드 xy) → 세기:", flush=True)
    for i in range(n):
        print(f"[view]   ({org[i, 0]:+5.1f}, {org[i, 1]:+5.1f})  s = {levels[i]:.2f}  "
              f"목표 {v15.STR_V_FULL * (lo + (1 - lo) * levels[i]):.2f} m/s", flush=True)

    obs = _obs(env)
    t_ep = torch.zeros(n, device=dev)
    t_kick = torch.full((n,), -1.0, device=dev)
    peak = torch.zeros(n, device=dev)
    dir_err = torch.zeros(n, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    reported = torch.zeros(n, dtype=torch.bool, device=dev)
    dt = u.step_dt
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]
        t_ep += dt
        v = ball.data.root_lin_vel_w[:, :2]
        sp = torch.norm(v, dim=-1)
        along = (v * u.kick_target_dir_w[:, :2]).sum(-1)
        start = (t_kick < 0) & (sp > 1.0)
        t_kick = torch.where(start, t_ep, t_kick)
        win = (t_kick >= 0) & (t_ep - t_kick <= 0.5)
        better = win & (along > peak)
        peak = torch.where(better, along, peak)
        ang = torch.rad2deg(torch.acos(torch.clamp(along / (sp + 1e-9), -1, 1)))
        dir_err = torch.where(better, ang, dir_err)
        g = robot.data.projected_gravity_b[:, 2].clamp(-1, 1)
        fell |= torch.rad2deg(torch.acos(-g)) > 45
        done = (t_kick >= 0) & (t_ep - t_kick > 1.5) & ~reported
        for i in done.nonzero().flatten().tolist():
            tgt = v15.STR_V_FULL * (lo + (1 - lo) * levels[i])
            print(f"[view] s {levels[i]:.2f}  공속 {peak[i]:.2f} m/s (목표 {tgt:.2f}, {100 * (peak[i] / tgt - 1):+.0f} %)  "
                  f"방향 오차 {dir_err[i]:4.1f}°{'  넘어짐' if fell[i] else ''}", flush=True)
        reported |= done
        rst = u.episode_length_buf == 0
        if bool(rst.any()):
            for i in (rst & ~reported).nonzero().flatten().tolist():
                print(f"[view] s {levels[i]:.2f}  못 참{'  넘어짐' if fell[i] else ''}", flush=True)
            t_ep[rst] = 0.0
            t_kick[rst] = -1.0
            peak[rst] = 0.0
            fell[rst] = False
            reported[rst] = False
        pos = robot.data.root_pos_w.clone()
        pos[:, 2] += 0.35
        markers.visualize(translations=pos, marker_indices=idx)
        slp = dt - (time.time() - t0)
        if slp > 0:
            time.sleep(slp)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
