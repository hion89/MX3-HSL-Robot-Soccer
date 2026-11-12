"""walk_kick v1_5 넘겨받기 상태 은행 — 최종 보행 (v0_6 13500) 이 킥을 넘겨줄 때의 상태를 모은다.

경기에서 행동 노드는 정렬 중 (옆걸음·회전) 킥 영역에 들어오는 순간 명령 0 을 보내고 (파라미터 응답 대기,
약 0.04 s) 킥 러너로 넘긴다. 그 순간의 관절·몸통 상태를 Isaac 에서 그대로 만든다:
  보행 정책 (v0_6, 실기와 같이 걸음 2.0 Hz 고정) 을 v1_5 로봇 (convexhull + 팔 내림) 에 올리고,
  명령 = 경기 실측 (kick_handoff_probe.py 84 건) 을 덮는 분포, hsl CommandShaper 처럼 가속 제한·최소 노름 0.12.
  1.5~3.0 s 걸은 뒤 명령 0 을 2 스텝 → 기록 → 새 명령으로 계속.

  conda activate isaac_env && cd ~/unitree_rl_lab
  python scripts/kick_v1_5/gen_handoff_bank.py --headless --num_envs 4096 --count 120000 \
      --checkpoint logs/rsl_rl/unitree_mx3_velocity_v0_6/2026-08-23_14-45-11/model_13500.pt
결과: source/.../robots/MX3/data/handoff_bank_v0_6_13500.pt (walk_kick_v1_5_env.HANDOFF_BANK)
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-MX3-Velocity-V0-6")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--count", type=int, default=120000, help="모을 상태 수")
parser.add_argument("--out", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402
from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.assets.robots.mx_bot import MX_BOT_KICK_CFG  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "source", "unitree_rl_lab",
                           "unitree_rl_lab", "tasks", "locomotion", "robots", "MX3", "data",
                           "handoff_bank_v0_6_13500.pt")
MIN_NORM = 0.12                 # hsl WalkLimits.min_norm
ACC_LIN, ACC_ANG = 0.6, 2.5     # hsl WalkLimits.accel_lin / accel_ang
ARM_STEPS = 2                   # 명령 0 (0.04 s) 뒤 기록


def _obs(env):
    out = env.get_observations()
    return out[0] if version("rsl-rl-lib").startswith("2.3.") else out


def sample_cmd(n, dev):
    """경기 넘겨받기 직전 명령 분포 (실측 몸통 속도 5~95 %: vx −0.14~0.22, vy ±0.2, wz ±0.93)."""
    kind = torch.multinomial(torch.tensor([0.25, 0.45, 0.15, 0.15], device=dev), n, replacement=True)
    u = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)
    cmd = torch.zeros(n, 3, device=dev)
    approach = torch.stack([u(0.10, 0.30), u(-0.05, 0.05), u(-0.3, 0.3)], -1)
    align = torch.stack([u(-0.15, 0.20), u(-0.25, 0.25), u(-1.0, 1.0)], -1)
    sgn = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    turn = torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev), sgn * u(0.4, 1.0)], -1)
    creep = torch.stack([u(-0.10, 0.15), u(-0.10, 0.10), u(-0.3, 0.3)], -1)
    for i, c in enumerate((approach, align, turn, creep)):
        cmd[kind == i] = c[kind == i]
    nrm = cmd.norm(dim=-1, keepdim=True)
    small = (nrm > 1e-6) & (nrm < MIN_NORM)
    return torch.where(small, cmd * MIN_NORM / nrm.clamp(min=1e-6), cmd)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.scene.robot = MX_BOT_KICK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    bv = env_cfg.commands.base_velocity
    bv.resampling_time_range = (1.0e9, 1.0e9)
    bv.rel_standing_envs = 0.0
    bv.heading_command = False
    env_cfg.commands.gait.frequency_range = None          # 실기 러너 gait_frequency 2.0 고정
    env_cfg.commands.gait.frequency = 2.0
    env_cfg.commands.gait.rel_stop_when_standing = 1.0    # 명령 0 = 정지 (러너와 같음)
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.curriculum.ang_vel_cmd_levels = None
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 1.0e6

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg, render_mode=None))
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    dev, n, dt = u.device, u.num_envs, u.step_dt
    robot = u.scene["robot"]
    vt = u.command_manager.get_term("base_velocity")
    target = sample_cmd(n, dev)
    cmd = torch.zeros(n, 3, device=dev)
    walk_left = torch.empty(n, device=dev).uniform_(1.5, 3.0)
    arm_left = torch.zeros(n, dtype=torch.long, device=dev)     # >0: 명령 0 을 남은 스텝만큼 더 보낸다
    keys = ("joint_pos", "joint_vel", "root_z", "root_roll", "root_pitch", "lin_vel_h", "ang_vel_h", "cmd")
    rec = {k: [] for k in keys}
    got, steps, fell = 0, 0, 0
    obs = _obs(env)
    step = torch.tensor([ACC_LIN * dt, ACC_LIN * dt, ACC_ANG * dt], device=dev)
    while got < args_cli.count and simulation_app.is_running():
        # 가속 제한 (hsl CommandShaper). 넘겨주기 직전 명령 0 은 즉시
        stop = arm_left > 0
        cmd = torch.where(stop.unsqueeze(-1), torch.zeros_like(cmd), cmd + torch.clamp(target - cmd, -step, step))
        vt.vel_command_b[:] = cmd
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        dones = dones.bool()                                 # 래퍼는 long 을 준다 — 마스크로 쓰려면 bool
        steps += 1
        walk_left -= dt
        arm_left[stop] -= 1
        done_arm = stop & (arm_left == 0) & ~dones
        if dones.any():                                      # 넘어져 리셋된 env: 처음부터 다시
            ids = dones.nonzero(as_tuple=False).squeeze(-1)
            fell += len(ids)
            cmd[ids] = 0.0
            walk_left[ids] = torch.empty(len(ids), device=dev).uniform_(2.0, 3.5)
            arm_left[ids] = 0
            target[ids] = sample_cmd(len(ids), dev)
        if done_arm.any():
            idx = done_arm.nonzero(as_tuple=False).squeeze(-1)
            q = robot.data.root_quat_w[idx]
            roll, pitch, yaw = euler_xyz_from_quat(q)
            tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[idx, 2]).clamp(-1, 1)))
            z = robot.data.root_pos_w[idx, 2] - u.scene.env_origins[idx, 2]
            ok = (tilt < 20.0) & (z > 0.33)
            ids, roll, pitch, yaw, z = idx[ok], roll[ok], pitch[ok], yaw[ok], z[ok]
            c, s = torch.cos(yaw), torch.sin(yaw)

            def to_h(v):     # 월드 → heading 프레임
                return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)

            rec["joint_pos"].append(robot.data.joint_pos[ids].cpu())
            rec["joint_vel"].append(robot.data.joint_vel[ids].cpu())
            rec["root_z"].append(z.cpu())
            rec["root_roll"].append(roll.cpu())
            rec["root_pitch"].append(pitch.cpu())
            rec["lin_vel_h"].append(to_h(robot.data.root_lin_vel_w[ids]).cpu())
            rec["ang_vel_h"].append(to_h(robot.data.root_ang_vel_w[ids]).cpu())
            rec["cmd"].append(target[ids].cpu())
            got += len(ids)
            # 정지 상태에서 새 명령으로 다시 출발 (shaper 가속 제한)
            walk_left[idx] = torch.empty(len(idx), device=dev).uniform_(1.5, 3.0)
            target[idx] = sample_cmd(len(idx), dev)
        start_arm = (arm_left == 0) & (walk_left <= 0) & ~done_arm
        arm_left[start_arm] = ARM_STEPS
        if steps % 250 == 0:
            print(f"[bank] step {steps}  모음 {got}/{args_cli.count}  넘어짐 리셋 {fell}", flush=True)

    bank = {k: torch.cat(v)[: args_cli.count] for k, v in rec.items()}
    bank["joint_names"] = list(robot.joint_names)
    bank["meta"] = dict(task=args_cli.task, checkpoint=args_cli.checkpoint, robot="XM_robot04_convexhull_kick",
                        arm_steps=ARM_STEPS, steps=steps, fell=fell)
    out = os.path.abspath(args_cli.out or DEFAULT_OUT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(bank, out)
    lv, av = bank["lin_vel_h"], bank["ang_vel_h"]
    q = lambda t: [round(float(x), 3) for x in torch.quantile(t, torch.tensor([0.05, 0.5, 0.95]))]
    print(f"[bank] {len(lv)} 상태 -> {out}", flush=True)
    print(f"[bank] vx 5/50/95% {q(lv[:, 0])}  vy {q(lv[:, 1])}  wz {q(av[:, 2])}  넘어짐 리셋 {fell}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
