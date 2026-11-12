# eval_getup_recovery.py
# ============================================================================
# 기립 후 **방향별 밀림 회복** 능력을 잰다.
#
#   실기 관찰 (2026-09-02): "일어나는 건 아주 좋다. 근데 일어나서 조금만
#   밀리거나 흔들리면 발산한다. 특히 뒤로 쏠리면 복구를 못한다."
#   -> sim 에서 재현되는지, 뒤쪽이 정말 더 약한지 확인한다.
#
#   ★ 임펄스는 **몸통 좌표계** 기준이다 ★
#     기상 리셋이 yaw 를 ±π 로 랜덤화하므로 월드 기준으로 주면
#     "앞/뒤/좌/우" 가 섞인다.
#
#   측정 (방향 x 세기 조건별)
#     fell      측정 창 안에 기립 높이 아래로 떨어진 비율 [%]
#     peak_tilt 임펄스 후 최대 몸통 기울기 [deg]
#     recover   기울기가 회복 문턱 아래로 돌아온 비율 [%]
#     step      한 발이라도 지면에서 떨어진 비율 [%] (발목전략 vs 스텝전략)
#
# 사용:
#   python scripts/rsl_rl/eval_getup_recovery.py --headless --num_envs 512 \
#       --checkpoint logs/.../model_44400.pt
# ============================================================================
"""Launch Isaac Sim Simulator first."""
import argparse, os, sys
sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="기립 후 방향별 밀림 회복")
parser.add_argument("--task", type=str, default="Unitree-MX3-Getup-V1-Play")
parser.add_argument("--robot_usd", type=str, default=None, help="로봇 USD 경로 덮어쓰기 (충돌 모델 비교용)")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--settle_steps", type=int, default=300, help="기립·안정화 스텝")
parser.add_argument("--window_steps", type=int, default=150, help="임펄스 후 측정 스텝(3s)")
parser.add_argument("--mags", type=str, default="0.2,0.4,0.6")
parser.add_argument("--stand_h", type=float, default=0.35)
parser.add_argument("--settle_deg", type=float, default=8.0, help="회복 판정 기울기")
parser.add_argument("--fall_angle", type=float, default=90.0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
import unitree_rl_lab.tasks  # noqa: F401, E402

_DIRS = [("전방", (1.0, 0.0)), ("후방", (-1.0, 0.0)),
         ("좌", (0.0, 1.0)), ("우", (0.0, -1.0))]


def _obs(env):
    o = env.get_observations()
    return o[0] if version("rsl-rl-lib").startswith("2.3.") else o


def _tilt_deg(robot):
    gz = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos((-gz).clamp(-1.0, 1.0)))


def main():
    if args_cli.checkpoint is None:
        raise SystemExit("--checkpoint 필요")
    mags = [float(x) for x in args_cli.mags.split(",")]
    n_cond = len(_DIRS) * len(mags)

    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    if args_cli.robot_usd:
        cfg.scene.robot.spawn.usd_path = args_cli.robot_usd
    cfg.events.reset_base_supine.params["fall_angle_deg"] = args_cli.fall_angle
    # 내장 외란은 전부 끈다 — 우리가 주는 임펄스만 본다
    for name in ("push_robot", "push_knockdown", "push_force"):
        if getattr(cfg.events, name, None) is not None:
            setattr(cfg.events, name, None)
    if getattr(cfg.curriculum, "disturbance_rampin", None) is not None:
        cfg.curriculum.disturbance_rampin = None
    cfg.episode_length_s = 1.0e6

    env = gym.make(args_cli.task, cfg=cfg); env = RslRlVecEnvWrapper(env)
    ag = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, ag.to_dict(), log_dir=None, device=ag.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped; dev = u.device; robot = u.scene["robot"]; N = u.num_envs
    cs = u.scene["contact_forces"]
    feet = [i for i, n in enumerate(cs.body_names) if "ankle_roll" in n]

    cond = torch.arange(N, device=dev) % n_cond
    di, mi = cond // len(mags), cond % len(mags)
    dir_b = torch.tensor([d[1] for d in _DIRS], device=dev, dtype=torch.float32)[di]
    mag_t = torch.tensor(mags, device=dev, dtype=torch.float32)[mi]

    obs = _obs(env)
    print(f"[eval] {args_cli.task} envs={N} 조건={n_cond} (방향 4 x 세기 {len(mags)}) "
          f"조건당 ~{N//n_cond} env")
    for _ in range(args_cli.settle_steps):
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]

    h0 = robot.data.root_pos_w[:, 2] - u.scene.env_origins[:, 2]
    up0 = h0 > args_cli.stand_h
    print(f"[eval] 임펄스 직전 기립 유지: {int(up0.sum())}/{N} ({up0.float().mean()*100:.1f}%)")

    with torch.inference_mode():
        d3 = torch.zeros(N, 3, device=dev); d3[:, :2] = dir_b
        dw = quat_apply(robot.data.root_quat_w, d3)
        vel = robot.data.root_vel_w.clone()
        vel[:, :2] += dw[:, :2] * mag_t.unsqueeze(-1)
        robot.write_root_velocity_to_sim(vel)

    peak = torch.zeros(N, device=dev)
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    stepped = torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(args_cli.window_steps):
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]
            peak = torch.maximum(peak, _tilt_deg(robot))
            h = robot.data.root_pos_w[:, 2] - u.scene.env_origins[:, 2]
            fell |= h < args_cli.stand_h
            if feet:
                c = cs.data.net_forces_w[:, feet, :].norm(dim=-1) > 1.0
                stepped |= ~c.all(dim=1)
    final_tilt = _tilt_deg(robot)
    recovered = (final_tilt < args_cli.settle_deg) & ~fell

    print("")
    print("=" * 74)
    print(f"  {os.path.basename(args_cli.checkpoint)}   임펄스 후 {args_cli.window_steps*u.step_dt:.1f}s 관찰")
    print("=" * 74)
    print(f"  {'방향':>5s} {'세기':>6s} {'넘어짐':>8s} {'회복':>8s} {'최대기울기':>10s} {'발뗌':>8s}")
    print("  " + "-" * 60)
    for i, (dn, _) in enumerate(_DIRS):
        for j, m in enumerate(mags):
            k = (di == i) & (mi == j) & up0
            if not k.any(): continue
            print(f"  {dn:>5s} {m:6.2f} {fell[k].float().mean()*100:7.1f}% "
                  f"{recovered[k].float().mean()*100:7.1f}% {peak[k].mean():9.1f}° "
                  f"{stepped[k].float().mean()*100:7.1f}%")
    print("=" * 74)
    os._exit(0)


if __name__ == "__main__":
    main()
