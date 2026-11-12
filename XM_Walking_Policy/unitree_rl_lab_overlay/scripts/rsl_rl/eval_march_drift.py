# eval_march_drift.py
# ============================================================================
# 제자리걸음 횡드리프트 정량 평가.
#
#   실기에서 "제자리걸음 시 왼쪽으로 조금씩 간다" 는 증상이 관측됐다(2026-08-24).
#   sim 에서도 같은 편향이 있는지 재는 스크립트다.
#
#   ★ 반드시 **몸통 좌표계** 로 재야 한다. reset_base 가 yaw 를 -pi~pi 로
#     랜덤 초기화하므로 월드 좌표계로 재면 좌우 편향이 평균 0 으로 상쇄된다.
#
#   측정:
#     vy_body  몸통 y 속도 [m/s]  (+ 가 좌측). 대칭 정책이면 평균 0 이어야 한다
#     vx_body  몸통 x 속도 [m/s]  (+ 가 전방). 앞뒤 쏠림 확인용
#     left_ratio  vy > 0 인 샘플 비율. 대칭이면 0.50
#
#   제자리걸음 강제:
#     base_velocity ranges 를 전부 (0,0) 으로 두면 gait term 이 정지로 판정하고,
#     rel_stop_when_standing=0.0 이면 그중 완전정지가 0% 라 전부 제자리걸음이 된다.
#
# 사용:
#   python scripts/rsl_rl/eval_march_drift.py \
#       --task Unitree-MX3-Velocity-V0-6 --num_envs 512 --steps 500 --headless \
#       --checkpoint logs/.../model_1600.pt
# ============================================================================
"""Launch Isaac Sim Simulator first."""
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="제자리걸음 횡드리프트 평가")
parser.add_argument("--task", type=str, default="Unitree-MX3-Velocity-V0-6")
parser.add_argument("--robot_usd", type=str, default=None, help="로봇 USD 경로 덮어쓰기 (충돌 모델 비교용)")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--warmup", type=int, default=50, help="초기 과도구간 제외 스텝")
parser.add_argument("--no_noise", action="store_true", default=False,
                    help="관측 노이즈를 끄고 잰다(노이즈가 원인인지 분리)")
parser.add_argument("--disable_fabric", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402


def _get_obs(env):
    out = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        return out[0]
    return out


def main():
    if args_cli.checkpoint is None:
        raise SystemExit("--checkpoint 필요")

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if args_cli.robot_usd:
        env_cfg.scene.robot.spawn.usd_path = args_cli.robot_usd

    # ── 제자리걸음 강제 ────────────────────────────────────────────────
    bv = env_cfg.commands.base_velocity
    bv.ranges.lin_vel_x = (0.0, 0.0)
    bv.ranges.lin_vel_y = (0.0, 0.0)
    bv.ranges.ang_vel_z = (0.0, 0.0)
    bv.rel_standing_envs = 0.0          # 명령 자체가 0 이라 무관하지만 명시
    env_cfg.commands.gait.rel_stop_when_standing = 0.0   # 완전정지 0% → 전부 제자리걸음
    # 커리큘럼이 범위를 다시 넓히지 못하게 끈다
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.curriculum.ang_vel_cmd_levels = None

    if args_cli.no_noise:
        env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    dev = uenv.device
    robot = uenv.scene["robot"]

    z = lambda: torch.zeros((), dtype=torch.float64, device=dev)  # noqa: E731
    vy_sum, vx_sum, n = z(), z(), z()
    vy_abs_sum = z()
    left_n = z()
    freq_sum, freq_n = z(), z()

    obs = _get_obs(env)
    print(f"[eval] task={args_cli.task}  envs={args_cli.num_envs}  "
          f"steps={args_cli.steps}  noise={'OFF' if args_cli.no_noise else 'ON'}")

    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            out = env.step(actions)
            obs = out[0]

            if step >= args_cli.warmup:
                v = robot.data.root_lin_vel_b        # (N,3) 몸통 좌표계
                vy, vx = v[:, 1].double(), v[:, 0].double()
                vy_sum += vy.sum(); vx_sum += vx.sum()
                vy_abs_sum += vy.abs().sum()
                left_n += (vy > 0).double().sum()
                n += vy.numel()
                gt = uenv.command_manager.get_term("gait")
                freq_sum += gt.effective_frequency.double().sum()
                freq_n += gt.effective_frequency.numel()

    vy_m = (vy_sum / n).item(); vx_m = (vx_sum / n).item()
    vy_a = (vy_abs_sum / n).item(); lr = (left_n / n).item()
    fr = (freq_sum / freq_n).item()

    print("\n" + "=" * 68)
    print(f"  제자리걸음 횡드리프트  ({int(n.item()):,} 샘플, warmup {args_cli.warmup} 제외)")
    print("=" * 68)
    print(f"  vy_body 평균 : {vy_m:+.5f} m/s   ← + 가 좌측. 대칭이면 0")
    print(f"  vx_body 평균 : {vx_m:+.5f} m/s   ← + 가 전방")
    print(f"  |vy| 평균    : {vy_a:.5f} m/s    ← 좌우 흔들림 크기(편향 아님)")
    print(f"  좌측 비율    : {lr*100:.2f} %      ← 대칭이면 50.00")
    print(f"  평균 주파수  : {fr:.3f} Hz        ← 0 이면 제자리걸음이 아님!")
    print("-" * 68)
    bias = abs(vy_m) / max(vy_a, 1e-9)
    print(f"  편향 비율    : {bias*100:.1f} %   (|평균| / 평균절대. 0 이면 순수 흔들림)")
    if abs(vy_m) < 0.005:
        print("  판정: 유의한 횡드리프트 없음 (sim 은 정상)")
    else:
        d = "좌측" if vy_m > 0 else "우측"
        print(f"  판정: {d}으로 {abs(vy_m)*100:.2f} cm/s 드리프트 — 10 초에 "
              f"{abs(vy_m)*10*100:.1f} cm")
    print("=" * 68)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
