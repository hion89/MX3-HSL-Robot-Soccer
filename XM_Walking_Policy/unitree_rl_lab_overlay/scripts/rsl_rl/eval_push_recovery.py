# eval_push_recovery.py
# ============================================================================
# 정지 상태 밀림 회복 정량 평가.
#
#   실기 증상: "살짝 밀면 반응이 없다가 10° 넘게 기울어야 균형을 잡으려 한다."
#   (부스터 K1 은 살짝만 밀어도 즉시 반응) — 이걸 sim 에서 숫자로 만든다.
#
#   프로토콜: 완전정지로 안정화 -> 한 번에 속도 임펄스 -> 회복 과정 측정.
#   env 를 (방향 x 세기) 조건으로 균등 분할해 한 번에 전부 잰다.
#
#   측정 (조건별):
#     peak_tilt   임펄스 후 최대 몸통 기울기 [deg]. 작을수록 잘 버틴 것
#     settle_s    기울기가 settle_deg 아래로 돌아와 유지되기까지 [s]. -1 = 미복귀
#     fell        측정 창 안에 종료(전도)된 비율 [%]
#     drift_cm    임펄스 직전 위치 대비 최대 수평 이동 [cm]
#     stepped     한 발이라도 지면에서 떨어진 비율 [%] (발목전략 vs 스텝전략)
#
#   ★ 임펄스는 **몸통 좌표계** 기준으로 준다. reset_base 가 yaw 를 랜덤
#     초기화하므로 월드 기준으로 주면 "앞/뒤/좌/우" 가 섞인다.
#
# 사용:
#   python scripts/rsl_rl/eval_push_recovery.py \
#       --task Unitree-MX3-Velocity-V0-6 --num_envs 512 --headless \
#       --checkpoint logs/.../model_2500.pt
# ============================================================================
"""Launch Isaac Sim Simulator first."""
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="정지 밀림 회복 평가")
parser.add_argument("--task", type=str, default="Unitree-MX3-Velocity-V0-6")
parser.add_argument("--robot_usd", type=str, default=None, help="로봇 USD 경로 덮어쓰기 (충돌 모델 비교용)")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--settle_steps", type=int, default=150, help="임펄스 전 안정화 스텝")
parser.add_argument("--window_steps", type=int, default=150, help="임펄스 후 측정 스텝(=3s)")
parser.add_argument("--settle_deg", type=float, default=2.0, help="복귀 판정 기울기 [deg]")
parser.add_argument("--mags", type=str, default="0.20,0.35,0.50",
                    help="임펄스 세기 [m/s] 쉼표구분")
parser.add_argument("--march_on_push", type=float, default=None,
                    help="임펄스 후 이 시간[s] 뒤에 제자리걸음을 켠다(외부 충격감지 시뮬레이션). "
                         "미지정이면 완전정지 유지 = 기본 동작.")
parser.add_argument("--march_dur", type=float, default=1.5,
                    help="제자리걸음 유지 시간 [s]. 이후 완전정지로 복귀.")
parser.add_argument("--no_noise", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import math  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402

_DIRS = [("전방", (1.0, 0.0)), ("후방", (-1.0, 0.0)),
         ("좌", (0.0, 1.0)), ("우", (0.0, -1.0))]


def _get_obs(env):
    out = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        return out[0]
    return out


def _tilt_deg(robot):
    """몸통 기울기 [deg]. 직립이면 projected_gravity_b = (0,0,-1)."""
    gz = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos((-gz).clamp(-1.0, 1.0)))


def main():
    if args_cli.checkpoint is None:
        raise SystemExit("--checkpoint 필요")
    mags = [float(x) for x in args_cli.mags.split(",")]
    n_cond = len(_DIRS) * len(mags)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if args_cli.robot_usd:
        env_cfg.scene.robot.spawn.usd_path = args_cli.robot_usd
    # ── 완전정지 강제 ──────────────────────────────────────────────────
    bv = env_cfg.commands.base_velocity
    bv.ranges.lin_vel_x = (0.0, 0.0)
    bv.ranges.lin_vel_y = (0.0, 0.0)
    bv.ranges.ang_vel_z = (0.0, 0.0)
    bv.rel_standing_envs = 1.0
    bv.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.gait.rel_stop_when_standing = 1.0   # 제자리걸음 아닌 완전정지
    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.curriculum.ang_vel_cmd_levels = None
    env_cfg.events.push_robot = None                     # 내장 외란 끔(우리가 준다)
    env_cfg.episode_length_s = 1000.0                    # 측정 중 타임아웃 방지
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
    N = uenv.num_envs
    dt = uenv.step_dt

    # env -> 조건 배정
    cond = torch.arange(N, device=dev) % n_cond
    dir_i, mag_i = cond // len(mags), cond % len(mags)
    dir_b = torch.tensor([d[1] for d in _DIRS], device=dev, dtype=torch.float32)[dir_i]
    mag_t = torch.tensor(mags, device=dev, dtype=torch.float32)[mag_i]

    obs = _get_obs(env)
    print(f"[eval] {args_cli.task}  envs={N}  조건={n_cond} (방향 4 x 세기 {len(mags)})  "
          f"조건당 ~{N // n_cond} env  noise={'OFF' if args_cli.no_noise else 'ON'}")

    for _ in range(args_cli.settle_steps):
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]

    # ── 임펄스 (몸통 좌표계 -> 월드) ───────────────────────────────────
    with torch.inference_mode():
        d3 = torch.zeros(N, 3, device=dev)
        d3[:, :2] = dir_b
        d_w = quat_apply(robot.data.root_quat_w, d3)
        vel = robot.data.root_vel_w.clone()
        vel[:, :2] += d_w[:, :2] * mag_t.unsqueeze(-1)
        robot.write_root_velocity_to_sim(vel)
        p0 = robot.data.root_pos_w[:, :2].clone()

    peak = torch.zeros(N, device=dev)
    drift = torch.zeros(N, device=dev)
    settle = torch.full((N,), -1.0, device=dev)
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    stepped = torch.zeros(N, dtype=torch.bool, device=dev)
    cs = uenv.scene["contact_forces"]
    feet = [i for i, n in enumerate(cs.body_names) if "ankle_roll" in n]

    gait_term = uenv.command_manager.get_term("gait")
    march_on = int(args_cli.march_on_push / dt) if args_cli.march_on_push is not None else None
    march_off = march_on + int(args_cli.march_dur / dt) if march_on is not None else None
    if march_on is not None:
        print(f"[eval] 제자리걸음: 임펄스 +{args_cli.march_on_push:.2f}s 에 ON, "
              f"{args_cli.march_dur:.1f}s 유지 (step {march_on}~{march_off})")

    for k in range(args_cli.window_steps):
        with torch.inference_mode():
            # ── 외부 충격감지 시뮬레이션 ────────────────────────────────
            #   _want_stop=False 로 두면 속도명령이 0 이어도 gait 가 주파수를
            #   유지해 제자리걸음이 된다(완전정지는 want_stop=True 일 때만).
            #   실기에서 "충격 감지 -> 제자리걸음 잠깐 켜기" 를 그대로 흉내낸다.
            if march_on is not None:
                if k == march_on:
                    gait_term._want_stop[:] = False
                elif k == march_off:
                    gait_term._want_stop[:] = True
            out = env.step(policy(obs))
            obs, dones = out[0], out[2].bool()
            t = _tilt_deg(robot)
            alive = ~fell
            peak = torch.where(alive, torch.maximum(peak, t), peak)
            d = torch.norm(robot.data.root_pos_w[:, :2] - p0, dim=-1)
            drift = torch.where(alive, torch.maximum(drift, d), drift)
            # 복귀: 아직 미기록이고 기울기가 문턱 아래로 내려온 첫 시점
            ok = (settle < 0) & (t < args_cli.settle_deg) & alive & (k > 5)
            settle = torch.where(ok, torch.full_like(settle, k * dt), settle)
            f = cs.data.net_forces_w_history[:, 0, feet, :].norm(dim=-1)
            # min = 두 발 중 힘이 작은 쪽. 그게 문턱 아래면 "한 발을 뗐다".
            # (max 를 쓰면 두 발이 동시에 뜬 비행 구간만 세어 보행에서는
            #  거의 0 이 나온다 — 2026-08-25 수정)
            stepped |= alive & (f.min(dim=-1).values < 1.0)
            fell |= dones

    print("\n" + "=" * 78)
    print(f"  정지 밀림 회복  (임펄스 후 {args_cli.window_steps * dt:.1f}s 관측, "
          f"복귀문턱 {args_cli.settle_deg}°)")
    print("=" * 78)
    print(f"{'방향':>5}{'세기':>7}{'peak_tilt':>11}{'settle':>9}{'전도%':>8}{'drift':>9}{'스텝%':>8}")
    for di, (dn, _) in enumerate(_DIRS):
        for mi, mg in enumerate(mags):
            m = (dir_i == di) & (mag_i == mi)
            if m.sum() == 0:
                continue
            s = settle[m]
            ok = s >= 0
            st = f"{s[ok].mean().item():.2f}s" if ok.any() else "미복귀"
            print(f"{dn:>5}{mg:>7.2f}{peak[m].mean().item():>10.2f}°{st:>9}"
                  f"{fell[m].float().mean().item() * 100:>7.1f}"
                  f"{drift[m].mean().item() * 100:>8.1f}cm"
                  f"{stepped[m].float().mean().item() * 100:>7.1f}")
    print("-" * 78)
    ok = settle >= 0
    print(f"  전체 : peak_tilt {peak.mean().item():.2f}°  "
          f"복귀율 {ok.float().mean().item() * 100:.1f}%  "
          f"평균복귀 {settle[ok].mean().item() if ok.any() else float('nan'):.2f}s  "
          f"전도 {fell.float().mean().item() * 100:.1f}%  "
          f"스텝전략 {stepped.float().mean().item() * 100:.1f}%")
    print("=" * 78)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
