"""Isaac Sim 4.5 ↔ 5.x 킥 차이 원인 찾기 — PhysX 관절 속성 + 킥 중 관절 속도·발속도·공속.

eval_strength_isaac.py 와 같은 조건 (서서, 공 정상, 외란·종료 끔) 에 관절 DR (게인·armature·마찰 랜덤) 도 끈다.
--friction 으로 다리 관절 마찰을 바꿔 본다 (5.x 에선 unitree_rl_lab 이 기본으로 4.5 계수 모델을 쓴다 —
utils/joint_friction_compat.py. 아래 zero/dyn_eq_static/legacy 를 5.x 새 모델 기준으로 보려면 UNITREE_JOINT_FRICTION=native):
  default        빌드된 그대로 (actuator friction 1.1)
  zero           마찰 0
  dyn_eq_static  (5.x 전용) 운동 마찰 = 정지 마찰 — 4.5 의 단일 계수에 가까운지 보려고
  legacy         (5.x 전용) 새 마찰 (정지·운동·점성) 0 + 폐기 예정 계수 API 로 4.5 와 같은 계수 1.1

  python scripts/kick_v1_5/probe_sim_version.py --headless --checkpoint <model.pt> [--friction zero] [--levels 1]
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Unitree-MX3-WalkKick-V1-5B-Play")
parser.add_argument("--num_envs", type=int, default=300)
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--levels", type=str, default="0,0.5,1")
parser.add_argument("--friction", choices=["default", "zero", "dyn_eq_static", "legacy"], default="default")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.version import get_isaac_sim_version  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: F401, E402
from unitree_rl_lab.tasks.locomotion.robots.MX3 import walk_kick_v1_4_env as v14  # noqa: E402


def _obs(env):
    out = env.get_observations()
    return out[0] if version("rsl-rl-lib").startswith("2.3.") else out


def main():
    simv = get_isaac_sim_version()
    tag = f"[probe {simv.major}.{simv.minor} {args_cli.friction}]"
    levels = tuple(float(x) for x in args_cli.levels.split(","))
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    ev = cfg.events
    ev.kick_strength.params["eval_levels"] = levels
    ev.ball_belief.params["probs"] = {"nominal": 1.0, "bias": 0.0, "roll": 0.0, "whiff": 0.0, "duel": 0.0}
    ev.reset_robot_joints.params["standing_prob"] = 1.0
    for k in ("push_robot", "push_on_kick", "base_external_force_torque", "duel_knock",
              "randomize_joint_params", "randomize_actuator_gains"):
        if hasattr(ev, k):
            setattr(ev, k, None)
    t = cfg.terminations
    for k in list(vars(t)):
        if k != "time_out":
            setattr(t, k, None)
    cfg.episode_length_s = args_cli.seconds + 1.0
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg, render_mode=None))
    u = env.unwrapped
    robot, ball = u.scene["robot"], u.scene["ball"]
    n, dev = u.num_envs, u.device
    leg_ids, leg_names = robot.find_joints(v14.MX_LEG_JOINTS, preserve_order=True)
    leg_t = torch.tensor(leg_ids, device=dev)

    if args_cli.friction == "zero":
        if simv.major < 5:
            robot.write_joint_friction_coefficient_to_sim(0.0, joint_ids=leg_ids)
        else:
            robot.write_joint_friction_coefficient_to_sim(0.0, joint_dynamic_friction_coeff=0.0,
                                                          joint_viscous_friction_coeff=0.0, joint_ids=leg_ids)
    elif args_cli.friction == "dyn_eq_static":
        assert simv.major >= 5, "dyn_eq_static 은 5.x 전용"
        mu = robot.data.joint_friction_coeff[:, leg_t].clone()
        robot.write_joint_friction_coefficient_to_sim(mu, joint_dynamic_friction_coeff=mu, joint_ids=leg_ids)
    elif args_cli.friction == "legacy":
        assert simv.major >= 5, "legacy 는 5.x 전용"
        mu = robot.data.joint_friction_coeff[:, leg_t].clone()
        robot.write_joint_friction_coefficient_to_sim(0.0, joint_dynamic_friction_coeff=0.0,
                                                      joint_viscous_friction_coeff=0.0, joint_ids=leg_ids)
        idx = torch.arange(n, dtype=torch.int32)
        c = robot.root_physx_view.get_dof_friction_coefficients().clone()
        c[:, leg_t.cpu()] = mu.cpu()
        robot.root_physx_view.set_dof_friction_coefficients(c, idx)

    v = robot.root_physx_view
    print(f"{tag} PhysX 다리 관절 속성 (env 0):", flush=True)
    mv = v.get_dof_max_velocities()[0]
    mf = v.get_dof_max_forces()[0]
    kp = v.get_dof_stiffnesses()[0]
    kd = v.get_dof_dampings()[0]
    ar = v.get_dof_armatures()[0]
    if simv.major < 5:
        fr = v.get_dof_friction_coefficients()[0].unsqueeze(-1)
        fr_lbl = "friction coeff"
    else:
        fr = torch.cat([v.get_dof_friction_properties()[0], v.get_dof_friction_coefficients()[0].unsqueeze(-1)], -1)
        fr_lbl = "friction (static, dynamic, viscous | legacy coeff)"
    for j, nm in zip(leg_ids, leg_names):
        print(f"{tag}   {nm:22s} max_vel {float(mv[j]):7.3f}  max_force {float(mf[j]):6.2f}  kp {float(kp[j]):6.2f}  "
              f"kd {float(kd[j]):5.2f}  armature {float(ar[j]):5.3f}  {fr_lbl} {[round(float(x), 3) for x in fr[j]]}",
              flush=True)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=dev)
    obs = _obs(env)
    s = u.kick_strength_cmd.clone()
    t_kick = torch.full((n,), -1.0, device=dev)
    peak = torch.zeros(n, device=dev)
    max_tilt = torch.zeros(n, device=dev)
    jv_peak = torch.zeros(n, len(leg_ids), device=dev)       # 킥 전후 0.4 s 의 |관절 속도| 최대
    over = torch.zeros(n, device=dev)                         # |관절 속도| > max_vel 인 (스텝 × 관절) 수
    steps = torch.zeros(n, device=dev)
    lim = mv[leg_ids].to(dev)
    jv_all = torch.zeros(len(leg_ids), device=dev)            # 전 구간·전 env |관절 속도| 최대
    hist = []
    for k in range(int(args_cli.seconds / u.step_dt)):
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]
        tnow = (k + 1) * u.step_dt
        bv = ball.data.root_lin_vel_w[:, :2]
        sp = torch.norm(bv, dim=-1)
        along = (bv * u.kick_target_dir_w[:, :2]).sum(-1)
        start = (t_kick < 0) & (sp > 1.0)
        t_kick = torch.where(start, torch.full_like(t_kick, tnow), t_kick)
        win = (t_kick >= 0) & (tnow - t_kick <= 0.5)
        peak = torch.where(win & (along > peak), along, peak)
        g = robot.data.projected_gravity_b[:, 2].clamp(-1, 1)
        max_tilt = torch.maximum(max_tilt, torch.rad2deg(torch.acos(-g)))
        jv = robot.data.joint_vel[:, leg_t].abs()
        hist.append(jv)
        jv_all = torch.maximum(jv_all, jv.max(0).values)
        hist = hist[-20:]
        over += (jv > lim * 1.001).float().sum(-1)
        steps += 1
        near = start.unsqueeze(-1)                            # 킥 시작 순간: 직전 0.4 s 최대를 기록
        if bool(start.any()):
            jv_peak = torch.where(near, torch.stack(hist).max(0).values, jv_peak)
    strike = getattr(u, "ins_strike_speed", torch.zeros(n, device=dev))
    print(f"{tag} {'s':>5} {'공속':>6} {'발속도':>6} {'킥%':>5} {'넘어짐':>6} | 킥 직전 0.4 s |관절속도| 최대 (오른/왼 pelvis_pitch, knee, ankle_pitch)",
          flush=True)
    ip = [leg_names.index(x) for x in ("r_pelvis_pitch_joint", "r_knee_pitch_joint", "r_ankle_pitch_joint",
                                       "l_pelvis_pitch_joint", "l_knee_pitch_joint", "l_ankle_pitch_joint")]
    for lv in levels:
        m = (s - lv).abs() < 1e-4
        kk = m & (t_kick >= 0)
        jp = jv_peak[kk].mean(0) if kk.any() else torch.full((len(leg_ids),), float("nan"), device=dev)
        print(f"{tag} {lv:5.2f} {peak[kk].mean().item():6.2f} {strike[kk & (strike > 0)].mean().item():6.2f} "
              f"{100 * kk.float().sum().item() / max(m.sum().item(), 1):4.0f}% {100 * (max_tilt[m] > 45).float().mean().item():5.1f}% | "
              + " ".join(f"{float(jp[i]):5.2f}" for i in ip), flush=True)
    print(f"{tag} |관절 속도| > max_vel 비율: {100 * over.sum().item() / (steps.sum().item() * len(leg_ids)):.3f} % "
          f"(전 스텝·다리 관절), 전 구간 최대 |관절 속도| {jv_all.max().item():.2f} ({leg_names[int(jv_all.argmax())]})", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
