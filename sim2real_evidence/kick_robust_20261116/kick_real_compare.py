"""실물 킥 시험(2026-11-25)과 같은 조건을 MuJoCo 에서 재현해 관절·토크를 기록한다.

실물 조건: 공 0.35 m 정면, 목표 정면, cmd_vx 0.2, 시작 hold 0.6 s, 필터 0.02, 킥 구동 2.0 s 뒤 기본 자세 hold.
비교 지표 (실물 bag 과 같은 정의):
  * 차는 다리 골반 roll 최대 벌림 [deg] (기본 자세 대비 |q|)
  * 디딤발 골반 roll 명령-실제 최대 차이 [deg], 디딤발 골반 roll 토크가 한계의 95 % 이상인 시간 비율
  * 몸 roll 최대 [deg], 넘어짐 (기울기 > 0.6 rad 또는 높이 < 0.28 m)
로봇·ROS 와 무관한 시뮬레이션 측정이다.

  python kick_real_compare.py --policy a.onnx --policy b.onnx --trials 20 --out results_real_compare
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from collections import deque

import numpy as np

import mujoco_kick_stress as ks

R_ROLL, L_ROLL = ks.POLICY_JOINTS.index("r_pelvis_roll_joint"), ks.POLICY_JOINTS.index("l_pelvis_roll_joint")
R_PITCH, L_PITCH = ks.POLICY_JOINTS.index("r_pelvis_pitch_joint"), ks.POLICY_JOINTS.index("l_pelvis_pitch_joint")


_DEF = {}


def apply_model(sim, base_add_kg=0.0, kp=None, kv=None, cap=None, head_add_kg=0.0):
    """실물 맞춤 모델 변경. 매 조건마다 원래 값에서 다시 적용한다.
    base_add_kg: 몸통(base_link)에 더하는 질량 (카메라·배터리 등). 관성은 같은 비율로 키운다.
    kp, kv: 다리 position 액추에이터 강성·감쇠 [Nm/rad, Nm·s/rad]. cap: 다리 토크 한계 [Nm] (DC 모델의 상한도 같이)."""
    m = sim.m
    if not _DEF:
        _DEF.update(mass=m.body_mass.copy(), inertia=m.body_inertia.copy(), gain=m.actuator_gainprm.copy(),
                    bias=m.actuator_biasprm.copy(), fr=sim.default_forcerange.copy(), cap=ks.EFFORT_CAP)
    m.body_mass[:] = _DEF["mass"]; m.body_inertia[:] = _DEF["inertia"]
    m.actuator_gainprm[:] = _DEF["gain"]; m.actuator_biasprm[:] = _DEF["bias"]
    sim.default_forcerange[:] = _DEF["fr"]; ks.EFFORT_CAP = _DEF["cap"]
    if base_add_kg:
        b = sim.base
        f = (m.body_mass[b] + base_add_kg) / m.body_mass[b]
        m.body_mass[b] *= f; m.body_inertia[b] *= f
    if head_add_kg:
        h = m.body(ks.PREFIX + "neck_pitch_link").id
        f = (m.body_mass[h] + head_add_kg) / m.body_mass[h]
        m.body_mass[h] *= f; m.body_inertia[h] *= f
    for a in sim.leg_a:
        if kp is not None:
            m.actuator_gainprm[a, 0] = kp; m.actuator_biasprm[a, 1] = -kp
        if kv is not None:
            m.actuator_biasprm[a, 2] = -kv
        if cap is not None:
            sim.default_forcerange[a] = [-cap, cap]
    if cap is not None:
        ks.EFFORT_CAP = cap
    sim.mj.mj_setConst(m, sim.d)


def roll_of(R):
    return math.degrees(math.atan2(R[2, 1], R[2, 2]))


def run(sim, sess, c: ks.Cond, seed: int, kick_s: float, after_s: float, jitter: float, ball_x: float = 0.35):
    rng = np.random.default_rng(seed)
    sim.apply_cond(c)
    sim.place(rng, c)
    prof = np.full(sim.m.nu, c.profile_rpm * ks.RPM)
    for _ in range(int(1.0 / sim.dt)):
        sim.step_physics(c, prof)
    R0 = sim.R_base()
    yaw0 = ks.yaw_of(R0)
    p0 = sim.d.xpos[sim.base].copy()
    # 실물처럼 공은 거의 정면 0.35 m. 놓는 오차만 jitter 로 준다.
    bel_b = np.array([ball_x, 0.0])
    true_b = bel_b + rng.uniform(-jitter, jitter, 2)
    cy, sy = math.cos(yaw0), math.sin(yaw0)
    bw = p0[:2] + np.array([cy * true_b[0] - sy * true_b[1], sy * true_b[0] + cy * true_b[1]])
    sim.d.qpos[sim.bq:sim.bq + 3] = [bw[0], bw[1], 0.065]
    sim.d.qvel[sim.bv:sim.bv + 6] = 0.0
    sim.mj.mj_forward(sim.m, sim.d)
    tgt_w = yaw0

    gait = ks.Gait()
    last_a = np.zeros(12, np.float32)
    filt = np.zeros(12, np.float32)
    alpha = math.exp(-ks.POLICY_DT / c.filter_tau) if c.filter_tau > 0 else 0.0
    goal_q = deque([sim.default12.copy()] * (c.act_delay + 1), maxlen=c.act_delay + 1)
    n_sub = int(round(ks.POLICY_DT / sim.dt))
    imu_buf = deque(maxlen=64)
    imu_lag = int(round(c.imu_delay_ms / 1000.0 / sim.dt))
    hist = None

    def sense():
        imu_buf.append(sim.imu())

    sense()
    push = ks.KickPush(sim, c)
    rows = []
    fell_t = None
    steps = int(round((kick_s + after_s) / ks.POLICY_DT))
    for k in range(steps):
        t = k * ks.POLICY_DT
        active = t < kick_s
        in_hold = t < c.hold_s
        cmd = np.zeros(3, np.float32) if in_hold else np.array([c.cmd_vx, 0, 0], np.float32)
        freq = 0.0 if (in_hold or np.linalg.norm(cmd) < ks.GAIT_STAND_TH) else ks.GAIT_FREQ
        gait.update(ks.POLICY_DT if k > 0 else 0.0, freq)
        R = sim.R_base()
        dyaw = ks.yaw_of(R) - yaw0
        tb = np.array([math.cos(tgt_w - yaw0 - dyaw), math.sin(tgt_w - yaw0 - dyaw)])
        g, grav = imu_buf[-1 - min(imu_lag, len(imu_buf) - 1)]
        q = sim.d.qpos[sim.leg_q].astype(np.float32)
        qd = sim.d.qvel[sim.leg_v].astype(np.float32)

        def frame(la):
            return np.concatenate([0.2 * g, grav, cmd, gait.obs(), q - sim.default12, 0.05 * qd,
                                   la, bel_b, tb, [c.strength]]).astype(np.float32)

        if active:
            if hist is None:
                hist = deque([frame(last_a)] * 5, maxlen=5)
            obs = ks.build_obs(np.stack(hist))
            a = sess.run(None, {sess.get_inputs()[0].name: obs[None]})[0].reshape(12).astype(np.float32)
            a = np.clip(a, -10.0, 10.0)
            hist.append(frame(a))
            filt = alpha * filt + (1 - alpha) * a if c.filter_tau > 0 else a
            q_tgt = sim.default12 + 0.25 * filt
            if in_hold:
                q_tgt = sim.default12.copy()
            last_a = a
        else:
            q_tgt = sim.default12.copy()   # 실물 러너: 킥 끝나면 기본 자세 hold
        goal_q.append(q_tgt)
        applied = goal_q[0]
        for _ in range(n_sub):
            sim.step_physics(c, prof)
            sense()
        for i, n in enumerate(ks.POLICY_JOINTS):
            sim.target[sim.aid[n]] = applied[i]
        push.update(t)
        R = sim.R_base()
        tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
        h = sim.d.xpos[sim.base][2]
        if fell_t is None and (tilt > ks.FALL_TILT or h < ks.FALL_HEIGHT):
            fell_t = t
        qn = sim.d.qpos[sim.leg_q]
        frc = sim.d.actuator_force[sim.leg_a]
        lim_hi = sim.m.actuator_forcerange[sim.leg_a, 1]
        lim_lo = sim.m.actuator_forcerange[sim.leg_a, 0]
        sat = np.where(frc >= 0, frc / np.maximum(lim_hi, 1e-6), frc / np.minimum(lim_lo, -1e-6))
        rows.append(dict(t=t, roll=roll_of(R), tilt=math.degrees(tilt), fn=sim.foot_contact().tolist(),
                         q=np.degrees(qn - sim.default12).tolist(), goal=np.degrees(applied - sim.default12).tolist(),
                         tau=frc.tolist(), sat=sat.tolist(), vball=float(np.linalg.norm(sim.d.qvel[sim.bv:sim.bv + 2])),
                         push=float(np.linalg.norm(sim.d.xfrc_applied[sim.base, :3]))))
        if fell_t is not None and t - fell_t > 0.4:
            break
    return rows, fell_t


def summarize(rows, fell_t):
    q = np.array([r["q"] for r in rows])
    goal = np.array([r["goal"] for r in rows])
    sat = np.array([r["sat"] for r in rows])
    tau = np.array([r["tau"] for r in rows])
    # 차는 다리 = 골반 pitch 를 더 크게 휘두른 쪽
    r_sw, l_sw = np.abs(q[:, R_PITCH]).max(), np.abs(q[:, L_PITCH]).max()
    kick_r = r_sw >= l_sw
    kr, sr = (R_ROLL, L_ROLL) if kick_r else (L_ROLL, R_ROLL)
    act = slice(0, max(1, int(2.0 / ks.POLICY_DT)))
    return dict(
        kick_leg="R" if kick_r else "L",
        kick_hip_roll_max=float(np.abs(q[act, kr]).max()),
        kick_hip_roll_goal_max=float(np.abs(goal[act, kr]).max()),
        support_hip_roll_err_max=float(np.abs(goal[act, sr] - q[act, sr]).max()),
        support_hip_roll_tau_max=float(np.abs(tau[act, sr]).max()),
        support_hip_roll_sat95_frac=float((np.abs(sat[act, sr]) >= 0.95).mean()),
        roll_max=float(np.abs([r["roll"] for r in rows[:len(q)]]).max()),
        fell=fell_t is not None, fell_t=fell_t if fell_t is not None else -1.0,
        vball_max=float(max(r["vball"] for r in rows)),
        pushed=float(any(r.get("push", 0.0) > 0 for r in rows)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--strength", type=float, action="append")
    ap.add_argument("--jitter", type=float, default=0.03, help="공 놓는 오차 ± [m]")
    ap.add_argument("--out", default="results_real_compare")
    ap.add_argument("--model-sweep", action="store_true", help="질량·강성·토크 한계를 실물 쪽으로 바꾼 조건들")
    ap.add_argument("--model-sweep2", action="store_true", help="질량 위치·감쇠·지연·Profile Velocity 조건들")
    ap.add_argument("--model-sweep3", action="store_true", help="바닥 접촉 부드러움 (잔디) 조건들")
    args = ap.parse_args()
    import onnxruntime as ort
    sim = ks.Sim(ks.DEFAULT_SCENE)
    base = ks.Cond()
    real = dataclasses.replace(base, name="real_like", volt=11.1, imu_delay_ms=10.0, hold_s=0.6,
                               cmd_vx=0.20, filter_tau=0.02, mu=0.6, mu_torsion=0.1)
    v103 = dataclasses.replace(real, name="v10.3", volt=10.3)
    # (이름, 조건, 킥 구동 s, 이후 s, 모델 변경)
    conds = [("real_like_cut2.0", real, 2.0, 1.5, {}),
             ("real_like_10.3V_cut2.0", v103, 2.0, 1.5, {})]
    if args.model_sweep:
        conds = [
            ("A_sim_model", v103, 2.0, 1.5, {}),
            ("B_mass6.5", v103, 2.0, 1.5, dict(base_add_kg=0.8)),
            ("C_kp30", v103, 2.0, 1.5, dict(kp=30.0)),
            ("D_mass6.5_kp30", v103, 2.0, 1.5, dict(base_add_kg=0.8, kp=30.0)),
            ("E_mass6.5_kp30_kv1.5", v103, 2.0, 1.5, dict(base_add_kg=0.8, kp=30.0, kv=1.5)),
            ("F_mass6.5_kp30_cap6", v103, 2.0, 1.5, dict(base_add_kg=0.8, kp=30.0, cap=6.0)),
            ("G_mass6.5_kp40", v103, 2.0, 1.5, dict(base_add_kg=0.8, kp=40.0)),
        ]
    if args.model_sweep2:
        D = dict(base_add_kg=0.8, kp=30.0)
        conds = [
            ("D_mass6.5_kp30", v103, 2.0, 1.5, D),
            ("H_head0.4_base0.4_kp30", v103, 2.0, 1.5, dict(base_add_kg=0.4, head_add_kg=0.4, kp=30.0)),
            ("I_D_actdelay2", dataclasses.replace(v103, act_delay=2), 2.0, 1.5, D),
            ("J_D_kv1.0", v103, 2.0, 1.5, dict(D, kv=1.0)),
            ("K_D_imu30ms", dataclasses.replace(v103, imu_delay_ms=30.0), 2.0, 1.5, D),
            ("L_D_kv1.5_actdelay1", dataclasses.replace(v103, act_delay=1), 2.0, 1.5, dict(D, kv=1.5)),
            ("M_D_profile30rpm", dataclasses.replace(v103, profile_rpm=30.0), 2.0, 1.5, D),
        ]
    if args.model_sweep3:
        D = dict(base_add_kg=0.8, kp=30.0)
        conds = [(f"T_D_solref{tc:g}", dataclasses.replace(v103, solref_tc=tc), 2.0, 1.5, D) for tc in (0.02, 0.04, 0.06, 0.08, 0.12)]
        conds.append(("T_D_solref0.08_kv1.5", dataclasses.replace(v103, solref_tc=0.08), 2.0, 1.5, dict(D, kv=1.5)))
    os.makedirs(args.out, exist_ok=True)
    summary = []
    for pol in args.policy:
        sess = ort.InferenceSession(pol, providers=["CPUExecutionProvider"])
        for s in (args.strength or [1.0]):
            for name, c, kick_s, after_s, mod in conds:
                apply_model(sim, **mod)
                c2 = dataclasses.replace(c, strength=s)
                res = []
                for i in range(args.trials):
                    rows, fell_t = run(sim, sess, c2, 1000 + i, kick_s, after_s, args.jitter)
                    res.append(summarize(rows, fell_t))
                    if i == 0:
                        with open(os.path.join(args.out, f"{os.path.basename(pol)}_{name}_s{s}_trial0.json"), "w") as f:
                            json.dump(rows, f)
                agg = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != "kick_leg"}
                agg.update(policy=os.path.basename(pol), cond=name, strength=s,
                           kick_leg_R=float(np.mean([r["kick_leg"] == "R" for r in res])))
                summary.append(agg)
                print(f"{os.path.basename(pol):32s} {name:24s} s{s}: fall {100*agg['fell']:.0f}% | "
                      f"kick-leg hip roll {agg['kick_hip_roll_max']:.0f} deg (goal {agg['kick_hip_roll_goal_max']:.0f}) | "
                      f"support hip roll err {agg['support_hip_roll_err_max']:.0f} deg, tau {agg['support_hip_roll_tau_max']:.1f} Nm, "
                      f"sat>=95% {100*agg['support_hip_roll_sat95_frac']:.0f}% | roll max {agg['roll_max']:.0f} | "
                      f"R-kick {100*agg['kick_leg_R']:.0f}% | vball {agg['vball_max']:.2f}", flush=True)
    apply_model(sim)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
