"""실물 공중 계단 응답(2026-11-25)과 같은 시험을 MuJoCo 에서 한다. 몸통 고정, Profile Velocity 45 rpm, 다리 position 액추에이터."""
import math, sys
import numpy as np
import mujoco
import mujoco_kick_stress as ks
import kick_real_compare as kc

TESTS = [("l_pelvis_roll_joint", 10), ("l_pelvis_roll_joint", -30), ("r_pelvis_roll_joint", 30),
         ("l_pelvis_pitch_joint", -30), ("l_knee_pitch_joint", -30), ("l_ankle_roll_joint", 15)]


def run(sim, j, deg, delay_s=0.02):
    m, d = sim.m, sim.d
    c = ks.Cond()
    sim.apply_cond(c)
    sim.place(np.random.default_rng(0), c)
    d.qpos[sim.rq:sim.rq + 3] = [0, 0, 0.8]
    prof = np.full(m.nu, 45.0 * ks.RPM)
    root_q = d.qpos[sim.rq:sim.rq + 7].copy()
    a = sim.aid[j]; qa = sim.qadr[j]
    q0 = ks.INIT_POSE[j]
    out = []
    n = int(1.5 / m.opt.timestep)
    for k in range(n):
        t = k * m.opt.timestep - 0.3
        if t >= delay_s:
            sim.target[a] = q0 + math.radians(deg)
        sim.step_physics(c, prof)
        d.qpos[sim.rq:sim.rq + 7] = root_q; d.qvel[sim.rv:sim.rv + 6] = 0
        if k % 10 == 0:
            out.append((t, math.degrees(d.qpos[qa] - q0)))
    t = np.array([o[0] for o in out]); q = np.array([o[1] for o in out]) / deg
    return t, q


def metrics(t, r):
    def cross(f):
        i = np.argmax(r >= f); return t[i] if r.max() >= f else float("nan")
    return cross(0.1) * 1000, cross(0.9) * 1000, (r.max() - 1) * 100, np.mean(r[t > 0.8]) * 100


if __name__ == "__main__":
    sim = ks.Sim(ks.DEFAULT_SCENE)
    for label, mod in (("sim kp50 kv2.5", {}), ("kp30 kv2.5", dict(kp=30.0)), ("kp30 kv1.5", dict(kp=30.0, kv=1.5)),
                       ("kp30 kv1.0", dict(kp=30.0, kv=1.0)), ("kp50 kv1.0", dict(kv=1.0))):
        kc._DEF.clear(); kc.apply_model(sim, **mod)
        print("==", label)
        for j, deg in TESTS:
            t, r = run(sim, j, deg)
            t10, t90, ov, fin = metrics(t, r)
            samp = r[(t >= 0) & (t <= 0.3)][::2]
            print(f"  {j:22s} {deg:+3d}: t10 {t10:4.0f} t90 {t90:4.0f} ms  overshoot {ov:+5.1f}%  final {fin:5.1f}%  | " +
                  " ".join("%.0f" % (x * 100) for x in samp))
