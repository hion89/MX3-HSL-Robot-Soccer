"""실물 킥 bag (npz) 의 /joint_goals 를 MuJoCo 에 그대로 재생해 실물과 비교한다 (열린 루프, 정책 없음).

다리·팔 (XM540-W150, id 1~18) 은 Dynamixel 위치 제어를 흉내 낸 토크 모델로 돈다:
  setpoint  : 목표를 Profile Velocity (rpm) 로 따라가는 내부 궤적
  pwm       = clip( kp_pwm·(setpoint − q) − kd_pwm·q̇ , ±1 )          (P 800 → kp_pwm 4.60 /rad = 12.4° 에서 100 %)
  τ         = τ_stall·s·(V/12)·pwm − k_emf·q̇ − τ_fric·sign(q̇)       (τ_stall 7.3, k_emf 7.3/5.55 = 1.32)
  전류 추정 I = τ_motor / 1.159  (7.3 N·m / 6.3 A)
목은 원래 위치 액추에이터 그대로 둔다.

비교 (트리거 0 s 기준, 0~t_end): 몸 roll·pitch (IMU 자세, 서 있을 때 기준 차이), 디딤발 관절각, PWM, 전류.

  python kick_replay_calib.py --npz ../kick_trials_20261126/npz/push31800_r2_162012.npz [--sweep]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import kick_real_compare as kc
import mujoco_kick_stress as ks

TICK_PER_RAD = 4096 / (2 * math.pi)
BODY = [n for n in ks.INIT_POSE if not n.startswith("neck")]          # 18 XM540 관절
COMPARE = ["l_pelvis_roll_joint", "l_pelvis_pitch_joint", "l_knee_pitch_joint", "l_ankle_pitch_joint",
           "l_ankle_roll_joint", "r_pelvis_roll_joint", "r_pelvis_pitch_joint"]


@dataclasses.dataclass
class Motor:
    P: float = 800.0            # Position P gain (register)
    kd_pwm: float = 0.02        # PWM 비율 / (rad/s)  (D 500 의 실효값, 시간 단위를 몰라 맞춘다)
    stall: float = 7.3          # N·m @12 V
    s: float = 1.0              # 스톨 토크 배율 (효율·기어 손실)
    k_emf: float = 7.3 / 5.55   # N·m·s/rad (역기전력)
    emf_s: float = 1.0
    fric: float = 0.0           # 쿨롱 마찰 N·m
    volt: float = 11.3
    prof_rpm: float = 45.0
    delay_s: float = 0.010      # 목표가 버스로 나가기까지 (GoalDelay p50 약 7~8 ms)
    add_kg: float = 0.8         # 몸통 추가 질량 (실측 6.5 kg)
    mu: float = 0.8             # 발-바닥 마찰
    armature: float = 0.05

    @property
    def kp_pwm(self):
        return self.P / 128.0 * TICK_PER_RAD / 885.0


def rpy_from_quat(q):
    w, x, y, z = q
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    return roll, pitch


def load_real(path):
    z = np.load(path, allow_pickle=True)
    names = list(z["names"])
    r = dict(names=names, goal_t=z["goal_t"], goal=z["goal"], js_t=z["js_t"], q=z["q"], mf_t=z["mf_t"],
             pwm=z["pwm"], cur=z["cur"], imu_t=z["imu_t"], quat=z["quat"], volt=float(np.nanmean(z["volt"])) if len(z["volt"]) else 11.3)
    rp = np.array([rpy_from_quat(q) for q in r["quat"]])
    base = (r["imu_t"] > -1.0) & (r["imu_t"] < 0.0)
    r["roll"] = rp[:, 0] - rp[base, 0].mean()
    r["pitch"] = rp[:, 1] - rp[base, 1].mean()
    return r


class Replay:
    def __init__(self, mot: Motor):
        self.sim = ks.Sim(ks.DEFAULT_SCENE)
        self.mot = mot
        sim, m = self.sim, self.sim.m
        kc._DEF.clear()
        kc.apply_model(sim, base_add_kg=mot.add_kg)
        self.body_a = np.array([sim.aid[n] for n in BODY])
        self.body_q = np.array([sim.qadr[n] for n in BODY])
        self.body_v = np.array([sim.vadr[n] for n in BODY])
        # 몸 관절 액추에이터 → 순수 토크 (force = ctrl)
        m.actuator_gainprm[self.body_a, 0] = 1.0
        m.actuator_biasprm[self.body_a, :] = 0.0
        m.actuator_forcerange[self.body_a] = [-50.0, 50.0]
        m.actuator_ctrllimited[self.body_a] = 0
        m.dof_armature[self.body_v] = mot.armature
        for g in sim.robot_geoms + [sim.floor]:
            m.geom_friction[g] = [mot.mu, 0.1, 0.0001]
        self.idx = {n: i for i, n in enumerate(BODY)}

    def torque(self, setpoint):
        mot, d = self.mot, self.sim.d
        q = d.qpos[self.body_q]; qd = d.qvel[self.body_v]
        pwm = np.clip(mot.kp_pwm * (setpoint - q) - mot.kd_pwm * qd, -1.0, 1.0)
        tau_m = mot.stall * mot.s * (mot.volt / 12.0) * pwm - mot.k_emf * mot.emf_s * qd
        tmax = mot.stall * mot.s * (mot.volt / 12.0)
        tau_m = np.clip(tau_m, -tmax, tmax)                 # 전류 제한
        tau = tau_m - mot.fric * np.tanh(qd / 0.05)
        return pwm, tau, tau_m / 1.159

    def run(self, real, t_start=-0.8, t_end=2.4, settle_s=1.5):
        sim, mot, m, d = self.sim, self.mot, self.sim.m, self.sim.d
        mj = sim.mj
        names = real["names"]
        col = np.array([names.index(n) for n in BODY])
        neck = [n for n in ks.INIT_POSE if n.startswith("neck")]
        gt, G = real["goal_t"], real["goal"][:, col]
        # 시작 자세 = 실물 관절각 (t_start 근처 평균)
        w = (real["js_t"] > t_start - 0.3) & (real["js_t"] < t_start)
        with np.errstate(all="ignore"):
            q0 = np.nanmean(real["q"][w][:, col], 0)
            g0 = np.nanmean(G[(gt > t_start - 0.3) & (gt < t_start)], 0)
        init = np.array([ks.INIT_POSE[n] for n in BODY])
        q0 = np.where(np.isnan(q0), init, q0)
        g0 = np.where(np.isnan(g0), q0, g0)          # 목표가 없는 관절 (팔 등) 은 시작 자세 유지
        G = np.where(np.isnan(G), g0[None, :], G)
        mj.mj_resetData(m, d)
        d.qpos[sim.rq:sim.rq + 3] = [0.0, 0.0, 0.42]
        d.qpos[sim.rq + 3:sim.rq + 7] = [1, 0, 0, 0]
        for i, n in enumerate(BODY):
            d.qpos[sim.qadr[n]] = q0[i]
        for n in neck:
            d.qpos[sim.qadr[n]] = ks.INIT_POSE[n]
            d.ctrl[sim.aid[n]] = ks.INIT_POSE[n]
        d.qpos[sim.bq:sim.bq + 3] = [10.0, 10.0, 0.065]
        mj.mj_forward(m, d)
        # 발이 바닥에 닿게 내린다
        for _ in range(200):
            mj.mj_forward(m, d)
            if sim.foot_contact().sum() > 1.0:
                break
            d.qpos[sim.rq + 2] -= 0.002
        setpoint = q0.copy()
        dt = sim.dt
        step_lim = mot.prof_rpm * ks.RPM * dt
        n_settle = int(settle_s / dt)
        for k in range(n_settle):          # 서 있는 목표 g0 로 정착 (내력 포함)
            setpoint += np.clip(g0 - setpoint, -step_lim, step_lim)
            _, tau, _ = self.torque(setpoint)
            d.ctrl[self.body_a] = tau
            mj.mj_step(m, d)
        # 재생
        out = dict(t=[], roll=[], pitch=[], q=[], pwm=[], cur=[], fell_t=None)
        rp0 = None
        n_sub = int(round(0.02 / dt))
        t = t_start
        target = g0.copy()
        k = 0
        while t < t_end:
            if k % n_sub == 0:                       # 50 Hz 버스 사이클: 그 시점 (지연 반영) 최신 목표
                j = np.searchsorted(gt, t - mot.delay_s) - 1
                if j >= 0:
                    target = np.where(np.isnan(G[j]), target, G[j])
            setpoint += np.clip(target - setpoint, -step_lim, step_lim)
            pwm, tau, cur = self.torque(setpoint)
            d.ctrl[self.body_a] = tau
            mj.mj_step(m, d)
            t += dt; k += 1
            if k % n_sub == 0:
                quat = d.sensordata[sim.s_quat:sim.s_quat + 4]
                r_, p_ = rpy_from_quat(quat)
                if rp0 is None:
                    rp0 = (r_, p_)
                R = sim.R_base()
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
                if out["fell_t"] is None and t > 0 and (tilt > 50 or d.qpos[sim.rq + 2] < ks.FALL_HEIGHT - 0.05):
                    out["fell_t"] = t
                out["t"].append(t); out["roll"].append(r_ - rp0[0]); out["pitch"].append(p_ - rp0[1])
                out["q"].append(d.qpos[self.body_q].copy()); out["pwm"].append(pwm.copy()); out["cur"].append(cur.copy())
        for kk in ("t", "roll", "pitch", "q", "pwm", "cur"):
            out[kk] = np.array(out[kk])
        return out


def compare(real, sim, t0=0.0, t1=1.6):
    names = real["names"]
    ts = sim["t"]
    sel = (ts >= t0) & (ts <= t1)
    tt = ts[sel]
    ri = lambda x, y: np.interp(tt, x, y)
    res = {}
    res["roll_rmse"] = float(np.sqrt(np.mean((sim["roll"][sel] - ri(real["imu_t"], real["roll"])) ** 2)))
    res["pitch_rmse"] = float(np.sqrt(np.mean((sim["pitch"][sel] - ri(real["imu_t"], real["pitch"])) ** 2)))
    res["roll_max_sim"] = float(sim["roll"][(ts >= 0.4) & (ts <= 1.5)].max())
    res["roll_max_real"] = float(real["roll"][(real["imu_t"] >= 0.4) & (real["imu_t"] <= 1.5)].max())
    jr = []
    for n in COMPARE:
        i, c = BODY.index(n), names.index(n)
        ok = ~np.isnan(real["q"][:, c])
        jr.append(np.degrees(np.sqrt(np.mean((sim["q"][sel, i] - ri(real["js_t"][ok], real["q"][ok, c])) ** 2))))
    res["joint_rmse_deg"] = float(np.mean(jr))
    pr = []
    for n in COMPARE:
        i, c = BODY.index(n), names.index(n)
        ok = ~np.isnan(real["pwm"][:, c])
        pr.append(np.sqrt(np.mean((sim["pwm"][sel, i] - ri(real["mf_t"][ok], real["pwm"][ok, c])) ** 2)))
    res["pwm_rmse"] = float(np.mean(pr))
    res["fell_t"] = sim["fell_t"] if sim["fell_t"] is not None else -1.0
    res["cost"] = res["roll_rmse"] / 5.0 + res["pitch_rmse"] / 5.0 + res["joint_rmse_deg"] / 5.0 + res["pwm_rmse"] / 0.2
    return res


def _job(args):
    path, mot = args
    real = load_real(path)
    if mot.volt <= 0:
        mot = dataclasses.replace(mot, volt=real["volt"])
    sim = Replay(mot).run(real)
    return os.path.basename(path), dataclasses.asdict(mot), compare(real, sim)


def detail(path, mot):
    real = load_real(path)
    mot = dataclasses.replace(mot, volt=real["volt"]) if mot.volt <= 0 else mot
    sim = Replay(mot).run(real)
    r = compare(real, sim)
    print(os.path.basename(path), "V %.2f" % mot.volt, json.dumps({k: round(v, 3) for k, v in r.items()}))
    names = real["names"]
    print("t     roll real/sim  pitch real/sim | " + " | ".join("%s q real/sim pwm real/sim" % n.replace("_joint", "")[:10] for n in COMPARE[:4]))
    for tq in np.arange(0.0, 2.21, 0.2):
        i = int(np.argmin(np.abs(sim["t"] - tq)))
        row = "%.1f  %+6.1f/%+6.1f  %+6.1f/%+6.1f |" % (
            tq, np.interp(tq, real["imu_t"], real["roll"]), sim["roll"][i], np.interp(tq, real["imu_t"], real["pitch"]), sim["pitch"][i])
        for n in COMPARE[:4]:
            b, c = BODY.index(n), names.index(n)
            ok = ~np.isnan(real["q"][:, c])
            okp = ~np.isnan(real["pwm"][:, c])
            row += " %+6.1f/%+6.1f %+4.0f/%+4.0f |" % (
                math.degrees(np.interp(tq, real["js_t"][ok], real["q"][ok, c])), math.degrees(sim["q"][i, b]),
                100 * np.interp(tq, real["mf_t"][okp], real["pwm"][okp, c]), 100 * sim["pwm"][i, b])
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", action="append", default=[])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--out", default="results_replay_calib_20261126")
    ap.add_argument("--set", action="append", default=[], help="Motor 필드=값 (예: P=700)")
    args = ap.parse_args()
    paths = args.npz or sorted(glob.glob(os.path.join(ks.REPO, "sim2real_evidence", "kick_trials_20261126", "npz", "*_r*.npz")))
    base = Motor(volt=0.0)
    for kv in args.set:
        k, v = kv.split("="); base = dataclasses.replace(base, **{k: float(v)})
    if not args.sweep:
        for p in paths:
            detail(p, base)
        return
    os.makedirs(args.out, exist_ok=True)
    grid = []
    for P, s, emf, kd, add, mu, dl in itertools.product((800.0, 560.0, 400.0), (1.0, 0.8, 0.65), (1.0, 1.5),
                                                      (0.02, 0.1), (0.8,), (0.8, 0.5), (0.01, 0.03)):
        grid.append(dataclasses.replace(base, P=P, s=s, emf_s=emf, kd_pwm=kd, add_kg=add, mu=mu, delay_s=dl))
    jobs = [(p, g) for g in grid for p in paths]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for name, mot, res in ex.map(_job, jobs, chunksize=2):
            rows.append(dict(trial=name, **{k: mot[k] for k in ("P", "s", "emf_s", "kd_pwm", "add_kg", "mu", "delay_s")}, **res))
    json.dump(rows, open(os.path.join(args.out, "rows.json"), "w"), indent=1)
    keys = ("P", "s", "emf_s", "kd_pwm", "add_kg", "mu", "delay_s")
    agg = {}
    for r in rows:
        agg.setdefault(tuple(r[k] for k in keys), []).append(r)
    ranked = sorted(agg.items(), key=lambda kv: np.mean([r["cost"] for r in kv[1]]))
    for kk, rs in ranked[:15]:
        print("%s | cost %.2f roll %.1f pitch %.1f joint %.1f pwm %.2f | roll max sim %.0f real %.0f | fell %s" % (
            " ".join("%s=%g" % (k, v) for k, v in zip(keys, kk)), np.mean([r["cost"] for r in rs]),
            np.mean([r["roll_rmse"] for r in rs]), np.mean([r["pitch_rmse"] for r in rs]),
            np.mean([r["joint_rmse_deg"] for r in rs]), np.mean([r["pwm_rmse"] for r in rs]),
            np.mean([r["roll_max_sim"] for r in rs]), np.mean([r["roll_max_real"] for r in rs]),
            "/".join("%.1f" % r["fell_t"] for r in rs)), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
