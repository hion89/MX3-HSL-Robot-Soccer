"""MX3 kick policy sim2sim stress evaluator (MuJoCo, no ROS).

Runs a kick ONNX policy (obs 265 -> 12) the same way the real runner does
(actor_onnx_walkkick_v1_runner.py) on the humanoid_sim MuJoCo scene, and sweeps
physical / deployment conditions that could make the support foot spin or slip:

  * ground sliding friction, torsional friction, contact softness (turf)
  * action delay (policy steps), IMU (gyro + gravity) observation delay
  * battery voltage sag + DC motor speed-torque limit on leg actuators
  * runner contract: start hold, cmd_vx, action low-pass filter, Profile Velocity

It does NOT send anything to a robot and does not need ROS.  Results are
simulation measurements only; they rank hypotheses, they do not prove the
real-robot cause.

Usage (Windows PowerShell or Linux):
  python mujoco_kick_stress.py --policy <kick.onnx> --suite quick --trials 40 --out results
  python mujoco_kick_stress.py --policy a.onnx --policy b.onnx --suite full --trials 100
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_SCENE = os.path.join(REPO, "MX3_Match_Sim", "src", "humanoid_sim", "models", "scene_1v0.xml")
DEFAULT_POLICY = os.path.join(REPO, "XM_Kick_Policy", "policy", "policy.onnx")
PREFIX = "robo02_"

# Policy joint order = deploy.yaml / runner POLICY_JOINTS (right leg then left leg).
POLICY_JOINTS = [
    "r_pelvis_pitch_joint", "r_pelvis_roll_joint", "r_pelvis_yaw_joint",
    "r_knee_pitch_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "l_pelvis_pitch_joint", "l_pelvis_roll_joint", "l_pelvis_yaw_joint",
    "l_knee_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
]
INIT_POSE = {  # robot_init_behavior.py / sim_bridge.INIT_POSE
    "r_pelvis_pitch_joint": 0.62, "r_pelvis_roll_joint": 0.0, "r_pelvis_yaw_joint": 0.0,
    "r_knee_pitch_joint": 1.2, "r_ankle_pitch_joint": 0.62, "r_ankle_roll_joint": 0.0,
    "l_pelvis_pitch_joint": -0.62, "l_pelvis_roll_joint": 0.0, "l_pelvis_yaw_joint": 0.0,
    "l_knee_pitch_joint": -1.2, "l_ankle_pitch_joint": -0.62, "l_ankle_roll_joint": 0.0,
    "r_shoulder_pitch_joint": -0.3, "r_shoulder_roll_joint": 0.0, "r_elbow_pitch_joint": 1.0,
    "l_shoulder_pitch_joint": 0.3, "l_shoulder_roll_joint": 0.0, "l_elbow_pitch_joint": -1.0,
    "neck_yaw_joint": 0.0, "neck_pitch_joint": 0.4363,
}
FEET = ("r_ankle_roll_link", "l_ankle_roll_link")
RPM = 2.0 * math.pi / 60.0
POLICY_DT = 0.02
GAIT_FREQ, GAIT_STOP_RATE, GAIT_FREQ_SCALE, GAIT_STAND_TH = 2.0, 2.0, 2.0, 0.1
FALL_TILT = 0.6        # rad, same as training termination (v1_4 STAGE_HARD)
FALL_HEIGHT = 0.28     # m, base height
KICK_SPEED = 0.8       # m/s, kick latch threshold (KICK_SPEED_THRESHOLD)
STALL_12V, NOLOAD_12V = 7.3, 5.55   # XM540-W150 @12V (README / kick_stability_mdp)
EFFORT_CAP = 5.0


@dataclasses.dataclass
class Cond:
    name: str = "baseline"
    # ground / foot contact
    mu: float = 1.0            # sliding friction (floor and robot geoms)
    mu_torsion: float = 0.2    # MuJoCo torsional friction (condim 4), bridge default 0.2
    solref_tc: float = 0.02    # contact time constant [s]; larger = softer (turf)
    # timing
    act_delay: int = 0         # extra policy-step delay of joint goals (bridge already has 1 cycle)
    imu_delay_ms: float = 0.0  # gyro + gravity observation delay
    # actuators
    volt: float = 0.0          # battery voltage for DC speed-torque limit; 0 = flat 5 N.m (bridge)
    profile_rpm: float = 45.0  # Dynamixel Profile Velocity
    # runner contract
    hold_s: float = 0.3        # kick_start_hold / resume_hold_seconds
    cmd_vx: float = 0.30
    filter_tau: float = 0.0    # 0 = off; real launch uses 0.02
    duration_s: float = 3.0    # evaluation window after trigger (kick policy keeps control)
    runner_stale_obs: bool = True  # True = real runner history order (sensors 1 step old)
    # task sampling
    ball_dist: tuple = (0.20, 0.50)
    ball_bearing_deg: tuple = (-70.0, 70.0)
    target_deg: tuple = (-90.0, 90.0)
    strength: float = 1.0
    ball_bias_x: tuple = (0.0, 0.0)   # true ball - believed ball [m], robot frame
    # 킥 디딤발 쪽 외력 (실물 디딤발 바깥 모서리 들림 대용). 0 = 끔.
    # 한쪽 발이 떠 있고 그 다리 골반 roll 벌림이 push_trig_deg 를 넘는 순간 시작,
    # 몸통에 디딤발 방향(몸 좌우축, 수평) 힘 push_N 을 push_s 동안 준다. 에피소드당 한 번.
    push_N: float = 0.0
    push_s: float = 0.5
    push_trig_deg: float = 40.0       # 걷는 중 골반 roll ~30° 보다 커야 킥에만 걸린다


def suite(name: str) -> list[Cond]:
    base = Cond()
    deploy = dataclasses.replace(base, name="deploy_contract", hold_s=0.6, cmd_vx=0.20, filter_tau=0.02)
    fresh = dataclasses.replace(base, name="train_fresh_obs", runner_stale_obs=False)
    c = [base, fresh, deploy]
    for t in (0.0, 0.01, 0.03, 0.08):
        c.append(dataclasses.replace(base, name=f"torsion_{t:g}", mu_torsion=t))
    for mu in (0.4, 0.6):
        c.append(dataclasses.replace(base, name=f"mu_{mu:g}", mu=mu))
    c.append(dataclasses.replace(base, name="soft_turf", solref_tc=0.05))
    c.append(dataclasses.replace(base, name="imu_delay_20ms", imu_delay_ms=20.0))
    c.append(dataclasses.replace(base, name="act_delay_2", act_delay=2))
    c.append(dataclasses.replace(base, name="volt_12_dc", volt=12.0))
    c.append(dataclasses.replace(base, name="volt_11.1_dc", volt=11.1))
    c.append(dataclasses.replace(base, name="volt_10.6_dc", volt=10.6))
    turf = dataclasses.replace(base, name="turf_combo", mu=0.6, mu_torsion=0.01, solref_tc=0.04)
    c.append(turf)
    c.append(dataclasses.replace(turf, name="turf_combo_deploy", hold_s=0.6, cmd_vx=0.20, filter_tau=0.02,
                                 volt=11.1, imu_delay_ms=10.0))
    if name == "quick":
        keep = {"baseline", "train_fresh_obs", "deploy_contract", "torsion_0.01", "mu_0.4", "imu_delay_20ms", "volt_12_dc", "volt_10.6_dc", "torsion_0.03",
                "turf_combo", "turf_combo_deploy"}
        return [x for x in c if x.name in keep]
    if name == "full":
        for t in (0.0, 0.005, 0.02, 0.05, 0.1, 0.5):
            c.append(dataclasses.replace(base, name=f"torsion_{t:g}", mu_torsion=t))
        for d in (10.0, 40.0):
            c.append(dataclasses.replace(base, name=f"imu_delay_{d:g}ms", imu_delay_ms=d))
        for a in (1, 3):
            c.append(dataclasses.replace(base, name=f"act_delay_{a}", act_delay=a))
        c.append(dataclasses.replace(base, name="filter_only", filter_tau=0.02))
        c.append(dataclasses.replace(base, name="hold_0.6_only", hold_s=0.6))
        c.append(dataclasses.replace(base, name="vx_0.2_only", cmd_vx=0.20))
        c.append(dataclasses.replace(base, name="profile_35rpm", profile_rpm=35.0))
        c.append(dataclasses.replace(base, name="ball_far_4cm", ball_bias_x=(0.04, 0.04)))
        seen, out = set(), []
        for x in c:
            if x.name not in seen:
                seen.add(x.name)
                out.append(x)
        return out
    if name == "baseline":
        return [base]
    if name == "gate":
        # checkpoint selection gate for v1_6 candidates (each cell >= 300 trials recommended)
        real = dataclasses.replace(base, name="real_like", volt=11.1, imu_delay_ms=10.0, hold_s=0.6,
                                   cmd_vx=0.20, filter_tau=0.02, mu=0.6, mu_torsion=0.1)
        return [
            fresh, base, deploy,
            dataclasses.replace(base, name="volt_11.1_dc", volt=11.1),
            dataclasses.replace(base, name="imu_delay_20ms", imu_delay_ms=20.0),
            dataclasses.replace(base, name="act_delay_2", act_delay=2),
            dataclasses.replace(base, name="mu_0.5", mu=0.5),
            dataclasses.replace(base, name="torsion_0.1", mu_torsion=0.1),
            real,
            dataclasses.replace(real, name="real_like_low_mu", mu=0.45),
            dataclasses.replace(real, name="real_like_strength_0.5", strength=0.5),
        ]
    if name == "push":
        # 킥 디딤발 쪽 외력 (KickPush). 실물 디딤발 쪽 전도 대용. 30899 는 8 N, 27500 은 4 N 부터 거의 다 넘어진다.
        real = dataclasses.replace(base, name="real_like", volt=11.1, imu_delay_ms=10.0, hold_s=0.6,
                                   cmd_vx=0.20, filter_tau=0.02, mu=0.6, mu_torsion=0.1)
        return [dataclasses.replace(real, name=f"push_{f:g}N", push_N=f) for f in (4.0, 6.0, 8.0, 10.0, 12.0)]
    raise SystemExit(f"unknown suite {name}")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class Gait:
    """Copy of runner GaitPhase (= Isaac PhaseOnlyGaitCommand)."""

    def __init__(self):
        self.phase, self.freq = 0.0, 0.0

    def update(self, dt, freq_cmd):
        self.freq = max(0.0, freq_cmd)
        if self.freq > 1e-6:
            self.phase = math.fmod(self.phase + dt * self.freq, 1.0)
            return
        step = GAIT_STOP_RATE * dt
        if self.phase > 0.5:
            self.phase = math.fmod(self.phase + min(1.0 - self.phase, step), 1.0)
        else:
            self.phase = self.phase - min(self.phase, step)
        if self.phase < 1e-6:
            self.phase = 0.0

    def obs(self):
        a = 2.0 * math.pi * self.phase
        return [math.cos(a), math.sin(a), self.freq / GAIT_FREQ_SCALE]


def build_obs(frames: np.ndarray) -> np.ndarray:
    """(5, 53) oldest->newest -> 265, term-major like Isaac concatenate_terms."""
    sl = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 24), (24, 36), (36, 48), (48, 50), (50, 52), (52, 53)]
    return np.concatenate([frames[:, a:b].reshape(-1) for a, b in sl]).astype(np.float32)


class Sim:
    def __init__(self, scene_path: str):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(scene_path)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.dt = m.opt.timestep
        jid = lambda n: m.joint(PREFIX + n).id
        self.all_joints = list(INIT_POSE)
        self.qadr = {n: m.jnt_qposadr[jid(n)] for n in self.all_joints}
        self.vadr = {n: m.jnt_dofadr[jid(n)] for n in self.all_joints}
        self.aid = {n: m.actuator(PREFIX + n).id for n in self.all_joints}
        self.leg_q = np.array([self.qadr[n] for n in POLICY_JOINTS])
        self.leg_v = np.array([self.vadr[n] for n in POLICY_JOINTS])
        self.leg_a = np.array([self.aid[n] for n in POLICY_JOINTS])
        self.rq = m.jnt_qposadr[m.joint(PREFIX + "root").id]
        self.rv = m.jnt_dofadr[m.joint(PREFIX + "root").id]
        self.bq = m.jnt_qposadr[m.joint("ball_root").id]
        self.bv = m.jnt_dofadr[m.joint("ball_root").id]
        self.base = m.body(PREFIX + "base_link").id
        self.feet = [m.body(PREFIX + f).id for f in FEET]
        self.floor = m.geom("floor").id
        sid = lambda n: m.sensor(PREFIX + n).id
        self.s_gyro = m.sensor_adr[sid("gyro")]
        self.s_quat = m.sensor_adr[sid("quat")]
        self.robot_geoms = [g for g in range(m.ngeom)
                            if m.body_rootid[m.geom_bodyid[g]] == self.base]
        self.ball_geom = m.geom("ball").id
        self.ball_mass = float(m.body_mass[m.body("ball").id])
        self.ball_I = m.body_inertia[m.body("ball").id].copy()
        self.default_fr = m.geom_friction.copy()
        self.default_solref = m.geom_solref.copy()
        self.default_forcerange = m.actuator_forcerange.copy()
        # position actuators are not ctrllimited; clamp to joint range like sim_bridge
        self.lo = np.array([m.jnt_range[m.actuator_trnid[a, 0]][0] for a in range(m.nu)])
        self.hi = np.array([m.jnt_range[m.actuator_trnid[a, 0]][1] for a in range(m.nu)])
        self.default12 = np.array([INIT_POSE[n] for n in POLICY_JOINTS], dtype=np.float32)

    def apply_cond(self, c: Cond):
        m = self.m
        m.geom_friction[:] = self.default_fr
        m.geom_solref[:] = self.default_solref
        for g in self.robot_geoms + [self.floor]:
            m.geom_friction[g] = [c.mu, c.mu_torsion, 0.0001]
            m.geom_solref[g] = [c.solref_tc, 1.0]
        m.actuator_forcerange[:] = self.default_forcerange

    def place(self, rng, c: Cond):
        mj, m, d = self.mj, self.m, self.d
        mj.mj_resetData(m, d)
        d.xfrc_applied[:] = 0.0
        d.qpos[self.rq:self.rq + 3] = [0.0, 0.0, 0.42]
        d.qpos[self.rq + 3:self.rq + 7] = [1.0, 0.0, 0.0, 0.0]
        for n in self.all_joints:
            d.qpos[self.qadr[n]] = INIT_POSE[n]
        self.setpoint = np.array([INIT_POSE.get(m.actuator(a).name[len(PREFIX):], 0.0)
                                  for a in range(m.nu)])
        self.target = self.setpoint.copy()
        d.qpos[self.bq:self.bq + 3] = [10.0, 10.0, 0.065]   # parked during settle
        d.qpos[self.bq + 3:self.bq + 7] = [1, 0, 0, 0]
        mj.mj_forward(m, d)

    def step_physics(self, c: Cond, prof_rad_s: np.ndarray):
        m, d = self.m, self.d
        step = np.where(prof_rad_s > 0, prof_rad_s * self.dt, np.inf)
        self.setpoint += np.clip(self.target - self.setpoint, -step, step)
        d.ctrl[:] = np.clip(self.setpoint, self.lo, self.hi)
        if c.volt > 0:
            # Isaac Lab DCMotor rule (quadrant aware): driving torque falls with speed,
            # braking torque (opposite to velocity) is only capped by EFFORT_CAP.
            k = c.volt / 12.0
            w = d.qvel[self.leg_v]
            sat, vnl = STALL_12V * k, NOLOAD_12V * k
            top = np.minimum(sat * (1.0 - w / vnl), EFFORT_CAP)
            bot = np.maximum(sat * (-1.0 - w / vnl), -EFFORT_CAP)
            m.actuator_forcerange[self.leg_a, 0] = np.minimum(bot, -0.05)
            m.actuator_forcerange[self.leg_a, 1] = np.maximum(top, 0.05)
        v = d.qvel[self.bv:self.bv + 6]
        d.qfrc_applied[self.bv:self.bv + 3] = -0.2 * self.ball_mass * v[:3]
        d.qfrc_applied[self.bv + 3:self.bv + 6] = -0.2 * self.ball_I * v[3:]
        self.mj.mj_step(m, d)

    def R_base(self):
        return self.d.xmat[self.base].reshape(3, 3).copy()

    def imu(self):
        g = self.d.sensordata[self.s_gyro:self.s_gyro + 3].copy()
        q = self.d.sensordata[self.s_quat:self.s_quat + 4].copy()
        R = quat_to_R(q)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return g, grav

    def foot_contact(self):
        """normal force per foot from floor contacts."""
        m, d = self.m, self.d
        out = np.zeros(2)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = con.geom1, con.geom2
            if self.floor not in (g1, g2):
                continue
            other = g2 if g1 == self.floor else g1
            b = m.geom_bodyid[other]
            for k, fb in enumerate(self.feet):
                if b == fb:
                    self.mj.mj_contactForce(m, d, i, f6)
                    out[k] += abs(f6[0])
        return out


class KickPush:
    """킥 디딤발 쪽 외력. 정책 스텝마다 update() (물리 스텝 뒤) 를 부른다."""
    _ROLL = None

    def __init__(self, sim, c):
        self.sim, self.c = sim, c
        self.t_left = -1.0          # 남은 시간, -1 = 대기
        self.done = False
        self.F = np.zeros(3)
        self.start_t = -1.0
        if KickPush._ROLL is None:
            KickPush._ROLL = (POLICY_JOINTS.index("r_pelvis_roll_joint"), POLICY_JOINTS.index("l_pelvis_roll_joint"))

    def update(self, t):
        sim, c = self.sim, self.c
        if c.push_N <= 0:
            return
        if self.t_left < 0 and not self.done:
            dev = np.abs(sim.d.qpos[sim.leg_q] - sim.default12)
            fn = sim.foot_contact()          # (r, l)
            for kick, sup in ((0, 1), (1, 0)):
                if fn[kick] < 5.0 and fn[sup] > 5.0 and math.degrees(dev[KickPush._ROLL[kick]]) > c.push_trig_deg:
                    R = sim.R_base()
                    y = R[:, 1].copy(); y[2] = 0.0
                    y /= max(np.linalg.norm(y), 1e-6)
                    sgn = 1.0 if sup == 1 else -1.0       # 몸 +y = 왼쪽
                    self.F = sgn * c.push_N * y
                    self.t_left = c.push_s
                    self.start_t = t
                    break
        if self.t_left >= 0:
            sim.d.xfrc_applied[sim.base, :3] = self.F
            self.t_left -= POLICY_DT
            if self.t_left < 0:
                self.done = True
        else:
            sim.d.xfrc_applied[sim.base, :3] = 0.0


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def run_trial(sim: Sim, sess, c: Cond, seed: int, log_rows: list | None = None) -> dict:
    rng = np.random.default_rng(seed)
    sim.apply_cond(c)
    sim.place(rng, c)
    nu = sim.m.nu
    prof_all = np.full(nu, c.profile_rpm * RPM)
    # settle standing 1.0 s at default pose (robot_init done, torque on)
    for _ in range(int(1.0 / sim.dt)):
        sim.step_physics(c, prof_all)
    # sample task, place ball relative to robot heading
    R0 = sim.R_base()
    yaw0 = yaw_of(R0)
    p0 = sim.d.xpos[sim.base].copy()
    dist = rng.uniform(*c.ball_dist)
    brg = math.radians(rng.uniform(*c.ball_bearing_deg))
    tgt = math.radians(rng.uniform(*c.target_deg))
    bel_b = np.array([dist * math.cos(brg), dist * math.sin(brg)])
    bias = np.array([rng.uniform(*c.ball_bias_x), 0.0])
    true_b = bel_b + bias
    cy, sy = math.cos(yaw0), math.sin(yaw0)
    bw = p0[:2] + np.array([cy * true_b[0] - sy * true_b[1], sy * true_b[0] + cy * true_b[1]])
    sim.d.qpos[sim.bq:sim.bq + 3] = [bw[0], bw[1], 0.065]
    sim.d.qvel[sim.bv:sim.bv + 6] = 0.0
    sim.mj.mj_forward(sim.m, sim.d)
    tgt_w = yaw0 + tgt
    tgt_dir_w = np.array([math.cos(tgt_w), math.sin(tgt_w)])

    gait = Gait()
    last_a = np.zeros(12, np.float32)
    filt = np.zeros(12, np.float32)
    alpha = math.exp(-POLICY_DT / c.filter_tau) if c.filter_tau > 0 else 0.0
    goal_q = deque([sim.default12.copy()] * (c.act_delay + 1), maxlen=c.act_delay + 1)
    n_sub = int(round(POLICY_DT / sim.dt))
    imu_buf = deque(maxlen=64)
    imu_lag = int(round(c.imu_delay_ms / 1000.0 / sim.dt))
    hist = None
    steps = int(round(c.duration_s / POLICY_DT))
    fell_t, kick_t, peak_v = None, None, 0.0
    foot_yaw0 = [None, None]
    max_stance_spin = 0.0
    max_stance_slip = 0.0
    stance_xy0 = [None, None]
    base_yaw_rate_max = 0.0
    min_height = 9.0
    max_tilt = 0.0
    sp_rate = c.profile_rpm * RPM

    def sense():
        g, grav = sim.imu()
        imu_buf.append((g, grav))
        lag = min(imu_lag, len(imu_buf) - 1)
        return imu_buf[-1 - lag]

    sense()
    push = KickPush(sim, c)
    for k in range(steps):
        t = k * POLICY_DT
        in_hold = t < c.hold_s
        cmd = np.zeros(3, np.float32) if in_hold else np.array([c.cmd_vx, 0, 0], np.float32)
        freq = 0.0 if (in_hold or np.linalg.norm(cmd) < GAIT_STAND_TH) else GAIT_FREQ
        gait.update(POLICY_DT if k > 0 else 0.0, freq)
        R = sim.R_base()
        dyaw = yaw_of(R) - yaw0
        tb = np.array([math.cos(tgt_w - yaw0 - dyaw), math.sin(tgt_w - yaw0 - dyaw)])
        g, grav = imu_buf[-1 - min(imu_lag, len(imu_buf) - 1)]
        q = sim.d.qpos[sim.leg_q].astype(np.float32)
        qd = sim.d.qvel[sim.leg_v].astype(np.float32)
        def make_frame(la):
            return np.concatenate([0.2 * g, grav, cmd, gait.obs(), q - sim.default12, 0.05 * qd,
                                   la, bel_b, tb, [c.strength]]).astype(np.float32)

        if hist is None:
            hist = deque([make_frame(last_a)] * 5, maxlen=5)
        elif not c.runner_stale_obs:
            hist.append(make_frame(last_a))      # Isaac: newest frame = sensors now
        obs = build_obs(np.stack(hist))
        a = sess.run(None, {sess.get_inputs()[0].name: obs[None]})[0].reshape(12).astype(np.float32)
        a = np.clip(a, -10.0, 10.0)
        if c.runner_stale_obs:
            # actor_onnx_walkkick_v1_runner: history.push(one_frame()) at the END of the tick,
            # so the next inference sees sensors of this tick (one policy step = 20 ms old).
            hist.append(make_frame(a))
        if c.filter_tau > 0:
            filt = alpha * filt + (1 - alpha) * a
            a_cmd = filt
        else:
            a_cmd = a
        q_tgt = sim.default12 + 0.25 * a_cmd
        if in_hold:
            q_tgt = sim.default12.copy()
        last_a = a
        goal_q.append(q_tgt)
        applied = goal_q[0]
        # bridge: goals published now are written at the next 20 ms cycle (1 cycle lag)
        for _ in range(n_sub):
            sim.step_physics(c, prof_all)
            sense()
        for i, n in enumerate(POLICY_JOINTS):
            sim.target[sim.aid[n]] = applied[i]
        push.update(t)
        # metrics
        R = sim.R_base()
        tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
        h = sim.d.xpos[sim.base][2]
        max_tilt, min_height = max(max_tilt, tilt), min(min_height, h)
        if fell_t is None and (tilt > FALL_TILT or h < FALL_HEIGHT):
            fell_t = t
        bvel = sim.d.qvel[sim.bv:sim.bv + 2]
        vproj = float(bvel @ tgt_dir_w)
        if kick_t is None and np.linalg.norm(bvel) > KICK_SPEED:
            kick_t = t
        if kick_t is not None and t - kick_t <= 0.5:
            peak_v = max(peak_v, vproj)
        base_yaw_rate_max = max(base_yaw_rate_max, abs(float(sim.d.qvel[sim.rv + 5])))
        fn = sim.foot_contact()
        for i, fb in enumerate(sim.feet):
            Rf = sim.d.xmat[fb].reshape(3, 3)
            if fn[i] > 5.0:
                if foot_yaw0[i] is None:
                    foot_yaw0[i] = yaw_of(Rf)
                    stance_xy0[i] = sim.d.xpos[fb][:2].copy()
                dy = (yaw_of(Rf) - foot_yaw0[i] + math.pi) % (2 * math.pi) - math.pi
                if kick_t is not None and t - kick_t < 1.0 and fell_t is None:
                    max_stance_spin = max(max_stance_spin, abs(dy))
                    max_stance_slip = max(max_stance_slip, float(np.linalg.norm(sim.d.xpos[fb][:2] - stance_xy0[i])))
            else:
                foot_yaw0[i] = None
                stance_xy0[i] = None
        if log_rows is not None:
            log_rows.append(dict(t=t, tilt=tilt, h=h, yaw_rate=float(sim.d.qvel[sim.rv + 5]),
                                 fn_r=fn[0], fn_l=fn[1], vball=float(np.linalg.norm(bvel)),
                                 q_tgt_max=float(np.max(np.abs(q_tgt - sim.default12)))))
        if fell_t is not None and t - fell_t > 0.3:
            break
    return dict(seed=seed, fell=fell_t is not None, fell_t=fell_t if fell_t is not None else -1.0,
                kicked=kick_t is not None, kick_t=kick_t if kick_t is not None else -1.0,
                fell_after_kick=(fell_t is not None and kick_t is not None and fell_t >= kick_t),
                peak_v=peak_v, stance_spin_deg=math.degrees(max_stance_spin),
                stance_slip_cm=100.0 * max_stance_slip, yaw_rate_max=base_yaw_rate_max,
                max_tilt_deg=math.degrees(max_tilt), dist=dist, bearing_deg=math.degrees(brg),
                target_deg=math.degrees(tgt), pushed=push.start_t >= 0)


_W = {}


def _init_worker(scene, policies):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    _W["sim"] = Sim(scene)
    _W["sess"] = {p: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for p in policies}


def _work(job):
    pol, cond, seed = job
    r = run_trial(_W["sim"], _W["sess"][pol], cond, seed)
    r.update(policy=os.path.basename(pol), cond=cond.name)
    return r


def summarize(rows):
    out = {}
    for r in rows:
        key = (r["policy"], r["cond"])
        out.setdefault(key, []).append(r)
    res = []
    for (pol, cond), rs in out.items():
        n = len(rs)
        f = sum(r["fell"] for r in rs)
        kicked = [r for r in rs if r["kicked"]]
        fak = sum(r["fell_after_kick"] for r in rs)
        p = f / n
        ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)
        res.append(dict(
            policy=pol, cond=cond, n=n, fall_pct=100 * p, fall_ci95_pct=100 * ci,
            kick_pct=100 * len(kicked) / n, fall_after_kick_pct=100 * fak / n,
            pushed_pct=100 * sum(bool(r.get("pushed", False)) for r in rs) / n,
            peak_v_mean=float(np.mean([r["peak_v"] for r in kicked])) if kicked else 0.0,
            stance_spin_deg_p90=float(np.percentile([r["stance_spin_deg"] for r in kicked], 90)) if kicked else 0.0,
            stance_slip_cm_p90=float(np.percentile([r["stance_slip_cm"] for r in kicked], 90)) if kicked else 0.0,
            yaw_rate_p90=float(np.percentile([r["yaw_rate_max"] for r in rs], 90)),
        ))
    return res


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", help="kick ONNX (repeatable). default: XM_Kick_Policy/policy/policy.onnx")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--suite", default="quick", choices=["baseline", "quick", "full", "gate", "push"])
    ap.add_argument("--only", default="", help="comma separated condition names to keep")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--trace", default="", help="cond name: save per-step trace of first seed")
    ap.add_argument("--strength", type=float, default=None, help="override kick_strength (0..1) for all conditions")
    args = ap.parse_args()
    pols = [os.path.abspath(p) for p in (args.policy or [DEFAULT_POLICY])]
    conds = suite(args.suite)
    if args.only:
        keep = set(args.only.split(","))
        conds = [c for c in conds if c.name in keep]
    if args.strength is not None:
        conds = [dataclasses.replace(c, name=f"{c.name}@s{args.strength:g}", strength=args.strength) for c in conds]
    os.makedirs(args.out, exist_ok=True)
    jobs = [(p, c, args.seed + i) for p in pols for c in conds for i in range(args.trials)]
    print(f"[stress] {len(pols)} policy x {len(conds)} cond x {args.trials} = {len(jobs)} trials, "
          f"{args.workers} workers", flush=True)
    t0 = time.time()
    if args.trace:
        c = next(x for x in conds if x.name == args.trace)
        _init_worker(args.scene, pols)
        rows = []
        run_trial(_W["sim"], _W["sess"][pols[0]], c, args.seed, rows)
        with open(os.path.join(args.out, f"trace_{c.name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(args.scene, pols)) as pool:
            rows = []
            for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
                rows.append(r)
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    else:
        _init_worker(args.scene, pols)
        rows = [_work(j) for j in jobs]
    summ = summarize(rows)
    order = {c.name: i for i, c in enumerate(conds)}
    summ.sort(key=lambda s: (s["policy"], order.get(s["cond"], 99)))
    with open(os.path.join(args.out, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = dict(policies={os.path.basename(p): sha256(p) for p in pols}, scene=os.path.relpath(args.scene, REPO),
                scene_sha256=sha256(args.scene), suite=args.suite, trials=args.trials, seed=args.seed,
                conditions=[dataclasses.asdict(c) for c in conds], wall_s=time.time() - t0,
                mujoco=__import__("mujoco").__version__)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, summary=summ), f, indent=1, ensure_ascii=False)
    print(f"{'policy':<22}{'cond':<22}{'n':>4}{'fall%':>8}{'±95':>6}{'kick%':>7}{'v':>6}{'spin90':>8}{'slip90':>8}")
    for s in summ:
        print(f"{s['policy'][:21]:<22}{s['cond']:<22}{s['n']:>4}{s['fall_pct']:>8.1f}{s['fall_ci95_pct']:>6.1f}"
              f"{s['kick_pct']:>7.1f}{s['peak_v_mean']:>6.2f}{s['stance_spin_deg_p90']:>8.1f}{s['stance_slip_cm_p90']:>8.1f}")
    print(f"[stress] done {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
