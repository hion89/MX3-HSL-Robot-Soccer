"""Offline unit tests for XM_Kick_Policy/.../mdp/kick_robust_mdp.py (no Isaac Sim / Isaac Lab needed).

Isaac Lab modules are replaced by minimal stubs so the real source file is imported unchanged and its
pure tensor parts are checked: DC clipping, delay interpolation, DelayedObs buffering, turf torsion torque,
low-pass alpha (vs runner formula), random hold, voltage sampling, curriculum stepping.
Also checks that the MuJoCo evaluator uses the same DC rule.

  python test_kick_robust_math.py      (needs torch, numpy)
"""
import importlib.util
import math
import os
import sys
import types
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MDP = os.path.join(REPO, "XM_Kick_Policy", "unitree_rl_lab_overlay", "source", "unitree_rl_lab", "unitree_rl_lab",
                   "tasks", "locomotion", "mdp", "kick_robust_mdp.py")


def _stub_modules():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class DCMotor:
        def __init__(self, cfg, *a, **k):
            self.cfg = cfg
            self.computed_effort = torch.zeros(k.get("num_envs", 2), 2)

    class DCMotorCfg:
        pass

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    class SceneEntityCfg:
        def __init__(self, name, **kw):
            self.name = name
            self.__dict__.update(kw)

    def configclass(c):
        return c

    def quat_apply_inverse(q, v):
        # q (w,x,y,z): rotate v by q^-1
        w, x, y, z = q.unbind(-1)
        qv = torch.stack([-x, -y, -z], -1)
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + w.unsqueeze(-1) * t + torch.cross(qv, t, dim=-1)

    mod("isaaclab")
    mod("isaaclab.actuators", DCMotor=DCMotor, DCMotorCfg=DCMotorCfg)
    mod("isaaclab.managers", ManagerTermBase=ManagerTermBase, SceneEntityCfg=SceneEntityCfg)
    mod("isaaclab.utils", configclass=configclass)
    mod("isaaclab.utils.math", quat_apply_inverse=quat_apply_inverse)
    mod("isaaclab.envs", mdp=types.SimpleNamespace(joint_pos_rel=None, joint_vel_rel=None))
    pkg = mod("fakemdp")
    pkg.__path__ = []
    mod("fakemdp.kick_clock", kick_clock_s=lambda env: env.episode_length_buf.float() * env.step_dt)

    class PV:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    mod("fakemdp.kick_handoff_mdp", ProfileVelocityJointPositionAction=PV, ProfileVelocityJointPositionActionCfg=object)
    mod("fakemdp.kick_stability_mdp", base_ang_vel_biased=None, projected_gravity_biased=None)


def _load():
    _stub_modules()
    spec = importlib.util.spec_from_file_location("fakemdp.kick_robust_mdp", MDP)
    m = importlib.util.module_from_spec(spec)
    sys.modules["fakemdp.kick_robust_mdp"] = m
    spec.loader.exec_module(m)
    return m


kr = _load()


class Env:
    def __init__(self, n=4):
        self.num_envs, self.device, self.step_dt = n, "cpu", 0.02
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)


class TestDC(unittest.TestCase):
    def test_flat_when_beta0(self):
        e = torch.tensor([[9.0, -9.0, 3.0]])
        qd = torch.tensor([[4.0, 4.0, 4.0]])
        out = kr.dc_clip_effort(e, qd, 7.3, 5.55, 5.0, torch.ones(1, 1), 0.0)
        self.assertTrue(torch.allclose(out, torch.tensor([[5.0, -5.0, 3.0]])))

    def test_curve_quadrants(self):
        qd = torch.tensor([[0.0, 4.0, 4.0, -4.0]])
        e = torch.tensor([[9.0, 9.0, -9.0, -9.0]])
        out = kr.dc_clip_effort(e, qd, 7.3, 5.55, 5.0, torch.ones(1, 1), 1.0)
        drive = 7.3 * (1 - 4 / 5.55)
        # stall capped by 5, driving at 4 rad/s limited, braking keeps cap, mirrored driving limited
        self.assertTrue(torch.allclose(out, torch.tensor([[5.0, drive, -5.0, -drive]]), atol=1e-5))

    def test_voltage_scale(self):
        k = torch.tensor([[10.6 / 12.0]])
        out = kr.dc_clip_effort(torch.tensor([[9.0]]), torch.tensor([[4.0]]), 7.3, 5.55, 5.0, k, 1.0)
        s, v = 7.3 * k.item(), 5.55 * k.item()
        self.assertAlmostEqual(out.item(), s * (1 - 4 / v), places=5)

    def test_tensor_cap_and_blend(self):
        cap = torch.full((2, 1), 5.0)
        out = kr.dc_clip_effort(torch.tensor([[9.0], [9.0]]), torch.tensor([[4.0], [4.0]]), 7.3, 5.55, cap,
                                torch.ones(2, 1), torch.tensor([[0.5], [1.0]]))
        drive = 7.3 * (1 - 4 / 5.55)
        self.assertAlmostEqual(out[0].item(), 5.0 + 0.5 * (drive - 5.0), places=5)
        self.assertAlmostEqual(out[1].item(), drive, places=5)

    def test_matches_mujoco_evaluator_rule(self):
        # same rule as mujoco_kick_stress.Sim.step_physics (forcerange from quadrant-aware DC curve)
        rng = np.random.default_rng(0)
        for _ in range(200):
            w = rng.uniform(-6, 6)
            e = rng.uniform(-10, 10)
            volt = rng.uniform(10.6, 12.4)
            k = volt / 12.0
            sat, vnl = 7.3 * k, 5.55 * k
            top = min(sat * (1 - w / vnl), 5.0)
            bot = max(sat * (-1 - w / vnl), -5.0)
            ref = min(max(e, bot), top)
            got = kr.dc_clip_effort(torch.tensor([[e]]), torch.tensor([[w]]), 7.3, 5.55, 5.0,
                                    torch.tensor([[k]]), 1.0).item()
            self.assertAlmostEqual(got, ref, places=4)


class TestDelay(unittest.TestCase):
    def test_interp(self):
        buf = torch.tensor([[[10.0], [8.0], [6.0], [4.0]]])
        for d, ref in ((0.0, 10.0), (1.0, 8.0), (1.5, 7.0), (3.0, 4.0), (9.0, 4.0)):
            self.assertAlmostEqual(kr.interp_delayed(buf, torch.tensor([d])).item(), ref, places=5)

    def test_delayed_obs_buffer(self):
        env = Env(n=2)
        cfg = types.SimpleNamespace(params={"inner": "ang_vel", "kind": "imu", "max_delay_steps": 3})
        term = kr.DelayedObs(cfg, env)
        state = {"v": torch.zeros(2, 1)}
        term._inner = lambda e, a: state["v"].clone()
        env.kick_obs_delay = {"imu": torch.tensor([0.0, 1.0]), "joint": torch.zeros(2)}
        outs = []
        for step in range(5):
            env.common_step_counter = step
            state["v"] = torch.full((2, 1), float(step))
            o = term(env, inner="ang_vel")
            o2 = term(env, inner="ang_vel")            # same step: must not push twice
            self.assertTrue(torch.equal(o, o2))
            outs.append(o.squeeze(-1).tolist())
        # env0 no delay -> current, env1 one step delay -> previous (first step fills with current)
        self.assertEqual([x[0] for x in outs], [0, 1, 2, 3, 4])
        self.assertEqual([x[1] for x in outs], [0, 0, 1, 2, 3])
        # reset env1: next call refills its buffer with the current value (no stale pre-reset data)
        term.reset(torch.tensor([1]))
        env.common_step_counter = 5
        state["v"] = torch.full((2, 1), 50.0)
        o = term(env, inner="ang_vel")
        self.assertEqual(o.squeeze(-1).tolist(), [50.0, 50.0])

    def test_sample_obs_delays(self):
        env = Env(n=2000)
        torch.manual_seed(0)
        kr.sample_obs_delays(env, torch.arange(2000), 0.5, 1.0, (0.0, 1.5))
        j, i = env.kick_obs_delay["joint"], env.kick_obs_delay["imu"]
        self.assertTrue(set(j.unique().tolist()) <= {0.0, 1.0})
        self.assertAlmostEqual(j.mean().item(), 0.5, delta=0.05)
        self.assertTrue(torch.all(i >= j) and torch.all(i <= j + 1.5))


class TestTurfAndMisc(unittest.TestCase):
    def test_torsion(self):
        fn = torch.tensor([[50.0, 0.0], [50.0, 50.0]])
        c = torch.tensor([0.02, 0.0])
        wz = torch.tensor([-3.0, 2.0])
        tz = kr.torsion_deficit_torque_z(fn, c, wz, 0.5, 5.0)
        self.assertAlmostEqual(tz[0, 0].item(), 0.02 * 50 * math.tanh(-6.0), places=5)
        self.assertEqual(tz[0, 1].item(), 0.0)                 # foot in the air
        self.assertTrue(torch.all(tz[1] == 0))                 # hard floor env

    def test_quat_body_frame(self):
        # foot yawed 90 deg: world z torque stays body z (yaw axis is shared)
        h = math.radians(90) / 2
        q = torch.tensor([[math.cos(h), 0.0, 0.0, math.sin(h)]])
        v = kr._quat_apply_inverse(q, torch.tensor([[0.0, 0.0, 1.5]]))
        self.assertTrue(torch.allclose(v, torch.tensor([[0.0, 0.0, 1.5]]), atol=1e-6))

    def test_lowpass_matches_runner(self):
        tau = torch.tensor([0.02, 0.0, 0.03])
        a = kr.lowpass_alpha(0.02, tau)
        self.assertAlmostEqual(a[0].item(), math.exp(-1.0), places=6)   # runner: exp(-policy_dt/tau)
        self.assertEqual(a[1].item(), 0.0)
        # filtered sequence equals runner recursion filt = a*filt + (1-a)*x
        x = torch.randn(30)
        f, r = torch.zeros(()), 0.0
        al = a[0].item()
        for v in x:
            f = a[0] * f + (1 - a[0]) * v
            r = al * r + (1 - al) * v.item()
        self.assertAlmostEqual(f.item(), r, places=5)

    def test_hold_random(self):
        env = Env(n=500)
        kr.set_reset_hold_random(env, torch.arange(500), (0.3, 0.6))
        s = env.reset_hold_steps
        self.assertTrue(int(s.min()) >= 15 and int(s.max()) <= 30)

    def test_voltage_and_curriculum(self):
        env = Env(n=8)
        motor = kr.VoltageDCMotor.__new__(kr.VoltageDCMotor)
        motor.volt_scale = torch.ones(8, 1)
        motor.dc_blend = 0.25
        env.scene = {"robot": types.SimpleNamespace(actuators={"knee": motor, "arm": object()})}
        kr.randomize_motor_voltage(env, torch.arange(8), (10.6, 12.4))
        self.assertTrue(torch.all((motor.volt_scale >= 10.6 / 12 - 1e-6) & (motor.volt_scale <= 12.4 / 12 + 1e-6)))

        class TM:
            active_terms = ["time_out", "fall_stand", "fall_nominal", "kick_resolved"]

            def __init__(self):
                self.v = {n: torch.zeros(8, dtype=torch.bool) for n in self.active_terms}

            def get_term(self, n):
                return self.v[n]

        env.termination_manager = TM()
        b = kr.dc_motor_blend_curriculum(env, None, start=0.25, step=0.25, fall_threshold=0.12, min_episodes=16,
                                         ema_window=16.0)
        self.assertEqual(b, 0.25)
        env.common_step_counter = 10
        for _ in range(3):                      # 24 episodes with no falls -> one step up
            kr.dc_motor_blend_curriculum(env, torch.arange(8), start=0.25, step=0.25, fall_threshold=0.12,
                                         min_episodes=16, ema_window=16.0)
        self.assertEqual(motor.dc_blend, 0.5)
        env.termination_manager.v["fall_nominal"][:] = True      # all falling -> stays
        for _ in range(10):
            kr.dc_motor_blend_curriculum(env, torch.arange(8), start=0.25, step=0.25, fall_threshold=0.12,
                                         min_episodes=16, ema_window=16.0)
        self.assertEqual(motor.dc_blend, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
