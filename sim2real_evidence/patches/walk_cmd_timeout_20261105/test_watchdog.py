"""Exercise the actual patched methods without importing ROS or opening hardware."""
import ast
from collections import deque
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
TREE = ast.parse((ROOT / 'actor_onnx_walking_runner.py').read_text(encoding='utf-8'))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'ActorOnnxWalkingNode')
METHODS = [n for n in CLASS.body if isinstance(n, ast.FunctionDef) and
           n.name in {'cb_cmd_vel', 'effective_cmd_vel', 'control_tick'}]
FAKE_CLASS = ast.ClassDef(name='OfflineActor', bases=[], keywords=[], body=METHODS, decorator_list=[])
HELPERS = [n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
           n.name in {'FrameHistory', 'build_obs_from_frames', '_make_frame_spec'}]


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.clock = 100.0
        ns = {'np': np, 'math': math, 'deque': deque, 'HISTORY_LEN': 5,
              'Twist': object,
              'time': SimpleNamespace(monotonic=lambda: self.clock,
                                      time=lambda: self.clock, perf_counter=lambda: self.clock)}
        module = ast.fix_missing_locations(ast.Module(body=HELPERS + [FAKE_CLASS], type_ignores=[]))
        exec(compile(module, 'actual_patched_methods', 'exec'), ns)
        self.actor = ns['OfflineActor']()
        a = self.actor
        a.cmd = np.array([0.3, 0.1, -0.2], dtype=np.float32)
        a.cmd_vel_timeout_s = 0.5
        a._last_cmd_vel_monotonic = None
        self.ns = ns

    def send(self, vx=0.2, vy=0.0, wz=0.0):
        self.actor.cb_cmd_vel(SimpleNamespace(linear=SimpleNamespace(x=vx, y=vy),
                                             angular=SimpleNamespace(z=wz)))

    def test_startup_without_command_zeroes_parameter_velocity(self):
        command, fresh = self.actor.effective_cmd_vel()
        self.assertFalse(fresh)
        np.testing.assert_array_equal(command, [0, 0, 0])

    def test_fresh_command_preserved(self):
        self.send()
        self.clock += 0.49
        command, fresh = self.actor.effective_cmd_vel()
        self.assertTrue(fresh)
        self.assertAlmostEqual(float(command[0]), 0.2)

    def test_expiry_clears_cached_command_and_fresh_command_recovers(self):
        self.send()
        self.clock += 0.501
        command, fresh = self.actor.effective_cmd_vel()
        self.assertFalse(fresh)
        np.testing.assert_array_equal(command, [0, 0, 0])
        self.send(0.1)
        self.assertTrue(self.actor.effective_cmd_vel()[1])

    def test_nonfinite_command_invalidates_old_motion(self):
        self.send()
        self.send(float('nan'))
        self.assertIsNone(self.actor._last_cmd_vel_monotonic)
        np.testing.assert_array_equal(self.actor.effective_cmd_vel()[0], [0, 0, 0])

    def test_explicit_zero_is_a_fresh_command(self):
        self.send(0)
        command, fresh = self.actor.effective_cmd_vel()
        self.assertTrue(fresh)
        np.testing.assert_array_equal(command, [0, 0, 0])

    def configure_tick(self):
        a = self.actor
        a.ready = lambda: True
        a.policy_enabled = lambda: True
        a._kick_yield = False
        a.extract_q_qd_from_joint_state = lambda: (np.zeros(12, np.float32), np.zeros(12, np.float32))
        a.start_wall_time = 0
        a.startup_hold_seconds = 0
        a.resume_time = None
        a._resume_hold_dur = 0
        a.policy_dt = 0.02
        self.resolved = []
        a.resolve_gait_frequency = lambda *args: self.resolved.append(args) or 2.0
        self.frequencies = []
        a.gait = SimpleNamespace(update=lambda dt, freq: self.frequencies.append(freq))
        a.one_frame = lambda cmd: np.zeros(48, np.float32)
        a.history_initialized = False
        a.history = self.ns['FrameHistory'](5, 48)
        a.frame_slices, _ = self.ns['_make_frame_spec'](3)
        a.obs_dim = 240
        a.input_name, a.output_name = 'obs', 'actions'
        self.inferences = []
        a.session = SimpleNamespace(run=lambda *args: self.inferences.append(args) or [np.zeros((1, 12), np.float32)])
        a.clip_actions = 10
        a.enable_action_filter = False
        a.default_q12 = np.zeros(12, np.float32)
        a.default_qd12 = np.zeros(12, np.float32)
        a.joint_vel_scale = 0.05
        a.action_scale = 0.25
        a.pelvis_pitch_offset = 0
        a.max_target_abs_from_default = 10.80
        a.max_target_delta_from_current = 10.35
        self.goals = []
        a.publish_joint_goals = lambda q: self.goals.append(q)
        a.publish_gait_debug = lambda: None
        a._log_csv_row = lambda *args, **kwargs: None
        a.tick_count, a.debug_print_every = 0, 0

    def test_control_tick_expiry_forces_zero_frequency_and_keeps_standing_inference(self):
        self.configure_tick()
        self.send()
        self.clock += 0.6
        self.actor.control_tick()
        self.assertEqual(self.resolved, [])
        self.assertEqual(self.frequencies, [0.0])
        self.assertEqual(len(self.inferences), 1)
        self.assertEqual(len(self.goals), 1)
        np.testing.assert_array_equal(self.actor.cmd, [0, 0, 0])

    def test_control_tick_fresh_motion_retains_normal_frequency(self):
        self.configure_tick()
        self.send()
        self.actor.control_tick()
        self.assertEqual(len(self.resolved), 1)
        self.assertEqual(self.frequencies, [2.0])


if __name__ == '__main__':
    unittest.main()
