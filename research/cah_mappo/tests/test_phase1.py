"""Analytical expectations are fixed before measurement; no ROS or ML packages."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cah_mappo.core import Config, Simulator, can_respond, scenario_config, write_jsonl
from cah_mappo.metrics import evaluate, load_jsonl


def trial(scenario='normal', policy='ready', **kwargs):
    rows = Simulator(scenario_config(scenario, policy, **kwargs)).run()
    return rows, evaluate(rows)


class ExpectedScenarios(unittest.TestCase):
    def test_normal_ready_no_gap_response8_complete12(self):
        rows, m = trial()
        self.assertEqual(m['coverage_gap_ratio'], 0)
        self.assertEqual(m['task_success_rate'], 1)
        self.assertEqual(m['response_times_s'], [8])
        self.assertEqual(m['handovers'][0]['release'], 6)
        self.assertEqual(m['handovers'][0]['initial_gap_s'], 0)
        self.assertEqual(next(e['time'] for e in rows if e.get('event') == 'task_completed'), 12)

    def test_early_release_gap4_and_response4(self):
        _, m = trial(policy='immediate')
        self.assertEqual(m['coverage_gap_s'], 4)
        self.assertEqual(m['coverage_gap_ratio'], .25)
        self.assertEqual(m['ownership_gap_ratio'], 0)
        self.assertEqual(m['declared_owner_not_ready_ratio'], .25)
        self.assertEqual(m['response_times_s'], [4])
        self.assertEqual(m['task_success_rate'], 1)
        self.assertEqual(m['handovers'][0]['initial_gap_s'], 4)

    def test_late_release_delays_response2(self):
        _, normal = trial()
        _, late = trial(policy='late')
        self.assertEqual(late['coverage_gap_ratio'], 0)
        self.assertEqual(late['response_times_s'], [10])
        self.assertEqual(late['response_time_mean_s'] - normal['response_time_mean_s'], 2)
        self.assertEqual(late['task_success_rate'], 1)

    def test_urgent_tradeoff_ready_misses_immediate_succeeds(self):
        _, ready = trial('urgent')
        _, fast = trial('urgent', 'immediate')
        self.assertEqual(ready['task_success_rate'], 0)
        self.assertEqual(ready['coverage_gap_ratio'], 0)
        self.assertEqual(ready['response_time_mean_s'], None)
        self.assertEqual(ready['unresponded_tasks'], 1)
        self.assertEqual(ready['restricted_response_mean_s'], 7)
        self.assertEqual(fast['task_success_rate'], 1)
        self.assertEqual(fast['response_times_s'], [4])
        self.assertEqual(fast['coverage_gap_ratio'], .25)

    def test_failed_successor_ready_retains_old_responsibility(self):
        _, m = trial('successor_failure')
        self.assertEqual(m['coverage_gap_ratio'], 0)
        self.assertEqual(m['handover_count'], 0)
        self.assertEqual(m['task_success_rate'], 0)
        self.assertEqual(m['unresponded_tasks'], 1)
        self.assertEqual(m['failure_recoveries'][0]['coverage_recovery_s'], 0)

    def test_failed_successor_immediate_censored_gap14(self):
        _, m = trial('successor_failure', 'immediate')
        self.assertEqual(m['coverage_gap_s'], 14)
        self.assertEqual(m['handovers'][0]['initial_gap_s'], 14)
        self.assertTrue(m['handovers'][0]['right_censored'])
        self.assertTrue(m['failure_recoveries'][0]['right_censored'])
        self.assertIsNone(m['failure_recoveries'][0]['coverage_recovery_s'])

    def test_roundtrip_delay1_request3_ready7_release8_complete14(self):
        rows, m = trial('communication_delay')
        self.assertEqual(m['response_times_s'], [10])
        self.assertEqual(m['task_success_rate'], 1)
        self.assertEqual(m['handovers'][0]['release'], 8)
        received = [e for e in rows if e.get('event') == 'message_received']
        self.assertEqual([e['time'] for e in received], [3, 8])
        self.assertTrue(all(e['age'] == 1 for e in received))

    def test_temporary_failure_coverage_restored_at9(self):
        _, m = trial('temporary_failure', 'immediate')
        self.assertEqual(m['coverage_gap_s'], 7)
        self.assertEqual(m['failure_recoveries'][0]['coverage_recovery_s'], 5)
        self.assertFalse(m['failure_recoveries'][0]['right_censored'])

    def test_fixed_policy_zero_gap_but_task_failure(self):
        _, m = trial(policy='fixed')
        self.assertEqual(m['coverage_gap_ratio'], 0)
        self.assertEqual(m['task_success_rate'], 0)
        self.assertIsNone(m['response_time_mean_s'])
        self.assertEqual(m['unresponded_tasks'], 1)
        self.assertEqual(m['restricted_response_mean_s'], 12)

    def test_fault_after_ready_has_zero_initial_gap_but_union9(self):
        _, m = trial(failure=7.0)
        self.assertEqual(m['handovers'][0]['initial_gap_s'], 0)
        self.assertEqual(m['handovers'][0]['physical_gap_union_s'], 9)
        self.assertEqual(m['coverage_gap_s'], 9)

    def test_arrival_at_deadline_is_not_a_response_without_service(self):
        _, m = trial(delay=2.0)
        self.assertEqual(m['task_success_rate'], 0)
        self.assertIsNone(m['response_time_mean_s'])
        self.assertEqual(m['unresponded_tasks'], 1)


class LogAndCapacityTests(unittest.TestCase):
    def test_one_robot_cannot_serve_collocated_task_and_base(self):
        rows, m = trial(policy='immediate', task=(0, 0))
        at2 = next(e['truth'] for e in rows if e['type'] == 'interval' and e['start'] == 2)
        self.assertEqual(at2['robots']['A']['position'], [0, 0])
        self.assertFalse(can_respond(at2['robots']['A'], 'base', (0, 0), .25))
        self.assertEqual(m['coverage_gap_s'], 2)
        self.assertEqual(m['task_success_rate'], 1)

    def test_assignment_does_not_imply_physical_readiness(self):
        rows, _ = trial(policy='immediate')
        at2 = next(e['truth'] for e in rows if e['type'] == 'interval' and e['start'] == 2)
        self.assertEqual(at2['base']['owner'], 'B')
        self.assertFalse(can_respond(at2['robots']['B'], 'base', (0, 0), .25))

    def test_observation_and_policy_flags_cannot_change_evaluation(self):
        rows, m = trial(policy='immediate')
        altered = copy.deepcopy(rows)
        for row in altered:
            if row['type'] == 'interval':
                row['observations'] = {'false_ready': True, 'position': [0, 0]}
        self.assertEqual(m, evaluate(altered))

    def test_same_seed_byte_identical_jsonl(self):
        a, _ = trial(seed=17, jitter=.2, observation_noise=.1)
        b, _ = trial(seed=17, jitter=.2, observation_noise=.1)
        c, _ = trial(seed=18, jitter=.2, observation_noise=.1)
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td) / f'{i}.jsonl' for i in range(3)]
            for rows, path in zip((a, b, c), paths):
                write_jsonl(rows, path)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            self.assertNotEqual(paths[0].read_bytes(), paths[2].read_bytes())

    def test_paired_policies_have_same_exogenous_plan(self):
        a, _ = trial(seed=3, jitter=.2)
        b, _ = trial(seed=3, jitter=.2, policy='immediate')
        self.assertEqual(a[0]['exogenous_plan'], b[0]['exogenous_plan'])

    def test_two_dimensional_diagonal_motion_is_speed_limited(self):
        rows, _ = trial(initial_b=(-3.0, -4.0))
        previous = None
        import math
        for row in rows:
            if row['type'] == 'interval':
                position = row['truth']['robots']['B']['position']
                if previous is not None:
                    self.assertLessEqual(math.dist(previous, position), 1.0 + 1e-8)
                previous = position
        self.assertEqual(next(e['time'] for e in rows if e.get('event') == 'successor_ready'), 7)

    def test_smaller_timestep_preserves_analytic_case(self):
        _, m = trial(dt=.5)
        self.assertEqual(m['coverage_gap_s'], 0)
        self.assertEqual(m['task_success_rate'], 1)
        self.assertEqual(m['response_times_s'], [8])

    def test_robot_stops_moving_while_service_consumes_capacity(self):
        rows, _ = trial(policy='immediate', dt=.25)
        serving = [r['truth']['robots']['A']['position'] for r in rows
                   if r['type'] == 'interval' and r['truth']['task']['work'] > 0
                   and r['truth']['task']['status'] == 'active']
        self.assertGreater(len(serving), 1)
        self.assertTrue(all(p == serving[0] for p in serving))

    def test_log_only_cli_runs_in_separate_process(self):
        import subprocess
        rows, expected = trial(policy='immediate')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'raw.jsonl'
            write_jsonl(rows, path)
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'run.py'),
                                     'evaluate', str(path)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), expected)

    def test_noise_is_logged_separately_from_ground_truth(self):
        rows, _ = trial(seed=7, observation_noise=.1)
        row = next(r for r in rows if r['type'] == 'interval')
        self.assertNotEqual(row['observations']['A']['self_position'],
                            row['truth']['robots']['A']['position'])

    def test_serialized_log_alone_reproduces_metrics(self):
        rows, expected = trial('communication_delay')
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'events.jsonl'
            write_jsonl(rows, path)
            del rows
            self.assertEqual(evaluate(load_jsonl(path)), expected)

    def test_logs_never_overwrite_existing_file(self):
        rows, _ = trial()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'events.jsonl'
            write_jsonl(rows, path)
            before = hashlib.sha256(path.read_bytes()).digest()
            with self.assertRaises(FileExistsError):
                write_jsonl(rows, path)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).digest())

    def test_truncated_and_noncontiguous_logs_rejected(self):
        rows, _ = trial()
        with self.assertRaises(ValueError):
            evaluate(rows[:-1])
        bad = copy.deepcopy(rows)
        first = next(e for e in bad if e['type'] == 'interval')
        first['start'] = .5
        with self.assertRaises(ValueError):
            evaluate(bad)

    def test_multiple_assignments_rejected_by_evaluator(self):
        rows, _ = trial()
        row = next(e for e in rows if e['type'] == 'interval')
        row['truth']['robots']['A']['active_assignment'] = ['base', 'dynamic']
        with self.assertRaises(ValueError):
            evaluate(rows)

    def test_invalid_configurations_rejected(self):
        for changes in ({'dt': 0}, {'delay': -.1}, {'deadline': 20}, {'failure': 4, 'recovery': 3}, {'dt': .3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                scenario_config(**changes)


if __name__ == '__main__':
    unittest.main()
