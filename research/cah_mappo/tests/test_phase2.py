"""Expected allocations, analytic timelines, capacity and log-only regression tests."""
import copy
from dataclasses import FrozenInstanceError, asdict, replace
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cah_mappo.core import write_jsonl
from cah_mappo.metrics import evaluate as phase1_evaluate
from cah_mappo.phase2.engine import Environment
from cah_mappo.phase2.metrics import evaluate, load_jsonl, physical_coverage
from cah_mappo.phase2.model import (Action, Config, EventSpec, JobSpec, JobView, Observation,
                                  RobotSpec, RobotView, config_from_dict, eta)
from cah_mappo.phase2.policies import (POLICIES, FixedRole, GreedyETA, ExactAssignment,
                                     RuleBasedHandover, assignment_cost, make_policy)
from cah_mappo.phase2.scenarios import SCENARIOS, scenario


def trial(name='single', policy='exact', **changes):
    c = replace(scenario(name), **changes)
    rows = Environment(c, make_policy(policy, c), policy).run()
    return rows, evaluate(rows)


def allocation_fixture():
    robots = (RobotView('A', (0, 0), 0, 1, 1000, True, None, None, 'worker'),
              RobotView('B', (2, 0), math.pi, 1, 1000, True, None, None, 'worker'))
    jobs = (JobView('a', (1, 0), 'task', 1, 0, 20, 1, 'A', None),
            JobView('z', (-2, 0), 'task', 1, 0, 20, 1, 'B', None))
    return Observation(0, 0, 1, .25, robots, jobs)


class PolicyTests(unittest.TestCase):
    def test_greedy_priority_and_eta_with_stable_tie(self):
        obs = allocation_fixture()
        self.assertEqual(dict(GreedyETA().decide(obs).assignments), {'A': 'a', 'B': 'z'})
        important = replace(obs.jobs[1], priority=10)
        changed = replace(obs, jobs=(obs.jobs[0], important))
        self.assertEqual(dict(GreedyETA().decide(changed).assignments), {'A': 'z', 'B': 'a'})

    def test_exact_is_better_than_greedy_on_known_counterexample(self):
        obs = allocation_fixture()
        greedy = GreedyETA().decide(obs)
        exact = ExactAssignment(0).decide(obs)
        self.assertEqual(dict(exact.assignments), {'A': 'z', 'B': 'a'})
        self.assertEqual(assignment_cost(obs, dict(greedy.assignments), 0), 7)
        self.assertEqual(exact.allocation_cost, 5)

    def test_exact_matches_independent_permutation_oracle(self):
        obs = allocation_fixture()
        # Unique dummy slots permit idle in a permutation enumeration.
        candidates = []
        for p in itertools.permutations(['a', 'z', 'idle0', 'idle1'], 2):
            allocation = {r.id: value if value in ('a', 'z') else None for r, value in zip(obs.robots, p)}
            candidates.append(assignment_cost(obs, allocation))
        self.assertEqual(ExactAssignment().decide(obs).allocation_cost, min(candidates))

    def test_switch_cost_retains_existing_allocation(self):
        obs = allocation_fixture()
        obs = replace(obs, robots=(replace(obs.robots[0], assignment='a'), replace(obs.robots[1], assignment='z')))
        self.assertEqual(dict(ExactAssignment(0).decide(obs).assignments), {'A': 'z', 'B': 'a'})
        self.assertEqual(dict(ExactAssignment(5).decide(obs).assignments), {'A': 'a', 'B': 'z'})

    def test_fixed_preserves_guard_and_static_worker_queue(self):
        rows, m = trial(policy='fixed')
        self.assertEqual(m['coverage_gap_ratio'], 0)
        self.assertEqual(m['role_switch_count'], 0)
        self.assertEqual(m['response_time_mean_s'], 8)
        self.assertEqual(m['all_tasks_makespan_s'], 10)
        self.assertTrue(all(r['truth']['robots']['A']['assignment'] == 'base' for r in rows if r['type'] == 'interval'))

    def test_policy_observations_are_immutable(self):
        obs = allocation_fixture()
        with self.assertRaises(FrozenInstanceError):
            obs.robots[0].speed = 99

    def test_nonoperational_robot_never_selected(self):
        obs = allocation_fixture()
        obs = replace(obs, robots=(replace(obs.robots[0], operational=False), obs.robots[1]))
        for name in POLICIES:
            c = scenario(); p = make_policy(name, c)
            self.assertIsNone(dict(p.decide(obs).assignments)['A'])

    def test_eta_uses_robot_speed_and_turning(self):
        obs = allocation_fixture(); robot, job = obs.robots[0], obs.jobs[0]
        self.assertEqual(eta(robot, job, obs), 1)
        self.assertEqual(eta(replace(robot, speed=.25), job, obs), 3)
        self.assertEqual(eta(replace(robot, heading=math.pi, turn_speed=math.pi), job, obs), 2)


class ScenarioTests(unittest.TestCase):
    def test_single_greedy_gap4_response4_and_success(self):
        _, m = trial(policy='greedy')
        self.assertEqual(m['coverage_gap_s'], 4)
        self.assertAlmostEqual(m['coverage_gap_ratio'], 4 / 24)
        self.assertEqual(m['response_times_s'], [4])
        self.assertEqual(m['task_success_rate'], 1)
        self.assertEqual(m['all_tasks_makespan_s'], 6)
        self.assertEqual(m['role_switch_count'], 2)

    def test_rule_waits_for_successor_then_releases_at6(self):
        rows, m = trial(policy='handover')
        self.assertEqual(m['coverage_gap_s'], 0)
        self.assertEqual(m['response_times_s'], [8])
        release = next(e for e in rows if e.get('event') == 'responsibility_released')
        self.assertEqual((release['time'], release['old'], release['new']), (6, 'A', 'B'))
        self.assertEqual(m['handovers'][0]['initial_gap_s'], 0)

    def test_urgent_deadline_override_succeeds_fixed_fails(self):
        rows, m = trial('urgent', 'handover')
        self.assertEqual(m['task_success_rate'], 1)
        self.assertEqual(m['all_tasks_makespan_s'], 6)
        self.assertEqual(m['coverage_gap_s'], 4)
        self.assertTrue(any('deadline_override:base' in e.get('action', {}).get('notes', []) for e in rows))
        _, fixed = trial('urgent', 'fixed')
        self.assertEqual(fixed['coverage_gap_s'], 0)
        self.assertEqual(fixed['task_success_rate'], 0)
        self.assertIsNone(fixed['response_time_mean_s'])
        self.assertEqual(fixed['unresponded_tasks'], 1)
        self.assertEqual(fixed['deadline_miss_rate'], 1)
        self.assertIsNone(fixed['all_tasks_makespan_s'])

    def test_four_robots_three_simultaneous_tasks(self):
        rows, m = trial('simultaneous', 'exact')
        self.assertEqual(m['task_count'], 3)
        self.assertEqual(m['task_success_rate'], 1)
        self.assertEqual(m['all_tasks_makespan_s'], 6)
        self.assertEqual(m['task_duplicate_rate'], 0)
        self.assertEqual(len(rows[-1]['truth']['robots']), 4)

    def test_priorities_save_urgent_job_fixed_misses_it(self):
        _, fixed = trial('priorities', 'fixed')
        _, greedy = trial('priorities', 'greedy')
        self.assertAlmostEqual(fixed['task_success_rate'], 2 / 3)
        self.assertFalse(next(j for j in fixed['task_details'] if j['id'] == 'task1')['success'])
        self.assertEqual(greedy['task_success_rate'], 1)

    def test_speed_drop_changes_measured_completion_time(self):
        _, normal = trial('simultaneous', 'fixed')
        rows, slow = trial('slow_robot', 'fixed')
        self.assertEqual(normal['all_tasks_makespan_s'], 10)
        self.assertEqual(slow['all_tasks_makespan_s'], 14)
        self.assertEqual(rows[-1]['truth']['robots']['C']['speed'], .25)

    def test_owner_failure_fixed_never_repairs_coverage(self):
        _, fixed = trial('owner_failure', 'fixed')
        _, exact = trial('owner_failure', 'exact')
        self.assertEqual(fixed['coverage_gap_s'], 23)
        self.assertTrue(fixed['handovers'][0]['right_censored'])
        self.assertEqual(fixed['handovers'][0]['initial_gap_s'], 23)
        self.assertLess(exact['coverage_gap_s'], fixed['coverage_gap_s'])

    def test_rising_risk_cancels_in_progress_handover(self):
        rows, m = trial('rising_risk', 'handover')
        notes = [(e['time'], e['action']['notes']) for e in rows if e.get('event') == 'policy_selection']
        self.assertIn((2, ['hold_prepare:base']), notes)
        self.assertIn((3, ['risk_cancel:base']), notes)
        self.assertEqual(m['coverage_gap_s'], 0)
        self.assertEqual(m['handover_count'], 0)
        self.assertTrue(all(e['truth']['jobs']['base']['owner'] == 'A' for e in rows if e['type'] == 'interval'))

    def test_delayed_observation_causes_real_response_delay(self):
        _, normal = trial('single', 'handover')
        rows, delayed = trial('communication_delay', 'handover')
        self.assertEqual(normal['response_time_mean_s'], 8)
        self.assertEqual(delayed['response_time_mean_s'], 10)
        selections = [e for e in rows if e.get('event') == 'policy_selection' and e['time'] == 2]
        self.assertEqual(selections[0]['observation']['observed_at'], 1)
        self.assertNotIn('task1', [j['id'] for j in selections[0]['observation']['jobs']])

    def test_three_robots_and_task_spawn_after_zero(self):
        c = scenario('simultaneous')
        jobs = tuple(j for j in c.jobs if j.fixed_owner != 'D')
        c = replace(c, robots=c.robots[:3], jobs=jobs)
        rows = Environment(c, make_policy('exact', c)).run()
        self.assertEqual(evaluate(rows)['task_count'], 2)
        first = next(e for e in rows if e.get('event') == 'policy_selection')
        self.assertEqual([j['id'] for j in first['observation']['jobs']], ['base'])


class CapacityLogTests(unittest.TestCase):
    def test_idle_robot_cannot_cover_two_collocated_bases(self):
        c = scenario(); rows, _ = trial(); state = copy.deepcopy(rows[0]['exogenous_plan'])
        state['radius'] = .25
        state['robots']['A']['assignment'] = None
        state['robots']['B']['operational'] = False
        state['jobs']['other'] = {**state['jobs']['base'], 'id': 'other', 'priority': 3}
        self.assertEqual(physical_coverage(state), {'other': 'A'})

    def test_duplicate_proposals_logged_but_arbitrated(self):
        class Duplicate(GreedyETA):
            def decide(self, obs):
                return Action(tuple((r.id, next((j.id for j in obs.jobs if j.kind == 'task'), 'base')) for r in obs.robots))
        c = scenario(); rows = Environment(c, Duplicate()).run(); m = evaluate(rows)
        self.assertGreater(m['requested_duplicate_rate'], 0)
        self.assertEqual(m['task_duplicate_rate'], 0)
        self.assertTrue(any(e.get('reason') == 'duplicate_assignment' for e in rows))

    def test_intentional_preparation_is_not_unnecessary_duplicate(self):
        rows, m = trial(policy='handover')
        interval = next(e for e in rows if e['type'] == 'interval' and e['start'] == 2)
        self.assertEqual([r['assignment'] for r in interval['truth']['robots'].values()], ['base', 'base'])
        self.assertEqual(m['task_duplicate_rate'], 0)
        self.assertEqual(m['requested_duplicate_rate'], 0)

    def test_all_policies_share_exogenous_events_independent_of_rng_use(self):
        class ConsumesRandom(GreedyETA):
            def decide(self, obs):
                for _ in range(100):
                    self.rng.random()
                return super().decide(obs)
        c = replace(scenario('slow_robot'), seed=19, jitter=.15)
        plans = []
        for p in [make_policy(name, c) for name in POLICIES] + [ConsumesRandom()]:
            rows = Environment(c, p).run()
            plans.append(rows[0]['exogenous_plan'])
        self.assertTrue(all(plan == plans[0] for plan in plans))

    def test_same_seed_log_bytes_and_different_seed_plan(self):
        a, _ = trial(seed=12, jitter=.15)
        b, _ = trial(seed=12, jitter=.15)
        c, _ = trial(seed=13, jitter=.15)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0]['exogenous_plan'], c[0]['exogenous_plan'])
        with tempfile.TemporaryDirectory() as td:
            p, q = Path(td) / 'a.jsonl', Path(td) / 'b.jsonl'
            write_jsonl(a, p); write_jsonl(b, q)
            self.assertEqual(p.read_bytes(), q.read_bytes())

    def test_log_only_cli_and_phase1_backwards_compatibility(self):
        rows, expected = trial('simultaneous')
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'raw.jsonl'; write_jsonl(rows, p)
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'phase2.py'), 'evaluate', str(p)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout), expected)
        for p in (ROOT / 'outputs/phase1_final').glob('*.jsonl'):
            self.assertEqual(evaluate(load_jsonl(p)), phase1_evaluate(load_jsonl(p)))

    def test_selected_notes_and_observation_cannot_change_gt_metrics(self):
        rows, expected = trial()
        for e in rows:
            if e.get('event') == 'policy_selection':
                e['observation'] = {'fake': 'ready'}
                e['action']['notes'] = ['fake_ready']
        self.assertEqual(evaluate(rows), expected)

    def test_weighted_risk_denominator_and_explicit_task_owner(self):
        rows, m = trial('rising_risk', 'exact')
        self.assertEqual(m['coverage_weighted_time_s'], 3 + 21 * 10)
        self.assertEqual(m['weighted_coverage_gap_s'], 31)
        self.assertAlmostEqual(m['coverage_gap_ratio'], 31 / 213)
        at2 = next(r['truth'] for r in rows if r['type'] == 'interval' and r['start'] == 2)
        self.assertEqual(at2['jobs']['task1']['owner'], 'A')

    def test_capacity_limit_through_every_scenario(self):
        for name in SCENARIOS:
            rows, _ = trial(name)
            for row in rows:
                if row['type'] != 'interval':
                    continue
                state = row['truth']
                active = [r['assignment'] for r in state['robots'].values() if r['assignment'] is not None]
                self.assertEqual(len(active), len(set(active)))  # exact has no preparation overlap
                for r in state['robots'].values():
                    if not r['operational']:
                        self.assertIsNone(r['assignment'])

    def test_truncated_and_overlapping_logs_rejected(self):
        rows, _ = trial()
        with self.assertRaises(ValueError):
            evaluate(rows[:-1])
        e = next(e for e in rows if e['type'] == 'interval')
        e['end'] = 2
        with self.assertRaises(ValueError):
            evaluate(rows)

    def test_config_serialization_and_invalid_inputs(self):
        c = scenario()
        self.assertEqual(config_from_dict(json.loads(json.dumps(asdict(c)))), c)
        for bad in [replace(c, robots=c.robots[:1]), replace(c, dt=.3),
                    replace(c, robots=(replace(c.robots[0], capacity=2), c.robots[1])),
                    replace(c, jobs=(replace(c.jobs[0], priority=-1), *c.jobs[1:]))]:
            with self.assertRaises(ValueError):
                bad.validate()


if __name__ == '__main__':
    unittest.main()
