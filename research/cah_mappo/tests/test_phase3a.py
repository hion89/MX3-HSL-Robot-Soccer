"""CTDE leakage, sampled-action, failure-mask and return-boundary contract tests."""
from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cah_mappo.core import write_jsonl
from cah_mappo.phase2.metrics import evaluate, load_jsonl
from cah_mappo.phase2.model import Config, RobotSpec as R, JobSpec as J, EventSpec as E
from cah_mappo.phase2.scenarios import scenario
from cah_mappo.phase3a.env import ParallelCore
from cah_mappo.phase3a.contracts import Settings, SampledAction, validate_ppo_record
from cah_mappo.phase3a.adapters import padded_batch
from cah_mappo.phase3a.rewards import RewardWeights, combine


def hidden_config():
    return Config('hidden', (R('A', (0, 0)), R('B', (20, 0))),
                  (J('base', (0, 0), kind='base', spawn=0, service=0, deadline=24),
                   J('hidden', (20, 0), spawn=0, deadline=20)))


def noops(env):
    return {a: 0 for a in env.agents}


def uniform_sample(action, mask, token=None):
    probs = tuple(1 / sum(mask) if valid else 0 for valid in mask)
    return SampledAction(action, math.log(probs[action]), tuple(mask), probs, 'uniform-test-v0', token)


def hold_episode(config=None, **kwargs):
    env = ParallelCore(config, **kwargs); obs, _ = env.reset(7)
    total = 0
    while env.agents:
        actions = {a: 1 if obs[a]['action_mask'][1] else 0 for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        total += rewards['A']
    return env, total


class InformationTests(unittest.TestCase):
    def test_hidden_gt_is_not_actor_observation_or_mask(self):
        env = ParallelCore(hidden_config(), Settings(sensor_radius=1, radio_enabled=False))
        obs, _ = env.reset(0)
        old_actor, old_critic = copy.deepcopy(obs['A']), env.state()
        env.engine.robots['B']['position'] = [30, 10]
        env.engine.jobs['hidden']['priority'] = 17
        env.perception.update(env.engine.truth(), 0, env.engine.event)
        self.assertEqual(env.perception.actor('A', 0), old_actor)
        from cah_mappo.phase3a.perception import build_critic
        critic = build_critic(env.engine.truth(), 0, 24, env.settings, env.global_slots)
        self.assertNotEqual(critic, old_critic)

    def test_future_events_and_unspawned_tasks_do_not_change_actor(self):
        c = hidden_config()
        c = replace(c, jobs=(c.jobs[0], replace(c.jobs[1], spawn=10)))
        a = ParallelCore(c, Settings(sensor_radius=1, radio_enabled=False)); oa, _ = a.reset(0)
        changed = replace(c, events=(E(15, 'failure', 'B'),),
                          jobs=(c.jobs[0], replace(c.jobs[1], position=(99, 99), priority=99)))
        b = ParallelCore(changed, a.settings); ob, _ = b.reset(0)
        self.assertEqual(oa, ob)
        self.assertEqual(a.state(), b.state())
        self.assertNotIn('hidden', oa['A']['job_ids'])
        self.assertNotIn('hidden', a.state()['job_ids'])

    def test_local_slots_do_not_reveal_unknown_job_order(self):
        c = hidden_config()
        a = ParallelCore(c, Settings(sensor_radius=1, radio_enabled=False)); oa, _ = a.reset(0)
        extra = J('aaa_hidden', (100, 100), spawn=0, deadline=20)
        b = ParallelCore(replace(c, jobs=(extra, *c.jobs)), a.settings); ob, _ = b.reset(0)
        self.assertEqual(oa['A'], ob['A'])

    def test_radio_is_invisible_until_delivery_and_carries_old_measurement(self):
        env = ParallelCore(hidden_config(), Settings(sensor_radius=1, sensor_noise=0, radio_delay=2))
        obs, _ = env.reset(0)
        self.assertEqual(obs['A']['peer_fresh_mask'][1], 0)
        obs, *_ = env.step(noops(env))
        self.assertEqual(obs['A']['peer_fresh_mask'][1], 0)
        env.engine.robots['B']['position'] = [30, 0]
        obs, *_ = env.step(noops(env))
        peer = env.perception.memory['A']['peers']['B']
        self.assertEqual(peer['observed_at'], 0)
        self.assertEqual(peer['position'], [20, 0])
        self.assertIn('hidden', obs['A']['job_ids'])  # now legitimately received

    def test_unseen_expiry_does_not_oracle_mask_cached_job(self):
        c = hidden_config(); c = replace(c, jobs=(c.jobs[0], replace(c.jobs[1], deadline=3)))
        env = ParallelCore(c, Settings(sensor_radius=1, radio_delay=2, cache_ttl=8))
        obs, _ = env.reset(0)
        for _ in range(3):obs, *_ = env.step(noops(env))
        slot = obs['A']['job_ids'].index('hidden')
        self.assertEqual(env.engine.jobs['hidden']['status'], 'expired')
        self.assertEqual(obs['A']['action_mask'][slot + 2], 1)
        actions = noops(env); actions['A'] = slot + 2
        _, _, _, _, infos = env.step(actions)
        rec = infos['A']['transition']
        self.assertEqual(rec['sampled_action'], slot + 2)
        self.assertIsNone(rec['executed_assignment'])
        self.assertTrue(rec['arbitration_changed'])

    def test_actor_and_critic_fixed_shapes_for_two_and_four_robots(self):
        for name in ('single', 'simultaneous'):
            env = ParallelCore(scenario(name)); obs, _ = env.reset(0)
            self.assertTrue(all(len(o['observation']) == 146 and len(o['action_mask']) == 10 for o in obs.values()))
            self.assertEqual(len(env.state()['state']), 141)
            batch = padded_batch(obs, env.state(), env.settings)
            self.assertEqual(len(batch['actor_observations']), 4)
            self.assertEqual(batch['present_masks'], [[1], [1], [int(name == 'simultaneous')], [int(name == 'simultaneous')]])
            self.assertEqual(len(batch['critic_states'][0]), 141)
            if name == 'single':self.assertEqual(batch['available_actions'][2], [1] + [0] * 9)

    def test_returned_observations_and_state_cannot_mutate_environment(self):
        env = ParallelCore(); obs, _ = env.reset(0)
        obs['A']['observation'][0] = 999
        state = env.state(); state['state'][0] = 999
        self.assertNotEqual(env.observations['A']['observation'][0], 999)
        self.assertNotEqual(env.state()['state'][0], 999)


class ActionsAndFaults(unittest.TestCase):
    def test_no_jobs_still_has_noop_and_invalid_actions_rejected_atomically(self):
        c = hidden_config(); c = replace(c, jobs=(replace(c.jobs[1], spawn=10),))
        env = ParallelCore(c, Settings(sensor_radius=0, radio_enabled=False)); obs, _ = env.reset(0)
        self.assertEqual(obs['A']['action_mask'], [1] + [0] * 9)
        before = copy.deepcopy(env.engine.rows)
        for action in (-1, 99, True, 2):
            with self.assertRaises(ValueError):env.step({'A': action, 'B': 0})
            self.assertEqual(env.time, 0)
            self.assertEqual(env.engine.rows, before)

    def test_new_local_task_adds_slot_without_changing_dimensions(self):
        env = ParallelCore(); obs, _ = env.reset(0)
        self.assertNotIn('task1', obs['A']['job_ids'])
        for _ in range(2):obs, *_ = env.step(noops(env))
        self.assertIn('task1', obs['A']['job_ids'])
        self.assertEqual(len(obs['A']['observation']), 146)

    def test_duplicate_arbitration_does_not_replace_sample_or_logprob(self):
        c = hidden_config(); c = replace(c, robots=(R('A', (0, 0)), R('B', (0, 0))),
                                        jobs=(J('task', (0, 0), spawn=0, deadline=10),))
        env = ParallelCore(c); obs, _ = env.reset(0)
        actions = {a: uniform_sample(o['job_ids'].index('task') + 2, o['action_mask'], o['observation_token']) for a, o in obs.items()}
        _, _, _, _, infos = env.step(actions)
        loser = infos['B']['transition']
        self.assertEqual(loser['sampled_action'], actions['B'].action)
        self.assertEqual(loser['log_probability'], actions['B'].log_probability)
        self.assertEqual(loser['executed_action'], 0)
        self.assertTrue(loser['arbitration_changed'])
        self.assertTrue(validate_ppo_record(loser))  # still a valid sampled action for PPO

    def test_behavior_distribution_stale_mask_and_wrong_logprob_rejected(self):
        env = ParallelCore(); obs, _ = env.reset(0)
        mask = obs['A']['action_mask']; sample = uniform_sample(0, mask, obs['A']['observation_token'])
        bad = [replace(sample, action_mask=(0,) * 10), replace(sample, log_probability=-99),
               replace(sample, probabilities=(float('nan'),) * 10)]
        for item in bad:
            with self.assertRaises(ValueError):env.step({'A': item, 'B': 0})
        self.assertEqual(env.time, 0)

    def test_old_sample_rejected_even_when_action_mask_is_unchanged(self):
        env = ParallelCore(); obs, _ = env.reset(0)
        old = uniform_sample(0, obs['A']['action_mask'], obs['A']['observation_token'])
        next_obs, *_ = env.step({'A': 1, 'B': 0})
        self.assertEqual(obs['A']['action_mask'], next_obs['A']['action_mask'])
        with self.assertRaises(ValueError):env.step({'A': old, 'B': 0})

    def test_sample_cannot_be_reused_for_another_agent(self):
        env = ParallelCore(); obs, _ = env.reset(0)
        sample = uniform_sample(0, obs['B']['action_mask'], obs['B']['observation_token'])
        with self.assertRaises(ValueError):env.step({'A': sample, 'B': 0})

    def test_integer_actions_are_not_silently_training_records(self):
        env = ParallelCore(); env.reset(0)
        _, _, _, _, info = env.step(noops(env))
        with self.assertRaises(ValueError):validate_ppo_record(info['A']['transition'])

    def test_failure_recovery_does_not_terminate_individual_agent(self):
        c = replace(scenario(), events=(E(1, 'failure', 'B'), E(3, 'recovery', 'B')))
        env = ParallelCore(c); obs, _ = env.reset(0)
        obs, _, terms, trunc, info = env.step(noops(env))
        self.assertFalse(any(terms.values()) or any(trunc.values()))
        self.assertIn('B', env.agents)
        self.assertEqual(obs['B']['action_mask'], [1] + [0] * 9)
        self.assertEqual(info['B']['actor_active_mask'], 0)
        self.assertTrue(info['B']['transition']['rnn_reset'])
        obs, _, _, _, info = env.step(noops(env))
        self.assertEqual(info['B']['transition']['policy_loss_mask'], 0)
        self.assertEqual(info['B']['transition']['critic_loss_mask'], 1)
        self.assertEqual(info['B']['transition']['bootstrap_mask'], 1)
        obs, _, _, _, info = env.step(noops(env))
        self.assertEqual(obs['B']['active_mask'], 1)
        self.assertEqual(info['B']['actor_active_mask'], 1)
        self.assertTrue(info['B']['transition']['rnn_reset'])

    def test_all_agents_failed_still_advance_to_recovery(self):
        c = replace(scenario(), events=(E(1, 'failure', 'A'), E(1, 'failure', 'B'),
                                         E(3, 'recovery', 'A'), E(3, 'recovery', 'B')))
        env = ParallelCore(c); obs, _ = env.reset(0)
        for _ in range(3):obs, _, terms, trunc, _ = env.step(noops(env))
        self.assertEqual(len(env.agents), 2)
        self.assertTrue(all(o['active_mask'] == 1 for o in obs.values()))
        self.assertFalse(any(terms.values()) or any(trunc.values()))


class RewardsAndEpisodeTests(unittest.TestCase):
    def test_time_limit_bootstraps_but_breaks_episode_trace(self):
        env, _ = hold_episode()
        transition = next(r for r in reversed(env.engine.rows) if r['type'] == 'transition')
        self.assertFalse(transition['terminated']); self.assertTrue(transition['truncated'])
        for record in transition['records'].values():
            self.assertEqual(record['bootstrap_mask'], 1)
            self.assertEqual(record['continuation_mask'], 0)
        self.assertEqual(env.agents, [])
        with self.assertRaises(RuntimeError):env.step({})

    def test_normal_completion_terminates_and_disables_bootstrap(self):
        c = hidden_config(); c = replace(c, robots=(R('A', (0, 0)), R('B', (10, 0))),
                                        jobs=(J('task', (0, 0), spawn=0, deadline=10),))
        env = ParallelCore(c); obs, _ = env.reset(0, {'terminate_on_tasks_resolved': True})
        action = obs['A']['job_ids'].index('task') + 2
        obs, _, terms, truncs, _ = env.step({'A': action, 'B': 0})
        obs, _, terms, truncs, infos = env.step({'A': 1, 'B': 0})
        self.assertTrue(all(terms.values())); self.assertFalse(any(truncs.values()))
        self.assertEqual(infos['A']['transition']['bootstrap_mask'], 0)
        self.assertEqual(infos['A']['final_state'], env.state())
        self.assertEqual(env.time, 2)
        self.assertEqual(env.metrics()['task_success_rate'], 1)

    def test_future_task_prevents_premature_natural_end(self):
        c = hidden_config(); c = replace(c, robots=(R('A', (0, 0)), R('B', (10, 0))),
                                        jobs=(J('task', (0, 0), spawn=0, deadline=10), J('later', (20, 0), spawn=5, deadline=12)))
        env = ParallelCore(c); obs, _ = env.reset(0, {'terminate_on_tasks_resolved': True})
        action = obs['A']['job_ids'].index('task') + 2
        env.step({'A': action, 'B': 0})
        _, _, terms, truncs, _ = env.step({'A': 1, 'B': 0})
        self.assertFalse(any(terms.values()) or any(truncs.values()))

    def test_finite_rewards_components_and_large_scale_detection(self):
        with self.assertRaises(ValueError):combine({'completion': float('nan')}, RewardWeights())
        with self.assertRaises(ValueError):combine({'completion': float('inf')}, RewardWeights())
        with self.assertRaises(ValueError):combine({'completion': 1}, RewardWeights(completion=1000))
        env, total = hold_episode()
        self.assertTrue(math.isfinite(total))
        for row in env.engine.rows:
            if row['type'] == 'transition':
                self.assertAlmostEqual(row['reward'], sum(row['reward_contributions'].values()))
                self.assertTrue(all(math.isfinite(x) for x in row['reward_terms'].values()))
                self.assertLessEqual(abs(row['reward']), 100)

    def test_excessive_episode_accumulation_is_rejected(self):
        env = ParallelCore(weights=RewardWeights(max_abs_episode=.001))
        env.reset(0)
        with self.assertRaises(ValueError):env.step(noops(env))

    def test_termination_and_truncation_have_distinct_numeric_bootstrap_targets(self):
        truncated, _ = hold_episode()
        last = next(r for r in reversed(truncated.engine.rows) if r['type'] == 'transition')['records']['A']
        self.assertEqual(1 + .9 * 5 * last['bootstrap_mask'], 5.5)
        terminal = {**last, 'bootstrap_mask': 0}
        self.assertEqual(1 + .9 * 5 * terminal['bootstrap_mask'], 1)
        self.assertEqual(last['continuation_mask'], 0)

    def test_probability_mask_never_all_zero_even_for_padding(self):
        c = replace(scenario(), robots=tuple(replace(r, operational=False) for r in scenario().robots))
        env = ParallelCore(c); obs, _ = env.reset(0)
        batch = padded_batch(obs, env.state(), env.settings)
        self.assertTrue(all(mask == [1] + [0] * 9 for mask in batch['available_actions']))
        self.assertEqual(batch['active_masks'], [[0]] * 4)

    def test_radio_cache_age_eventually_masks_selection(self):
        env = ParallelCore(hidden_config(), Settings(sensor_radius=1, radio_delay=2, cache_ttl=1))
        obs, _ = env.reset(0)
        for _ in range(2):obs, *_ = env.step(noops(env))
        slot = obs['A']['job_ids'].index('hidden')
        self.assertEqual(obs['A']['job_mask'][slot], 1)
        self.assertEqual(obs['A']['job_fresh_mask'][slot], 0)
        self.assertEqual(obs['A']['action_mask'][slot + 2], 0)

    def test_successful_service_reward_exceeds_static_holding_in_controlled_case(self):
        c = replace(scenario(), jobs=(scenario().jobs[0], replace(scenario().jobs[1], position=(0, 0), spawn=0)))
        held, hold_reward = hold_episode(c)
        env = ParallelCore(c); obs, _ = env.reset(7); total = 0
        first = True
        while env.agents:
            if first:
                actions = {'A': obs['A']['job_ids'].index('task1') + 2, 'B': 0}; first = False
            else:actions = {a: 1 if obs[a]['action_mask'][1] else 0 for a in env.agents}
            obs, reward, *_ = env.step(actions); total += reward['A']
        self.assertEqual(held.metrics()['task_success_rate'], 0)
        self.assertEqual(env.metrics()['task_success_rate'], 1)
        self.assertGreater(total, hold_reward)

    def test_noncollocated_actor_only_goal_demo_completes_task(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('phase3a_cli_test', ROOT / 'phase3a.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        env, result = module.goal_demo()
        self.assertEqual(result['task_success_rate'], 1)
        self.assertEqual(result['response_time_mean_s'], 4)
        self.assertEqual(result['coverage_gap_s'], 4)
        self.assertGreater(env.episode_return, 0)

    def test_rewards_do_not_modify_independent_metric_values(self):
        a, _ = hold_episode()
        b, _ = hold_episode(weights=RewardWeights(coverage=5, completion=20))
        self.assertEqual(a.metrics(), b.metrics())

    def test_seed_initial_state_transitions_and_jsonl_reproducible(self):
        c = replace(scenario('simultaneous'), jitter=.15)
        a = ParallelCore(c); b = ParallelCore(c)
        oa, _ = a.reset(12); ob, _ = b.reset(12)
        self.assertEqual(oa, ob)
        while a.agents:
            aa, ab = noops(a), noops(b)
            self.assertEqual(a.step(aa), b.step(ab))
        self.assertEqual(a.log_rows(), b.log_rows())
        with tempfile.TemporaryDirectory() as td:
            p, q = Path(td) / 'a.jsonl', Path(td) / 'b.jsonl'
            a.export_log(p); b.export_log(q)
            self.assertEqual(p.read_bytes(), q.read_bytes())
            self.assertEqual(evaluate(load_jsonl(p)), a.metrics())

    def test_limits_and_reset_options_rejected(self):
        with self.assertRaises(ValueError):ParallelCore(settings=Settings(max_jobs=1)).reset(0)
        with self.assertRaises(ValueError):ParallelCore().reset(0, {'unknown': True})
        with self.assertRaises(RuntimeError):ParallelCore().step({})

    def test_legacy_evaluator_replays_phase1_and_phase2_outputs(self):
        for folder in ('outputs/phase1_final', 'phase2_outputs/final_seed0'):
            for p in (ROOT / folder).glob('*.jsonl'):
                result = evaluate(load_jsonl(p))
                self.assertIn('coverage_gap_ratio', result)
                self.assertTrue(0 <= result['task_success_rate'] <= 1)


if __name__ == '__main__':
    unittest.main()
