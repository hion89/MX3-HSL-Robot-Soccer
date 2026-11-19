"""Split safety, actor-only baselines, varying slots, failures, and frozen PPO reuse."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from cah_mappo.phase3b2a.scenarios import generate, build_manifest, make_env, STAGES
from cah_mappo.phase3b2a.baselines import LocalPolicy
from cah_mappo.phase3b2a.evaluation import actor_inputs, evaluate_one, aggregate
from cah_mappo.phase3b2a.learner import PoolLearner
from cah_mappo.phase3b2a.experiment import selection_key, train_pilot
from cah_mappo.phase3b1.learner import Learner, TrainConfig
from cah_mappo.phase3b1.networks import ActorCritic


class ScenariosAndViews(unittest.TestCase):
    def test_split_seeds_and_ids_are_disjoint(self):
        manifest = build_manifest(3, 3, 3)['scenarios']
        seeds = {split: {s['event_seed'] for stage in STAGES for s in manifest[stage][split]}
                 for split in ('train', 'validation', 'test')}
        self.assertFalse(seeds['train'] & seeds['validation'] or seeds['train'] & seeds['test'] or seeds['validation'] & seeds['test'])

    def test_generator_reproducible_and_real_randomization(self):
        for stage in STAGES:
            self.assertEqual(generate(stage, 'train', 0), generate(stage, 'train', 0))
            a, b = generate(stage, 'train', 0), generate(stage, 'train', 1)
            self.assertNotEqual(a['config']['robots'], b['config']['robots'])
            self.assertNotEqual(a['config']['jobs'], b['config']['jobs'])
        for bad in (-1, 1000):
            with self.assertRaises(ValueError):generate('A', 'train', bad)

    def test_dimensions_and_masks_for_all_team_and_job_sizes(self):
        counts = []
        for stage in STAGES:
            env, obs = make_env(generate(stage, 'train', 0))
            counts.append(len(obs))
            for tick in range(9):
                vectors, masks = actor_inputs(obs, env.settings)
                self.assertEqual(tuple(vectors.shape), (4, 146)); self.assertEqual(tuple(masks.shape), (4, 10))
                self.assertEqual(len(env.state()['state']), 141)
                self.assertTrue(masks[:, 0].all())
                obs, *_ = env.step({name: 1 if packet['action_mask'][1] else 0 for name, packet in obs.items()})
            self.assertLessEqual(len(env.state()['job_ids']), 8)
        self.assertEqual(counts, [2, 2, 3, 4, 4])

    def test_scenario_plan_independent_of_policy_and_information(self):
        spec = generate('D', 'test', 0)
        a, _ = evaluate_one(spec, 'local_fixed')
        b, _ = evaluate_one(spec, 'local_random')
        c, _ = evaluate_one(spec, 'central_exact')
        self.assertEqual(a['exogenous_sha256'], b['exogenous_sha256'])
        self.assertEqual(a['exogenous_sha256'], c['exogenous_sha256'])

    def test_local_slots_survive_later_job_arrival_and_decode_correctly(self):
        env, obs = make_env(generate('B3', 'train', 0))
        old = {name: {} for name in obs}
        for _ in range(9):
            for name, packet in obs.items():
                for jid, slot in old[name].items():self.assertEqual(packet['job_ids'][slot], jid)
                old[name] = {jid: i for i, jid in enumerate(packet['job_ids']) if jid is not None}
            actions = {name: next((i for i in range(2, 10) if packet['action_mask'][i]), 0) for name, packet in obs.items()}
            previous = obs
            obs, _, _, _, infos = env.step(actions)
            for name, action in actions.items():
                expected = None if action == 0 else previous[name]['job_ids'][action - 2]
                self.assertEqual(infos[name]['transition']['requested_assignment'], expected)

    def test_no_undelivered_message_or_critic_leak(self):
        a = generate('D', 'train', 0); b = copy.deepcopy(a)
        for spec in (a, b):
            for i, robot in enumerate(spec['config']['robots']):robot['position'] = [i * 10., 0.]
            spec['settings']['sensor_radius'] = 1.
        b['config']['robots'][1]['position'] = [15., 0.]
        ea, oa = make_env(a); eb, ob = make_env(b)
        self.assertEqual(oa['A'], ob['A']); self.assertNotEqual(ea.state(), eb.state())
        oa, *_ = ea.step({name: 0 for name in oa}); ob, *_ = eb.step({name: 0 for name in ob})
        self.assertEqual(oa['A'], ob['A'])
        net = ActorCritic()
        va, ma = actor_inputs({'A': oa['A']}, ea.settings)
        vb, mb = actor_inputs({'A': ob['A']}, eb.settings)
        self.assertTrue(torch.equal(net.distribution(va, ma).probs, net.distribution(vb, mb).probs))
        oa, *_ = ea.step({name: 0 for name in oa}); ob, *_ = eb.step({name: 0 for name in ob})
        self.assertNotEqual(oa['A'], ob['A'])

    def test_fixed_horizon_prevents_completion_denominator_change(self):
        spec = generate('A', 'test', 0)
        for mode in ('local_hold', 'local_eta', 'central_exact'):
            _, rows = evaluate_one(spec, mode)
            self.assertEqual(rows[-1]['time'], 32)
            self.assertEqual(rows[0]['config']['horizon'], 32)
            if mode.startswith('local'):
                last = next(r for r in reversed(rows) if r['type'] == 'transition')
                self.assertTrue(last['truncated']); self.assertFalse(last['terminated'])
                self.assertTrue(all(r['bootstrap_mask'] == 1 and r['continuation_mask'] == 0 for r in last['records'].values()))

    def test_fresh_base_risk_is_in_actor_packet(self):
        env, obs = make_env(generate('D', 'train', 0))
        for _ in range(6):obs, *_ = env.step({name: 1 if packet['action_mask'][1] else 0 for name, packet in obs.items()})
        i = obs['A']['job_ids'].index('base0')
        self.assertEqual(obs['A']['observation'][55 + 9 * i], .8)

    def test_local_baselines_use_only_packet_and_respect_mask(self):
        env, obs = make_env(generate('B3', 'train', 0))
        for mode in ('local_hold', 'local_fixed', 'local_eta', 'local_random'):
            policy = LocalPolicy(mode)
            for name, packet in obs.items():
                action = policy.decide(name, copy.deepcopy(packet))
                self.assertTrue(packet['action_mask'][action])
        with self.assertRaises(ValueError):LocalPolicy('central_exact')

    def test_single_base_absolute_risk_cancels_in_normalized_gap_penalty(self):
        measured = []
        for priority in (1., 8.):
            spec = generate('A', 'train', 0)
            spec['config']['jobs'][0]['priority'] = priority
            spec['config']['jobs'][1]['spawn'] = 0.
            env, obs = make_env(spec)
            action = obs['A']['job_ids'].index('task0') + 2
            _, _, _, _, infos = env.step({'A': action, 'B': 0})
            measured.append(infos['A']['reward_contributions']['coverage'])
        self.assertEqual(measured, [-2 / 32, -2 / 32])

    def test_standard_actions_cannot_move_successor_to_occupied_base(self):
        env, obs = make_env(generate('A', 'train', 0))
        position = copy.deepcopy(env.engine.robots['B']['position'])
        actions = {name: packet['job_ids'].index('base0') + 2 for name, packet in obs.items()}
        _, _, _, _, infos = env.step(actions)
        self.assertTrue(infos['B']['transition']['arbitration_changed'])
        self.assertEqual(infos['B']['transition']['executed_action'], 0)
        self.assertEqual(env.engine.robots['B']['position'], position)


class LearningAndSelection(unittest.TestCase):
    def test_reward_sensitivity_keeps_initial_actions_and_scenarios_paired(self):
        specs = [generate('A', 'train', 0)]
        a = PoolLearner(TrainConfig(), specs, 2.)
        b = PoolLearner(TrainConfig(), specs, 8.)
        ba, bb = a.collect(32), b.collect(32)
        for x, y in zip(ba, bb):
            for key in ('obs', 'state', 'actions', 'old_logp', 'mask', 'scenario_id'):
                self.assertEqual(x[key], y[key])
        self.assertNotEqual([x['reward'] for x in ba], [x['reward'] for x in bb])

    def test_collector_reuses_frozen_ppo_update_exactly(self):
        config = TrainConfig()
        a = PoolLearner(config, [generate('A', 'train', 0)])
        b = Learner(config)
        batch = a.collect(32)
        ra, rb = a.update(batch), b.update(copy.deepcopy(batch))
        self.assertEqual(ra, rb)
        self.assertTrue(all(torch.equal(p, b.net.state_dict()[k]) for k, p in a.net.state_dict().items()))

    def test_real_all_failed_training_batch_skips_actor(self):
        spec = generate('A', 'train', 0)
        for r in spec['config']['robots']:r['operational'] = False
        learner = PoolLearner(TrainConfig(), [spec])
        batch = learner.collect(8)
        self.assertTrue(all(row['active'] == [0] * 4 for row in batch))
        before = copy.deepcopy(learner.net.actor.state_dict())
        report = learner.update(batch)
        self.assertEqual(report['active_actor_samples'], 0)
        self.assertTrue(all(torch.equal(p, before[k]) for k, p in learner.net.actor.state_dict().items()))

    def test_failures_and_recovery_preserve_team_bootstrap(self):
        learner = PoolLearner(TrainConfig(), [generate('D', 'train', 0)])
        batch = learner.collect(12)
        self.assertEqual(batch[5]['active'][2], 0); self.assertEqual(batch[10]['active'][2], 1)
        self.assertTrue(all(row['bootstrap'] == 1 and row['continuation'] == 1 for row in batch))
        self.assertEqual(batch[5]['actions'][2], 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in learner.update(batch).values()))

    def test_scenario_cycle_and_logs_reproduce(self):
        specs = [generate('A', 'train', i) for i in range(2)]
        a = PoolLearner(TrainConfig(), specs); b = PoolLearner(TrainConfig(), specs)
        ba, bb = a.collect(65), b.collect(65)
        self.assertEqual(ba, bb); self.assertEqual(a.last_episode_logs, b.last_episode_logs)
        self.assertEqual([ba[i]['scenario_id'] for i in (0, 32, 64)], [specs[0]['scenario_id'], specs[1]['scenario_id'], specs[0]['scenario_id']])
        self.assertEqual(ba[31]['bootstrap'], 1); self.assertEqual(ba[31]['continuation'], 0)

    def test_test_split_rejected_for_training_and_selection(self):
        with self.assertRaises(ValueError):PoolLearner(TrainConfig(), [generate('A', 'test', 0)])
        with self.assertRaises(ValueError):train_pilot(Path('never_created'), [], [generate('A', 'test', 0)], 'A', 2, 1)

    def test_selection_ignores_return_and_prioritizes_success(self):
        a = {'task_success_rate': 0., 'coverage_gap_ratio': 0., 'restricted_response_mean_s': 0., 'mean_return': 999}
        b = {**a, 'task_success_rate': 1., 'coverage_gap_ratio': 1., 'mean_return': -999}
        self.assertGreater(selection_key(b), selection_key(a))
        self.assertEqual(selection_key(b), selection_key({**b, 'mean_return': 1e10}))

    def test_unresponded_rate_not_zero_latency(self):
        row, _ = evaluate_one(generate('A', 'test', 0), 'local_hold')
        summary = aggregate([row])
        self.assertEqual(summary['task_success_rate'], 0.)
        self.assertIsNone(summary['response_time_mean_s'])
        self.assertEqual(summary['unresponded_rate'], 1.)
        self.assertGreater(summary['restricted_response_mean_s'], 0.)


if __name__ == '__main__':unittest.main()
