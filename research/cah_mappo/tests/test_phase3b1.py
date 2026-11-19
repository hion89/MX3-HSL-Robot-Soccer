"""Real CPU forward/backward/optimizer tests; no simulation or historical outputs overwritten."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cah_mappo.phase3b1.math_utils import gae
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from cah_mappo.phase3b1.learner import Learner, TrainConfig
    from cah_mappo.phase3b1.networks import ActorCritic


class ReturnTests(unittest.TestCase):
    def test_true_terminal_no_bootstrap_or_episode_leak(self):
        adv, ret = gae([1, 100], [0, 0], [5, 0], [0, 0], [0, 0], .9, 1)
        self.assertEqual(adv, [1, 100]); self.assertEqual(ret, [1, 100])

    def test_truncation_bootstraps_final_value_without_next_episode_gae(self):
        adv, _ = gae([1, 100], [0, 0], [5, 0], [1, 0], [0, 0], .9, 1)
        self.assertEqual(adv, [5.5, 100])

    def test_failed_agent_does_not_cut_team_return(self):
        adv, _ = gae([1, 2], [0, 0], [0, 0], [1, 0], [1, 0], .9, 1)
        self.assertAlmostEqual(adv[0], 2.8)

    def test_invalid_gae_inputs_rejected(self):
        for rewards, masks in [([float('nan')], [1]), ([1], [2])]:
            with self.assertRaises(ValueError):gae(rewards, [0], [0], masks, [0])


@unittest.skipUnless(HAS_TORCH, 'Requires separately approved Phase 3B-1 torch environment')
class TorchTests(unittest.TestCase):
    def test_real_failure_recovery_reaches_the_training_masks(self):
        from dataclasses import replace
        from cah_mappo.phase2.model import EventSpec
        from cah_mappo.phase2.scenarios import scenario
        from cah_mappo.phase3a.env import ParallelCore
        learner = Learner()
        events = tuple(EventSpec(t, kind, robot) for t, kind in ((1, 'failure'), (3, 'recovery')) for robot in ('A', 'B'))
        learner.env = ParallelCore(replace(scenario(), events=events))
        learner.obs, _ = learner.env.reset(100)
        batch = learner.collect(4)
        self.assertEqual([b['active'] for b in batch], [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]])
        for row in batch[1:3]:
            self.assertEqual(row['actions'], [0] * 4)
            self.assertEqual(row['old_logp'], [0] * 4)
            self.assertEqual(row['bootstrap'], 1)
            self.assertEqual(row['continuation'], 1)
        self.assertTrue(all(math.isfinite(v) for v in learner.update(batch).values()))

    def test_truncation_bootstraps_final_state_before_reset(self):
        from dataclasses import replace
        from cah_mappo.phase2.scenarios import scenario
        from cah_mappo.phase3a.env import ParallelCore
        learner = Learner()
        config = scenario()
        learner.env = ParallelCore(replace(config, jobs=(config.jobs[0],), horizon=2))
        learner.obs, _ = learner.env.reset(100)
        batch = learner.collect(3)
        self.assertEqual([r['bootstrap'] for r in batch], [1, 1, 1])
        self.assertEqual([r['continuation'] for r in batch], [1, 0, 1])
        final_transition = next(r for r in reversed(learner.last_episode_logs[0][1]) if r['type'] == 'transition')
        final_state = final_transition['critic_after']['state']
        with torch.no_grad():
            expected = learner.net.value(torch.tensor([final_state])).item()
        self.assertEqual(batch[1]['next_value'], expected)
        self.assertNotEqual(final_state, batch[2]['state'])
        self.assertTrue(all(math.isfinite(v) for v in learner.update(batch).values()))

    def test_real_forward_backward_and_parameter_updates(self):
        learner = Learner(TrainConfig(rollout_steps=32))
        batch = learner.collect()
        before_a = {k: p.clone() for k, p in learner.net.actor.state_dict().items()}
        before_c = {k: p.clone() for k, p in learner.net.critic.state_dict().items()}
        result = learner.update(batch)
        self.assertTrue(any(not torch.equal(p, before_a[k]) for k, p in learner.net.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(p, before_c[k]) for k, p in learner.net.critic.state_dict().items()))
        self.assertTrue(all(math.isfinite(v) for v in result.values()))
        self.assertLessEqual(result['actor_clipped_gradient_norm'], .50001)
        self.assertLessEqual(result['critic_clipped_gradient_norm'], .50001)

    def test_same_forward_sampling_logp_and_preupdate_ratio_one(self):
        learner = Learner(); batch = learner.collect(24); data = learner.tensors(batch)
        error = learner.ratio_before_update(data)
        self.assertLess(error, 1e-6)
        dist = learner.net.distribution(data['obs'].reshape(-1, 146), data['mask'].reshape(-1, 10))
        ratio = (dist.log_prob(data['actions'].reshape(-1)) - data['old_logp'].reshape(-1)).exp()
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6, rtol=0))

    def test_noop_only_distribution_and_all_masked_error(self):
        learner = Learner(); mask = torch.zeros(2, 10); mask[:, 0] = 1
        obs = torch.zeros(2, 146); dist = learner.net.distribution(obs, mask)
        self.assertTrue(torch.equal(dist.probs[:, 0], torch.ones(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(dist.entropy(), torch.zeros(2, dtype=torch.float64)))
        with self.assertRaises(ValueError):learner.net.distribution(obs, torch.zeros_like(mask))

    def test_arbitration_never_rewrites_buffer_sample(self):
        learner = Learner(); batch = learner.collect(32); original = [b['actions'].copy() for b in batch]
        for b in batch:b['executed_actions'] = [0, 0, 0, 0]
        data = learner.tensors(batch)
        self.assertEqual(data['actions'].tolist(), original)
        self.assertLess(learner.ratio_before_update(data), 1e-6)

    def test_all_inactive_skips_actor_and_updates_one_team_critic(self):
        learner = Learner(); batch = learner.collect(16)
        for b in batch:b['active'] = [0, 0, 0, 0]
        actor = copy.deepcopy(learner.net.actor.state_dict())
        critic = copy.deepcopy(learner.net.critic.state_dict())
        report = learner.update(batch)
        self.assertTrue(all(torch.equal(p, actor[k]) for k, p in learner.net.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(p, critic[k]) for k, p in learner.net.critic.state_dict().items()))
        self.assertEqual(report['active_actor_samples'], 0)

    def test_padding_and_failure_rows_have_no_actor_gradient(self):
        a = Learner(); b = Learner(); batch = a.collect(16); other = copy.deepcopy(batch)
        for row in other:
            row['obs'][2] = [999] * 146; row['obs'][3] = [-999] * 146
        a.update(batch); b.update(other)
        self.assertTrue(all(torch.equal(p, b.net.actor.state_dict()[k]) for k, p in a.net.actor.state_dict().items()))

    def test_critic_is_not_reweighted_by_number_of_active_robots(self):
        a = Learner(); b = Learner(); batch = a.collect(16); other = copy.deepcopy(batch)
        for row in other:row['active'] = [1, 0, 0, 0]
        report_a = a.update(batch); report_b = b.update(other)
        self.assertEqual(report_a['critic_loss'], report_b['critic_loss'])
        self.assertTrue(all(torch.equal(p, b.net.critic.state_dict()[k]) for k, p in a.net.critic.state_dict().items()))

    def test_invalid_shape_mask_nan_and_action_detected(self):
        learner = Learner(); original = learner.collect(4)
        for key, value in [('obs', [[float('nan')] * 146] * 4), ('active', [2, 1, 0, 0]),
                           ('mask', [[0] * 10] * 4), ('state', [0] * 140), ('actions', [99] * 4)]:
            batch = copy.deepcopy(original); batch[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):learner.update(batch)

    def test_same_seed_collect_update_and_checkpoint_reload(self):
        a = Learner(); b = Learner(); ba = a.collect(16); bb = b.collect(16)
        self.assertEqual(ba, bb)
        self.assertEqual(a.env.engine.rows, b.env.engine.rows)
        self.assertEqual(a.last_episode_logs, b.last_episode_logs)
        a.update(ba); b.update(bb)
        self.assertTrue(all(torch.equal(p, b.net.state_dict()[k]) for k, p in a.net.state_dict().items()))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'checkpoint.pt'; a.checkpoint(path)
            data = torch.load(path, map_location='cpu', weights_only=True)
            net = ActorCritic(); net.load_state_dict(data['network'])
            self.assertTrue(all(torch.equal(p, net.state_dict()[k]) for k, p in a.net.state_dict().items()))
            with self.assertRaises(FileExistsError):a.checkpoint(path)


if __name__ == '__main__':unittest.main()
