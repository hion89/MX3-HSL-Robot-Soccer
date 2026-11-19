"""New collector for scenario pools; reuse frozen Phase 3B-1 PPO math unchanged."""
import torch

from ..phase3b1.learner import Learner
from ..phase3a.adapters import padded_batch
from ..phase3a.contracts import SampledAction, validate_ppo_record
from .scenarios import make_env


class PoolLearner(Learner):
    def __init__(self, config, training_specs, coverage_weight=2.):
        if not training_specs or any(s['split'] != 'train' for s in training_specs):
            raise ValueError('Training collector accepts train split only')
        super().__init__(config)
        self.training_specs = training_specs
        self.coverage_weight = coverage_weight
        self.policy_updates = 0
        self.env, self.obs = make_env(self.training_specs[0], coverage_weight)

    @torch.no_grad()
    def collect(self, n=None):
        batch = []; self.completed = []; self.last_episode_logs = []
        for _ in range(n or self.c.rollout_steps):
            packed = padded_batch(self.obs, self.env.state(), self.env.settings)
            observations = torch.tensor(packed['actor_observations'], dtype=torch.float32)
            masks = torch.tensor(packed['available_actions'], dtype=torch.bool)
            state = self.env.state()['state']
            sampled, logp, probs = self.net.sample(observations, masks, self.sample_rng)
            value = self.net.value(torch.tensor([state])).item()
            actions = {}
            for name in self.env.agents:
                i = self.env.settings.roster.index(name)
                actions[name] = SampledAction(int(sampled[i]), float(logp[i]), tuple(masks[i].int().tolist()),
                                              tuple(probs[i].tolist()), f'ppo-update-{self.policy_updates}',
                                              self.obs[name]['observation_token'])
            next_obs, rewards, _, _, infos = self.env.step(actions)
            next_value = self.net.value(torch.tensor([self.env.state()['state']])).item()
            first = next(iter(infos.values()))['transition']
            executed, changed = [0] * 4, [False] * 4
            for name, info in infos.items():
                record = info['transition']; validate_ppo_record(record)
                i = self.env.settings.roster.index(name)
                executed[i], changed[i] = record['executed_action'], record['arbitration_changed']
                assert record['sampled_action'] == int(sampled[i])
                assert record['policy_loss_mask'] == packed['active_masks'][i][0]
            batch.append({'obs': packed['actor_observations'], 'mask': packed['available_actions'],
                          'state': state, 'actions': sampled.tolist(), 'old_logp': logp.tolist(),
                          'active': [v[0] for v in packed['active_masks']], 'value': value,
                          'next_value': next_value, 'reward': next(iter(rewards.values())),
                          'bootstrap': first['bootstrap_mask'], 'continuation': first['continuation_mask'],
                          'executed_actions': executed, 'arbitration_changed': changed,
                          'scenario_id': self.training_specs[self.episodes % len(self.training_specs)]['scenario_id'],
                          'policy_update': self.policy_updates})
            self.steps += 1
            if self.env.done:
                self.completed.append({'episode': self.episodes, 'return': self.env.episode_return,
                                       'scenario_id': batch[-1]['scenario_id'], **self.env.metrics()})
                self.last_episode_logs.append((self.episodes, self.env.log_rows()))
                self.episodes += 1
                self.env, self.obs = make_env(self.training_specs[self.episodes % len(self.training_specs)], self.coverage_weight)
            else:
                self.obs = next_obs
        return batch

    def update(self, batch):
        report = super().update(batch)
        self.policy_updates += 1
        return report
