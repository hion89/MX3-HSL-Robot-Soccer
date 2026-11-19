"""On-policy feed-forward MAPPO: masked shared Actor, one team Critic sample/tick.

Algorithm reference: marlbenchmark/on-policy (MIT, MAPPO contributors 2021).
Independent compact implementation; not the upstream runner or a copied module.
"""
from dataclasses import dataclass, asdict
import copy
import math
import random
import torch

from ..phase2.scenarios import scenario
from ..phase3a.env import ParallelCore
from ..phase3a.contracts import Settings, SampledAction, validate_ppo_record
from ..phase3a.adapters import padded_batch
from .math_utils import gae
from .networks import ActorCritic


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    environment_seed: int = 100
    hidden: int = 64
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy: float = .01
    value_coef: float = .5
    learning_rate: float = .0003
    max_grad_norm: float = .5
    rollout_steps: int = 256
    epochs: int = 4
    minibatch_steps: int = 64


def minimal_env(seed=100, jitter=0.0):
    from dataclasses import replace
    c = replace(scenario('single'), jitter=jitter)
    env = ParallelCore(c, Settings(sensor_radius=10, sensor_noise=0, radio_enabled=True, radio_delay=0))
    obs, _ = env.reset(seed, {'terminate_on_tasks_resolved': True})
    return env, obs


def finite_tensor(tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError('Nonfinite training tensor')


class Learner:
    def __init__(self, config=TrainConfig()):
        self.c = config
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        self.net = ActorCritic(config.hidden).cpu()
        self.actor_optimizer = torch.optim.Adam(self.net.actor.parameters(), lr=config.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.net.critic.parameters(), lr=config.learning_rate, eps=1e-5)
        self.sample_rng = torch.Generator(device='cpu').manual_seed(config.seed + 10000)
        self.shuffle_rng = torch.Generator(device='cpu').manual_seed(config.seed + 20000)
        self.env, self.obs = minimal_env(config.environment_seed)
        self.steps, self.episodes = 0, 0
        self.completed = []
        self.last_episode_logs = []

    @torch.no_grad()
    def collect(self, n=None):
        batch = []
        self.completed = []
        self.last_episode_logs = []
        for _ in range(n or self.c.rollout_steps):
            packed = padded_batch(self.obs, self.env.state(), self.env.settings)
            observations = torch.tensor(packed['actor_observations'], dtype=torch.float32)
            masks = torch.tensor(packed['available_actions'], dtype=torch.bool)
            states = torch.tensor([self.env.state()['state']], dtype=torch.float32)
            sampled, logp, probs = self.net.sample(observations, masks, self.sample_rng)
            value = self.net.value(states).item()
            actions = {}
            for name in self.env.agents:
                i = self.env.settings.roster.index(name)
                actions[name] = SampledAction(int(sampled[i]), float(logp[i]), tuple(masks[i].int().tolist()),
                                              tuple(probs[i].tolist()), f'update_steps_{self.steps}',
                                              self.obs[name]['observation_token'])
            next_obs, rewards, terms, truncs, infos = self.env.step(actions)
            next_value = self.net.value(torch.tensor([self.env.state()['state']], dtype=torch.float32)).item()
            first = next(iter(infos.values()))['transition']
            executed, changed = [0] * 4, [False] * 4
            for name, info in infos.items():
                validate_ppo_record(info['transition'])
                i = self.env.settings.roster.index(name)
                executed[i] = info['transition']['executed_action']
                changed[i] = info['transition']['arbitration_changed']
            batch.append({'obs': packed['actor_observations'], 'mask': packed['available_actions'],
                          'state': states[0].tolist(), 'actions': sampled.tolist(), 'old_logp': logp.tolist(),
                          'active': [v[0] for v in packed['active_masks']], 'value': value,
                          'next_value': next_value, 'reward': next(iter(rewards.values())),
                          'bootstrap': first['bootstrap_mask'], 'continuation': first['continuation_mask'],
                          'executed_actions': executed, 'arbitration_changed': changed})
            self.steps += 1
            if self.env.done:
                self.completed.append({'episode': self.episodes, 'return': self.env.episode_return, **self.env.metrics()})
                self.last_episode_logs.append((self.episodes, self.env.log_rows()))
                self.episodes += 1
                self.env, self.obs = minimal_env(self.c.environment_seed)
            else:
                self.obs = next_obs
        return batch

    def tensors(self, batch):
        if not batch:
            raise ValueError('Empty rollout')
        adv, returns = gae(*([b[key] for b in batch] for key in ('reward', 'value', 'next_value', 'bootstrap', 'continuation')),
                           self.c.gamma, self.c.gae_lambda)
        data = {key: torch.tensor([b[key] for b in batch], dtype=torch.float32)
                for key in ('obs', 'mask', 'state', 'active')}
        data['actions'] = torch.tensor([b['actions'] for b in batch], dtype=torch.long)
        data['old_logp'] = torch.tensor([b['old_logp'] for b in batch], dtype=torch.float64)
        data['returns'] = torch.tensor(returns, dtype=torch.float32)
        data['advantages'] = torch.tensor(adv, dtype=torch.float64).unsqueeze(-1).expand(-1, 4).clone()
        live = data['active'].bool()
        selected = data['advantages'][live]
        if selected.numel():
            data['advantages'] = (data['advantages'] - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8)
        for tensor in data.values():finite_tensor(tensor)
        if data['obs'].shape != (len(batch), 4, 146) or data['state'].shape != (len(batch), 141):
            raise ValueError('Rollout shapes disagree')
        if (data['mask'].shape != (len(batch), 4, 10) or data['active'].shape != (len(batch), 4)
                or data['actions'].shape != (len(batch), 4) or data['old_logp'].shape != (len(batch), 4)):
            raise ValueError('Action or mask shapes disagree')
        if not ((data['active'] == 0) | (data['active'] == 1)).all():
            raise ValueError('Nonbinary active mask')
        if not ((data['mask'] == 0) | (data['mask'] == 1)).all() or not (data['mask'].sum(-1) > 0).all():
            raise ValueError('Invalid available actions')
        if (data['actions'] < 0).any() or (data['actions'] >= 10).any():
            raise ValueError('Action out of range')
        if not data['mask'].gather(-1, data['actions'].unsqueeze(-1)).all():
            raise ValueError('Stored action was masked')
        return data

    def ratio_before_update(self, data):
        with torch.no_grad():
            dist = self.net.distribution(data['obs'].reshape(-1, 146), data['mask'].reshape(-1, 10))
            logs = dist.log_prob(data['actions'].reshape(-1)).reshape(-1, 4)
            live = data['active'].bool()
            error = (logs - data['old_logp']).abs()[live]
            return float(error.max()) if error.numel() else 0.0

    def update(self, batch):
        data = self.tensors(batch)
        error = self.ratio_before_update(data)
        if error > 1e-6:
            raise ValueError(f'Behavior policy mismatch before update: {error}')
        reports = []
        for _ in range(self.c.epochs):
            indices = torch.randperm(len(batch), generator=self.shuffle_rng)
            for ids in indices.split(self.c.minibatch_steps):
                report = self.minibatch(data, ids)
                reports.append(report)
        return {key: sum(r[key] for r in reports) / len(reports) for key in reports[0]} | {'preupdate_logp_max_error': error}

    def minibatch(self, data, ids):
        live = data['active'][ids].reshape(-1).bool()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = entropy = kl = clip_fraction = actor_grad = actor_clipped_grad = 0.0
        if live.any():
            obs = data['obs'][ids].reshape(-1, 146)[live]
            mask = data['mask'][ids].reshape(-1, 10)[live]
            actions = data['actions'][ids].reshape(-1)[live]
            old = data['old_logp'][ids].reshape(-1)[live]
            advantages = data['advantages'][ids].reshape(-1)[live]
            dist = self.net.distribution(obs, mask)
            logratio = dist.log_prob(actions) - old
            ratio = logratio.exp()
            surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - self.c.clip, 1 + self.c.clip) * advantages)
            policy_loss = -surrogate.mean()
            ent = dist.entropy().mean()
            loss = policy_loss - self.c.entropy * ent
            finite_tensor(loss); loss.backward()
            self._finite_grad(self.net.actor)
            actor_grad = float(torch.nn.utils.clip_grad_norm_(self.net.actor.parameters(), self.c.max_grad_norm))
            actor_clipped_grad = self._grad_norm(self.net.actor)
            self.actor_optimizer.step()
            actor_loss, entropy = float(policy_loss.detach()), float(ent.detach())
            kl = float(((ratio - 1) - logratio).mean().detach())
            clip_fraction = float(((ratio - 1).abs() > self.c.clip).double().mean())
        # Critic: one scalar team-state sample per ENVIRONMENT tick, not per agent.
        self.critic_optimizer.zero_grad(set_to_none=True)
        values = self.net.value(data['state'][ids])
        value_loss = .5 * ((values - data['returns'][ids]) ** 2).mean()
        finite_tensor(value_loss)
        (self.c.value_coef * value_loss).backward()
        self._finite_grad(self.net.critic)
        critic_grad = float(torch.nn.utils.clip_grad_norm_(self.net.critic.parameters(), self.c.max_grad_norm))
        critic_clipped_grad = self._grad_norm(self.net.critic)
        self.critic_optimizer.step()
        return {'actor_loss': actor_loss, 'critic_loss': float(value_loss.detach()), 'policy_entropy': entropy,
                'approximate_kl': kl, 'clip_fraction': clip_fraction, 'actor_gradient_norm': actor_grad,
                'critic_gradient_norm': critic_grad, 'actor_clipped_gradient_norm': actor_clipped_grad,
                'critic_clipped_gradient_norm': critic_clipped_grad, 'active_actor_samples': int(live.sum())}

    @staticmethod
    def _finite_grad(net):
        for p in net.parameters():
            if p.grad is not None:finite_tensor(p.grad)

    @staticmethod
    def _grad_norm(net):
        return math.sqrt(sum(float((p.grad.detach().double() ** 2).sum()) for p in net.parameters() if p.grad is not None))

    def checkpoint(self, path):
        from pathlib import Path
        path = Path(path)
        with path.open('xb') as stream:
            torch.save({'config': asdict(self.c), 'network': self.net.state_dict(),
                        'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict(),
                        'sample_rng': self.sample_rng.get_state(), 'shuffle_rng': self.shuffle_rng.get_state(),
                        'steps': self.steps, 'episodes': self.episodes,
                        'resume_scope': 'weights/optimizer/RNG only; a mid-episode environment is not serialized'}, stream)
