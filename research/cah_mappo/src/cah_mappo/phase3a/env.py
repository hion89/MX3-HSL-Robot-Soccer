"""Framework-independent parallel core reusing the frozen Phase 2 dynamics."""
from dataclasses import asdict, replace
import copy
import hashlib
import math
from pathlib import Path

from ..core import write_jsonl
from ..phase2.engine import Environment
from ..phase2.metrics import evaluate
from ..phase2.model import Action, config_from_dict
from ..phase2.scenarios import scenario
from .contracts import Settings, SampledAction, validate_sample
from .perception import Perception, build_critic
from .rewards import RewardWeights, reward_terms, combine, covered_sites


class _NoPlanner:
    def reset(self, seed):
        pass


class ParallelCore:
    metadata = {'name': 'cah_phase3a_v0', 'is_parallelizable': True}

    def __init__(self, config=None, settings=None, weights=None):
        self.base_config = config or scenario('single')
        self.settings = settings or Settings()
        self.weights = weights or RewardWeights()
        self.possible_agents = list(self.settings.roster)
        self.agents = []
        self.engine = None
        self.done = False

    @property
    def observation_size(self):
        return 50 + 12 * self.settings.max_jobs

    @property
    def state_size(self):
        return 37 + 13 * self.settings.max_jobs

    def reset(self, seed=0, options=None):
        options = dict(options or {})
        if set(options) - {'config', 'terminate_on_tasks_resolved'}:
            raise ValueError('Unknown reset option')
        cfg = options.get('config', self.base_config)
        cfg = config_from_dict(cfg) if isinstance(cfg, dict) else cfg
        cfg = replace(cfg, seed=0 if seed is None else seed)
        cfg.validate(); self.settings.validate(cfg); self.weights.validate()
        self.c = cfg
        self.natural_end = bool(options.get('terminate_on_tasks_resolved', False))
        self.engine = Environment(cfg, _NoPlanner(), 'parallel_external_actions')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
        self.engine.rows.append({'type': 'header', 'schema_version': 2, 'interface_version': 'phase3a-v1',
                                 'config': asdict(cfg), 'planned_horizon': cfg.horizon,
                                 'policy': 'parallel_external_actions', 'settings': asdict(self.settings),
                                 'reward_weights': asdict(self.weights), 'exogenous_plan': self.engine.plan,
                                 'implementation_sha256': hashes,
                                 'phase2_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in sorted(Path(__file__).parents[1].joinpath('phase2').glob('*.py'))}})
        self.time, self.tick, self.done = 0.0, 0, False
        self.episode_return, self.absolute_reward_sum = 0.0, 0.0
        self.agents = [name for name in self.possible_agents if name in self.engine.robots]
        self.perception = Perception(self.settings, cfg)
        self.global_slots = []
        self.engine.boundary(0)
        self.engine.update_owners(0)
        self.perception.update(self.engine.truth(), 0, self.engine.event)
        self.observations = {a: self.perception.actor(a, 0) for a in self.agents}
        self.global_state = build_critic(self.engine.truth(), 0, cfg.horizon, self.settings, self.global_slots)
        infos = {a: {'actor_active_mask': self.observations[a]['active_mask'], 'rnn_reset': True,
                     'critic_state': copy.deepcopy(self.global_state), 'privileged_info': True}
                 for a in self.agents}
        return copy.deepcopy(self.observations), infos

    def state(self):
        if self.engine is None:
            raise RuntimeError('Call reset first')
        return copy.deepcopy(self.global_state)

    def step(self, actions):
        if self.engine is None or self.done:
            raise RuntimeError('reset required before step or after episode end')
        if set(actions) != set(self.agents):
            raise ValueError('Provide one action for every present agent, including failed robots')
        names = list(self.agents)
        old_obs, old_state = copy.deepcopy(self.observations), copy.deepcopy(self.global_state)
        decoded, records = {}, {}
        # Validate all samples before any environment mutation.
        for name in names:
            item = actions[name]; mask = old_obs[name]['action_mask']
            if isinstance(item, SampledAction):
                validate_sample(item, mask, old_obs[name]['observation_token'])
                action, logp, probs, version = item.action, item.log_probability, list(item.probabilities), item.policy_version
            else:
                action, logp, probs, version = item, None, None, 'scripted_not_training'
                if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(mask) or not mask[action]:
                    raise ValueError('Invalid or masked action')
            target = None if action == 0 else (old_obs[name]['self_assignment'] if action == 1
                                               else old_obs[name]['job_ids'][action - 2])
            decoded[name] = target
            records[name] = {'sampled_action': action, 'action_mask': list(mask), 'log_probability': logp,
                             'observation_token': old_obs[name]['observation_token'],
                             'sample_observation_token': item.observation_token if isinstance(item, SampledAction) else None,
                             'probabilities': probs, 'policy_version': version,
                             'requested_assignment': target, 'policy_loss_mask': old_obs[name]['active_mask'],
                             'critic_loss_mask': 1, 'actor_observation': old_obs[name]['observation']}
        t = self.time
        start_events = len(self.engine.rows)
        selection = self.engine.apply(t, Action(tuple(sorted(decoded.items())), notes=('standard_discrete_assignment',)))
        self.engine.event(t, 'policy_selection', actor_observations=old_obs, action=selection,
                          sampled_actions=copy.deepcopy(records))
        for job in self.engine.jobs.values():
            if job['kind'] != 'task' or job['status'] != 'active' or job['response'] is not None:
                continue
            for robot in self.engine.robots.values():
                if (robot['operational'] and robot['assignment'] == job['id']
                        and math.dist(robot['position'], job['position']) <= self.c.radius + 1e-9):
                    job['response'] = t
                    self.engine.event(t, 'task_response_started', job=job['id'], robot=robot['id'])
                    break
        before = self.engine.truth()
        end = round(t + self.c.dt, 10)
        self.engine.rows.append({'type': 'interval', 'tick': self.tick, 'start': t, 'end': end,
                                 'truth': before, 'selection': selection})
        self.engine.integrate(t)
        self.engine.boundary(end)
        self.engine.update_owners(end)
        after = self.engine.truth()
        events = [r for r in self.engine.rows[start_events:] if r['type'] == 'event']
        terms = reward_terms(before, after, events, self.c.dt, self.c.horizon,
                             self.settings.max_jobs, covered_sites(before))
        total, contributions = combine(terms, self.weights)
        self.episode_return += total
        self.absolute_reward_sum += abs(total)
        if not math.isfinite(self.episode_return) or self.absolute_reward_sum > self.weights.max_abs_episode:
            raise ValueError('Episode reward accumulation exceeds declared safety threshold')
        self.time, self.tick = end, self.tick + 1
        self.perception.update(after, end, self.engine.event)
        self.observations = {a: self.perception.actor(a, end) for a in names}
        self.global_state = build_critic(after, end, self.c.horizon, self.settings, self.global_slots)
        tasks = [j for j in self.engine.jobs.values() if j['kind'] == 'task']
        resolved = bool(tasks) and all(j['status'] in ('completed', 'expired') for j in tasks)
        terminated = self.natural_end and resolved
        truncated = not terminated and end >= self.c.horizon - 1e-9
        self.done = terminated or truncated
        reason = 'workload_resolved' if terminated else ('time_limit' if truncated else None)
        infos = {}
        for name in names:
            record = records[name]
            accepted = selection['accepted'][name]
            record.update(executed_assignment=accepted,
                          executed_action=record['sampled_action'] if accepted == decoded[name] else 0,
                          arbitration_changed=accepted != decoded[name],
                          bootstrap_mask=int(not terminated), continuation_mask=int(not self.done),
                          terminated=bool(terminated), truncated=bool(truncated),
                          rnn_reset=bool(self.done or not self.observations[name]['active_mask']
                                         or old_obs[name]['active_mask'] != self.observations[name]['active_mask']))
            infos[name] = {'transition': copy.deepcopy(record), 'reward_terms': terms.copy(),
                           'reward_contributions': contributions.copy(), 'critic_state': self.state(),
                           'actor_active_mask': self.observations[name]['active_mask'], 'present_mask': 1,
                           'rnn_reset': record['rnn_reset'],
                           'privileged_info': True, 'episode': {'time': end, 'reason': reason,
                                                                'team_return': self.episode_return}}
            if self.done:
                infos[name]['final_observation'] = copy.deepcopy(self.observations[name])
                infos[name]['final_state'] = self.state()
        self.engine.rows.append({'type': 'transition', 'start': t, 'end': end,
                                 'records': copy.deepcopy(records), 'critic_before': old_state,
                                 'critic_after': self.state(), 'reward': total,
                                 'reward_terms': terms, 'reward_contributions': contributions,
                                 'terminated': bool(terminated), 'truncated': bool(truncated)})
        if self.done:
            self.engine.rows.append({'type': 'terminal', 'time': end, 'truth': after})
            self.agents = []
        return (copy.deepcopy(self.observations), {a: total for a in names},
                {a: bool(terminated) for a in names}, {a: bool(truncated) for a in names}, infos)

    def log_rows(self):
        if not self.done:
            raise RuntimeError('Only completed episode logs can be exported or scored')
        rows = copy.deepcopy(self.engine.rows)
        # Schema 2 evaluates the actual observed horizon. Planned settings remain recorded.
        rows[0]['config']['horizon'] = self.time
        return rows

    def export_log(self, path):
        write_jsonl(self.log_rows(), path)

    def metrics(self):
        return evaluate(self.log_rows())

    def close(self):
        pass
