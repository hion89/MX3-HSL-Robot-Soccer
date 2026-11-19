"""Paired local comparisons and explicitly privileged, unchanged Phase 2 references."""
from dataclasses import asdict
import math
from pathlib import Path
from statistics import mean
import torch

from ..core import write_jsonl
from ..phase2.engine import Environment
from ..phase2.model import config_from_dict
from ..phase2.policies import make_policy
from ..phase2.metrics import evaluate as score
from ..phase3a.contracts import SampledAction
from .baselines import LocalPolicy
from .scenarios import make_env, digest


def actor_inputs(observations, settings):
    """Inference path has no global state or privileged infos argument."""
    vectors, masks = [], []
    for name in settings.roster:
        packet = observations.get(name)
        vectors.append(packet['observation'] if packet else [0.] * 146)
        masks.append(packet['action_mask'] if packet else [1] + [0] * 9)
    return torch.tensor(vectors, dtype=torch.float32), torch.tensor(masks, dtype=torch.bool)


@torch.no_grad()
def evaluate_one(spec, mode, network=None, coverage_weight=2., policy_seed=7):
    if mode.startswith('central_'):
        config = config_from_dict(spec['config'])
        policy = make_policy(mode.removeprefix('central_'), config)
        engine = Environment(config, policy, mode)
        rows = engine.run()
        rows[0]['phase3b2a_scenario'] = {k: spec[k] for k in ('scenario_id', 'sha256', 'split', 'stage')}
        result = {'scenario_id': spec['scenario_id'], 'stage': spec['stage'], 'split': spec['split'],
                  'mode': mode, 'information': 'central_perfect_current_team_reference',
                  'return': None, 'exogenous_sha256': digest(engine.plan), **score(rows)}
        return result, rows
    env, obs = make_env(spec, coverage_weight)
    rng = torch.Generator(device='cpu').manual_seed(policy_seed * 100000 + spec['event_seed'])
    policy = LocalPolicy(mode, policy_seed + spec['event_seed']) if mode.startswith('local_') else None
    if policy is None and (network is None or mode not in ('mappo_greedy', 'mappo_stochastic')):
        raise ValueError('Unknown policy or missing network')
    while env.agents:
        actions = {}
        dist = selected = None
        if policy is None:
            vectors, masks = actor_inputs(obs, env.settings)
            dist = network.distribution(vectors, masks)
            selected = torch.multinomial(dist.probs, 1, generator=rng).squeeze(-1) if mode == 'mappo_stochastic' else dist.probs.argmax(-1)
        for name in env.agents:
            packet = obs[name]
            i = env.settings.roster.index(name)
            action = policy.decide(name, packet) if policy else int(selected[i])
            if mode == 'mappo_stochastic':
                probabilities = tuple(dist.probs[i].tolist())
            elif mode == 'local_random':
                count = sum(packet['action_mask']); probabilities = tuple(v / count for v in packet['action_mask'])
            else:
                probabilities = tuple(float(j == action) for j in range(10))
            actions[name] = SampledAction(action, math.log(probabilities[action]), tuple(packet['action_mask']),
                                          probabilities, f'eval-{mode}-not-training', packet['observation_token'])
        obs, *_ = env.step(actions)
    return ({'scenario_id': spec['scenario_id'], 'stage': spec['stage'], 'split': spec['split'],
             'mode': mode, 'information': 'local_actor_packets_only', 'return': env.episode_return,
             'exogenous_sha256': digest(env.engine.plan), **env.metrics()}, env.log_rows())


def aggregate(rows):
    def avg(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return mean(values) if values else None
    tasks = sum(r['task_count'] for r in rows)
    return {'episodes': len(rows), 'task_success_rate': avg('task_success_rate'),
            'deadline_compliance_rate': None if not rows else 1 - avg('deadline_miss_rate'),
            'coverage_gap_ratio': avg('coverage_gap_ratio'), 'response_time_mean_s': avg('response_time_mean_s'),
            'restricted_response_mean_s': avg('restricted_response_mean_s'),
            'unresponded_rate': sum(r['unresponded_tasks'] for r in rows) / tasks if tasks else None,
            'all_tasks_makespan_mean_s_complete_only': avg('all_tasks_makespan_s'),
            'completed_task_flow_time_sum_mean_s': avg('completed_task_flow_time_sum_s'),
            'role_switch_count': avg('role_switch_count'), 'assignment_change_count': avg('assignment_change_count'),
            'requested_duplicate_rate': avg('requested_duplicate_rate'), 'mean_return': avg('return')}


def evaluate_many(specs, mode, network=None, coverage_weight=2., log_dir=None):
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=False)
    results = []
    for spec in specs:
        row, log = evaluate_one(spec, mode, network, coverage_weight)
        results.append(row)
        if log_dir is not None:
            write_jsonl(log, Path(log_dir) / f"{spec['scenario_id']}.jsonl")
    return {'summary': aggregate(results), 'episodes': results}
