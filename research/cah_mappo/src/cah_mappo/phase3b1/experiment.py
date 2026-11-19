"""Bounded CPU overfit experiment, paired baselines, checkpoints and measured curves."""
from dataclasses import asdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
from statistics import mean
import sys
import time
import torch

from ..core import write_jsonl
from ..phase3a.contracts import SampledAction
from ..phase3a.adapters import padded_batch
from .learner import Learner, TrainConfig, minimal_env


@torch.no_grad()
def evaluate_policy(net, mode, seeds, jitter=0.0, logs=None):
    results = []
    if logs is not None:Path(logs).mkdir(parents=True, exist_ok=False)
    for seed in seeds:
        env, obs = minimal_env(seed, jitter)
        rng = random.Random(f'evaluation:{mode}:{seed}')
        generator = torch.Generator(device='cpu').manual_seed(seed + 50000)
        while env.agents:
            actions = {}
            dist = None
            if mode.startswith('learned'):
                packed = padded_batch(obs, env.state(), env.settings)
                dist = net.distribution(torch.tensor(packed['actor_observations']), torch.tensor(packed['available_actions']))
                sampled = torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1) if mode == 'learned_stochastic' else dist.probs.argmax(-1)
            for name in env.agents:
                o = obs[name]; mask = o['action_mask']; valid = [i for i, v in enumerate(mask) if v]
                if dist is not None:
                    i = env.settings.roster.index(name); action = int(sampled[i])
                    probs = tuple(dist.probs[i].tolist()) if mode == 'learned_stochastic' else tuple(float(j == action) for j in range(10))
                elif mode == 'random':
                    action = rng.choice(valid); probs = tuple(1 / len(valid) if v else 0 for v in mask)
                else:
                    action = 1 if mask[1] else 0
                    if mode == 'manual':
                        job = 'task1' if name == 'A' else 'base'
                        if job in o['job_ids'] and mask[o['job_ids'].index(job) + 2]:action = o['job_ids'].index(job) + 2
                    probs = tuple(float(j == action) for j in range(10))
                actions[name] = SampledAction(action, math_log(probs[action]), tuple(mask), probs,
                                               f'eval-{mode}-not-training', o['observation_token'])
            obs, *_ = env.step(actions)
        result = {'environment_seed': seed, 'mode': mode, 'jitter': jitter, 'return': env.episode_return, **env.metrics()}
        results.append(result)
        if logs is not None:env.export_log(Path(logs) / f'seed{seed}.jsonl')
    return results


def math_log(value):
    import math
    return math.log(value)


def aggregate(rows):
    def m(key):
        values = [r[key] for r in rows if r[key] is not None]
        return mean(values) if values else None
    return {'episodes': len(rows), 'mean_return': m('return'), 'task_success_rate': m('task_success_rate'),
            'response_time_mean_s': m('response_time_mean_s'), 'coverage_gap_ratio': m('coverage_gap_ratio'),
            'unresponded_tasks': sum(r['unresponded_tasks'] for r in rows)}


def run(output, config=TrainConfig(), total_steps=4096, eval_every=4):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoints').mkdir(); (output / 'trajectories').mkdir()
    learner = Learner(config)
    environment, _ = minimal_env(config.environment_seed)
    source_root = Path(__file__).resolve().parents[3]
    metadata = {'train_config': asdict(config), 'total_env_steps_requested': total_steps,
                'training_environment_config': asdict(environment.c), 'settings': asdict(environment.settings),
                'reward_weights': asdict(environment.weights), 'terminate_on_tasks_resolved': True,
                'evaluation_environment_seeds': list(range(1000, 1020)), 'evaluation_jitter': .05,
                'python': sys.version, 'platform': platform.platform(),
                'dependencies': {n: importlib.metadata.version(n) for n in ('torch', 'numpy')},
                'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                'network': {'actor': [146, config.hidden, config.hidden, 10], 'critic': [141, config.hidden, config.hidden, 1],
                            'activation': 'Tanh', 'actor_parameters': sum(p.numel() for p in learner.net.actor.parameters()),
                            'critic_parameters': sum(p.numel() for p in learner.net.critic.parameters())},
                'source_sha256': {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [*Path(__file__).parent.glob('*.py'), *Path(__file__).parents[1].joinpath('phase3a').glob('*.py')]}}
    with (output / 'config.json').open('x', encoding='utf-8') as f:json.dump(metadata, f, indent=2)
    learner.checkpoint(output / 'checkpoints/initial.pt')
    baseline = {}
    for mode in ('random', 'hold', 'manual', 'learned_greedy', 'learned_stochastic'):
        rows = evaluate_policy(learner.net, mode, range(1000, 1020), .05, output / f'initial_{mode}')
        baseline[mode] = {'summary': aggregate(rows), 'episodes': rows}
    with (output / 'baselines.json').open('x', encoding='utf-8') as f:json.dump(baseline, f, indent=2)
    best = None; best_file = None; curves = []; train_episode_count = 0
    start = time.perf_counter()
    with (output / 'train_episodes.jsonl').open('x', encoding='utf-8') as episode_stream:
        while learner.steps < total_steps:
            batch = learner.collect(min(config.rollout_steps, total_steps - learner.steps))
            if not curves:
                with (output / 'first_rollout.pt').open('xb') as f:torch.save(batch, f)
            metrics = learner.update(batch)
            for row in learner.completed:
                episode_stream.write(json.dumps({'total_env_steps': learner.steps, **row}, allow_nan=False) + '\n')
            for episode, log in learner.last_episode_logs:
                if episode < 2 or any(e.get('event') == 'task_completed' for e in log) and train_episode_count < 4:
                    write_jsonl(log, output / 'trajectories' / f'episode{episode}.jsonl')
                    train_episode_count += 1
            report = {'update': len(curves) + 1, 'total_env_steps': learner.steps,
                      'train_episodes': len(learner.completed), **metrics}
            if learner.completed:
                stats = aggregate(learner.completed)
                report.update({'train_' + k: v for k, v in stats.items()})
            if not curves or report['update'] % eval_every == 0 or learner.steps == total_steps:
                debug = aggregate(evaluate_policy(learner.net, 'learned_greedy', [config.environment_seed]))
                report.update({'debug_' + k: v for k, v in debug.items()})
                # Selection uses training/debug scenario only, never held-out evaluation seeds.
                key = (debug['task_success_rate'], debug['mean_return'])
                if best is None or key > best:
                    best = key; best_file = output / f"checkpoints/best_step{learner.steps}.pt"
                    learner.checkpoint(best_file)
            curves.append(report)
            with (output / f"update_{learner.steps:06d}.json").open('x', encoding='utf-8') as f:json.dump(report, f, indent=2, allow_nan=False)
            if report['update'] % eval_every == 0 or learner.steps == total_steps:
                print(json.dumps({'step': learner.steps, 'training_success': report.get('train_task_success_rate'),
                                  'debug_success': report.get('debug_task_success_rate'),
                                  'debug_return': report.get('debug_mean_return'),
                                  'entropy': report['policy_entropy']}, allow_nan=False), flush=True)
    final_file = output / f'checkpoints/final_step{learner.steps}.pt'; learner.checkpoint(final_file)
    fields = sorted({k for r in curves for k in r})
    with (output / 'learning_curve.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(curves)
    with (output / 'learning_curve.json').open('x', encoding='utf-8') as f:json.dump(curves, f, indent=2, allow_nan=False)
    # Save final and training-selected candidate separately; never tune on these held-out results.
    results = {}
    for label, path in [('final', final_file), ('best_debug', best_file)]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        learner.net.load_state_dict(checkpoint['network'])
        for mode in ('learned_greedy', 'learned_stochastic'):
            rows = evaluate_policy(learner.net, mode, range(1000, 1020), .05, output / f'{label}_{mode}')
            results[f'{label}_{mode}'] = {'checkpoint': str(path.relative_to(output)), 'summary': aggregate(rows), 'episodes': rows}
    result = {'total_env_steps': learner.steps, 'training_wall_seconds': time.perf_counter() - start,
              'best_debug_checkpoint': str(best_file.relative_to(output)), 'results': results}
    with (output / 'evaluation.json').open('x', encoding='utf-8') as f:json.dump(result, f, indent=2, allow_nan=False)
    return result
