"""Predeclared bounded pilots; validation selection precedes any test evaluation."""
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import torch

from ..core import write_jsonl
from ..phase3b1.learner import TrainConfig
from ..phase3b1.networks import ActorCritic
from .learner import PoolLearner
from .scenarios import build_manifest, STAGES, digest
from .evaluation import evaluate_many, aggregate
from .prior_analysis import analyze


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def selection_key(summary):
    # Predeclared lexicographic model-selection rule, not a composite paper score.
    return (summary['task_success_rate'], -summary['coverage_gap_ratio'],
            -summary['restricted_response_mean_s'])


def train_pilot(output, train, validation, stage, weight, steps, seed=7):
    if any(s['split'] != 'validation' for s in validation):
        raise ValueError('Only validation scenarios may select checkpoints')
    output.mkdir(exist_ok=False)
    (output / 'checkpoints').mkdir(); (output / 'trajectories').mkdir()
    learner = PoolLearner(TrainConfig(seed=seed), train, weight)
    source_root = Path(__file__).resolve().parents[3]
    source = [*Path(__file__).parent.glob('*.py'),
              *Path(__file__).parents[1].joinpath('phase3b1').glob('*.py'),
              *Path(__file__).parents[1].joinpath('phase3a').glob('*.py'),
              *Path(__file__).parents[1].joinpath('phase2').glob('*.py')]
    config = {'train_config': asdict(learner.c), 'stage': stage, 'coverage_weight': weight,
              'steps': steps, 'train_ids': [s['scenario_id'] for s in train],
              'validation_ids': [s['scenario_id'] for s in validation], 'test_used_for_selection': False,
              'python': sys.version, 'device': 'cpu', 'dependencies': {n: importlib.metadata.version(n) for n in ('torch', 'numpy')},
              'selection': 'maximize(validation success, -coverage gap, -restricted response); no return ranking',
              'source_sha256': {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}}
    save(output / 'config.json', config)
    learner.checkpoint(output / 'checkpoints/initial.pt')
    initial = evaluate_many(validation, 'mappo_greedy', learner.net, weight)
    save(output / 'validation_initial.json', initial)
    best = selection_key(initial['summary']); best_path = output / 'checkpoints/initial.pt'
    curve = []; validation_history = [{'step': 0, **initial['summary']}]
    start = time.perf_counter(); update_seconds = collect_seconds = validation_seconds = 0.
    with (output / 'train_episodes.jsonl').open('x', encoding='utf-8') as episode_stream:
        while learner.steps < steps:
            t = time.perf_counter(); batch = learner.collect(min(256, steps - learner.steps)); collect_seconds += time.perf_counter() - t
            if not curve:
                with (output / 'first_rollout.pt').open('xb') as f: torch.save(batch, f)
            t = time.perf_counter(); metrics = learner.update(batch); update_seconds += time.perf_counter() - t
            for row in learner.completed:
                episode_stream.write(json.dumps({'total_env_steps': learner.steps, **row}, allow_nan=False) + '\n')
            for episode, log in learner.last_episode_logs:
                if episode < 2 or learner.steps == steps and episode >= learner.episodes - 2:
                    write_jsonl(log, output / 'trajectories' / f'episode{episode}.jsonl')
            report = {'update': learner.policy_updates, 'total_env_steps': learner.steps, **metrics,
                      **{'train_' + k: v for k, v in aggregate(learner.completed).items()}}
            if learner.steps % 1024 == 0 or learner.steps == steps:
                t = time.perf_counter(); validation_result = evaluate_many(validation, 'mappo_greedy', learner.net, weight)
                validation_seconds += time.perf_counter() - t
                summary = validation_result['summary']; validation_history.append({'step': learner.steps, **summary})
                report.update({'validation_' + k: v for k, v in summary.items()})
                path = output / f'checkpoints/step{learner.steps}.pt'; learner.checkpoint(path)
                save(output / f'validation_step{learner.steps}.json', validation_result)
                if selection_key(summary) > best:
                    best, best_path = selection_key(summary), path
                print(json.dumps({'stage': stage, 'weight': weight, 'step': learner.steps,
                                  'validation_success': summary['task_success_rate'],
                                  'validation_gap': summary['coverage_gap_ratio']}, allow_nan=False), flush=True)
            curve.append(report)
            save(output / f"update_{learner.steps:06d}.json", report)
    learner.checkpoint(output / 'checkpoints/final.pt')
    save(output / 'learning_curve.json', curve)
    import csv
    with (output / 'learning_curve.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in curve for key in row}))
        writer.writeheader(); writer.writerows(curve)
    summary = {'stage': stage, 'coverage_weight': weight, 'environment_steps': learner.steps,
               'completed_episodes': learner.episodes, 'training_and_validation_wall_seconds': time.perf_counter() - start,
               'collection_seconds': collect_seconds, 'optimizer_seconds': update_seconds,
               'intermediate_validation_seconds': validation_seconds,
               'selected_checkpoint': str(best_path), 'selected_validation_key': best,
               'initial_validation': initial['summary'], 'validation_history': validation_history}
    save(output / 'selection.json', summary)
    return summary


def load_net(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    network = ActorCritic(checkpoint['config']['hidden'])
    network.load_state_dict(checkpoint['network']); network.eval()
    return network


def run(output, old_outputs):
    output.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(24, 8, 12)
    save(output / 'scenario_manifest.json', manifest)
    plan = {'version': 'phase3b2a-pilot-v1', 'training_seed': 7, 'train_scenarios_per_stage': 24,
            'validation_scenarios_per_stage': 8, 'test_scenarios_per_stage': 12,
            'stage_A': {'steps_per_weight': 8192, 'coverage_weights': [2., 8.]},
            'stage_B3': {'steps': 4096, 'gate': 'A default selected validation success >= .75'},
            'stage_C': {'steps': 2048, 'gate': 'B3 selected validation success >= .60'},
            'stage_B2_D': 'difficulty evaluation only; no training',
            'max_training_steps': 22528, 'selection': 'lexicographic validation success, -gap, -restricted response',
            'test_policy': 'evaluate only after selection_freeze.json; no subsequent tuning',
            'episode_rule': 'fixed horizon 32 seconds for all new policies',
            'sensitivity': 'two predeclared reward settings, identical optimizer/network/scenarios/seed'}
    save(output / 'predeclared_plan.json', plan)
    save(output / 'phase3b1_diagnostics.json', analyze(old_outputs))
    runs = []
    for weight in (2., 8.):
        runs.append(train_pilot(output / f'A_weight{int(weight)}', manifest['scenarios']['A']['train'],
                                manifest['scenarios']['A']['validation'], 'A', weight, 8192))
    if runs[0]['selected_validation_key'][0] >= .75:
        runs.append(train_pilot(output / 'B3_weight2', manifest['scenarios']['B3']['train'],
                                manifest['scenarios']['B3']['validation'], 'B3', 2., 4096))
        if runs[-1]['selected_validation_key'][0] >= .60:
            runs.append(train_pilot(output / 'C_weight2', manifest['scenarios']['C']['train'],
                                    manifest['scenarios']['C']['validation'], 'C', 2., 2048))
    freeze = {'manifest_sha256': digest(manifest), 'plan_sha256': digest(plan), 'runs': runs,
              'nominal_weight_selected_on_validation': max(runs[:2], key=lambda r: tuple(r['selected_validation_key']))['coverage_weight'],
              'training_finished_before_test': True,
              'checkpoint_sha256': {r['selected_checkpoint']: hashlib.sha256(Path(r['selected_checkpoint']).read_bytes()).hexdigest() for r in runs}}
    save(output / 'selection_freeze.json', freeze)
    # Test is first evaluated below. No training/selection occurs after this line.
    t = time.perf_counter(); benchmarks = {}
    modes = ('local_hold', 'local_fixed', 'local_eta', 'local_random',
             'central_fixed', 'central_greedy', 'central_exact', 'central_handover')
    for stage in STAGES:
        benchmarks[stage] = {}
        for mode in modes:
            benchmarks[stage][mode] = evaluate_many(manifest['scenarios'][stage]['test'], mode,
                                                   log_dir=output / 'test_logs' / stage / mode)
    comparisons = {}
    for entry in runs:
        label = f"{entry['stage']}_weight{int(entry['coverage_weight'])}"
        comparisons[label] = {}
        paths = [('selected', entry['selected_checkpoint']), ('final', output / label / 'checkpoints/final.pt')]
        for point, path in paths:
            network = load_net(path)
            for mode in ('mappo_greedy', 'mappo_stochastic'):
                key = f'{point}_{mode}'
                comparisons[label][key] = evaluate_many(manifest['scenarios'][entry['stage']]['test'], mode, network,
                                                        entry['coverage_weight'], output / 'test_logs' / label / key)
    transfer = {}
    old_network = load_net(Path(old_outputs) / 'debug_seed7_steps4096/checkpoints/final_step4096.pt')
    a_network = load_net(runs[0]['selected_checkpoint'])
    for stage in STAGES:
        transfer[stage] = {}
        for label, network in [('fixed_3b1_checkpoint', old_network), ('randomized_A_checkpoint', a_network)]:
            transfer[stage][label] = evaluate_many(manifest['scenarios'][stage]['test'], 'mappo_greedy', network,
                                                  log_dir=output / 'test_logs' / 'transfer' / stage / label)
    result = {'benchmarks': benchmarks, 'trained_policies': comparisons, 'transfer': transfer,
              'test_evaluation_wall_seconds': time.perf_counter() - t}
    save(output / 'test_results.json', result)
    print(json.dumps({'complete': True, 'training_steps': sum(r['environment_steps'] for r in runs),
                      'test_seconds': result['test_evaluation_wall_seconds']}, allow_nan=False), flush=True)
    return result
