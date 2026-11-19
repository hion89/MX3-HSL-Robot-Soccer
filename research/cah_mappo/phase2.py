"""Run Phase 2 policies and replay schema 1 or 2 logs; no robot imports."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cah_mappo.core import write_jsonl
from cah_mappo.phase2.engine import Environment
from cah_mappo.phase2.metrics import evaluate, load_jsonl
from cah_mappo.phase2.model import config_from_dict
from cah_mappo.phase2.policies import POLICIES, make_policy
from cah_mappo.phase2.scenarios import SCENARIOS, scenario


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--scenario', choices=SCENARIOS, default='single')
    run.add_argument('--policy', choices=POLICIES, default='exact')
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--jitter', type=float, default=0)
    run.add_argument('--config', type=Path)
    run.add_argument('--output', type=Path, required=True)
    score = commands.add_parser('evaluate')
    score.add_argument('log', type=Path)
    suite = commands.add_parser('suite')
    suite.add_argument('--output', type=Path, required=True)
    suite.add_argument('--seeds', type=int, nargs='+', default=[0])
    suite.add_argument('--jitter', type=float, default=0)
    args = p.parse_args()
    if args.command == 'evaluate':
        print(json.dumps(evaluate(load_jsonl(args.log)), indent=2, allow_nan=False))
        return
    if args.command == 'run':
        c = config_from_dict(json.loads(args.config.read_text(encoding='utf-8'))) if args.config else scenario(args.scenario, args.seed, args.jitter)
        rows = Environment(c, make_policy(args.policy, c), args.policy).run()
        write_jsonl(rows, args.output)
        print(json.dumps(evaluate(load_jsonl(args.output)), indent=2, allow_nan=False))
        return
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Repeated seeds would overwrite output')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in SCENARIOS:
        for seed in args.seeds:
            c = scenario(name, seed, args.jitter)
            for policy in POLICIES:
                file = f'{name}__{policy}__seed{seed}.jsonl'
                rows = Environment(c, make_policy(policy, c), policy).run()
                write_jsonl(rows, args.output / file)
                results.append({'scenario': name, 'seed': seed, 'policy': policy, 'log': file,
                                **evaluate(load_jsonl(args.output / file))})
    with (args.output / 'summary.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(results, f, indent=2, allow_nan=False)
        f.write('\n')
    # No composite score or policy ordering: each metric remains separate.
    aggregates = []
    for name in SCENARIOS:
        for policy in POLICIES:
            subset = [r for r in results if r['scenario'] == name and r['policy'] == policy]
            out = {'scenario': name, 'policy': policy, 'seeds': args.seeds, 'runs': len(subset)}
            for metric in ('coverage_gap_ratio', 'task_success_rate', 'deadline_miss_rate',
                           'response_time_mean_s', 'restricted_response_mean_s', 'role_switch_count',
                           'task_duplicate_rate', 'all_tasks_makespan_s'):
                values = [r[metric] for r in subset if r[metric] is not None]
                out[metric] = {'mean': mean(values) if values else None,
                               'min': min(values) if values else None, 'max': max(values) if values else None,
                               'defined_runs': len(values)}
            aggregates.append(out)
    with (args.output / 'aggregates.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(aggregates, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'runs': len(results), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
