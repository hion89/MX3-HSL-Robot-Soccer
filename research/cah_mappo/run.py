"""Run scripted experiments or recompute metrics from JSONL, using only stdlib."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from cah_mappo.core import Config, POLICIES, SCENARIOS, Simulator, scenario_config, write_jsonl
from cah_mappo.metrics import evaluate, load_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--scenario', choices=SCENARIOS)
    run.add_argument('--policy', choices=POLICIES)
    run.add_argument('--seed', type=int)
    run.add_argument('--config', type=Path)
    run.add_argument('--output', type=Path, required=True)
    suite = sub.add_parser('suite')
    suite.add_argument('--output', type=Path, required=True)
    suite.add_argument('--seed', type=int, default=0)
    score = sub.add_parser('evaluate')
    score.add_argument('log', type=Path)
    args = parser.parse_args()
    if args.command == 'evaluate':
        print(json.dumps(evaluate(load_jsonl(args.log)), indent=2, allow_nan=False))
        return
    if args.command == 'run':
        if args.config:
            c = Config(**json.loads(args.config.read_text(encoding='utf-8')))
            changes = {k: getattr(args, k) for k in ('scenario', 'policy', 'seed') if getattr(args, k) is not None}
            c = replace(c, **changes)
        else:
            c = scenario_config(args.scenario or 'normal', args.policy or 'ready', args.seed or 0)
        rows = Simulator(c).run()
        write_jsonl(rows, args.output)
        print(json.dumps(evaluate(load_jsonl(args.output)), indent=2, allow_nan=False))
        return
    # A fresh directory is mandatory; previous evidence is never overwritten.
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            path = args.output / f'{scenario}__{policy}.jsonl'
            rows = Simulator(scenario_config(scenario, policy, args.seed)).run()
            write_jsonl(rows, path)
            metrics = evaluate(load_jsonl(path))
            summary.append({'scenario': scenario, 'policy': policy, 'log': path.name, **metrics})
    with (args.output / 'summary.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
