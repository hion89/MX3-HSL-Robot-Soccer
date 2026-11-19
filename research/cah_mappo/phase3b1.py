"""Explicit CPU-only minimal MAPPO experiment; never invoked automatically."""
import argparse
from pathlib import Path
import sys
import json

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=4096)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--environment-seed', type=int, default=100)
    args = p.parse_args()
    if not 1 <= args.steps <= 16384:
        raise ValueError('This debug entry point limits each experiment to 1..16384 environment steps')
    from cah_mappo.phase3b1.learner import TrainConfig
    from cah_mappo.phase3b1.experiment import run
    result = run(args.output, TrainConfig(seed=args.seed, environment_seed=args.environment_seed), args.steps)
    print(json.dumps({**{k: v for k, v in result.items() if k != 'results'},
                      'results': {k: v['summary'] for k, v in result['results'].items()}},
                     indent=2, allow_nan=False))


if __name__ == '__main__':main()
