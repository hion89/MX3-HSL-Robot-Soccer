"""Controlled CPU Standard MAPPO baseline; all operations confined to research outputs."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('calibrate');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p=sub.add_parser('train-all');p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--run',type=Path,required=True);p.add_argument('--stream',type=int,required=True)
    for name in ('benchmark-seed','benchmark-slots'):
        p=sub.add_parser(name);p.add_argument('--run',type=Path,required=True);p.add_argument('--stream',type=int,required=True)
    p=sub.add_parser('benchmark-reference');p.add_argument('--run',type=Path,required=True);p.add_argument('--policy',required=True)
    p=sub.add_parser('benchmark-all');p.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    from cah_mappo.phase3b2b import runner
    if args.command=='calibrate':runner.calibrate(args.output.resolve())
    elif args.command=='prepare':runner.prepare(args.output.resolve(),args.calibration.resolve(),ROOT)
    elif args.command=='train-all':runner.train_all(ROOT,args.run.resolve())
    elif args.command=='train':
        from cah_mappo.phase3b2b.training import train_worker
        train_worker(args.run.resolve(),args.stream)
    elif args.command=='benchmark-all':runner.benchmark_all(ROOT,args.run.resolve())
    elif args.command.startswith('benchmark-'):
        from cah_mappo.phase3b2b import benchmark
        if args.command=='benchmark-seed':benchmark.seed_benchmark(args.run.resolve(),args.stream)
        elif args.command=='benchmark-slots':benchmark.slot_benchmark(args.run.resolve(),args.stream)
        else:benchmark.reference_benchmark(args.run.resolve(),args.policy)
