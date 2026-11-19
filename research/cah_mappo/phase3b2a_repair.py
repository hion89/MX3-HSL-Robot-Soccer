"""Run bounded diagnostics and two equal-budget Standard MAPPO repair pilots."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from cah_mappo.phase3b2a_repair.experiment import run
    run(args.output,ROOT)
