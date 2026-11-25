"""Phase 4C: offline diagnostics and a final research-direction recommendation."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('protect','evidence','rules','plans','information','sensitivity','classify','audit'))
    p.add_argument('--output',type=Path,default=ROOT/'phase4c_outputs/study_v1')
    a=p.parse_args()
    from cah_mappo.phase4c import experiment
    getattr(experiment,a.command)(ROOT,a.output)
