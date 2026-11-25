"""Phase 4B: validity experiments only, isolated CPU environment, no training."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('protect','prepare','compare','offline','sensitivity','audit','report','legacy'))
    parser.add_argument('--output',type=Path,default=ROOT/'phase4b_outputs/study_v1')
    args=parser.parse_args()
    from cah_mappo.phase4b import experiment
    getattr(experiment,args.command)(ROOT,args.output)
