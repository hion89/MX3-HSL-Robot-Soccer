"""CPU-only Phase 4A diagnostics. No training, package installation or holdout input."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('legacy','prepare','compare','audit','report'))
    parser.add_argument('--output',type=Path,default=ROOT/'phase4a_outputs/study_v1')
    args=parser.parse_args()
    if args.command=='legacy':
        from cah_mappo.phase4a.legacy import analyze
        analyze(ROOT/'phase3b2b_outputs/main_v1',args.output/'legacy')
    else:
        from cah_mappo.phase4a.experiment import prepare,compare,audit,report
        {'prepare':prepare,'compare':compare,'audit':audit,'report':report}[args.command](ROOT,args.output)
