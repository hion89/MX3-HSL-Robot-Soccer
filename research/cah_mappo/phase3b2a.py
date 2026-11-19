"""Bounded CPU Standard MAPPO pilots; no installation, remote access or CAH."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from cah_mappo.phase3b2a.experiment import run
    run(args.output, ROOT / 'phase3b1_outputs')
