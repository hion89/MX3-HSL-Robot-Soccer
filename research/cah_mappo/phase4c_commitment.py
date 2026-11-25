"""One-factor follow-up: do not steal a selected finite-task worker for a quiet base.

The failed first diagnostic rule and all its outputs remain intact.
No timing, reward, physical parameter, or neural weight is changed.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.phase4c.runs import execute
from cah_mappo.phase4c.policies import RiskDirectedSwap
from cah_mappo.phase4b.experiment import read,grouped
from cah_mappo.phase3b2b.io import save,write_raw

def run(output):
    target=output/'rules/commitment_safe_swap';target.mkdir(exist_ok=False);records=[]
    for spec in read(ROOT/'phase4b_outputs/study_v1/design/diagnostic_manifest.json'):
        p=RiskDirectedSwap(protect_active_tasks=True);m,rows=execute(spec,p);records.append(m)
        write_raw(target/f"{spec['scenario_id']}.jsonl.gz",rows)
    save(target/'results.json',{'episodes':records,'summary':grouped(records),
         'interface':'current_public_with_preparation',
         'post_hoc_diagnostic_not_final_unseen_test':True,
         'single_change':'quiet-base replacement excludes workers provisionally assigned to active finite tasks'})
    print('commitment-preserving diagnostic complete',flush=True)

if __name__=='__main__':run(ROOT/'phase4c_outputs/study_v1')
