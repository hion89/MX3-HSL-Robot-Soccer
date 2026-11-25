"""Explicit new-output-only reproduction; never retrains or opens protected holdout."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def main(output):
    output=output.resolve()
    if not output.is_relative_to(ROOT.resolve()):raise ValueError('Output must stay inside research/cah_mappo')
    if output.exists():raise FileExistsError('Existing output is preserved; choose a new path')
    from cah_mappo.phase4c import experiment
    from cah_mappo.phase4c.runs import run_sensitivity
    import phase4c_commitment,phase4c_reset_audit,phase4c_service_evidence,phase4c_finalize
    for command in ('protect','evidence','rules','plans','information','sensitivity'):
        print('Phase 4C reproduction:',command,flush=True);getattr(experiment,command)(ROOT,output)
    phase4c_commitment.run(output);run_sensitivity(ROOT,output,True)
    phase4c_reset_audit.run(output);phase4c_service_evidence.collect(output)
    experiment.classify(ROOT,output);experiment.audit(ROOT,output);phase4c_finalize.finalize(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    main(p.parse_args().output)
