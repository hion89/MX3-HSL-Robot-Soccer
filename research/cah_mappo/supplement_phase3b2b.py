"""Descriptive tail risk and paired scenario uncertainty, separate from seed CIs."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean,variance
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def summarize(run,output):
    from cah_mappo.phase3b2b.scenarios import SEEDS,FAMILIES
    from cah_mappo.phase3b2b.io import save
    output.mkdir(parents=True,exist_ok=False)
    trained={str(seed):json.loads((run/f'seed_{seed}/benchmark_results.json').read_text())['modes'] for seed in SEEDS}
    tails={}
    for seed,modes in trained.items():
        for mode,group in modes.items():
            rows=group['episodes'];values=sorted(r['actual_gap_region_seconds'] for r in rows)
            worst=max(rows,key=lambda r:r['actual_gap_region_seconds'])
            tails[f'{seed}_{mode}']={'episodes':len(rows),'episodes_with_gap':sum(v>0 for v in values),
                'p95_gap_region_seconds':values[int((len(values)-1)*.95)],'p99_gap_region_seconds':values[int((len(values)-1)*.99)],
                'max_gap_region_seconds':max(values),'max_cgr':max(r['coverage_gap_ratio'] for r in rows),
                'worst_gap_benchmark_scenario_id':worst['scenario_id'],'worst_gap_repeat':worst['repeat']}
    paired={}
    for folder in sorted((run/'references').iterdir()):
        if not folder.is_dir():continue
        reference=json.loads((folder/'results.json').read_text());ref={r['scenario_id']:r for r in reference['episodes']}
        paired[folder.name]={}
        for seed,modes in trained.items():
            paired[folder.name][seed]={}
            for mode,group in modes.items():
                result={}
                for metric in ('task_success_rate','coverage_gap_ratio','response_time_mean_s'):
                    scenes={}
                    for row in group['episodes']:
                        if row[metric] is not None:scenes.setdefault(row['scenario_id'],[]).append(row[metric])
                    by_family={f:[] for f in FAMILIES}
                    for sid,values in scenes.items():
                        if ref[sid][metric] is not None:by_family[ref[sid]['family']].append(mean(values)-ref[sid][metric])
                    if any(len(v)<2 for v in by_family.values()):continue
                    avg=sum(mean(v) for v in by_family.values())/2
                    se=math.sqrt(sum(variance(v)/len(v) for v in by_family.values())/4)
                    result[metric]={'paired_delta_mappo_minus_reference':avg,'stratified_scenario_standard_error':se,
                        'scenario_normal_approx_95':[avg-1.95996398454*se,avg+1.95996398454*se],
                        'paired_scene_counts':{f:len(v) for f,v in by_family.items()}}
                paired[folder.name][seed][mode]=result
    save(output/'tail_responsibility_gaps.json',tails)
    save(output/'paired_scenario_uncertainty.json',{'estimand':'scenario-sampling uncertainty conditional on each fixed trained network; separate from n=5 seed uncertainty',
        'method':'equal-family stratified paired mean; normal approximation with independent physical-world strata',
        'not_training_replications':True,'comparisons':paired})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.run.resolve(),a.output.resolve())
