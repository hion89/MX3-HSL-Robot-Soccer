"""Post-run descriptive accounting, conditional handover metrics and export tables.

No training or checkpoint selection is performed here.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import math
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def handover_details(rows):
    events=[h for r in rows for h in r['handovers']]
    observed=[h['initial_gap_s'] for h in events if not h['right_censored']]
    return {'handover_events':len(events),'right_censored_events':sum(h['right_censored'] for h in events),
            'episodes_without_handover':sum(not r['handovers'] for r in rows),
            'initial_ready_gap_event_mean_including_censored_lower_bounds_s':mean(h['initial_gap_s'] for h in events) if events else None,
            'uncensored_ready_gap_event_mean_s':mean(observed) if observed else None,
            'physical_gap_union_per_handover_mean_s':mean(h['physical_gap_union_s'] for h in events) if events else None}


def task_details(rows):
    result={}
    for row in rows:
        for task in row['detailed_tasks']:
            d=result.setdefault(task['id'],{'attempts':0,'successes':0,'selection_requests':0,'failure_categories':{}})
            d['attempts']+=1;d['successes']+=int(task['success']);d['selection_requests']+=task['selection_count']
            if not task['success']:d['failure_categories'][task['category']]=d['failure_categories'].get(task['category'],0)+1
    for d in result.values():d['success_rate']=d['successes']/d['attempts']
    return result


def world_complete_wilson(rows):
    # One Bernoulli outcome per physical world within a SINGLE repeat, not per task.
    n=len(rows);success=sum(r['task_success_rate']==1 for r in rows);z=1.959963984540054
    p=success/n;denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return {'independent_physical_scenarios':n,'complete_worlds':success,'complete_world_rate':p,
            'wilson95':[center-half,center+half],
            'estimand':'all tasks completed in a physical world, one evaluation repeat; not a training-seed CI or Task Success Rate CI'}


def analyze(run,output):
    from cah_mappo.phase3b2b.scenarios import SEEDS,FAMILIES
    from cah_mappo.phase3b2b.metrics import aggregate,SCALARS
    from cah_mappo.phase3b2b.io import save
    output.mkdir(parents=True,exist_ok=False)
    all_rows=[];training={};handover={};tasks={};repeats={};epoch_validation=[];scenario_intervals={}
    for seed in SEEDS:
        folder=run/f'seed_{seed}';selection=json.loads((folder/'selection.json').read_text())
        curve=json.loads((folder/'learning_curve.json').read_text());validations=[]
        for path in sorted(folder.glob('validation_step*.json')):
            step=int(path.stem.removeprefix('validation_step'));record=json.loads(path.read_text())
            for mode in ('greedy','stochastic'):
                entry={'seed':seed,'step':step,'mode':mode,**record[mode]['summary']}
                epoch_validation.append(entry);validations.append(entry)
        training[str(seed)]={'selection':selection,'first_update':curve[0],'last_update':curve[-1],
                            'max_approximate_kl':max(r['approximate_kl'] for r in curve),
                            'max_clip_fraction':max(r['clip_fraction'] for r in curve),
                            'max_preupdate_logp_error':max(r['preupdate_logp_max_error'] for r in curve),
                            'validation_history':validations}
        main=json.loads((folder/'benchmark_results.json').read_text())['modes']
        for mode,group in main.items():
            key=f'seed{seed}_{mode}';rows=group['episodes']
            handover[key]=handover_details(rows);tasks[key]=task_details(rows)
            for family in ('overall',*FAMILIES):
                all_rows.append({'policy':'Standard MAPPO','training_seed':seed,'mode':mode,'family':family,
                                 **group['summaries'][family]})
            if mode=='stochastic':repeats[str(seed)]={str(i):aggregate([r for r in rows if r['repeat']==i]) for i in range(3)}
            for repeat in range(3 if mode=='stochastic' else 1):
                scenario_intervals[f'{key}_repeat{repeat}']=world_complete_wilson([r for r in rows if r['repeat']==repeat])
    for folder in sorted((run/'references').iterdir()):
        if not folder.is_dir():continue
        group=json.loads((folder/'results.json').read_text());rows=group['episodes']
        handover[folder.name]=handover_details(rows);tasks[folder.name]=task_details(rows)
        scenario_intervals[folder.name]=world_complete_wilson(rows)
        for family in ('overall',*FAMILIES):
            all_rows.append({'policy':folder.name,'training_seed':'not trained','mode':'reference','family':family,**group['summaries'][family]})
    save(output/'training_summary.json',training);save(output/'handover_censoring.json',handover)
    save(output/'per_task_outcomes.json',tasks);save(output/'stochastic_repeat_variability.json',repeats)
    save(output/'scenario_complete_world_intervals.json',scenario_intervals)
    for name,rows in [('all_metrics.csv',all_rows),('validation_curves.csv',epoch_validation)]:
        keys=sorted({key for r in rows for key,value in r.items() if not isinstance(value,dict)})
        with (output/name).open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    save(output/'analysis_scope.json',{'training_or_model_selection':False,'final_holdout_access':False,
         'handover_warning':'legacy ready-time metric includes censored lower bounds; actual physical CGR is reported separately',
         'references_are_not_independent_training_replicates':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.run.resolve(),a.output.resolve())
