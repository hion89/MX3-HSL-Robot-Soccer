"""Post-hoc paired mechanism checks on existing gap cases, NOT an unbiased benchmark."""
import copy
import json
import math
from pathlib import Path
from statistics import mean
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from cah_mappo.phase2.model import eta
from cah_mappo.phase3b2b.io import save,read_raw,write_raw
from cah_mappo.phase4a.metrics import distribution,aggregate
from cah_mappo.phase4a.evaluation import run,load_network
from types import SimpleNamespace as NS


def main():
    torch.set_num_threads(1)
    base=ROOT/'phase4a_outputs/study_v1';old=ROOT/'phase3b2b_outputs/main_v1'
    target=base/'legacy_paired_replay';target.mkdir(exist_ok=False)
    specs={s['scenario_id']:s for s in json.loads((old/'benchmark_manifest.json').read_text())}
    freeze=json.loads((base/'design/freeze.json').read_text())
    gap_groups=json.loads((base/'legacy/mappo_gaps.json').read_text())
    results={};raw=target/'raw';raw.mkdir()
    for seed in ('101','211','307','401','503'):
        cp=freeze['checkpoints'][seed];net=load_network(ROOT/cp['path'],cp['sha256'])
        cases=gap_groups[f'{seed}_stochastic']['gap_cases'];by_gate={k:[] for k in ('original','ready','adaptive')}
        for case in cases:
            spec=copy.deepcopy(specs[case['scenario_id']]);spec.update(split='diagnostic',family='legacy_gap_selected',team=4,
                response_limit_s={},feasibility_completion_at={})
            for gate in (None,'ready','adaptive'):
                label=gate or 'original'
                result,rows=run(spec,'mappo_stochastic',net,gate,
                                freeze['parameters'] if gate=='adaptive' else None,case['repeat'],sampling_seed=7001)
                if gate is None:
                    assert result['actual_gap_region_seconds']==case['actual_gap_region_seconds']
                    assert result['task_details']==case['task_details']
                by_gate[label].append(result)
                write_raw(raw/f"seed{seed}_{spec['scenario_id']}_r{case['repeat']}_{label}.jsonl.gz",rows)
        results[seed]={k:{'summary':aggregate(v),'episodes':v} for k,v in by_gate.items()}
        print('paired legacy seed',seed,'gap-selected cases',len(cases),flush=True)
    save(target/'paired_results.json',{'scope':'Post-hoc selection on ORIGINAL gap cases only; not population efficacy or held-out evaluation',
        'sampling_base':7001,'exact_original_gap_and_task_replay':True,'seeds':results})
    travel=[]
    for sid,spec in specs.items():
        rows=read_raw(old/'references/central_fixed/raw'/f'{sid}_p234_r0.jsonl.gz')
        for job in spec['config']['jobs']:
            if job['kind']!='task':continue
            state=next(r['truth'] for r in rows if r['type']=='interval' and r['start']==job['spawn'])
            obs=NS(radius=state['radius'],dt=spec['config']['dt'])
            times={name:eta(NS(**r),NS(**state['jobs'][job['id']]),obs) for name,r in state['robots'].items()}
            owner=job['fixed_owner'];end=rows[-1]['truth']['jobs'][job['id']]['completed']
            travel.append({'scenario_id':sid,'job':job['id'],'window_s':job['deadline']-job['spawn'],
                'assigned_worker_eta_at_spawn_ignoring_other_commitment':times[owner],
                'minimum_guard_eta_at_spawn':min(times[n] for n,r in state['robots'].items() if r['base_role']=='guard'),
                'distance_assigned_worker_at_spawn':math.dist(state['robots'][owner]['position'],job['position']),
                'actual_fixed_completion_latency_s':end-job['spawn'],'actual_deadline_slack_s':job['deadline']-end})
    summary={key:distribution([r[key] for r in travel]) for key in travel[0] if key not in ('scenario_id','job')}
    summary['minimum_deadline_slack_s']=min(r['actual_deadline_slack_s'] for r in travel)
    save(target/'travel_deadline_relationships.json',{'summary':summary,'tasks':travel,
        'eta_note':'ETA ignores queue/work already in progress; actual fixed completion is the feasibility evidence'})


if __name__=='__main__':main()
