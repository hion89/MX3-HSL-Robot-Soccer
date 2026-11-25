"""One-factor transfer diagnostics on 12 OLD non-final worlds; no rule tuning."""
import copy
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from cah_mappo.phase3b2b.io import save,write_raw
from cah_mappo.phase3b2a.scenarios import digest
from cah_mappo.phase4a.evaluation import run,load_network
from cah_mappo.phase4a.metrics import aggregate


def main():
    torch.set_num_threads(1);base=ROOT/'phase4a_outputs/study_v1';old=ROOT/'phase3b2b_outputs/main_v1'
    output=base/'transfer_probe';output.mkdir(exist_ok=False)
    freeze=json.loads((base/'design/freeze.json').read_text())
    originals=[s for s in json.loads((old/'benchmark_manifest.json').read_text()) if s['family']=='conditions'][:12]
    conditions=('control','worker_speed_only','service_duration_only','arrival_order_only','combined')
    results={}
    for seed,cp in freeze['checkpoints'].items():
        net=load_network(ROOT/cp['path'],cp['sha256']);groups={}
        for condition in conditions:
            records=[];raw=output/f'seed_{seed}_{condition}';raw.mkdir()
            for original in originals:
                spec=copy.deepcopy(original)
                if condition in ('worker_speed_only','combined'):
                    for r in spec['config']['robots']:
                        if r['base_role']=='worker':r['speed']=.5
                if condition in ('service_duration_only','combined'):
                    next(j for j in spec['config']['jobs'] if j['id']=='task1')['service']=12
                if condition in ('arrival_order_only','combined'):
                    next(j for j in spec['config']['jobs'] if j['id']=='task1')['spawn']=0
                spec.update(split='diagnostic',family=condition,team=4,response_limit_s={},feasibility_completion_at={})
                spec['sha256']=digest(spec)
                for mode in ('greedy','stochastic'):
                    result,rows=run(spec,'mappo_'+mode,net,sampling_seed=7001)
                    result['task0_success']=next(t['success'] for t in result['task_details'] if t['id']=='task0')
                    first=next(r for r in rows if r.get('event')=='policy_selection')
                    result['first_job_ids']={n:p['job_ids'] for n,p in first['actor_observations'].items()}
                    records.append(result)
                    write_raw(raw/f"{original['scenario_id']}_{mode}.jsonl.gz",rows)
            groups[condition]={'summary':{mode:aggregate([r for r in records if r['policy']=='mappo_'+mode]) for mode in ('greedy','stochastic')},
                               'episodes':records}
        results[seed]=groups
        print('one-factor transfer probe seed',seed,flush=True)
    save(output/'results.json',{'scope':'Post-hoc one-factor diagnostics on 12 previously used benchmark worlds; not a new final test',
        'variables':'speed=.5; task1 service=12; task1 spawn=0; all other physical conditions unchanged',
        'seeds':results,'retrained':False,'gate_used':False})


if __name__=='__main__':main()
