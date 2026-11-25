"""Exclusive artifacts, validation-only threshold selection, frozen-policy evaluation."""
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from collections import defaultdict
from statistics import mean
import torch
from ..phase2.model import config_from_dict
from ..phase3b2a.scenarios import digest
from ..phase3b2b.io import save,write_raw,read_raw
from .scenarios import bank,witness,FAMILIES
from .evaluation import run,load_network
from .metrics import aggregate,score


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root,output):
    torch.set_num_threads(1);output.mkdir(parents=True,exist_ok=True)
    target=output/'design';target.mkdir(exist_ok=False)
    # Snapshot only known tracked research files, selected checkpoints and non-final manifests.
    repo=root.parents[1]
    tracked=subprocess.check_output(['git','ls-files','-z','research'],cwd=repo).decode().split('\0')
    old={name:sha(repo/name) for name in tracked if name}
    freeze=read(root/'phase3b2b_outputs/main_v1/all_seeds_selection_freeze.json')
    paths={}
    for seed,data in freeze['seeds'].items():
        path=root/f'phase3b2b_outputs/main_v1/seed_{seed}'/data['selection']['selected_checkpoint']
        assert sha(path)==data['checkpoint_sha256']
        paths[seed]={'path':path.relative_to(root).as_posix(),'sha256':sha(path)}
    protected=root/'phase3b2a_repair_outputs/pilot_v1/sealed_final_holdout_manifest.json'
    metadata={'bytes':protected.stat().st_size,'mtime_ns':protected.stat().st_mtime_ns}
    save(target/'preservation.json',{'tracked_files':old,'selected_checkpoints':paths,'sealed_metadata_only':metadata})
    validation=bank('validation',3);diagnostic=bank('diagnostic',6)
    save(target/'validation_manifest.json',validation);save(target/'diagnostic_manifest.json',diagnostic)
    candidates=[]
    for risk in (.5,1.,2.):
        for budget in (1.,2.,4.):
            p={'risk_ratio':risk,'eta_budget':budget};records=[]
            for spec in validation:
                result,rows=run(spec,'witness_immediate',gate_mode='adaptive',parameters=p)
                records.append(result)
            summary=aggregate(records)
            key=[summary['task_success_rate'],-summary['threat_limit_violations'],
                 -summary['actual_gap_region_seconds'],-summary['role_switch_count']]
            candidates.append({'parameters':p,'selection_key':key,'summary':summary})
    best=max(candidates,key=lambda c:c['selection_key'])
    save(target/'validation_selection.json',{'candidates':candidates,'selected':best,
        'selection_rule':'lexicographic validation task success, fewer threat violations, smaller gap, fewer role switches; ties retain declared candidate order',
        'selector_used':'deterministic immediate witness requests; not learned policy tuning'})
    # Explicit feasibility evidence on the SAME generated cases, no temporal optimality claim.
    feasibility=[]
    for spec in validation+diagnostic:
        results={}
        for mode in ('fixed','immediate','ready'):
            done,rows=witness(config_from_dict(spec['config']),mode)
            rows[0]['phase4a']={k:spec[k] for k in ('scenario_id','response_limit_s')}
            m=score(rows);results[mode]={'task0_completed_at':done,'success':m['task_success_rate'],
                                      'gap':m['actual_gap_region_seconds'],'task_details':m['task_details']}
        if spec['family']=='A':assert results['fixed']['success']==1
        if spec['family']=='B':assert results['fixed']['task0_completed_at'] is None and results['ready']['success']==1
        if spec['family'] in ('C','C_risk','R_risk'):
            assert results['immediate']['success']==1 and results['ready']['task0_completed_at'] is None
        feasibility.append({'scenario_id':spec['scenario_id'],'family':spec['family'],'team':spec['team'],'witnesses':results})
    save(target/'feasibility.json',feasibility)
    sources={p.relative_to(root).as_posix():sha(p) for p in sorted((root/'src/cah_mappo/phase4a').glob('*.py'))}
    sources['phase4a.py']=sha(root/'phase4a.py');sources['PHASE4A_PROTOCOL_KO.md']=sha(root/'PHASE4A_PROTOCOL_KO.md')
    save(target/'freeze.json',{'parameters':best['parameters'],'validation_sha256':sha(target/'validation_manifest.json'),
         'diagnostic_sha256':sha(target/'diagnostic_manifest.json'),'checkpoints':paths,'sources':sources,
         'samples_per_family_team':6,'stochastic_repeats':2,'phase3b2b_reward_changed':False,'training_performed':False,
         'holdout_opened':False,'python':__import__('sys').version,'torch':torch.__version__,
         'timestamp_ns':time.time_ns()})
    print('Frozen validation selection',best['parameters'],'diagnostic cases',len(diagnostic),flush=True)


def summarize(records):
    groups=defaultdict(list)
    for r in records:groups[f"{r['family']}_team{r['team']}"].append(r)
    return {'overall':aggregate(records),'by_condition':{k:aggregate(v) for k,v in groups.items()}}


def compare(root,output):
    torch.set_num_threads(1);freeze=read(output/'design/freeze.json')
    for name,h in freeze['sources'].items():assert sha(root/name)==h,name
    specs=read(output/'design/diagnostic_manifest.json')
    assert sha(output/'design/diagnostic_manifest.json')==freeze['diagnostic_sha256']
    target=output/'comparison';target.mkdir(exist_ok=False)
    plan=[]
    for policy in ('central_fixed','central_greedy','central_exact','central_handover','local_fixed','local_eta',
                   'witness_fixed','witness_immediate','witness_ready'):
        plan.append((policy,policy,None,None,1))
    for gate in ('ready','adaptive'):
        plan.append(('central_exact_'+gate,'central_exact',gate,None,1))
    for seed in freeze['checkpoints']:
        for mode,repeats in (('greedy',1),('stochastic',2)):
            for gate in (None,'ready','adaptive'):
                plan.append((f'seed_{seed}_{mode}_{gate or "original"}','mappo_'+mode,gate,seed,repeats))
    exogenous={};all_results={};networks={};start=time.perf_counter()
    for label,policy,gate,seed,repeats in plan:
        if seed and seed not in networks:
            cp=freeze['checkpoints'][seed];networks[seed]=load_network(root/cp['path'],cp['sha256'])
        network=networks.get(seed);folder=target/label;folder.mkdir();raw=folder/'raw';raw.mkdir();records=[]
        for spec in specs:
            for repeat in range(repeats):
                result,rows=run(spec,policy,network,gate,freeze['parameters'] if gate=='adaptive' else None,repeat)
                sid=spec['scenario_id']
                if sid in exogenous:assert result['exogenous_sha256']==exogenous[sid]
                exogenous[sid]=result['exogenous_sha256']
                result['seed']=seed;records.append(result)
                write_raw(raw/f'{sid}_r{repeat}.jsonl.gz',rows)
        saved={'label':label,'seed':seed,'policy':policy,'gate':gate,'repeats':repeats,
               'summary':summarize(records),'episodes':records}
        save(folder/'results.json',saved);all_results[label]=saved['summary']
        print(label,'TSR',round(saved['summary']['overall']['task_success_rate'],5),
              'gap',round(saved['summary']['overall']['actual_gap_region_seconds'],4),flush=True)
    save(target/'summary.json',all_results)
    save(target/'execution.json',{'wall_seconds':time.perf_counter()-start,'groups':len(plan),
                                 'episodes':sum(len(specs)*p[-1] for p in plan),'common_worlds':len(specs),
                                 'all_exogenous_digests_match':True,'retrained':False})


def audit(root,output):
    torch.set_num_threads(1);target=output/'audit';target.mkdir(exist_ok=False)
    protection=read(output/'design/preservation.json');repo=root.parents[1]
    for name,h in protection['tracked_files'].items():assert sha(repo/name)==h,name
    for data in protection['selected_checkpoints'].values():assert sha(root/data['path'])==data['sha256']
    p=root/'phase3b2a_repair_outputs/pilot_v1/sealed_final_holdout_manifest.json'
    assert {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}==protection['sealed_metadata_only']
    count=0;changed=0;actor_checks=0
    for folder in sorted((output/'comparison').iterdir()):
        if not folder.is_dir():continue
        saved=read(folder/'results.json')
        for entry in saved['episodes']:
            rows=read_raw(folder/'raw'/f"{entry['scenario_id']}_r{entry['repeat']}.jsonl.gz")
            actual=score(rows)
            for key,value in actual.items():assert value==entry[key],(folder.name,key)
            rewards=[r['reward'] for r in rows if r['type']=='transition']
            if rewards:assert abs(sum(rewards)-entry['episode_return'])<1e-12
            selections={r['time']:r for r in rows if r.get('event')=='policy_selection' and 'actor_observations' in r}
            for row in rows:
                if row['type']!='transition':continue
                for name,record in row['records'].items():
                    from ..phase3a.contracts import validate_ppo_record
                    validate_ppo_record(record);actor_checks+=1
                    if record['arbitration_changed']:changed+=1
                    packet=selections[row['start']]['actor_observations'][name];a=record['sampled_action']
                    decoded=None if a==0 else packet['self_assignment'] if a==1 else packet['job_ids'][a-2]
                    assert decoded==record['requested_assignment']
                    assert selections[row['start']]['action']['requested'][name]==decoded
            count+=1
        assert summarize(saved['episodes'])==saved['summary']
    old_hashes=read(output/'legacy/raw_source_hashes.json')
    for name,h in old_hashes.items():assert sha(root/'phase3b2b_outputs/main_v1'/name)==h
    save(target/'audit.json',{'status':'PASS','raw_episodes_recomputed':count,'actor_sample_records_checked':actor_checks,
         'arbitrated_records':changed,'legacy_raw_preserved':len(old_hashes),'tracked_files_preserved':len(protection['tracked_files']),
         'selected_checkpoints_preserved':len(protection['selected_checkpoints']),'holdout_opened':False})
    print('audit PASS',count,'episodes',actor_checks,'sample records',flush=True)


def report(root,output):
    target=output/'analysis';target.mkdir(exist_ok=False)
    groups={p.name:read(p/'results.json') for p in (output/'comparison').iterdir() if p.is_dir()}
    flat=[];pairs={}
    for label,data in groups.items():
        for condition,summary in data['summary']['by_condition'].items():
            flat.append({'label':label,'seed':data['seed'],'condition':condition,**{k:v for k,v in summary.items() if not isinstance(v,dict)}})
    keys=sorted({k for r in flat for k in r})
    with (target/'condition_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    seed_statistics={}
    for mode in ('greedy','stochastic'):
        for gate in ('original','ready','adaptive'):
            matched=[d for label,d in groups.items() if label.startswith('seed_') and label.endswith(f'_{mode}_{gate}')]
            for condition in ['overall',*[f'{f}_team{n}' for f in FAMILIES for n in (2,4)]]:
                rows=[d['summary']['overall'] if condition=='overall' else d['summary']['by_condition'][condition] for d in matched]
                values={k:[r[k] for r in rows] for k in ('task_success_rate','actual_gap_region_seconds','coverage_gap_ratio','response_time_mean_s','gap_episode_rate')}
                seed_statistics[f'{mode}_{gate}_{condition}']={k:{'seed_values':v,'mean':mean(x for x in v if x is not None) if any(x is not None for x in v) else None}
                                                              for k,v in values.items()}
        for gate in ('ready','adaptive'):
            per_seed={}
            for seed in ('101','211','307','401','503'):
                original=groups[f'seed_{seed}_{mode}_original']['episodes'];gated=groups[f'seed_{seed}_{mode}_{gate}']['episodes']
                source={(r['scenario_id'],r['repeat']):r for r in original}
                per_seed[seed]={'success_change':mean(r['task_success_rate']-source[r['scenario_id'],r['repeat']]['task_success_rate'] for r in gated),
                                'gap_change':mean(r['actual_gap_region_seconds']-source[r['scenario_id'],r['repeat']]['actual_gap_region_seconds'] for r in gated)}
            pairs[f'{mode}_{gate}']=per_seed
    save(target/'seed_statistics.json',seed_statistics);save(target/'paired_changes.json',pairs)
    print('analysis saved',len(flat),'condition rows',flush=True)
