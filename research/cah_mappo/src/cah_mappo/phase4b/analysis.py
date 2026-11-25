"""Outcome-based summaries, resolution diagnostics, and independent log audits."""
from collections import defaultdict
import csv
import math
from pathlib import Path
from statistics import mean,stdev
from ..phase3b2b.io import read_raw,save
from ..phase3b2a.scenarios import digest
from ..phase3a.contracts import SampledAction,validate_sample
from .metrics import evaluate_new,aggregate
from .experiment import read,sha,ensure_frozen,grouped,internal_validation_key,POLICIES


def audit_all(root,output):
    frozen=ensure_frozen(root,output);target=output/'audit';target.mkdir(exist_ok=False)
    counts=defaultdict(int);max_error=0.
    for folder in sorted((output/'comparison').iterdir()):
        if not folder.is_dir():continue
        saved=read(folder/'results.json');reconstructed=[]
        for entry in saved['episodes']:
            rows=read_raw(folder/'raw'/f"{entry['scenario_id']}_r{entry['repeat']}.jsonl.gz")
            m=evaluate_new(rows)
            for k,v in m.items():assert v==entry[k],(folder.name,k)
            reconstructed.append(entry);counts['comparison_raw']+=1
            # Original actor samples stay samples; executed assignments are separately logged.
            for e in rows:
                if e.get('event')!='policy_selection':continue
                meta=e['action'].get('policy_metadata',{})
                for name,item in meta.get('actor_samples',{}).items():
                    s=item['sample'];p=item['packet']
                    sample=SampledAction(s['action'],s['log_probability'],tuple(s['action_mask']),tuple(s['probabilities']),
                                         s['policy_version'],s['observation_token'])
                    validate_sample(sample,p['action_mask'],p['observation_token']);counts['actor_samples']+=1
                    a=sample.action;decoded=None if a==0 else p['self_assignment'] if a==1 else p['job_ids'][a-2]
                    assert e['action']['requested'][name]==decoded
        assert grouped(reconstructed)==saved['summary']
    selection=read(output/'design/validation_selection.json')
    for c in selection['candidates']:
        folder=output/'design/validation_raw'/f"{c['kind']}_{int(c['value'])}";records=[];logs=[]
        for path in sorted(folder.glob('*.jsonl.gz')):
            rows=read_raw(path);m=evaluate_new(rows);meta=rows[0]['phase4b'];m.update(family=meta['family'],team=meta['team'])
            records.append(m);logs.append(rows);counts['validation_raw']+=1
        # Selection order means are independent of input file sorting.
        assert grouped(records)==c['summary'];assert internal_validation_key(records,logs)==c['selection_key']
    resolution=read(output/'sensitivity/results.json')
    for entry in resolution['episodes_detail']:
        path=output/'sensitivity'/entry['policy']/f"{entry['scenario_id']}_dt{entry['physics_dt']}.jsonl.gz"
        m=evaluate_new(read_raw(path))
        for k,v in m.items():assert v==entry[k],(path.name,k)
        counts['resolution_raw']+=1
    offline=read(output/'offline/results.json');oracle_success={}
    for folder in (output/'offline').iterdir():
        if not folder.is_dir():continue
        results=[]
        for path in folder.glob('*.jsonl.gz'):
            m=evaluate_new(read_raw(path));results.append(m);counts['offline_raw']+=1
        oracle_success[folder.name]=any(m['joint_task_and_defense_success'] for m in results)
    for case in offline['cases']:assert oracle_success[case['scenario_id']]==case['has_joint_success']
    by_id={s['scenario_id']:s for s in read(output/'design/diagnostic_manifest.json')}
    comparison=read(output/'comparison/summary.json')
    for certificate in offline['certificates']:
        if certificate['type']=='required_service_exceeds_total_robot_time':
            assert certificate['required_robot_seconds']>certificate['available_robot_seconds']
        elif certificate['joint_impossible_proven']:
            assert not oracle_success[certificate['scenario_id']]
            for folder in (output/'comparison').iterdir():
                if folder.is_dir():
                    saved=read(folder/'results.json')
                    assert all(not r['joint_task_and_defense_success'] for r in saved['episodes'] if r['scenario_id']==certificate['scenario_id'])
        counts['certificates_checked']+=1
    # Replay real Actor forwards on one episode per model and execution mode.
    from ..phase4a.evaluation import load_network
    import torch
    torch.set_num_threads(1)
    for seed,cp in frozen['checkpoints'].items():
        net=load_network(root/cp['path'],cp['sha256'])
        for mode in ('greedy','stochastic'):
            folder=output/f'comparison/seed_{seed}_{mode}';entry=read(folder/'results.json')['episodes'][0]
            rows=read_raw(folder/'raw'/f"{entry['scenario_id']}_r0.jsonl.gz")
            for event in rows:
                if event.get('event')!='policy_selection':continue
                samples=event['action']['policy_metadata']['actor_samples'];vectors=[];masks=[]
                for name in ('A','B','C','D'):
                    item=samples.get(name);vectors.append(item['packet']['observation'] if item else [0.]*146)
                    masks.append(item['packet']['action_mask'] if item else [1]+[0]*9)
                with torch.no_grad():p=net.distribution(torch.tensor(vectors,dtype=torch.float32),torch.tensor(masks,dtype=torch.bool)).probs
                for i,name in enumerate(('A','B','C','D')):
                    if name in samples:
                        error=max(abs(a-b) for a,b in zip(p[i].tolist(),samples[name]['sample']['probabilities']))
                        max_error=max(max_error,error);assert error<1e-10
                counts['actor_forward_batches']+=1
    before=read(output/'preservation_before.json');repo=root.parents[1]
    for name,h in before['old_source_files_sha256'].items():assert sha(repo/name)==h,name
    for name,expected in before['old_output_metadata_only'].items():
        stat=(root/name).stat();assert {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}==expected,name
    for cp in before['checkpoints'].values():assert sha(root/cp['path'])==cp['sha256']
    save(target/'audit.json',{'status':'PASS','counts':dict(counts),'max_actor_probability_error':max_error,
        'old_source_hashes_preserved':len(before['old_source_files_sha256']),
        'old_output_metadata_preserved':len(before['old_output_metadata_only']),
        'checkpoint_hashes_preserved':5,'protected_contents_opened':False,'retrained':False})
    print('AUDIT PASS',dict(counts),flush=True)


def summarize_all(root,output):
    target=output/'analysis';target.mkdir(exist_ok=False);data={}
    for p in (output/'comparison').iterdir():
        if p.is_dir():data[p.name]=read(p/'results.json')
    flat=[]
    for label,d in data.items():
        for condition,s in d['summary']['by_condition'].items():
            flat.append({'label':label,'condition':condition,**{k:v for k,v in s.items() if not isinstance(v,dict)}})
    keys=sorted({k for r in flat for k in r})
    with (target/'condition_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    seed_stats={}
    for mode in ('greedy','stochastic'):
        for condition in data['fixed']['summary']['by_condition']:
            stats={}
            for key in ('task_success_rate','task0_success','threat_pass_rate','joint_task_and_defense_success','legacy_gap_region_seconds','physical_gap_region_seconds'):
                values=[data[f'seed_{s}_{mode}']['summary']['by_condition'][condition][key] for s in ('101','211','307','401','503')]
                valid=[v for v in values if v is not None]
                stats[key]={'values':values,'mean':mean(valid) if valid else None,'sd':stdev(valid) if len(valid)>1 else None}
            seed_stats[f'{mode}_{condition}']=stats
    save(target/'five_seed_transfer.json',{'unit':'five existing trained model seeds; no new training; no superiority claim','conditions':seed_stats})
    sensitivity=read(output/'sensitivity/results.json')['episodes_detail'];groups=defaultdict(dict)
    for r in sensitivity:groups[(r['policy'],r['scenario_id'])][r['physics_dt']]=r
    resolution={};continuous=('legacy_gap_region_seconds','ownership_gap_region_seconds','physical_gap_region_seconds',
                             'threat_weighted_gap_seconds','task0_response_s','task0_completed_at','first_handover_ready_at')
    for policy in ('fixed','exact','ready','adaptive','observed_reserve','lookahead'):
        resolution[policy]={}
        for dt in (1.,.5,.25):
            pairs=[(cases[dt],cases[.125]) for (p,_),cases in groups.items() if p==policy]
            changes={}
            for key in continuous:
                deltas=[abs(a[key]-b[key]) for a,b in pairs if a.get(key) is not None and b.get(key) is not None]
                changes[key]={'paired':len(deltas),'max_abs_delta':max(deltas) if deltas else None,'mean_abs_delta':mean(deltas) if deltas else None}
            changes.update(task0_success_flips=sum(a['task0_success']!=b['task0_success'] for a,b in pairs),
                           deadline_miss_rate_changes=sum(a['deadline_miss_rate']!=b['deadline_miss_rate'] for a,b in pairs),
                           threat_pass_rate_changes=sum(a['threat_pass_rate']!=b['threat_pass_rate'] for a,b in pairs),
                           worlds=len(pairs))
            resolution[policy][str(dt)]=changes
    save(target/'resolution_sensitivity.json',{'reference_dt':.125,'control_clock_seconds':1.,'results':resolution})
    specs=read(output/'design/diagnostic_manifest.json');index={(s['family'],s['team'],s['index']):s for s in specs}
    prefix=[]
    for policy in POLICIES:
        for team in (2,4):
            for i in range(6):
                cases=[index[f,team,i] for f in ('A_near','D_visible')];frames=[]
                for s in cases:
                    rows=read_raw(output/'comparison'/policy/'raw'/f"{s['scenario_id']}_r0.jsonl.gz")
                    frames.append([{'public':r['public_observation'],'requested':r['action']['requested']} for r in rows
                                   if r.get('event')=='policy_selection' and r['time']<2])
                assert frames[0]==frames[1],(policy,team,i)
                prefix.append({'policy':policy,'team':team,'index':i,'identical_prefix_before_2':True})
    save(target/'future_information_audit.json',{'pairs':len(prefix),'all_current_policy_prefixes_identical':True,
        'scope':'Different unseen task location/deadline cannot change public inputs or native actions before t=2','details':prefix})
    # Metrics describe potential, ownership, and outcomes separately; no causal inflation.
    connections={}
    for policy,d in data.items():
        relevant=[r for r in d['episodes'] if r['threat_arrivals']]
        connections[policy]={'episodes_with_threat':len(relevant),
            'legacy_gap_but_no_threat_pass':sum(r['legacy_gap_region_seconds']>0 and r['threat_pass_count']==0 for r in relevant),
            'legacy_gap_and_pass':sum(r['legacy_gap_region_seconds']>0 and r['threat_pass_count']>0 for r in relevant),
            'zero_physical_gap_but_pass':sum(r['physical_gap_region_seconds']==0 and r['threat_pass_count']>0 for r in relevant),
            'positive_threat_exposure_gap_but_defended':sum(r['threat_weighted_gap_seconds']>0 and r['threat_pass_count']==0 for r in relevant)}
    save(target/'metric_outcome_connections.json',connections)
    offline=read(output/'offline/results.json');paired=[]
    for i in range(6):
        near=index['A_near',2,i];far=index['D_visible',2,i]
        n=next(c for c in offline['cases'] if c['scenario_id']==near['scenario_id'])
        f=next(c for c in offline['cases'] if c['scenario_id']==far['scenario_id'])
        worker=next(r for r in near['config']['robots'] if r['id']=='B')
        paired.append({'index':i,'both_have_joint_witness':n['has_joint_success'] and f['has_joint_success'],
            'pre_spawn_prefix_identical':True,'worker_distance_per_control_tick':worker['speed'],
            'near_task_radius':near['config']['radius'],
            'interpretation':'near task needs immediate service at t=2; any full pre-t=2 baseward control tick moves worker outside radius; no arbitrary future waypoint action'})
    save(target/'online_vs_oracle_pairs.json',{'scope':'bounded action space, matched prefix branch diagnostic, not proof of a learning advantage',
        'pairs':paired,'both_feasible_pairs':sum(p['both_have_joint_witness'] for p in paired)})
    print('Analysis saved',len(flat),'condition rows;',len(prefix),'prefix pairs',flush=True)
