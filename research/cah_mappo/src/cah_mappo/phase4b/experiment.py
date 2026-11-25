"""Exclusive artifacts and staged diagnostics; sealed data is never read."""
import copy
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import time
from ..phase3b2b.io import save,write_raw,read_raw
from ..phase3b2a.scenarios import digest
from ..phase4a.evaluation import load_network
from ..phase2.metrics import evaluate as old_metrics
from .scenarios import bank,FAMILIES
from .engine import Simulation
from .policies import Native,FrozenActor
from .metrics import evaluate_new,aggregate
from .planning import OfflineSchedule
from .model import travel_time

POLICIES=('fixed','greedy','exact','legacy_handover','ready','adaptive','observed_reserve','standing_reserve','lookahead')
SEEDS=('101','211','307','401','503')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def setup():
    import torch
    torch.set_num_threads(1)


def protect(root,output):
    output.mkdir(parents=True,exist_ok=False);repo=root.parents[1]
    names=[n for n in subprocess.check_output(['git','ls-files','-z','research'],cwd=repo).decode().split('\0') if n]
    old_files={n:sha(repo/n) for n in names}
    phase4a=read(root/'phase4a_outputs/study_v1/final_integrity.json')
    for name,h in phase4a['code_document_tests_sha256'].items():
        assert sha(root/name)==h,name;old_files[(root/name).relative_to(repo).as_posix()]=h
    meta={}
    for directory in root.iterdir():
        if not directory.is_dir() or directory==output.parent or not (directory.name=='outputs' or directory.name.endswith('_outputs')):continue
        for base,dirs,files in os.walk(directory):
            dirs[:]=[d for d in dirs if 'holdout' not in d.lower()]
            for name in files:
                p=Path(base)/name;stat=p.stat()
                meta[p.relative_to(root).as_posix()]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    old=read(root/'phase3b2b_outputs/main_v1/all_seeds_selection_freeze.json');checkpoints={}
    for seed in SEEDS:
        d=old['seeds'][seed];p=root/f'phase3b2b_outputs/main_v1/seed_{seed}'/d['selection']['selected_checkpoint']
        assert sha(p)==d['checkpoint_sha256']
        checkpoints[seed]={'path':p.relative_to(root).as_posix(),'sha256':sha(p)}
    save(output/'preservation_before.json',{'old_source_files_sha256':old_files,'old_output_metadata_only':meta,
        'checkpoints':checkpoints,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip(),
        'protected_contents_read':False})
    print('Protected',len(old_files),'source files,',len(meta),'old output metadata records; no sealed contents read',flush=True)


def result(spec,policy,dt=None):
    engine=Simulation(spec,policy,dt);rows=engine.run();metrics=evaluate_new(rows)
    metrics.update(scenario_id=spec['scenario_id'],family=spec['family'],team=spec['team'],
                   policy=policy.name,physics_dt=engine.c.dt,exogenous_sha256=digest(engine.plan))
    return metrics,rows


def grouped(rows):
    groups=defaultdict(list)
    for r in rows:groups[f"{r['family']}_team{r['team']}"].append(r)
    return {'overall_debug_mixture_only':aggregate(rows),'by_condition':{k:aggregate(v) for k,v in groups.items()}}


def internal_validation_key(rows,raw_logs):
    losses=[]
    for row,raw in zip(rows,raw_logs):
        task_loss=sum(t['priority'] for t in row['task_details'] if not t['success'])
        threat_loss=sum(e['intensity'] for e in raw if e.get('event')=='threat_passed')
        losses.append(task_loss+threat_loss)
    return [mean(losses),-mean(r['joint_task_and_defense_success'] for r in rows),
            mean(r['physical_gap_region_seconds'] for r in rows)]


def prepare(root,output):
    setup();target=output/'design';target.mkdir(exist_ok=False)
    validation=bank('validation',2);diagnostic=bank('diagnostic',6)
    save(target/'validation_manifest.json',validation);save(target/'diagnostic_manifest.json',diagnostic)
    candidate=[];raw_folder=target/'validation_raw';raw_folder.mkdir()
    for kind,values in (('warning',(4.,8.,12.)),('lookahead',(2.,4.,8.))):
        for value in values:
            label=f'{kind}_{int(value)}';folder=raw_folder/label;folder.mkdir();records=[];logs=[]
            for spec in validation:
                p=Native('observed_reserve',warning=value) if kind=='warning' else Native('lookahead',lookahead=value)
                m,rows=result(spec,p);records.append(m);logs.append(rows)
                write_raw(folder/f"{spec['scenario_id']}.jsonl.gz",rows)
            candidate.append({'kind':kind,'value':value,'selection_key':internal_validation_key(records,logs),
                              'summary':grouped(records)})
    selected={kind:min((c for c in candidate if c['kind']==kind),key=lambda c:c['selection_key'])['value'] for kind in ('warning','lookahead')}
    save(target/'validation_selection.json',{'candidates':candidate,'selected':selected,
        'scope':'internal tuning only; priority-weighted explicit task/threat failures, then joint success, then physical gap; not algorithm ranking'})
    sources={p.relative_to(root).as_posix():sha(p) for p in sorted((root/'src/cah_mappo/phase4b').glob('*.py'))}
    for name in ('phase4b.py','PHASE4B_PROTOCOL_KO.md','tests/test_phase4b.py'):sources[name]=sha(root/name)
    save(target/'freeze.json',{'parameters':selected,'sources':sources,'validation_sha256':sha(target/'validation_manifest.json'),
        'diagnostic_sha256':sha(target/'diagnostic_manifest.json'),'checkpoints':read(output/'preservation_before.json')['checkpoints'],
        'physics_dt':.25,'control_period':1.,'timestamp_ns':time.time_ns(),'retraining':False,'holdout_opened':False})
    print('FROZEN',selected,'validation',len(validation),'diagnostic',len(diagnostic),flush=True)


def ensure_frozen(root,output):
    frozen=read(output/'design/freeze.json')
    for name,h in frozen['sources'].items():assert sha(root/name)==h,name
    assert sha(output/'design/diagnostic_manifest.json')==frozen['diagnostic_sha256']
    return frozen


def compare(root,output):
    setup();frozen=ensure_frozen(root,output);specs=read(output/'design/diagnostic_manifest.json')
    target=output/'comparison';target.mkdir(exist_ok=False);start=time.perf_counter();summary={};worlds={};count=0
    plan=[(name,None,None,1) for name in POLICIES]
    plan += [(f'seed_{seed}_{mode}',seed,mode,1 if mode=='greedy' else 2) for seed in SEEDS for mode in ('greedy','stochastic')]
    networks={}
    for label,seed,mode,repeats in plan:
        folder=target/label;folder.mkdir();raw=folder/'raw';raw.mkdir();records=[]
        if seed and seed not in networks:
            cp=frozen['checkpoints'][seed];networks[seed]=load_network(root/cp['path'],cp['sha256'])
        for spec in specs:
            for repeat in range(repeats):
                p=(Native(label,**frozen['parameters']) if seed is None else
                   FrozenActor(networks[seed],mode,int(seed)*100000000+spec['event_seed']+repeat*1000000))
                m,rows=result(spec,p);m.update(seed=seed,repeat=repeat)
                sid=spec['scenario_id']
                if sid in worlds:assert worlds[sid]==m['exogenous_sha256']
                worlds[sid]=m['exogenous_sha256'];records.append(m);count+=1
                write_raw(raw/f'{sid}_r{repeat}.jsonl.gz',rows)
        saved={'label':label,'seed':seed,'mode':mode,'repeats':repeats,'summary':grouped(records),'episodes':records}
        save(folder/'results.json',saved);summary[label]=saved['summary']
        print('compared',label,'episodes',len(records),flush=True)
    save(target/'summary.json',summary);save(target/'execution.json',{'episodes':count,'groups':len(plan),'worlds':len(specs),
        'wall_seconds':time.perf_counter()-start,'same_exogenous_worlds':True,'retrained':False})


def offline(root,output):
    setup();ensure_frozen(root,output);target=output/'offline';target.mkdir(exist_ok=False)
    specs=read(output/'design/diagnostic_manifest.json');results=[];executions=0
    for spec in specs:
        if spec['team']!=2 or spec['family'] not in ('A_near','B_normal','C_reactive','D_visible'):continue
        folder=target/spec['scenario_id'];folder.mkdir();plans=[]
        # Fixed is a no-preparation alternative; each other candidate has full future timing.
        m,rows=result(spec,Native('fixed'));write_raw(folder/'fixed.jsonl.gz',rows)
        plans.append({'prepare':None,'depart':None,'metrics':m});executions+=1
        task=next(j for j in spec['config']['jobs'] if j['id']=='task0')
        for prepare in (0,1,2):
            for depart in range(2,min(15,int(task['deadline'])+1)):
                p=OfflineSchedule(prepare,depart);m,rows=result(spec,p);plans.append({'prepare':prepare,'depart':depart,'metrics':m})
                write_raw(folder/f'p{prepare}_d{depart}.jsonl.gz',rows);executions+=1
        both=[p for p in plans if p['metrics']['joint_task_and_defense_success']]
        task_ok=[p for p in plans if p['metrics']['task_success_rate']==1]
        results.append({'scenario_id':spec['scenario_id'],'family':spec['family'],'candidate_count':len(plans),
            'has_joint_success':bool(both),'has_task_success':bool(task_ok),
            'has_zero_legacy_gap_task_success':any(p['metrics']['legacy_gap_region_seconds']==0 for p in task_ok),
            'has_zero_physical_gap_joint_success':any(p['metrics']['physical_gap_region_seconds']==0 for p in both),
            'best_joint':min(both,key=lambda p:p['metrics']['legacy_gap_region_seconds']) if both else None,
            'best_task':min(task_ok,key=lambda p:p['metrics']['legacy_gap_region_seconds']) if task_ok else None})
    certificates=[]
    for spec in specs:
        if 'infeasibility_certificate' in spec:
            certificates.append({'scenario_id':spec['scenario_id'],**spec['infeasibility_certificate']})
        if spec['team']!=2 or spec['family'] not in ('C_reactive','D_visible'):continue
        robots={r['id']:r for r in spec['config']['robots']};jobs={j['id']:j for j in spec['config']['jobs']}
        a=robots['A'];w=robots['B'];j=jobs['task0'];base=jobs['base0'];radius=spec['config']['radius'];impact=4.
        worker_to_base=travel_time(w,base['position'],radius)
        guard_after=impact+travel_time(a,j['position'],radius)+j['service']
        guard_before=j['spawn']+travel_time(a,j['position'],radius)+j['service']
        # Generous lower bound: teleport worker to base at spawn, then move right.
        worker_task=j['spawn']+max(0.,j['position'][0]-base['position'][0]-radius)/w['speed']+j['service']
        proven=worker_to_base>impact and guard_after>j['deadline'] and guard_before>impact and worker_task>j['deadline']
        certificates.append({'scenario_id':spec['scenario_id'],'type':'two_robot_arrival_and_service_lower_bounds',
            'joint_impossible_proven':proven,'worker_earliest_base':worker_to_base,'guard_completion_after_defense_lb':guard_after,
            'guard_completion_before_defense_lb':guard_before,'worker_task_completion_lb':worker_task,'deadline':j['deadline'],
            'scope':'two robots; pre-spawn legal destinations are the known base or idle; no arbitrary unknown waypoint action'})
    save(target/'results.json',{'scope':'restricted future-aware schedule witnesses, not a complete temporal optimum or a certified upper bound',
        'executions':executions,'cases':results,'certificates':certificates})
    print('offline witnesses',executions,'executions',flush=True)


def sensitivity(root,output):
    setup();frozen=ensure_frozen(root,output);target=output/'sensitivity';target.mkdir(exist_ok=False)
    specs=[s for s in read(output/'design/diagnostic_manifest.json') if s['index']<2];records=[];count=0
    for name in ('fixed','exact','ready','adaptive','observed_reserve','lookahead'):
        folder=target/name;folder.mkdir()
        for spec in specs:
            signatures=[]
            for dt in (1.,.5,.25,.125):
                m,rows=result(spec,Native(name,**frozen['parameters']),dt)
                signatures.append(digest({'plan':rows[0]['exogenous_plan'],'threats':spec['threats'],'control':spec['control_period']}))
                times=[r['time'] for r in rows if r.get('event')=='policy_selection']
                assert times==list(range(32))
                m['control_times']=times;records.append(m);count+=1
                write_raw(folder/f"{spec['scenario_id']}_dt{dt}.jsonl.gz",rows)
            assert len(set(signatures))==1
        print('resolution compared',name,flush=True)
    save(target/'results.json',{'scope':'same physical world and one-second control clock; physics dt only varies',
        'episodes':count,'worlds':len(specs),'episodes_detail':records,'all_physical_signatures_equal':True})


def legacy(root,output):
    target=output/'legacy';target.mkdir(exist_ok=False)
    old=root/'phase3b2b_outputs/main_v1';d=read(root/'phase4a_outputs/study_v1/legacy/worst_gap.json')
    m=d['metrics'];path=old/f"seed_{m['seed']}/benchmark_raw/stochastic"/f"{m['scenario_id']}_p234_r{m['repeat']}.jsonl.gz"
    rows=read_raw(path);original=old_metrics(rows);overlay=copy.deepcopy(rows)
    overlay[0]['phase4b']={'threat_plan':[],'response_budgets':{j['id']:2. for j in rows[0]['config']['jobs'] if j['kind']=='base'}}
    metric=evaluate_new(overlay);assert metric['legacy_coverage_gap_ratio']==original['coverage_gap_ratio']
    save(target/'worst_trace_metric_overlay.json',{'source_sha256':sha(path),'source':path.relative_to(root).as_posix(),
        'interpretation':'Old simulator had no threat model. Empty threat overlay means undefined, NOT proof of no real threat.',
        'metrics':metric,'old_header_unchanged':True,'source_file_modified':False})
    fixed=read(root/'phase4a_outputs/study_v1/legacy/fixed_feasibility.json')
    travel=read(root/'phase4a_outputs/study_v1/legacy_paired_replay/travel_deadline_relationships.json')
    save(target/'prior_evidence.json',{'fixed':{k:v for k,v in fixed.items() if k!='details'},'travel':travel['summary'],
        'phase4a_paired_gap_cases':read(root/'phase4a_outputs/study_v1/legacy_paired_replay/paired_results.json')['scope']})


def audit(root,output):
    from .analysis import audit_all
    return audit_all(root,output)


def report(root,output):
    from .analysis import summarize_all
    return summarize_all(root,output)
