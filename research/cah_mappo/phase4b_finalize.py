"""Audit post-hoc diagnostics, then seal code/doc hashes without modifying prior files."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.phase3b2b.io import save,read_raw
from cah_mappo.phase4b.experiment import read,sha,grouped,ensure_frozen
from cah_mappo.phase4b.metrics import evaluate_new


def audit(output):
    frozen=ensure_frozen(ROOT,output);target=output/'completion';target.mkdir(exist_ok=False)
    primary=read(output/'audit/audit.json');feedback=read(output/'sensor_feedback_transfer/audit.json')
    assert primary['status']==feedback['status']=='PASS'
    supplemental=0;fault_summaries={}
    for folder in (output/'fault_contingency').iterdir():
        if not folder.is_dir():continue
        if folder.name=='offline_fault':
            success={}
            for case in folder.iterdir():
                if not case.is_dir():continue
                metrics=[evaluate_new(read_raw(p)) for p in case.glob('*.jsonl.gz')]
                supplemental+=len(metrics);success[case.name]=any(m['joint_task_and_defense_success'] for m in metrics)
            for case in read(folder/'summary.json')['cases']:assert success[case['scenario_id']]==case['has_joint_witness']
        else:
            data=read(folder/'results.json')
            for e in data['episodes']:
                m=evaluate_new(read_raw(folder/f"{e['scenario_id']}.jsonl.gz"))
                for k,v in m.items():assert v==e[k],(folder.name,k)
                supplemental+=1
            assert grouped(data['episodes'])==data['summary'];fault_summaries[folder.name]=data['summary']
    # Explicitly separate "old model had no threat variable" from a new known-quiet world.
    overlay=read(output/'legacy/worst_trace_metric_overlay.json');corrected=copy.deepcopy(overlay)
    corrected['old_threat_model_defined']=False
    corrected['interpretation']='Old trace cannot establish threat/defense outcomes. V1 empty-overlay vacuous joint success and zero counts must not be interpreted as defense evidence.'
    for k in ('threat_arrivals','threat_pass_count','threat_defended_count','post_task_threat_passes','joint_task_and_defense_success'):
        corrected['metrics'][k]=None
    save(target/'legacy_unmodeled_threat_interpretation_v2.json',corrected)
    # Unified final tables select the corrected sensor-feedback reference; v1 stays separately available.
    combined={}
    for p in (output/'comparison').iterdir():
        if p.is_dir() and not p.name.startswith('seed_'):combined[p.name]=read(p/'results.json')['summary']
    for p in (output/'sensor_feedback_transfer').iterdir():
        if p.is_dir():combined['sensor_'+p.name]=read(p/'results.json')['summary']
    combined.update({'fault_'+k:v for k,v in fault_summaries.items()})
    save(target/'final_condition_summaries.json',combined)
    flat=[]
    for label,data in combined.items():
        for condition,d in data['by_condition'].items():
            flat.append({'label':label,'condition':condition,**{k:v for k,v in d.items() if not isinstance(v,dict)}})
    keys=sorted({k for r in flat for k in r})
    with (target/'final_condition_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    offline=read(output/'offline/results.json');evidence=[]
    for case in offline['cases']:
        native={}
        for name in ('fixed','greedy','exact','legacy_handover','ready','adaptive','observed_reserve','standing_reserve','lookahead'):
            saved=read(output/'comparison'/name/'results.json')
            m=next(e for e in saved['episodes'] if e['scenario_id']==case['scenario_id'])
            native[name]=m['joint_task_and_defense_success']
        evidence.append({'scenario_id':case['scenario_id'],'family':case['family'],'has_oracle_joint_witness':case['has_joint_success'],
                         'native_joint_success':native,'any_native_success_hindsight':any(native.values())})
    known=[e for e in evidence if e['family'] in ('A_near','B_normal','D_visible') and e['has_oracle_joint_witness']]
    save(target/'feasible_case_diagnostics.json',{'scope':'Any-policy hindsight is NOT one deployable portfolio or a causal improvement claim',
        'known_visible_joint_feasible_cases':len(known),'cases_no_native_policy_solved':sum(not e['any_native_success_hindsight'] for e in known),
        'details':evidence})
    # Recompute canonical actor distributions; the critic and future fields are absent.
    import torch
    from cah_mappo.phase4a.evaluation import load_network
    torch.set_num_threads(1);forward=0;error_max=0.
    for seed,cp in frozen['checkpoints'].items():
        net=load_network(ROOT/cp['path'],cp['sha256'])
        for mode in ('greedy','stochastic'):
            folder=output/f'sensor_feedback_transfer/seed_{seed}_{mode}';entry=read(folder/'results.json')['episodes'][0]
            rows=read_raw(folder/'raw'/f"{entry['scenario_id']}_r0.jsonl.gz")
            for e in rows:
                if e.get('event')!='policy_selection':continue
                records=e['action']['policy_metadata']['actor_samples'];vectors=[];masks=[]
                for name in ('A','B','C','D'):
                    item=records.get(name);vectors.append(item['packet']['observation'] if item else [0.]*146)
                    masks.append(item['packet']['action_mask'] if item else [1]+[0]*9)
                with torch.no_grad():p=net.distribution(torch.tensor(vectors,dtype=torch.float32),torch.tensor(masks,dtype=torch.bool)).probs
                for i,name in enumerate(('A','B','C','D')):
                    if name in records:
                        err=max(abs(a-b) for a,b in zip(p[i].tolist(),records[name]['sample']['probabilities']))
                        error_max=max(error_max,err);assert err<1e-10
                forward+=1
    tests=(output/'regression_tests_complete.txt').read_text(encoding='utf-8')
    assert re.search(r'Ran 216 tests in [0-9.]+s\s+OK\s*$',tests)
    before=read(output/'preservation_before.json');repo=ROOT.parents[1]
    for name,h in before['old_source_files_sha256'].items():assert sha(repo/name)==h,name
    for name,expected in before['old_output_metadata_only'].items():
        stat=(ROOT/name).stat();assert {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}==expected,name
    for cp in before['checkpoints'].values():assert sha(ROOT/cp['path'])==cp['sha256']
    raw=sum(primary['counts'][k] for k in ('comparison_raw','validation_raw','resolution_raw','offline_raw'))+feedback['raw_episodes']+supplemental
    save(target/'audit.json',{'status':'PASS','all_raw_recomputed_total':raw,'supplementary_contingency_raw':supplemental,
        'canonical_actor_forward_batches':forward,'canonical_probability_error_max':error_max,
        'regression_tests':216,'old_source_hashes_preserved':len(before['old_source_files_sha256']),
        'old_output_metadata_preserved':len(before['old_output_metadata_only']),'checkpoint_hashes_preserved':5,
        'holdout_contents_opened':False,'retraining':False,'git_commit_or_push':False})
    print('COMPLETE AUDIT PASS',raw,'raw episodes;',forward,'canonical actor batches;',len(known),'known feasible cases')


def seal(output):
    ensure_frozen(ROOT,output);audit=read(output/'completion/audit.json');assert audit['status']=='PASS'
    files=[*sorted((ROOT/'src/cah_mappo/phase4b').glob('*.py')),*sorted(ROOT.glob('phase4b*.py')),
           *sorted(ROOT.glob('PHASE4B*.md')),*sorted((ROOT/'tests').glob('test_phase4b*.py'))]
    save(output/'completion/final_integrity.json',{'status':'PASS','audit_sha256':sha(output/'completion/audit.json'),
        'code_document_tests_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'protected_holdout_opened':False,'next_phase_started':False})
    print('SEALED',len(files),'new source/doc/test files')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('audit','seal'))
    parser.add_argument('--output',type=Path,default=ROOT/'phase4b_outputs/study_v1');a=parser.parse_args()
    {'audit':audit,'seal':seal}[a.command](a.output)
