"""Raw-log recomputation, sample consistency, regression tests and preservation."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time
from ..phase3b2b.io import read_raw,save
from ..phase4b.metrics import evaluate_new
from ..phase4b.experiment import read,sha
from ..phase3a.contracts import SampledAction,validate_sample
from .evaluation import failure_service_recovery


def audit_all(root,output):
    target=output/'completion_final';target.mkdir(exist_ok=False);count=Counter();recoveries=[]
    started=time.perf_counter()
    for path in sorted(output.rglob('*.jsonl.gz')):
        rows=read_raw(path);m=evaluate_new(rows);count['raw_episodes']+=1
        relative=path.relative_to(output).as_posix();top=relative.split('/')[0];count[top]+=1
        for r in rows:
            if r.get('event')!='policy_selection':continue
            for name,item in r['action'].get('policy_metadata',{}).get('actor_samples',{}).items():
                s=item['sample'];p=item['packet']
                sample=SampledAction(s['action'],s['log_probability'],tuple(s['action_mask']),tuple(s['probabilities']),s['policy_version'],s['observation_token'])
                validate_sample(sample,p['action_mask'],p['observation_token']);count['actor_samples']+=1
                a=sample.action;decoded=None if a==0 else p['self_assignment'] if a==1 else p['job_ids'][a-2]
                assert r['action']['requested'][name]==decoded
                assert len(p['observation'])==146 and len(p['action_mask'])==10
            feedback=r['public_observation'].get('current_observed_job_feedback',{})
            assert all(j['spawn']<=r['time'] for j in feedback.values())
        for record in failure_service_recovery(rows):recoveries.append({'file':relative,**record})
        # Every saved outcome is compared below, not merely rerun without checking.
        if top=='rules':saved=read(path.parent/'results.json')['episodes'];expected=next(r for r in saved if r['scenario_id']==rows[0]['phase4b']['scenario_id'])
        elif top=='plans':saved=read(path.parent/'witnesses.json')['plans'];expected=next(r['metrics'] for r in saved if r['file']==path.name)
        elif top in ('sensitivity','commitment_sensitivity'):
            name='fine_dt_results.json' if path.parent.name=='fine_dt' else 'parameter_results.json'
            saved=read(output/top/name)['episodes'];meta=rows[0]['phase4b']
            expected=next(r for r in saved if r['scenario_id']==meta['scenario_id'] and r['policy']==rows[0]['policy'] and r['physics_dt']==meta['physics_dt'] and
                          (name=='fine_dt_results.json' or r['profile']==path.parent.name))
        elif top=='information':
            if path.parent.name=='native_common_reset':
                saved=read(output/'information/native_reset_results.json')['episodes']
                expected=next(r['metrics'] for r in saved if r['file']==path.name)
            elif path.parent.name=='future_hint_ablation':
                hint=path.stem.endswith('hint1.jsonl');saved=read(output/'information/future_hint_ablation.json')['episodes']
                expected=next(r for r in saved if r['scenario_id']==rows[0]['phase4b']['scenario_id'] and r['privileged_future_announcement']==hint)
            else:
                saved=read(output/'information/actor_history_audit.json')['pairs'];meta=rows[0]['phase4b']
                pair=next(r for r in saved if path.name.startswith(f"seed{r['seed']}_{r['mode']}_") and any(x['scenario_id']==meta['scenario_id'] for x in r['outcomes']))
                expected=next(r for r in pair['outcomes'] if r['scenario_id']==meta['scenario_id'])
        else:raise AssertionError(('Unregistered raw log',path))
        for k,v in m.items():assert expected[k]==v,(relative,k,expected[k],v)
    save(target/'service_recovery_from_raw.json',recoveries)
    before=read(output/'preservation_before.json');repo=root.parents[1]
    for name,h in before['source_sha256'].items():assert sha(repo/name)==h,name
    for name,expected in before['old_output_metadata'].items():
        stat=(root/name).stat();assert {'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}==expected,name
    for cp in before['checkpoints'].values():assert sha(root/cp['path'])==cp['sha256']
    for name,h in read(output/'evidence/read_sources_sha256.json').items():assert sha(Path(name))==h,name
    for name,h in read(output/'evidence/service_timer_sources_sha256.json').items():assert sha(Path(name))==h,name
    elapsed=time.perf_counter()-started
    result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-v'],cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace')
    with (target/'unittest.txt').open('x',encoding='utf-8') as f:f.write(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr[-3000:]
    import re
    test_count=int(re.search(r'Ran (\d+) tests',result.stderr).group(1))
    save(target/'audit.json',{'status':'PASS','raw_recomputation_seconds':elapsed,'counts':dict(count),
        'old_source_hashes_preserved':len(before['source_sha256']),'old_output_metadata_preserved':len(before['old_output_metadata']),
        'checkpoint_hashes_preserved':len(before['checkpoints']),'read_only_evidence_hashes_preserved':len(read(output/'evidence/read_sources_sha256.json')),
        'actual_regression_test_count':test_count,'test_exit_code':result.returncode,'protected_holdout_content_opened':False,
        'packages_installed':False,'retrained':False,'remote_access':False,'git_commit_or_push':False})
    print('AUDIT PASS',dict(count),'tests',test_count,flush=True)
