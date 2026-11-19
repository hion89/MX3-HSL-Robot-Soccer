"""Verify saved training/evaluation only; never open existing sealed holdout files."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def verify(run,output):
    import torch
    from cah_mappo.phase3b1.networks import ActorCritic
    from cah_mappo.phase3b2a.evaluation import actor_inputs
    from cah_mappo.phase3b2a.scenarios import digest
    from cah_mappo.phase3b2b.scenarios import SEEDS,STEPS,scenario
    from cah_mappo.phase3b2b.learner import BaselineLearner
    from cah_mappo.phase3b2b.training import selection_key
    from cah_mappo.phase3b2b.metrics import score,aggregate,SCALARS
    from cah_mappo.phase3b2b.io import read_raw,save
    from cah_mappo.phase3b2b.evaluation import evaluate_one
    torch.set_num_threads(1);output.mkdir(parents=True,exist_ok=False)
    snapshot=json.loads((ROOT/'phase3b2b_outputs/preservation_before.json').read_text())
    for name,expected in snapshot['files'].items():
        assert hashlib.sha256((ROOT.parents[1]/name).read_bytes()).hexdigest()==expected,name
    for name,metadata in snapshot['sealed_metadata_only'].items():
        stat=(ROOT.parents[1]/name).stat()
        assert stat.st_size==metadata['bytes'] and stat.st_mtime_ns==metadata['mtime_ns']
    source=json.loads((run/'source_manifest.json').read_text())
    for name,expected in source.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    pre=json.loads((run/'pretraining_freeze.json').read_text())
    assert hashlib.sha256((run/'frozen_plan.json').read_bytes()).hexdigest()==pre['plan_sha256']
    freeze=json.loads((run/'all_seeds_selection_freeze.json').read_text())
    assert (run/'benchmark_manifest.json').stat().st_mtime_ns >= (run/'all_seeds_selection_freeze.json').stat().st_mtime_ns
    groups=[];networks={};initial_states=[];training_reports={};trained={};slot={}
    trace_decisions=0
    for stream,seed in enumerate(SEEDS):
        folder=run/f'seed_{seed}';selection=json.loads((folder/'selection.json').read_text())
        assert selection['steps']==STEPS and selection['episodes']==STEPS//32
        assert folder.joinpath('selection.json').stat().st_mtime_ns<=(run/'all_seeds_selection_freeze.json').stat().st_mtime_ns
        best=None;path=None
        for file in sorted(folder.glob('validation_step*.json')):
            step=int(file.stem.removeprefix('validation_step'));data=json.loads(file.read_text())
            key=selection_key(data)
            if best is None or key>best:
                best=key;path='checkpoints/initial.pt' if step==0 else f'checkpoints/step{step:06d}.pt'
        assert path==selection['selected_checkpoint'] and list(best)==selection['selected_validation_key']
        selected=folder/path
        assert hashlib.sha256(selected.read_bytes()).hexdigest()==freeze['seeds'][str(seed)]['checkpoint_sha256']
        restored=BaselineLearner.restore_exact(selected);networks[seed]=restored.net
        initial=torch.load(folder/'checkpoints/initial.pt',weights_only=True,map_location='cpu')
        assert initial['steps']==0 and not initial['actor_optimizer']['state'] and not initial['critic_optimizer']['state']
        initial_states.append(initial['network'])
        checker=BaselineLearner(stream);batch=torch.load(folder/'first_rollout.pt',weights_only=True,map_location='cpu')
        error=checker.ratio_before_update(checker.tensors(batch));assert error<1e-6
        updates=[json.loads(p.read_text()) for p in sorted(folder.glob('update_*.json'))]
        assert len(updates)==STEPS//256
        assert max(r['actor_clipped_gradient_norm'] for r in updates)<=.50001
        assert max(r['critic_clipped_gradient_norm'] for r in updates)<=.50001
        count=0
        with gzip.open(folder/'training_decisions.jsonl.gz','rt',encoding='utf-8') as f:
            for count,line in enumerate(f,1):
                row=json.loads(line);step=count-1
                assert row['step']==step and row['episode']==step//32 and row['time']==step%32
                assert row['policy_update']==step//256
                for i,action in enumerate(row['actions']):
                    mask=row['mask'][i];probs=row['probabilities'][i]
                    assert mask[0]==1 and mask[action] and abs(sum(probs)-1)<1e-7
                    assert all(p==0 for p,m in zip(probs,mask) if not m)
                    assert abs(math.log(probs[action])-row['old_logp'][i])<1e-6
                    for jid in row['job_ids'][i]:
                        if jid and jid.startswith('task'):assert row['time']>=2+2*int(jid[4:])
        assert count==STEPS;trace_decisions+=count*4
        training_reports[str(seed)]={'steps':STEPS,'trace_frames':count,'first_rollout_logp_error':error,'selected':path}
        main=json.loads((folder/'benchmark_results.json').read_text());trained[str(seed)]=main
        for mode,group in main['modes'].items():
            assert len(group['episodes'])==(400 if mode=='greedy' else 1200)
            assert group['summary']==aggregate(group['episodes'])
            for e in group['episodes']:groups.append((folder/'benchmark_raw'/mode,e,seed))
        slots=json.loads((folder/'slot_results.json').read_text());slot[str(seed)]=slots
        for mode,group in slots.items():
            for e in group['episodes']:
                if tuple(e['slot_order'])!=(2,3,4):groups.append((folder/'slot_raw'/mode,e,seed))
    for i,a in enumerate(initial_states):
        for b in initial_states[i+1:]:assert any(not torch.equal(value,b[key]) for key,value in a.items())
    references={}
    for folder in sorted((run/'references').iterdir()):
        if not folder.is_dir():continue
        group=json.loads((folder/'results.json').read_text());references[folder.name]=group
        assert len(group['episodes'])==400 and group['summary']==aggregate(group['episodes'])
        for e in group['episodes']:groups.append((folder/'raw',e,None))
    paired={};raw_count=0;inference_checks=0;max_return_error=0.
    for folder,episode,seed in groups:
        suffix=''.join(map(str,episode['slot_order']))
        path=folder/f"{episode['scenario_id']}_p{suffix}_r{episode['repeat']}.jsonl.gz"
        assert path.stat().st_mtime_ns>=(run/'all_seeds_selection_freeze.json').stat().st_mtime_ns
        rows=read_raw(path);metrics=score(rows)
        # Python 3.12 sum(float) may round differently from the stepwise += accumulator.
        return_error=abs(metrics['episode_return']-episode['episode_return'])
        max_return_error=max(max_return_error,return_error)
        assert return_error<=1e-12,str(path)
        assert {k:v for k,v in metrics.items() if k!='episode_return'}=={key:episode[key] for key in metrics if key!='episode_return'},str(path)
        assert metrics['evaluation_seconds']==32 and metrics['responsibility_region_seconds']==64 and metrics['risk_weighted_region_seconds']==64
        assert metrics['task_count']==3 and metrics['unobserved_task_rate']==0
        plan_hash=paired.setdefault(episode['scenario_id'],digest(rows[0]['exogenous_plan']))
        assert plan_hash==episode['exogenous_sha256']==digest(rows[0]['exogenous_plan'])
        if seed is not None:
            transition=next(r for r in rows if r['type']=='transition' and r['start']==6)
            packets=transition['slot_adapter']['actor_before'] if 'slot_adapter' in transition else next(r['actor_observations'] for r in rows if r.get('event')=='policy_selection' and r['time']==6)
            vectors,masks=actor_inputs(packets,restored.env.settings)
            with torch.no_grad():prob=networks[seed].distribution(vectors,masks).probs
            recorded=transition['evaluation_model_probabilities']
            for i,name in enumerate(('A','B','C','D')):
                assert torch.allclose(prob[i],torch.tensor(recorded[name],dtype=torch.float64),atol=1e-10,rtol=0)
            inference_checks+=1
        raw_count+=1
        if raw_count%500==0:print(json.dumps({'verified_raw_logs':raw_count}),flush=True)
    stats=json.loads((run/'statistics.json').read_text())
    for mode in ('greedy','stochastic'):
        for metric in SCALARS:
            values=[trained[str(s)]['modes'][mode]['summary'][metric] for s in SEEDS]
            values=[v for v in values if v is not None];saved=stats['seed_statistics']['overall'][mode][metric]
            assert saved['n_training_seeds']==len(values)
            if values:
                assert abs(saved['mean']-statistics.mean(values))<1e-12
                if len(values)==5:
                    half=2.776445105*statistics.stdev(values)/math.sqrt(5)
                    assert abs(saved['t95'][0]-(statistics.mean(values)-half))<1e-12
    benchmark=json.loads((run/'benchmark_manifest.json').read_text());sample=benchmark[:4]
    replay_count=0
    for seed in SEEDS:
        saved={e['scenario_id']:e for e in trained[str(seed)]['modes']['stochastic']['episodes'] if e['repeat']==0}
        for spec in sample:
            result,_=evaluate_one(spec,'stochastic',networks[seed])
            assert result==saved[spec['scenario_id']];replay_count+=1
    report={'status':'PASS','preserved_files_sha256':len(snapshot['files']),'sealed_files_metadata_only':len(snapshot['sealed_metadata_only']),
            'holdout_content_opened':False,'holdout_evaluations':0,'independent_training_seeds':5,
            'training_agent_decisions_verified':trace_decisions,'all_raw_logs_recomputed':raw_count,'paired_physical_scenarios':len(paired),
            'actor_only_inference_reproductions':inference_checks,'stochastic_checkpoint_replay_episodes':replay_count,
            'episode_return_recalculation_max_roundoff':max_return_error,'episode_return_absolute_tolerance':1e-12,
            'other_metrics_compare_exactly':True,'seed_level_ci_verified':True,'training':training_reports}
    save(output/'audit.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.run.resolve(),a.output.resolve())
