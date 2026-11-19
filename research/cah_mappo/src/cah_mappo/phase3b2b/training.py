"""Fixed-budget independent training with validation-only checkpoint selection."""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import torch
from ..phase3b1.learner import TrainConfig
from .scenarios import SEEDS,STEPS,INTERVAL,validation_specs
from .learner import BaselineLearner
from .evaluation import evaluate_many
from .metrics import aggregate
from .io import save,write_raw


def selection_key(reports):
    results=[reports[mode]['summary'] for mode in ('greedy','stochastic')]
    rates=[r['task_success_rate'] for r in results]
    return (min(rates),sum(rates)/2,-max(r['coverage_gap_ratio'] for r in results),
            -sum(r['restricted_response_mean_s'] for r in results)/2)


def validate(network,specs):
    if any(s['split']!='validation' for s in specs):raise ValueError('Validation inputs only')
    return {mode:evaluate_many(specs,mode,network) for mode in ('greedy','stochastic')}


def train_worker(run,stream,steps=STEPS):
    run=Path(run);output=run/f'seed_{SEEDS[stream]}'
    output.mkdir(exist_ok=False);(output/'checkpoints').mkdir();(output/'trajectories').mkdir()
    plan=json.loads((run/'frozen_plan.json').read_text())
    if steps!=plan['steps_per_seed']:raise ValueError('Training budget differs from frozen plan')
    learner=BaselineLearner(stream);specs=validation_specs()
    save(output/'config.json',{'stream':stream,'seed':SEEDS[stream],'train_config':asdict(learner.c),
         'steps':steps,'initialization':'independent from scratch; no pretrained weights or optimizers',
         'frozen_plan_sha256':hashlib.sha256((run/'frozen_plan.json').read_bytes()).hexdigest(),
         'source_manifest_sha256':hashlib.sha256((run/'source_manifest.json').read_bytes()).hexdigest(),
         'checkpoint_protocol':'episode-boundary-v1','slot_augmentation':False})
    learner.save_exact(output/'checkpoints/initial.pt')
    initial=validate(learner.net,specs);save(output/'validation_step000000.json',initial)
    best=selection_key(initial);best_path='checkpoints/initial.pt';curves=[]
    start=time.perf_counter();collect_seconds=update_seconds=validation_seconds=0.
    with gzip.open(output/'training_decisions.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as decisions,(output/'episodes.jsonl').open('x',encoding='utf-8') as episodes:
        while learner.steps<steps:
            t=time.perf_counter();batch=learner.collect(min(256,steps-learner.steps));collect_seconds+=time.perf_counter()-t
            if not curves:
                with (output/'first_rollout.pt').open('xb') as f:torch.save(batch,f)
            t=time.perf_counter();report=learner.update(batch);update_seconds+=time.perf_counter()-t
            for row in learner.completed:episodes.write(json.dumps({'training_step':learner.steps,**row},allow_nan=False)+'\n')
            for offset,row in enumerate(batch,learner.steps-len(batch)):
                trace={k:row[k] for k in ('scenario_id','episode','time','policy_update','job_ids','assignments','actions','old_logp','mask','probabilities','executed_actions','arbitration_changed')}
                trace['step']=offset;decisions.write(json.dumps(trace,separators=(',',':'),allow_nan=False)+'\n')
            for episode,rows in learner.last_episode_logs:
                if episode==0 or learner.steps==steps and episode==learner.episodes-1:
                    write_raw(output/'trajectories'/f'episode{episode}.jsonl.gz',rows)
            training_summary=aggregate(learner.completed)
            report.update(total_env_steps=learner.steps,update=learner.policy_updates,
                          **{'train_'+k:v for k,v in training_summary.items() if not isinstance(v,dict)})
            if learner.steps%INTERVAL==0 or learner.steps==steps:
                path=f'checkpoints/step{learner.steps:06d}.pt';learner.save_exact(output/path)
                t=time.perf_counter();evaluation=validate(learner.net,specs);validation_seconds+=time.perf_counter()-t
                save(output/f'validation_step{learner.steps:06d}.json',evaluation)
                key=selection_key(evaluation)
                if key>best:best,best_path=key,path
                report['validation_key']=list(key)
                for mode in ('greedy','stochastic'):
                    for metric in ('task_success_rate','coverage_gap_ratio','restricted_response_mean_s'):
                        report[f'validation_{mode}_{metric}']=evaluation[mode]['summary'][metric]
                progress={'seed':SEEDS[stream],'step':learner.steps,'elapsed_seconds':time.perf_counter()-start,
                          'greedy_success':evaluation['greedy']['summary']['task_success_rate'],
                          'stochastic_success':evaluation['stochastic']['summary']['task_success_rate']}
                save(output/f'progress_{learner.steps:06d}.json',progress)
                print(json.dumps(progress),flush=True)
            curves.append(report);save(output/f'update_{learner.steps:06d}.json',report)
    save(output/'learning_curve.json',curves)
    import csv
    with (output/'learning_curve.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in curves for k in r}));writer.writeheader();writer.writerows(curves)
    result={'seed':SEEDS[stream],'stream':stream,'steps':learner.steps,'episodes':learner.episodes,
            'wall_seconds_including_intermediate_validation':time.perf_counter()-start,
            'collect_seconds':collect_seconds,'optimizer_seconds':update_seconds,'validation_seconds':validation_seconds,
            'selected_checkpoint':best_path,'selected_validation_key':list(best),'last_checkpoint':path,
            'source_manifest_sha256':hashlib.sha256((run/'source_manifest.json').read_bytes()).hexdigest()}
    save(output/'selection.json',result)
    print(json.dumps({'training_complete':True,**result}),flush=True)
