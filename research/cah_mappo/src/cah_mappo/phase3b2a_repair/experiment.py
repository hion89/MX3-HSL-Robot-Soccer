"""Sequential diagnosis then a single-factor, equal-budget CPU intervention."""
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import torch

from ..core import write_jsonl
from ..phase2.model import config_from_dict
from ..phase2.engine import Environment
from ..phase2.policies import make_policy
from ..phase3b2a.scenarios import generate as old_generate, digest
from ..phase3b2a.experiment import load_net, save
from ..phase3b2a.evaluation import aggregate as training_aggregate
from .slots import PERMUTATIONS
from .scenarios import generate
from .learner import RepairLearner
from .evaluation import evaluate_many, evaluate_one, score, coverage_windows


def selection_key(validation):
    summaries=[validation[m]['summary'] for m in ('greedy','stochastic')]
    success=[s['task_success_rate'] for s in summaries]
    return (min(success),sum(success)/2,-max(s['coverage_gap_ratio'] for s in summaries),
            -sum(s['restricted_response_mean_s'] for s in summaries)/2)


def validate(specs, net):
    if any(s['split']!='validation' for s in specs):raise ValueError('Validation split only')
    return {mode:evaluate_many(specs,net,PERMUTATIONS,mode) for mode in ('greedy','stochastic')}


def train(output, specs, validation, parent, augment, steps=8192):
    output.mkdir(exist_ok=False);(output/'checkpoints').mkdir();(output/'trajectories').mkdir()
    learner=RepairLearner(specs,parent,augment)
    source_root=Path(__file__).resolve().parents[3]
    source=[*Path(__file__).parent.glob('*.py'),*Path(__file__).parents[1].joinpath('phase3b1').glob('*.py')]
    config={'train_config':asdict(learner.c),'steps':steps,'augment_episode_slots':augment,
            'parent_checkpoint':str(parent),'parent_sha256':hashlib.sha256(Path(parent).read_bytes()).hexdigest(),
            'load_parent_optimizers':True,'reward_weights':asdict(learner.env.weights),
            'train_ids':[s['scenario_id'] for s in specs],'validation_ids':[s['scenario_id'] for s in validation],
            'source_sha256':{str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
            'python':sys.version,'torch':torch.__version__,'numpy':importlib.metadata.version('numpy'),'device':'cpu',
            'selection_rule':'min success of greedy/stochastic, mean success, -worst gap, -mean restricted response'}
    save(output/'config.json',config)
    learner.checkpoint(output/'checkpoints/initial.pt')
    initial=validate(validation,learner.net);save(output/'validation_step0.json',initial)
    best=selection_key(initial);best_path=output/'checkpoints/initial.pt';curves=[]
    start=time.perf_counter()
    with (output/'train_episodes.jsonl').open('x',encoding='utf-8') as episodes:
        while learner.steps<steps:
            batch=learner.collect(min(256,steps-learner.steps))
            if not curves:
                with (output/'first_rollout.pt').open('xb') as f:torch.save(batch,f)
            report=learner.update(batch)
            report.update(update=learner.policy_updates,total_env_steps=learner.steps,
                          **{'train_'+k:v for k,v in training_aggregate(learner.completed).items()})
            for row in learner.completed:episodes.write(json.dumps({'total_env_steps':learner.steps,**row})+'\n')
            for episode,log in learner.last_episode_logs:
                if episode<1 or learner.steps==steps and episode==learner.episodes-1:
                    write_jsonl(log,output/'trajectories'/f'episode{episode}.jsonl')
            if learner.steps%2048==0 or learner.steps==steps:
                evaluation=validate(validation,learner.net)
                save(output/f'validation_step{learner.steps}.json',evaluation)
                path=output/f'checkpoints/step{learner.steps}.pt';learner.checkpoint(path)
                key=selection_key(evaluation)
                if key>best:best,best_path=key,path
                report['validation_key']=list(key)
                print(json.dumps({'augment':augment,'step':learner.steps,
                                  'validation_greedy_success':evaluation['greedy']['summary']['task_success_rate'],
                                  'validation_stochastic_success':evaluation['stochastic']['summary']['task_success_rate']}),flush=True)
            curves.append(report);save(output/f'update_{learner.steps:06d}.json',report)
    learner.checkpoint(output/'checkpoints/final.pt')
    save(output/'learning_curve.json',curves)
    import csv
    with (output/'learning_curve.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in curves for k in r}));writer.writeheader();writer.writerows(curves)
    summary={'augment':augment,'environment_steps':learner.steps,'wall_seconds_including_intermediate_validation':time.perf_counter()-start,
             'selected_checkpoint':str(best_path),'selected_validation_key':list(best),'episodes':learner.episodes}
    save(output/'selection.json',summary)
    return summary


def legacy_spec(spec,family):
    payload={k:v for k,v in spec.items() if k not in ('scenario_id','sha256')}
    payload.update(split='development',family=family,source='previous-generator-development-only')
    return {**payload,'scenario_id':f"repair-legacy-{family}-{spec['index']}-{digest(payload)[:10]}",'sha256':digest(payload)}


def run(output, root):
    output.mkdir(parents=True,exist_ok=False)
    parent=root/'phase3b2a_outputs/pilot_v1/C_weight2/checkpoints/final.pt'
    training=[generate('positions','train',i) for i in range(12)]
    validation=[generate('positions','validation',i) for i in range(2)]
    development={family:[generate(family,'development',i) for i in range(1 if family=='fixed' else 4)]
                 for family in ('fixed','positions','priority_deadline')}
    holdout=[generate(family,'final_holdout',i) for family in ('positions','priority_deadline') for i in range(200)]
    save(output/'sealed_final_holdout_manifest.json',{'status':'NOT_EVALUATED','scenario_count':len(holdout),
                                                    'scenarios':holdout,'release':'Separate future approval only'})
    manifest={'train':training,'validation':validation,'development':development,
              'D_development':[legacy_spec(old_generate('D','train',100+i),'D') for i in range(6)],
              'B3_development':[legacy_spec(old_generate('B3','test',i),'B3') for i in range(12)]}
    save(output/'scenario_manifest.json',manifest)
    save(output/'predeclared_plan.json',{'major_intervention':'episode-stable task-slot permutations only',
        'train_steps_per_arm':8192,'max_total_training_steps':16384,'training_seed':7,'evaluation_seconds':32,
        'all_orders':[list(p) for p in PERMUTATIONS],'PPO_and_network_and_reward':'frozen Phase 3B-1 implementations',
        'initialization':'same Phase 3B-2A C final network and optimizer states',
        'selection':'both greedy and stochastic validation; no holdout evaluation',
        'diagnostic_order':['fixed_feasibility','fixed_slot_only','limited_positions','priority_deadline','information_control'],
        'holdout_status':'prepared but rejected by environment/training interfaces'})
    net=load_net(parent)
    exact=[]
    for family,specs in development.items():
        for spec in specs:
            c=config_from_dict(spec['config']);engine=Environment(c,make_policy('exact',c),'central_exact')
            rows=engine.run();measured=score(rows)
            exact.append({'scenario_id':spec['scenario_id'],'family':family,**measured})
            folder=output/'exact_logs';folder.mkdir(exist_ok=True)
            write_jsonl(rows,folder/f"{spec['scenario_id']}.jsonl")
    save(output/'exact_feasibility.json',exact)
    if not all(r['task_success_rate']==1 for r in exact):raise RuntimeError('Exact did not finish all tasks; diagnose feasibility before training')
    ladder={}
    for family,specs in development.items():
        ladder[family]={mode:evaluate_many(specs,net,PERMUTATIONS,mode,
                            log_dir=output/'baseline_logs'/family/mode) for mode in ('greedy','stochastic')}
    save(output/'baseline_ladder.json',ladder)
    b3={}
    for label,checkpoint in [('early',root/'phase3b2a_outputs/pilot_v1/B3_weight2/checkpoints/step1024.pt'),
                             ('final',root/'phase3b2a_outputs/pilot_v1/B3_weight2/checkpoints/final.pt')]:
        network=load_net(checkpoint)
        b3[label]={mode:evaluate_many(manifest['B3_development'],network,mode=mode,
                                      log_dir=output/'b3_logs'/label/mode) for mode in ('greedy','stochastic')}
    save(output/'B3_deterministic_stochastic.json',b3)
    arms={}
    for label,augment in [('identity',False),('augmented',True)]:
        arms[label]=train(output/label,training,validation,parent,augment)
    save(output/'selection_freeze.json',{'arms':arms,'manifest_sha256':digest(manifest),
         'selected_checkpoint_hashes':{label:hashlib.sha256(Path(arm['selected_checkpoint']).read_bytes()).hexdigest() for label,arm in arms.items()},
         'final_holdout_evaluated':False})
    comparisons={'old':ladder};information={};windows={}
    for label,checkpoint in [('old',parent),*[(name,Path(arm['selected_checkpoint'])) for name,arm in arms.items()]]:
        network=load_net(checkpoint)
        if label!='old':
            comparisons[label]={family:{mode:evaluate_many(specs,network,PERMUTATIONS,mode,
                                         log_dir=output/'comparison_logs'/label/family/mode) for mode in ('greedy','stochastic')}
                                for family,specs in development.items()}
        information[label]={condition:{mode:evaluate_many(manifest['D_development'],network,mode=mode,condition=condition,
                                          log_dir=output/'information_logs'/label/condition/mode) for mode in ('greedy','stochastic')}
                            for condition in ('controlled','zero_delay','realistic')}
        _,rows=evaluate_one(development['fixed'][0],network)
        windows[label]=coverage_windows(rows)
    save(output/'same_condition_comparisons.json',comparisons)
    save(output/'information_comparisons.json',information)
    save(output/'coverage_window_diagnostics.json',windows)
    # Last checkpoints are retained and validated; they are not silently substituted for selected ones.
    final_results={label:{mode:evaluate_many(development['positions'],load_net(output/label/'checkpoints/final.pt'),
                                           PERMUTATIONS,mode,log_dir=output/'final_logs'/label/mode)
                          for mode in ('greedy','stochastic')} for label in arms}
    save(output/'final_checkpoint_comparisons.json',final_results)
    print(json.dumps({'finished':True,'total_training_steps':16384,'holdout_evaluated':False}),flush=True)
