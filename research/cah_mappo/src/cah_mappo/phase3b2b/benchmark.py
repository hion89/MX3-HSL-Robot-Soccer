"""Non-final common benchmark after all model selections are frozen."""
from collections import defaultdict
import json
from pathlib import Path
import torch
from ..phase3b1.networks import ActorCritic
from ..phase3b2a_repair.slots import PERMUTATIONS
from .scenarios import SEEDS,benchmark_specs,FAMILIES
from .evaluation import evaluate_many
from .metrics import aggregate
from .io import save


def selected_network(run,stream):
    if not (run/'all_seeds_selection_freeze.json').is_file():raise RuntimeError('All seed selections must be frozen before evaluation')
    seed=SEEDS[stream];selection=json.loads((run/f'seed_{seed}/selection.json').read_text())
    path=run/f'seed_{seed}'/selection['selected_checkpoint']
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    torch.set_num_threads(1);network=ActorCritic(checkpoint['config']['hidden'])
    network.load_state_dict(checkpoint['network']);network.eval()
    return network,path


def summaries(group):
    return {'overall':group['summary'],**{family:aggregate([r for r in group['episodes'] if r['family']==family]) for family in FAMILIES}}


def seed_benchmark(run,stream):
    network,path=selected_network(run,stream);seed=SEEDS[stream];specs=benchmark_specs();result={}
    for mode in ('greedy','stochastic'):
        group=evaluate_many(specs,mode,network,repeats=3 if mode=='stochastic' else 1,log_dir=run/f'seed_{seed}/benchmark_raw'/mode)
        result[mode]={**group,'summaries':summaries(group)}
        print(json.dumps({'seed':seed,'benchmark_mode':mode,'summary':group['summary']}),flush=True)
    save(run/f'seed_{seed}/benchmark_results.json',{'checkpoint':str(path),'final_holdout_used':False,'modes':result})


def slot_benchmark(run,stream):
    network,_=selected_network(run,stream);seed=SEEDS[stream]
    specs=[s for s in benchmark_specs() if s['family']=='conditions'][:32]
    main=json.loads((run/f'seed_{seed}/benchmark_results.json').read_text())['modes']
    results={}
    for mode in ('greedy','stochastic'):
        alternate=evaluate_many(specs,mode,network,orders=PERMUTATIONS[1:],log_dir=run/f'seed_{seed}/slot_raw'/mode)
        ids={s['scenario_id'] for s in specs}
        identity=[r for r in main[mode]['episodes'] if r['scenario_id'] in ids and r['repeat']==0]
        all_rows=identity+alternate['episodes'];by_order={}
        baseline={r['scenario_id']:r for r in identity}
        for order in PERMUTATIONS:
            rows=[r for r in all_rows if tuple(r['slot_order'])==order]
            delta={key:sum(r[key]-baseline[r['scenario_id']][key] for r in rows)/len(rows)
                   for key in ('task_success_rate','deadline_miss_rate','coverage_gap_ratio')}
            choices=defaultdict(int);success=defaultdict(int)
            for r in rows:
                for task in r['detailed_tasks']:
                    choices[task['id']]+=task['selection_count'];success[task['id']]+=int(task['success'])
            by_order[''.join(map(str,order))]={'summary':aggregate(rows),'paired_mean_delta_from_identity':delta,
                                             'per_task_selection_counts':dict(choices),'per_task_completed':dict(success)}
        results[mode]={'physical_scenarios':32,'repeats_per_scenario_order':1,'by_order':by_order,'episodes':all_rows}
    save(run/f'seed_{seed}/slot_results.json',results)


def reference_benchmark(run,mode):
    if not (run/'all_seeds_selection_freeze.json').is_file():raise RuntimeError('Selection barrier required')
    if mode not in ('local_fixed','local_eta','central_fixed','central_greedy','central_exact','central_handover'):raise ValueError(mode)
    output=run/'references'/mode;output.mkdir(parents=True,exist_ok=False)
    result=evaluate_many(benchmark_specs(),mode,log_dir=output/'raw')
    save(output/'results.json',{**result,'summaries':summaries(result)})
    print(json.dumps({'reference':mode,'summary':result['summary']}),flush=True)
