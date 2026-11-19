"""Learning seeds are replication units; scenario pairing is kept separately."""
import json
import math
from statistics import mean,median,stdev
from .scenarios import SEEDS,FAMILIES
from .metrics import SCALARS
from .io import save


def seed_statistics(values):
    values=[v for v in values if v is not None];n=len(values)
    if not n:return {'n_training_seeds':0,'mean':None,'median':None,'sd':None,'t95':None}
    avg=mean(values);sd=stdev(values) if n>1 else None
    critical={2:12.706204736,3:4.30265273,4:3.182446305,5:2.776445105}
    interval=None if n<2 else [avg-critical[n]*sd/math.sqrt(n),avg+critical[n]*sd/math.sqrt(n)]
    return {'n_training_seeds':n,'values':values,'mean':avg,'median':median(values),'sd':sd,
            'min':min(values),'max':max(values),'t95':interval,'interval_is_descriptive_not_multiple_comparison_adjusted':True}


def make_report(run):
    trained={str(seed):json.loads((run/f'seed_{seed}/benchmark_results.json').read_text()) for seed in SEEDS}
    references={p.name:json.loads((p/'results.json').read_text()) for p in (run/'references').iterdir() if p.is_dir()}
    seed_stats={}
    for family in ('overall',*FAMILIES):
        seed_stats[family]={mode:{metric:seed_statistics([trained[str(seed)]['modes'][mode]['summaries'][family][metric] for seed in SEEDS])
                                  for metric in SCALARS} for mode in ('greedy','stochastic')}
    paired={}
    for name,reference in references.items():
        ref={r['scenario_id']:r for r in reference['episodes']};paired[name]={}
        for mode in ('greedy','stochastic'):
            data={};counts={}
            for metric in SCALARS:
                differences=[];pair_counts=[]
                for seed in SEEDS:
                    scene_values={}
                    for r in trained[str(seed)]['modes'][mode]['episodes']:
                        if r.get(metric) is not None:scene_values.setdefault(r['scenario_id'],[]).append(r[metric])
                    delta=[mean(v)-ref[sid][metric] for sid,v in scene_values.items() if ref[sid].get(metric) is not None]
                    differences.append(mean(delta) if delta else None);pair_counts.append(len(delta))
                data[metric]=seed_statistics(differences);counts[metric]=pair_counts
            paired[name][mode]={'delta_is_mappo_minus_reference':True,'per_seed_pair_counts':counts,'seed_level_differences':data,
                                'information_caveat':reference['episodes'][0]['information']}
    slot_results={str(seed):json.loads((run/f'seed_{seed}/slot_results.json').read_text()) for seed in SEEDS}
    slot_stats={}
    for mode in ('greedy','stochastic'):
        slot_stats[mode]={order:{key:seed_statistics([slot_results[str(seed)][mode]['by_order'][order]['summary'][key] for seed in SEEDS])
                                  for key in ('task_success_rate','deadline_miss_rate','coverage_gap_ratio')}
                          for order in ('234','243','324','342','423','432')}
    report={'replication_unit':'independently initialized/trained seed; n=5, conditional on fixed common benchmark scenarios',
            'stochastic_repeats':3,'common_physical_scenarios':400,'final_holdout_used':False,
            'uncertainty':'Student t interval on five seed aggregates; not an episode-level independent-learning CI; no superiority claim',
            'seed_statistics':seed_stats,'paired_comparisons':paired,'slot_statistics':slot_stats,
            'references':{k:v['summaries'] for k,v in references.items()}}
    save(run/'statistics.json',report)
    return report
