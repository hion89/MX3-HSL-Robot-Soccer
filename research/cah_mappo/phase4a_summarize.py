"""Supplementary validation trace replay and transparent, condition-separated summaries."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean,stdev
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from cah_mappo.phase3b2b.io import read_raw,write_raw,save
from cah_mappo.phase4a.evaluation import run
from cah_mappo.phase4a.metrics import score,aggregate,distribution

BASE=ROOT/'phase4a_outputs/study_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def validation():
    torch.set_num_threads(1);target=BASE/'validation_trace_replay';target.mkdir(exist_ok=False)
    specs=read(BASE/'design/validation_manifest.json');selection=read(BASE/'design/validation_selection.json');count=0
    for i,candidate in enumerate(selection['candidates']):
        folder=target/f'candidate_{i}';folder.mkdir();results=[]
        for spec in specs:
            m,rows=run(spec,'witness_immediate',gate_mode='adaptive',parameters=candidate['parameters'])
            results.append(m);write_raw(folder/f"{spec['scenario_id']}.jsonl.gz",rows);count+=1
        assert aggregate(results)==candidate['summary']
    save(target/'audit.json',{'status':'PASS','deterministic_replay_of_validation_not_original_trace':True,
                             'raw_episodes':count,'all_candidate_summaries_exact_match':True,'selected_thresholds_unchanged':True})
    print('validation raw replay verified',count)


def extra(rows):
    tasks={jid:[j for r in rows for j in r['task_details'] if j['id']==jid] for jid in ('task0','task1','task2')}
    task0=tasks['task0']
    return {'episodes':len(rows),'task_success_rate':mean(r['task_success_rate'] for r in rows),
        'task0_success_rate':mean(t['success'] for t in task0),
        'all_tasks_episode_success_rate':mean(r['task_success_rate']==1 for r in rows),
        'per_task_success':{k:mean(t['success'] for t in ts) if ts else None for k,ts in tasks.items()},
        'task0_response_mean_s':mean(t['response_s'] for t in task0 if t['response_s'] is not None) if any(t['response_s'] is not None for t in task0) else None,
        'task0_completion_mean_s':mean(t['completed_at']-t['spawn'] for t in task0 if t['success']) if any(t['success'] for t in task0) else None,
        'gap_region_seconds':mean(r['actual_gap_region_seconds'] for r in rows),
        'coverage_gap_ratio':mean(r['coverage_gap_ratio'] for r in rows),
        'region_seconds':mean(r['responsibility_region_seconds'] for r in rows),
        'risk_weighted_region_seconds':mean(r['risk_weighted_region_seconds'] for r in rows),
        'gap_episode_rate':mean(r['episode_has_gap'] for r in rows),
        'handover_gap_distribution':distribution([h['initial_gap_s'] for r in rows for h in r['handovers']]),
        'continuous_gap_distribution':distribution([g['duration'] for r in rows for g in r['gap_segments']]),
        'episode_gap_distribution':distribution([r['actual_gap_region_seconds'] for r in rows]),
        'episode_gap_conditional_distribution':distribution([r['actual_gap_region_seconds'] for r in rows if r['episode_has_gap']]),
        'controller_cancellations':sum(r['controller_cancel_count'] for r in rows),
        'controller_failures':sum(r['controller_failure_count'] for r in rows)}


def summary(output_name='analysis_extended'):
    target=BASE/output_name
    if target.parent.resolve()!=BASE.resolve():raise ValueError('Use a new direct child output directory')
    target.mkdir(exist_ok=False)
    data={p.name:read(p/'results.json') for p in (BASE/'comparison').iterdir() if p.is_dir()}
    out={}
    for label,d in data.items():
        out[label]={}
        for family in ('A','B','C','C_risk','R_risk','F_incumbent','F_successor'):
            for team in (2,4):
                rows=[r for r in d['episodes'] if r['family']==family and r['team']==team]
                out[label][f'{family}_team{team}']=extra(rows)
    save(target/'condition_task_metrics.json',out)
    seeds={}
    for mode in ('greedy','stochastic'):
        for gate in ('original','ready','adaptive'):
            for condition in next(iter(out.values())):
                rows=[out[f'seed_{s}_{mode}_{gate}'][condition] for s in (101,211,307,401,503)]
                values={}
                for key in ('task_success_rate','task0_success_rate','gap_region_seconds','coverage_gap_ratio','gap_episode_rate'):
                    v=[r[key] for r in rows];avg=mean(v);sd=stdev(v);half=2.7764451051977987*sd/math.sqrt(5)
                    values[key]={'mean':avg,'sd':sd,'seed_values':v,'t95':[avg-half,avg+half]}
                seeds[f'{mode}_{gate}_{condition}']=values
    save(target/'five_seed_statistics.json',{'unit':'training seed n=5, conditional on six diagnostic worlds per condition',
                                           'multiple_comparison_adjustment':False,'conditions':seeds})
    failures=[]
    for label,d in data.items():
        if not label.startswith('seed_') or not label.endswith('_greedy_original'):continue
        for e in d['episodes']:
            task=next(t for t in e['task_details'] if t['id']=='task0')
            if task['success']:continue
            rows=read_raw(BASE/'comparison'/label/'raw'/f"{e['scenario_id']}_r0.jsonl.gz")
            known=False;selected=False;guard=False;choices=[]
            for row in rows:
                if row.get('event')!='policy_selection':continue
                known|=any('task0' in p['job_ids'] for p in row.get('actor_observations',{}).values())
                for name,jid in row['action']['requested'].items():
                    if jid=='task0':
                        selected=True;choices.append([row['time'],name]);guard|=name=='A'
            failures.append({'label':label,'scenario_id':e['scenario_id'],'family':e['family'],'team':e['team'],
                'known':known,'selected':selected,'guard_A_requested':guard,'choices':choices,
                'category':'unobserved' if not known else 'known_not_selected' if not selected else 'selected_no_response' if task['response_at'] is None else 'responded_incomplete'})
    counts={}
    for row in failures:counts[row['category']]=counts.get(row['category'],0)+1
    save(target/'frozen_actor_task0_failures.json',{'scope':'deterministic original, five fixed checkpoints; failures conditional on new cases',
        'failure_count':len(failures),'category_counts':counts,'cases':failures})
    old=read(BASE/'legacy/mappo_gaps.json');all_stoch=[]
    for key,d in old.items():
        if key.endswith('stochastic'):all_stoch.extend(d['gap_cases'])
    total=6000;gaps=[r['actual_gap_region_seconds'] for r in all_stoch]
    save(target/'legacy_pooled_gaps.json',{'unit':'descriptive episodes; NOT 6000 independent training seeds',
        'episodes':total,'episodes_with_gap':len(gaps),'gap_episode_rate':len(gaps)/total,
        'all_episode_distribution':distribution([0]*(total-len(gaps))+gaps),
        'positive_episode_distribution':distribution(gaps),
        'continuous_gap_distribution':distribution([g['duration'] for r in all_stoch for g in r['gap_segments']])})
    # Data-derived SVGs: no new plotting dependencies or raster/image generation.
    figures=target/'figures';figures.mkdir()
    labels=[('central_fixed','Fixed'),('central_greedy','ETA'),('central_exact','Exact'),
            ('central_exact_ready','Exact+wait'),('central_exact_adaptive','Exact+adaptive')]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="510" viewBox="0 0 1100 510">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial,sans-serif" font-size="13" fill="#25313b">',
           '<text x="25" y="28" font-size="19">Phase 4A: controlled trade-offs, 4 robots / 3 tasks</text>',
           '<text x="25" y="50">Six shared diagnostic worlds per condition. Points are not an overall ranking.</text>']
    colors=['#333333','#4b77be','#26854b','#ac4a88','#e08020']
    for panel,family in enumerate(('A','B','C')):
        x0=65+350*panel;y0=350
        parts += [f'<text x="{x0}" y="88" font-size="16">Condition {family}</text>',
                  f'<path d="M{x0},{y0-220} V{y0} H{x0+270}" fill="none" stroke="#777"/>']
        for p in (0,50,100):parts.append(f'<text x="{x0-8}" y="{y0-2*p+4}" text-anchor="end">{p}%</text>')
        for i,(label,name) in enumerate(labels):
            v=out[label][family+'_team4'];x=x0+v['gap_region_seconds']*25;y=y0-v['task0_success_rate']*200
            parts.append(f'<circle cx="{x}" cy="{y}" r="{3+i}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
            parts.append(f'<text x="{x+8}" y="{y+10+i*14}" fill="{colors[i]}">{name}</text>')
        parts.append(f'<text x="{x0}" y="400">x: actual gap (region-seconds)</text>')
        parts.append(f'<text x="{x0}" y="420">y: urgent task0 success</text>')
    parts+=['<text x="25" y="470">All worlds: 32 seconds, 2 bases, 64 region-seconds. Central policies have privileged team inputs.</text>','</g></svg>']
    (figures/'controlled_tradeoffs.svg').write_text('\n'.join(parts),encoding='utf-8')
    print('extended summaries saved',len(out),'groups; original task0 failures',counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('validation','summary'))
    parser.add_argument('--output-name',default='analysis_extended')
    args=parser.parse_args()
    summary(args.output_name) if args.command=='summary' else validation()
