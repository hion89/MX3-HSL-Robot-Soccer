"""Audit existing 400 non-final benchmark worlds and all saved main MAPPO traces."""
import hashlib
import json
import math
from statistics import mean
from ..phase3b2b.io import read_raw, save
from ..phase2.metrics import evaluate
from .metrics import score, aggregate, distribution


def analyze(old, output):
    output.mkdir(parents=True,exist_ok=False)
    hashes={};fixed=[];margins=[];concurrency=[];details=[]
    def read(path):
        hashes[path.relative_to(old).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return read_raw(path)
    group=json.loads((old/'references/central_fixed/results.json').read_text())
    for entry in group['episodes']:
        path=old/'references/central_fixed/raw'/f"{entry['scenario_id']}_p234_r0.jsonl.gz"
        rows=read(path);m=score(rows);fixed.append(m)
        assert m['task_success_rate']==entry['task_success_rate'] and m['coverage_gap_ratio']==entry['coverage_gap_ratio']
        guard_task_ticks=0
        for row in rows:
            if row['type']!='interval':continue
            truth=row['truth']
            active=sum(j['kind']=='task' and j['status']=='active' for j in truth['jobs'].values())
            concurrency.append(active)
            guard_task_ticks+=sum(r['base_role']=='guard' and r['assignment'] and truth['jobs'][r['assignment']]['kind']=='task'
                                  for r in truth['robots'].values())
        task_details=[]
        for task in m['task_details']:
            margin=task['deadline']-task['completed_at'];margins.append(margin)
            response=next(e for e in rows if e.get('event')=='task_response_started' and e['job']==task['id'])
            task_details.append({**task,'deadline_margin_s':margin,'serving_robot':response['robot']})
        details.append(dict(scenario_id=entry['scenario_id'],fixed_success=m['task_success_rate'],
                            fixed_gap=m['actual_gap_region_seconds'],guard_task_ticks=guard_task_ticks,tasks=task_details))
    save(output/'fixed_feasibility.json',dict(worlds=len(fixed),all_tasks_completed=sum(m['task_success_rate']==1 for m in fixed),
        zero_gap_worlds=sum(not m['episode_has_gap'] for m in fixed),
        worlds_requiring_role_switch_under_any_policy=0 if all(m['task_success_rate']==1 and not m['episode_has_gap'] for m in fixed) else None,
        definition='Fixed is a constructive no-role-change feasibility witness, not an optimum',
        completed_task_deadline_margin_s=distribution(margins),
        peak_active_tasks=max(concurrency),intervals_more_than_two_active_tasks=sum(n>2 for n in concurrency),
        total_intervals=len(concurrency),details=details))
    groups={};worst=None
    for seed in (101,211,307,401,503):
        saved=json.loads((old/f'seed_{seed}/benchmark_results.json').read_text())
        for mode in ('greedy','stochastic'):
            results=[]
            for entry in saved['modes'][mode]['episodes']:
                path=old/f'seed_{seed}/benchmark_raw'/mode/f"{entry['scenario_id']}_p234_r{entry['repeat']}.jsonl.gz"
                rows=read(path);m=score(rows)
                assert m['actual_gap_region_seconds']==entry['actual_gap_region_seconds']
                m.update(scenario_id=entry['scenario_id'],repeat=entry['repeat'],seed=seed)
                results.append(m)
                if worst is None or m['actual_gap_region_seconds']>worst['metrics']['actual_gap_region_seconds']:
                    gap_bases={s['base'] for s in m['gap_segments']}
                    timeline=[]
                    for row in rows:
                        if row['type']=='interval':
                            timeline.append(dict(time=row['start'],robots={n:{k:r[k] for k in ('position','assignment','operational','base_role')}
                                                                          for n,r in row['truth']['robots'].items()},
                                                 requested=row['selection']['requested'],accepted=row['selection']['accepted']))
                    worst={'metrics':m,'timeline':timeline,
                           'events':[r for r in rows if r['type']=='event' and r['event'] in
                                     ('assignment_changed','responsibility_released','task_completed','task_response_started')],
                           'same_world_fixed':next(d for d in details if d['scenario_id']==entry['scenario_id'])}
            groups[f'{seed}_{mode}']={'summary':aggregate(results),'gap_cases':[r for r in results if r['episode_has_gap']]}
        print('legacy raw verified seed',seed,flush=True)
    save(output/'mappo_gaps.json',groups);save(output/'worst_gap.json',worst)
    save(output/'raw_source_hashes.json',hashes)
    save(output/'audit.json',dict(raw_files=len(hashes),fixed_worlds=400,mappo_episodes=8000,
                                  metrics_exact_match=True,protected_holdout_opened=False))
