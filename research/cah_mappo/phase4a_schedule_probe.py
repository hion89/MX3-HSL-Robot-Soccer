"""Bounded timing search: prepare at t=0 or task spawn, enumerate departure ticks.

This is a feasibility diagnostic with future-known planning, not a full temporal
optimum or a deployable partially-observed policy. No tuning of frozen rules.
"""
from pathlib import Path
import json
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.phase2.engine import Environment
from cah_mappo.phase2.model import Action,config_from_dict
from cah_mappo.phase2.policies import FixedRole
from cah_mappo.phase3b2b.io import save
from cah_mappo.phase4a.metrics import score


class Scheduled:
    def __init__(self,prepare_at,departure):self.prepare_at,self.departure=prepare_at,departure
    def reset(self,seed):self.fixed=FixedRole();self.fixed.reset(seed);self.released=False
    def decide(self,obs):
        allocation=dict(self.fixed.decide(obs).assignments);jobs={j.id:j for j in obs.jobs}
        if obs.time<self.prepare_at:return Action(tuple(sorted(allocation.items())))
        if obs.time>=self.departure and 'task0' in jobs:self.released=True
        allocation['B']='base0'
        allocation['A']='task0' if self.released and 'task0' in jobs else None if self.released else 'base0'
        prep=() if self.released else (('B','base0'),)
        return Action(tuple(sorted(allocation.items())),prep)


def main():
    root=ROOT/'phase4a_outputs/study_v1';target=root/'bounded_schedule_probe';target.mkdir(exist_ok=False)
    specs=json.loads((root/'design/diagnostic_manifest.json').read_text());result=[];executions=0
    for spec in specs:
        if spec['team']!=2 or spec['family'] not in ('A','B','C','C_risk','R_risk'):continue
        c=config_from_dict(spec['config']);plans=[]
        for prepare in (0,2):
            for departure in range(2,30):
                rows=Environment(c,Scheduled(prepare,departure),'bounded_schedule').run()
                rows[0]['phase4a']={'response_limit_s':spec['response_limit_s']}
                m=score(rows);executions+=1
                plans.append({'prepare_at':prepare,'departure_at':departure,'success':m['task_success_rate'],
                              'gap':m['actual_gap_region_seconds'],'max_gap_s':m['max_continuous_gap_s'],
                              'threat_violations':m['threat_limit_violations'],'task_details':m['task_details']})
        best={}
        for name,start in (('reactive',2),('preparation_from_zero',0)):
            success=[p for p in plans if p['prepare_at']==start and p['success']==1]
            best[name]=min(success,key=lambda p:(p['gap'],p['departure_at'])) if success else None
        result.append({'scenario_id':spec['scenario_id'],'family':spec['family'],'best_successful_schedules':best})
    summary={}
    for family in ('A','B','C','C_risk','R_risk'):
        selected=[r for r in result if r['family']==family]
        summary[family]={mode:{'cases':len(selected),
            'zero_gap_success_cases':sum(r['best_successful_schedules'][mode] is not None and r['best_successful_schedules'][mode]['gap']==0 for r in selected),
            'successful_plan_cases':sum(r['best_successful_schedules'][mode] is not None for r in selected)}
            for mode in ('reactive','preparation_from_zero')}
    save(target/'results.json',{'scope':'Restricted two-robot guard-to-task/worker-to-base schedules only; not a complete temporal optimum',
        'assumption':'Known-base preparation at t=0 is admissible but is not offered automatically by the learned execution wrapper',
        'executions':executions,'summary':summary,'cases':result})
    print('bounded timing probe',executions,'executions',summary)


if __name__=='__main__':main()
