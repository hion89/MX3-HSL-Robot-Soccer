"""Bounded non-learning ablations. Existing worlds/results are never overwritten."""
import copy
from dataclasses import replace
from pathlib import Path
from ..phase2.model import Action
from ..phase2.policies import FixedRole
from ..phase3b2b.io import save,write_raw
from ..phase3b2a.scenarios import digest
from ..phase4b.engine import Simulation
from ..phase4b.policies import Native
from ..phase4b.metrics import evaluate_new
from ..phase4b.experiment import read,grouped
from .policies import RiskDirectedSwap,DelayedExternalInformation,LocalDistribution


def execute(spec,policy,dt=None,feedback=False):
    from phase4b_feedback import FeedbackSimulation
    engine=(FeedbackSimulation if feedback else Simulation)(spec,policy,dt);rows=engine.run();m=evaluate_new(rows)
    m.update(scenario_id=spec['scenario_id'],family=spec['family'],team=spec['team'],policy=policy.name,
             physics_dt=engine.c.dt,exogenous_sha256=digest(engine.plan))
    return m,rows


def run_rules(root,output):
    import torch
    from phase4b_feedback import SensorActor
    torch.set_num_threads(1);target=output/'rules';target.mkdir(exist_ok=False)
    specs=read(root/'phase4b_outputs/study_v1/design/diagnostic_manifest.json');summaries={}
    for name in ('risk_directed_swap','local_fixed','local_eta'):
        folder=target/name;folder.mkdir();records=[]
        for s in specs:
            if name=='risk_directed_swap':policy=RiskDirectedSwap()
            else:policy=SensorActor(LocalDistribution(name));policy.name=name
            m,rows=execute(s,policy,feedback=name!='risk_directed_swap');records.append(m)
            write_raw(folder/f"{s['scenario_id']}.jsonl.gz",rows)
        d={'episodes':records,'summary':grouped(records),'interface':'current_public_with_preparation' if name=='risk_directed_swap' else 'same_146_features_10_actions_as_frozen_actor'}
        save(folder/'results.json',d);summaries[name]=d['summary'];print('diagnostic rules',name,flush=True)
    save(target/'summary.json',summaries)


class TeamSchedule:
    name='offline_team_schedule'
    information='future_task_identity_and_timing_privileged_witness'
    def __init__(self,guard,stage,successor,target,prepare,depart):
        self.guard,self.stage,self.successor,self.target,self.prepare,self.depart=guard,stage,successor,target,prepare,depart
    def reset(self,seed):self.fixed=FixedRole();self.fixed.reset(seed);self.released=False;self.last={}
    def decide(self,view):
        obs=view.allocation;robots={r.id:r for r in obs.robots};jobs={j.id:j for j in obs.jobs}
        allocation=dict(self.fixed.decide(obs).assignments)
        if obs.time<self.prepare:return Action(tuple(sorted(allocation.items())))
        task=jobs.get('task0')
        if task and obs.time>=self.depart:self.released=True
        g=robots[self.guard];s=robots[self.successor]
        allocation[g.id]=('task0' if task else None) if self.released else self.stage
        allocation[s.id]=self.target if s.operational else None
        for name in allocation:
            if name!=g.id and allocation[name]=='task0':allocation[name]=None
        # Fill a vacated quiet home if an uncommitted worker is available.
        for old in (r for r in obs.robots if r.base_role=='guard' and r.initial_job and allocation.get(r.id)!=r.initial_job):
            spare=[r for r in obs.robots if r.id not in (g.id,s.id) and r.operational and r.base_role=='worker'
                   and (r.assignment is None or r.assignment==old.initial_job or
                        r.assignment in jobs and jobs[r.assignment].kind=='task' and jobs[r.assignment].remaining==view.services[r.assignment])]
            if spare:allocation[min(spare,key=lambda r:r.id).id]=old.initial_job
        prep=[]
        for base in (j for j in obs.jobs if j.kind=='base'):
            names=[n for n,v in allocation.items() if v==base.id]
            if base.owner in names:prep.extend((n,base.id) for n in names if n!=base.owner)
        return Action(tuple(sorted(allocation.items())),tuple(prep),('bounded_offline_team_schedule',))


def run_plans(root,output):
    target=output/'plans';target.mkdir(exist_ok=False)
    specs=[s for s in read(root/'phase4b_outputs/study_v1/design/diagnostic_manifest.json')
           if s['team']==4 and s['family'] in ('A_near','B_normal','D_visible','E_incumbent','E_successor')]
    results=[];count=0
    for spec in specs:
        folder=target/spec['scenario_id'];folder.mkdir();plans=[]
        m,rows=execute(spec,Native('fixed'));plans.append({'file':'fixed.jsonl.gz','parameters':None,'metrics':m})
        write_raw(folder/'fixed.jsonl.gz',rows);count+=1
        for guard in ('A','B'):
            home='base0' if guard=='A' else 'base1';other='B' if guard=='A' else 'A'
            for stage in sorted({'base0',home}):
                for successor in ('C',other):
                    for to in sorted({'base0',home}):
                        for prepare in (0,2):
                            for depart in (2,4,6,8):
                                p=TeamSchedule(guard,stage,successor,to,prepare,depart);m,rows=execute(spec,p)
                                name=f'g{guard}_stage{stage}_s{successor}_to{to}_p{prepare}_d{depart}.jsonl.gz'
                                write_raw(folder/name,rows);count+=1
                                plans.append({'file':name,'parameters':[guard,stage,successor,to,prepare,depart],'metrics':m})
        joint=[p for p in plans if p['metrics']['joint_task_and_defense_success']]
        task=[p for p in plans if p['metrics']['task_success_rate']==1]
        results.append({'scenario_id':spec['scenario_id'],'family':spec['family'],'team':4,'plans':len(plans),
            'has_joint_witness':bool(joint),'has_task_witness':bool(task),
            'best_joint':min(joint,key=lambda p:p['metrics']['legacy_gap_region_seconds']) if joint else None})
        save(folder/'witnesses.json',{'scope':'restricted future-aware schedules; not a global temporal optimum','plans':plans})
        print('four-robot schedules',spec['family'],spec['index'],'joint',bool(joint),flush=True)
    save(target/'results.json',{'executions':count,'cases':results,'no_witness_is_not_infeasibility':True})


def run_sensitivity(root,output,commitment_only=False):
    target=output/('commitment_sensitivity' if commitment_only else 'sensitivity');target.mkdir(exist_ok=False)
    worlds=read(root/'phase4b_outputs/study_v1/design/diagnostic_manifest.json');specs=[s for s in worlds if s['index']<2]
    policies=('commitment_safe_swap',) if commitment_only else ('fixed','ready','adaptive','observed_reserve','risk_directed_swap')
    profiles=('baseline','speed_scale_04','speed_scale_02','turn_1','turn_05','external_information_delay_1')
    records=[]
    for profile in profiles:
        folder=target/profile;folder.mkdir()
        for name in policies:
            for spec in specs:
                s=copy.deepcopy(spec)
                for r in s['config']['robots']:
                    if profile=='speed_scale_04':r['speed']*=.4
                    if profile=='speed_scale_02':r['speed']*=.2
                    if profile=='turn_1':r['turn_speed']=1.
                    if profile=='turn_05':r['turn_speed']=.5
                p=RiskDirectedSwap(name=='commitment_safe_swap') if name in ('risk_directed_swap','commitment_safe_swap') else Native(name,warning=4.,lookahead=4.)
                if profile=='external_information_delay_1':p=DelayedExternalInformation(p,1.)
                m,rows=execute(s,p);m['profile']=profile;records.append(m)
                write_raw(folder/f"{name}_{s['scenario_id']}.jsonl.gz",rows)
        print('assumption sensitivity',profile,flush=True)
    save(target/'parameter_results.json',{'scope':'one-factor assumption stress tests; command limits are not measured velocities',
        'episodes':records,'geometry_times_and_deadlines_unchanged':True})
    fine=target/'fine_dt';fine.mkdir();resolution=[]
    subset=[s for s in worlds if s['index']<2 and s['family'] in ('A_near','B_normal','C_reactive','D_visible','E_incumbent','E_successor')]
    for name in (('commitment_safe_swap',) if commitment_only else ('ready','observed_reserve','risk_directed_swap')):
        for s in subset:
            signatures=[]
            for dt in (.125,.0625):
                p=RiskDirectedSwap(name=='commitment_safe_swap') if name in ('risk_directed_swap','commitment_safe_swap') else Native(name,warning=4.,lookahead=4.)
                m,rows=execute(s,p,dt);resolution.append(m);signatures.append(m['exogenous_sha256'])
                assert [r['time'] for r in rows if r.get('event')=='policy_selection']==list(range(32))
                write_raw(fine/f"{name}_{s['scenario_id']}_dt{dt}.jsonl.gz",rows)
            assert len(set(signatures))==1
    save(target/'fine_dt_results.json',{'episodes':resolution,'control_period_unchanged_s':1.,'same_physical_scenarios':True})
