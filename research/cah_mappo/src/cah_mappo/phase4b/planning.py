"""Observed-only finite candidate lookahead; separately labeled offline schedule witnesses."""
import copy
from dataclasses import asdict
import math
from types import SimpleNamespace
from ..phase2.engine import Environment
from ..phase2.model import Action
from ..phase2.policies import FixedRole,GreedyETA,ExactAssignment
from ..phase4a.rules import Gate
from .model import travel_time,match
from .policies import apply_public_gate


def forecast(view,action,horizon):
    """Only public CURRENT state and already visible trajectories are copied.

    Known actions are held over a bounded horizon. There are no future spawns,
    failures, recoveries, hidden threats or a simulator object in this input.
    """
    obs=view.allocation;engine=Environment.__new__(Environment)
    engine.c=SimpleNamespace(dt=.25,events=(),radius=obs.radius)
    engine.rows=[];engine.handover_id=0
    engine.robots={r.id:asdict(r) for r in obs.robots};engine.jobs={}
    for j in obs.jobs:
        service=view.services[j.id]
        engine.jobs[j.id]={**asdict(j),'service':service,'work':service-j.remaining,
                           'status':'active','response':None,'completed':None}
    engine.apply(obs.time,action)
    end=min(view.horizon,obs.time+horizon);passed=0.;seen=set();gap=0.
    steps=round((end-obs.time)/.25)
    for tick in range(steps+1):
        t=round(obs.time+tick*.25,10);engine.boundary(t);engine.update_owners(t)
        due=[q for q in view.threats if q['id'] not in seen and q['estimated_arrival']<=t+1e-9]
        query={q['id']:(q['target'],q['intensity']) for q in due}
        state={**engine.truth(),'time':t,'control_period':1.}
        covered=match(state,query,{k:0. for k in query},weighted=True,at_point=True)
        for q in due:
            seen.add(q['id'])
            if q['id'] not in covered:passed+=q['intensity']
        if tick==steps:break
        bases={j['id']:(j['id'],1.) for j in engine.jobs.values() if j['kind']=='base'}
        gap+=(len(bases)-len(match(state,bases,view.response_budgets)))*.25
        engine.integrate(t)
    failed=passed;completed=progress=0.
    for job in engine.jobs.values():
        if job['kind']!='task':continue
        if job['status']=='expired':failed+=job['priority']
        elif job['status']=='completed':completed+=job['priority']
        else:
            assigned=[r for r in engine.robots.values() if r['operational'] and r['assignment']==job['id']]
            eligible=assigned or [r for r in engine.robots.values() if r['operational']]
            lower=min((travel_time(r,job['position'],obs.radius) for r in eligible),default=math.inf)+job['service']-job['work']
            if end+lower>job['deadline']+1e-9:failed+=job['priority']
        progress+=job['priority']*job['work']/job['service']
    changes=sum(dict(action.assignments).get(r.id)!=r.assignment for r in obs.robots)
    return (failed,-completed,-progress,gap,changes)


def choose_current_plan(view,horizon,warning):
    from .policies import Native
    obs=view.allocation;candidate=[]
    for allocator in (FixedRole(),GreedyETA(),ExactAssignment(2,50)):
        allocator.reset(0);candidate.append(allocator.decide(obs))
    exact=candidate[-1]
    for mode in ('ready','adaptive'):
        gate=Gate(mode,risk_ratio=.5,eta_budget=4.,limits=view.response_budgets)
        candidate.append(apply_public_gate(gate,view,exact)[0])
    helper=Native('observed_reserve',warning=warning)
    candidate.append(helper.reserve(view,candidate[-1],False))
    candidate.append(helper.reserve(view,candidate[-1],True))
    candidate.append(Action(tuple((r.id,r.assignment if r.operational else None) for r in obs.robots)))
    unique={}
    for action in candidate:unique.setdefault((tuple(sorted(action.assignments)),tuple(sorted(action.preparations))),action)
    scored=[(forecast(view,a,horizon),a) for a in unique.values()]
    key,action=min(scored,key=lambda pair:pair[0])
    return action,{'lookahead_seconds':horizon,'candidate_count':len(scored),
                   'cost_tuple':list(key),'future_events_used':False,
                   'scope':'candidate-action finite lookahead; not exhaustive temporal MPC optimum'}


class OfflineSchedule:
    information='future_task_timing_and_identity_privileged_offline_witness'
    name='offline_schedule'
    def __init__(self,prepare_at,depart_at):self.prepare_at=prepare_at;self.depart_at=depart_at
    def reset(self,seed):self.fixed=FixedRole();self.fixed.reset(seed);self.released=False;self.last={}
    def decide(self,view):
        obs=view.allocation;allocation=dict(self.fixed.decide(obs).assignments)
        robots={r.id:r for r in obs.robots};worker='C' if 'C' in robots else 'B';jobs={j.id:j for j in obs.jobs}
        if obs.time<self.prepare_at:return Action(tuple(sorted(allocation.items())))
        if obs.time>=self.depart_at and 'task0' in jobs:self.released=True
        allocation[worker]='base0' if robots[worker].operational else None
        allocation['A']='task0' if self.released and 'task0' in jobs else None if self.released else 'base0'
        prep=((worker,'base0'),) if not self.released else ()
        return Action(tuple(sorted(allocation.items())),prep,('oracle_schedule_witness_not_online',))
