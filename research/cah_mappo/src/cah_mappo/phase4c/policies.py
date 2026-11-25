"""Minimal non-learning diagnostics; no scenario IDs, future schedules or new network."""
import copy
from dataclasses import replace
from ..phase2.model import Action,eta
from ..phase2.policies import FixedRole
from ..phase4b.policies import Native


class RiskDirectedSwap:
    name='risk_directed_swap'
    information='same_current_public_team_and_visible_threats'
    def __init__(self,protect_active_tasks=False):
        self.protect_active_tasks=protect_active_tasks
        if protect_active_tasks:self.name='commitment_safe_swap'
    def reset(self,seed):
        from phase4b_contingency import FaultAwareRelease
        self.regular=Native('observed_reserve',warning=4);self.regular.reset(seed)
        self.fault=FaultAwareRelease();self.fault.reset(seed);self.plans={};self.last={}

    def decide(self,view):
        obs=view.allocation;robots={r.id:r for r in obs.robots};jobs={j.id:j for j in obs.jobs}
        regular=self.regular.decide(view);fault=self.fault.decide(view)
        action=fault if any(not r.operational for r in obs.robots) else regular
        allocation=dict(action.assignments);preparations=list(action.preparations);self.last={}
        threatened={q['target'] for q in view.threats};reserved=set()
        for base in (j for j in obs.jobs if j.kind=='base' and j.id in threatened):
            remaining=min(q['estimated_arrival']-obs.time for q in view.threats if q['target']==base.id)
            plan=self.plans.get(base.id)
            if plan and not robots[plan['successor']].operational:
                self.plans.pop(base.id);plan=None
            owner=robots.get(base.owner)
            if plan is None and owner and owner.operational and owner.assignment==base.id:
                available=[]
                for r in obs.robots:
                    if not r.operational or r.id==owner.id or r.id in reserved:continue
                    old=jobs.get(r.assignment)
                    if old and old.kind=='task' and old.remaining<view.services[old.id]-1e-9:continue
                    if old and old.kind=='base' and old.id!=base.id and old.id in threatened:continue
                    if eta(r,base,obs)>remaining+1e-9:continue
                    available.append(r)
                if available:
                    successor=min(available,key=lambda r:(eta(r,base,obs),r.id))
                    old=jobs.get(successor.assignment)
                    # Worker-only reservation is already in the frozen Phase4B rule.
                    if old and old.kind=='base' and old.id!=base.id:
                        spare=[r for r in obs.robots if r.id not in (owner.id,successor.id) and r.operational
                               and (r.assignment is None or r.assignment==old.id)
                               and (not self.protect_active_tasks or
                                    allocation.get(r.id) not in jobs or jobs[allocation[r.id]].kind!='task')]
                        replacement=min(spare,key=lambda r:(eta(r,old,obs),r.id)) if spare else None
                        plan={'incumbent':owner.id,'successor':successor.id,'quiet_base':old.id,
                              'replacement':replacement.id if replacement else None}
                        self.plans[base.id]=plan
            if not plan:continue
            old=robots[plan['incumbent']];new=robots[plan['successor']];reserved.add(new.id)
            allocation[new.id]=base.id
            if plan['replacement'] and robots[plan['replacement']].operational:
                allocation[plan['replacement']]=plan['quiet_base']
            candidates=[j for j in obs.jobs if j.kind=='task' and j.remaining>=view.services[j.id]-1e-9
                        and old.operational and obs.time+eta(old,j,obs)+j.remaining<=j.deadline+1e-9]
            current=jobs.get(old.assignment)
            task=current if current and current.kind=='task' else min(candidates,key=lambda j:(j.deadline,-j.priority,j.id)) if candidates else None
            ready=eta(new,base,obs)==0
            urgent=task is not None and obs.time+eta(new,base,obs)+eta(old,task,obs)+task.remaining>task.deadline+1e-9
            safe_prediction=eta(new,base,obs)<=remaining+1e-9
            if task and old.operational and (ready or urgent and safe_prediction):
                for name in allocation:
                    if name!=old.id and allocation[name]==task.id:allocation[name]=None
                allocation[old.id]=task.id
            elif old.operational:
                allocation[old.id]=base.id;preparations.append((new.id,base.id))
            self.last.setdefault('borrowed_quiet_guard',[]).append(copy.deepcopy(plan))
        for base in list(self.plans):
            if base not in threatened:del self.plans[base]
        return Action(tuple(sorted(allocation.items())),tuple(sorted(set(preparations))),('public_risk_directed_swap',))


class DelayedExternalInformation:
    """Synthetic task/threat delivery delay only; robot odometry stays current."""
    def __init__(self,policy,delay=1.):self.policy=policy;self.delay=delay;self.name=policy.name;self.information='synthetic_external_delivery_delay'
    def reset(self,seed):self.policy.reset(seed);self.first={};self.last={}
    def decide(self,view):
        from ..phase4b.model import PublicView
        obs=view.allocation;visible=[]
        for q in view.threats:self.first.setdefault(q['id'],obs.time)
        for j in obs.jobs:
            if j.kind=='base' or obs.time>=j.spawn+self.delay-1e-9:visible.append(j)
        threats=tuple(q for q in view.threats if obs.time>=self.first[q['id']]+self.delay-1e-9)
        delayed=PublicView(replace(obs,jobs=tuple(visible)),{j.id:view.services[j.id] for j in visible},
                           threats,view.response_budgets,view.horizon)
        action=self.policy.decide(delayed);self.last={'delay_s':self.delay,'delivered_public':delayed.record(),
                                                    'underlying':copy.deepcopy(self.policy.last)}
        return action


class BranchHintPolicy:
    """Same rule with/without privileged announcement, for information ablation only."""
    name='branch_hint_diagnostic'
    def __init__(self,announcement=None):self.announcement=announcement;self.information='privileged_future_announcement' if announcement else 'current_public_only'
    def reset(self,seed):
        self.ready=Native('ready');self.standing=Native('standing_reserve');self.ready.reset(seed);self.standing.reset(seed);self.last={}
    def decide(self,view):
        if self.announcement is None:return self.ready.decide(view)
        obs=view.allocation;workers=[r for r in obs.robots if r.base_role=='worker']
        task=self.announcement
        near=any(__import__('math').dist(r.position,task['position'])<=obs.radius+1e-9 for r in workers)
        policy=self.ready if near else self.standing
        action=policy.decide(view);self.last={'privileged_announcement_used':True,'near_branch':near}
        return action


class LocalDistribution:
    """A one-hot distribution from the existing local rules, using 146 features only."""
    def __init__(self,mode):
        from ..phase3b2a.baselines import LocalPolicy
        self.policy=LocalPolicy(mode,0)
    def distribution(self,vectors,masks):
        import torch
        probabilities=torch.zeros((4,10),dtype=torch.float64)
        for row,name in enumerate(('A','B','C','D')):
            o=vectors[row].tolist();mask=[int(v) for v in masks[row].tolist()]
            ids=[f'slot_{i}' if o[50+i*9] else None for i in range(8)]
            own=max(range(9),key=lambda i:o[9+i]);assignment=None if own==0 else ids[own-1]
            packet={'observation':o,'action_mask':mask,'self_assignment':assignment,'active_mask':int(o[6]),
                    'job_ids':ids,'peer_fresh_mask':[int(v) for v in o[126:130]]}
            a=self.policy.decide(name,packet);probabilities[row,a]=1
        return torch.distributions.Categorical(probs=probabilities)
