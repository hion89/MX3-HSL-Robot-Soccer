"""All native policies receive the same current PublicView, never a scenario or engine."""
import copy
from dataclasses import asdict
import math
from types import SimpleNamespace
from ..phase2.model import Action,eta
from ..phase2.policies import FixedRole,GreedyETA,ExactAssignment,RuleBasedHandover
from ..phase4a.rules import Gate


def apply_public_gate(gate,view,candidate):
    events=[]
    # This facade has no event schedule, hidden jobs, RNG, scenario ID or GT engine.
    facade=SimpleNamespace(observe=lambda t:view.allocation,
        jobs={j.id:{'service':view.services[j.id]} for j in view.allocation.jobs},
        event=lambda t,name,**kw:events.append({'time':t,'event':name,**kw}))
    def fake_apply(t,a):
        assignments=dict(a.assignments)
        return {'requested':assignments,'accepted':assignments.copy(),'preparations':list(a.preparations),
                'notes':list(a.notes),'allocation_cost':a.allocation_cost,'rejections':[]}
    result=gate.apply(facade,view.allocation.time,candidate,fake_apply)
    return Action(tuple(sorted(result['controller_requested'].items())),tuple(tuple(p) for p in result['controller_preparations']),
                  ('public_execution_gate',)),{'events':events,'allocator_requested':dict(candidate.assignments),
                                               'controller_requested':result['controller_requested']}


class Native:
    information='same_current_public_team_and_visible_threats'
    def __init__(self,name,warning=8.,lookahead=4.):
        if name not in ('fixed','greedy','exact','legacy_handover','ready','adaptive','observed_reserve','standing_reserve','lookahead'):
            raise ValueError(name)
        self.name=name;self.warning=warning;self.lookahead=lookahead

    def reset(self,seed):
        self.fixed=FixedRole();self.greedy=GreedyETA();self.exact=ExactAssignment(2,50);self.legacy=RuleBasedHandover(2,50)
        for p in (self.fixed,self.greedy,self.exact,self.legacy):p.reset(seed)
        self.gate=Gate('ready' if self.name=='ready' else 'adaptive',risk_ratio=.5,eta_budget=4.)
        self.last={}

    def reserve(self,view,action,standing=False):
        obs=view.allocation;allocation=dict(action.assignments);prep=list(action.preparations);used=set()
        for base in (j for j in obs.jobs if j.kind=='base'):
            owner=next((r for r in obs.robots if r.id==base.owner and r.operational and r.assignment==base.id),None)
            if owner is None or allocation.get(owner.id)!=base.id:continue
            warnings=[q for q in view.threats if q['target']==base.id and q['estimated_arrival']-obs.time<=self.warning]
            if not standing and not warnings:continue
            candidates=[r for r in obs.robots if r.operational and r.id!=owner.id and r.id not in used
                        and r.assignment in (None,base.id) and allocation.get(r.id) in (None,base.id)]
            if not candidates:continue
            successor=min(candidates,key=lambda r:(eta(r,base,obs),r.id))
            allocation[successor.id]=base.id;prep.append((successor.id,base.id));used.add(successor.id)
        return Action(tuple(sorted(allocation.items())),tuple(sorted(set(prep))),action.notes)

    def decide(self,view):
        obs=view.allocation;self.last={}
        if self.name in ('fixed','greedy','exact','legacy_handover'):
            return getattr(self,{'legacy_handover':'legacy'}.get(self.name,self.name)).decide(obs)
        candidate=self.exact.decide(obs)
        self.gate.limits=dict(view.response_budgets)
        if self.name=='lookahead':
            from .planning import choose_current_plan
            action,self.last=choose_current_plan(view,self.lookahead,self.warning)
            return action
        action,self.last=apply_public_gate(self.gate,view,candidate)
        if self.name in ('observed_reserve','standing_reserve'):
            action=self.reserve(view,action,standing=self.name=='standing_reserve')
        return action


class FrozenActor:
    """Legacy 146-feature projection: no new threat channels, never a critic call."""
    information='legacy_146_local_projection_no_threat_features_transfer_reference'
    def __init__(self,network,mode='greedy',sampling_seed=7):
        self.network=network;self.mode=mode;self.sampling_seed=sampling_seed
        self.name='mappo_'+mode

    def reset(self,seed):
        import torch
        self.rng=torch.Generator(device='cpu').manual_seed(self.sampling_seed)
        self.memory={name:[] for name in ('A','B','C','D')};self.job_memory=None;self.last={}

    def decide(self,view):
        import torch
        from ..phase3a.contracts import Settings,SampledAction,validate_sample
        from ..phase3a.perception import build_actor
        obs=view.allocation;settings=Settings(sensor_radius=100,sensor_noise=0,radio_delay=0)
        robots={r.id:r for r in obs.robots};packets={};vectors=[];masks=[]
        # Construct only current public measurements. Persist previously observed job slots.
        if self.job_memory is None:self.job_memory={n:{} for n in robots}
        for name,r in robots.items():
            slots=self.memory[name]
            for j in obs.jobs:
                if j.id not in slots:slots.append(j.id)
                self.job_memory[name][j.id]={'position':j.position,'kind':j.kind,'status':'active','priority':j.priority,
                    'deadline':j.deadline,'remaining':j.remaining,'observed_at':obs.time,'source':'sensor'}
            active={j.id for j in obs.jobs}
            for jid,record in self.job_memory[name].items():
                if jid not in active:record.update(status='completed',observed_at=obs.time)
            peers={p.id:{'position':p.position,'heading':p.heading,'speed':p.speed,'operational':p.operational,
                         'observed_at':obs.time,'source':'message'} for p in obs.robots if p.id!=name}
            packet=build_actor(asdict(r),{'peers':peers,'jobs':self.job_memory[name]},slots,tuple(robots),settings,obs.time,view.horizon,name)
            packets[name]=packet
        for name in settings.roster:
            p=packets.get(name);vectors.append(p['observation'] if p else [0.]*146);masks.append(p['action_mask'] if p else [1]+[0]*9)
        with torch.no_grad():
            dist=self.network.distribution(torch.tensor(vectors,dtype=torch.float32),torch.tensor(masks,dtype=torch.bool))
            chosen=dist.probs.argmax(-1) if self.mode=='greedy' else torch.multinomial(dist.probs,1,generator=self.rng).squeeze(-1)
        assignments={};records={}
        for i,name in enumerate(settings.roster):
            if name not in packets:continue
            p=packets[name];a=int(chosen[i]);probs=tuple(dist.probs[i].tolist())
            sample=SampledAction(a,math.log(probs[a]),tuple(p['action_mask']),probs,'phase4b-frozen',p['observation_token'])
            validate_sample(sample,p['action_mask'],p['observation_token'])
            assignments[name]=None if a==0 else p['self_assignment'] if a==1 else p['job_ids'][a-2]
            records[name]={'sample':asdict(sample),'packet':p}
        self.last={'actor_samples':records,'new_threat_channels_used':False}
        return Action(tuple(sorted(assignments.items())),notes=('frozen_actor_projection',))
