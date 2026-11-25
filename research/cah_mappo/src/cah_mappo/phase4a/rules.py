"""Privileged execution gates. Never modify sampled actor action or its probability."""
from dataclasses import asdict
import math
from ..phase2.model import Action, eta


class Gate:
    def __init__(self, mode='ready', risk_ratio=1., eta_budget=2., limits=None):
        if (mode not in ('ready', 'adaptive') or not all(math.isfinite(v) and v>0 for v in (risk_ratio,eta_budget))
                or any(not math.isfinite(v) or v<=0 for v in (limits or {}).values())):
            raise ValueError('Invalid execution rule')
        self.mode, self.risk_ratio, self.eta_budget = mode, risk_ratio, eta_budget
        self.limits = dict(limits or {}); self.pending = {}; self.counter = 0

    def apply(self, engine, t, action, apply_original):
        obs = engine.observe(t)  # explicit privileged CURRENT observation; never future events
        robots = {r.id:r for r in obs.robots}; jobs = {j.id:j for j in obs.jobs}
        raw = dict(action.assignments); allocation = raw.copy(); preparations = []; events = []
        def event(kind, **kw):
            row = {'event':kind, **kw}; events.append(row); engine.event(t, kind, **kw)
        for base_id, plan in list(self.pending.items()):
            old_id, new_id, task_id, original_risk = plan
            base, task = jobs.get(base_id), jobs.get(task_id)
            old, new = robots[old_id], robots[new_id]
            reason = ('incumbent_failed' if not old.operational else 'successor_failed' if not new.operational
                      else 'task_resolved' if task is None else 'request_changed' if raw.get(old_id)!=task_id
                      else 'risk_increased' if base and base.priority > original_risk and base.priority > task.priority else None)
            if reason:
                event('controller_failed' if 'failed' in reason else 'controller_cancelled',
                      base=base_id, incumbent=old_id, successor=new_id, task=task_id, reason=reason)
                del self.pending[base_id]
        reserved = set()
        for base in sorted((j for j in obs.jobs if j.kind=='base'), key=lambda j:j.id):
            old = robots.get(base.owner)
            # A fault has no live incumbent who can grant a normal handover.
            if old is None or not old.operational or old.assignment != base.id:
                if base.owner is None or old is not None and not old.operational:
                    candidates = [r for r in obs.robots if r.operational and r.id not in reserved
                                  and (r.assignment is None or r.assignment==base.id)]
                    if candidates:
                        new=min(candidates,key=lambda r:(eta(r,base,obs),r.id))
                        allocation[new.id]=base.id; reserved.add(new.id)
                        if raw.get(new.id)!=base.id:
                            event('controller_fault_rescue',base=base.id,successor=new.id)
                continue
            target = jobs.get(raw.get(old.id))
            if raw.get(old.id) == base.id:
                continue
            if target is None or target.kind != 'task':
                allocation[old.id] = base.id
                event('controller_idle_departure_blocked',base=base.id,incumbent=old.id,requested=raw.get(old.id))
                continue
            if base.id in self.pending:
                new = robots[self.pending[base.id][1]]
            else:
                candidates = [r for r in obs.robots if r.id!=old.id and r.operational and r.id not in reserved
                              and (r.assignment is None or r.assignment==base.id or
                                   (r.assignment in jobs and jobs[r.assignment].kind=='task' and jobs[r.assignment].remaining==
                                    engine.jobs[r.assignment]['service']))]
                if not candidates:
                    allocation[old.id]=base.id
                    event('controller_no_successor',base=base.id,incumbent=old.id,task=target.id)
                    continue
                new = min(candidates,key=lambda r:(eta(r,base,obs),r.id))
                self.pending[base.id]=(old.id,new.id,target.id,base.priority)
                event('controller_requested',base=base.id,incumbent=old.id,successor=new.id,task=target.id)
            reserved.add(new.id)
            wait=eta(new,base,obs); direct=eta(old,target,obs)+target.remaining
            ready=(new.operational and math.dist(new.position,base.position)<=obs.radius+1e-9
                   and new.assignment in (None,base.id))
            urgent=(self.mode=='adaptive' and t+wait+direct>target.deadline+1e-9
                    and t+direct<=target.deadline+1e-9 and target.priority>=self.risk_ratio*base.priority
                    and wait<=self.eta_budget*self.limits.get(base.id,4.))
            allocation[new.id]=base.id
            if ready or urgent:
                event('controller_ready_release' if ready else 'controller_early_release',base=base.id,
                      incumbent=old.id,successor=new.id,task=target.id,successor_eta=wait,
                      completion_now=t+direct,completion_if_wait=t+wait+direct)
                self.pending.pop(base.id,None)
            else:
                allocation[old.id]=base.id; preparations.append((new.id,base.id))
                event('controller_holding',base=base.id,incumbent=old.id,successor=new.id,
                      task=target.id,successor_eta=wait,completion_if_wait=t+wait+direct)
        result=apply_original(t,Action(tuple(sorted(allocation.items())),tuple(preparations),('phase4a_'+self.mode,)))
        result['policy_requested']=raw
        result['controller_requested']=allocation.copy()
        result['requested']=raw  # requested-policy metrics retain ORIGINAL choices
        result['controller_preparations']=[list(p) for p in preparations]
        result['controller_observation']=asdict(obs)
        result['controller_mode']=self.mode
        result['controller_events']=events
        result['controller_changed']={n:allocation.get(n)!=raw.get(n) for n in robots}
        return result


def install_gate(engine, gate):
    """Per-instance adapter; existing class source and reward functions stay untouched."""
    original=engine.apply
    engine.apply=lambda t,action:gate.apply(engine,t,action,original)
