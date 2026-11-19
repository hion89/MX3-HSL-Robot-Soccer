"""Four non-learning policies sharing the same immutable observation/action API."""
import itertools
import random
from typing import Protocol

from .model import Action, Observation, eta


class Policy(Protocol):
    def reset(self, seed: int) -> None: ...
    def decide(self, observation: Observation) -> Action: ...


class BasePolicy:
    def reset(self, seed):
        self.rng = random.Random(f'policy:{seed}')


class FixedRole(BasePolicy):
    """Fixed guard slots and predetermined task owners; worker FIFO, no rescue."""
    def decide(self, obs):
        assignments = {}
        for r in obs.robots:
            available = [j for j in obs.jobs if j.fixed_owner == r.id]
            initial = next((j for j in obs.jobs if j.id == r.initial_job), None)
            old = next((j for j in available if j.id == r.assignment), None)
            selected = initial or old or (min(available, key=lambda j: (j.spawn, j.id)) if available else None)
            assignments[r.id] = selected.id if r.operational and selected else None
        return Action(tuple(sorted(assignments.items())), notes=('fixed_owner_fifo',))


class GreedyETA(BasePolicy):
    """Priority-first, nearest remaining capable robot; stable ID tie breaks."""
    def decide(self, obs):
        unused = {r.id: r for r in obs.robots if r.operational}
        assignments = {r.id: None for r in obs.robots}
        for job in sorted(obs.jobs, key=lambda j: (-j.priority, j.deadline, j.id)):
            if not unused:
                break
            robot = min(unused.values(), key=lambda r: (eta(r, job, obs), r.id))
            assignments[robot.id] = job.id
            del unused[robot.id]
        return Action(tuple(sorted(assignments.items())), notes=('priority_then_eta',))


def assignment_cost(obs, allocation, switching_cost=2.0, unassigned_cost=50.0):
    """Internal allocation objective in weighted seconds, NOT an evaluation rank.

    Unassigned: P*U. Assigned: P*(ETA + service remaining), plus P*U if
    estimated completion misses deadline. Switching a still-active commitment
    costs S per robot. Base priority is its current risk weight.
    """
    robots = {r.id: r for r in obs.robots}
    active_ids = {j.id for j in obs.jobs}
    total = 0.0
    for job in obs.jobs:
        names = [name for name, jid in allocation.items() if jid == job.id]
        if len(names) > 1:
            raise ValueError('Exact allocations require unique ownership')
        if not names:
            total += unassigned_cost * job.priority
            continue
        r = robots[names[0]]
        duration = eta(r, job, obs) + job.remaining
        total += job.priority * duration
        if job.kind == 'task' and obs.time + duration > job.deadline + 1e-9:
            total += unassigned_cost * job.priority
    for robot in obs.robots:
        new = allocation.get(robot.id)
        if robot.assignment in active_ids and new != robot.assignment:
            total += switching_cost
    return total


class ExactAssignment(BasePolicy):
    """Exhaustive small-team assignment including idle; this is NOT Hungarian."""
    def __init__(self, switching_cost=2.0, unassigned_cost=50.0):
        self.switching_cost = switching_cost
        self.unassigned_cost = unassigned_cost

    def decide(self, obs):
        robots = sorted(obs.robots, key=lambda r: r.id)
        jobs = sorted(j.id for j in obs.jobs)
        choices = [(None, *jobs) if r.operational else (None,) for r in robots]
        best_cost, best = float('inf'), None
        for vector in itertools.product(*choices):
            selected = [jid for jid in vector if jid is not None]
            if len(set(selected)) != len(selected):
                continue
            allocation = dict(zip((r.id for r in robots), vector))
            cost = assignment_cost(obs, allocation, self.switching_cost, self.unassigned_cost)
            if cost < best_cost - 1e-9:
                best_cost, best = cost, allocation
        return Action(tuple(sorted(best.items())), notes=('exact_enumeration',), allocation_cost=best_cost)


class RuleBasedHandover(BasePolicy):
    """Same exact allocator plus readiness/deadline/risk release gate.

    Holds the incumbent while successor moves. Releases early only when waiting
    would threaten a task deadline and task priority >= current base risk.
    """
    def __init__(self, switching_cost=2.0, unassigned_cost=50.0):
        self.allocator = ExactAssignment(switching_cost, unassigned_cost)

    def reset(self, seed):
        super().reset(seed)
        self.allocator.reset(seed)
        self.pending = {}

    def decide(self, obs):
        candidate = self.allocator.decide(obs)
        allocation = dict(candidate.assignments)
        jobs, robots = {j.id: j for j in obs.jobs}, {r.id: r for r in obs.robots}
        preparations, notes = [], []
        # Keep an accepted handover plan stable while positions change; otherwise
        # a symmetric ETA tie can send the prepared successor to the new task.
        for jid, plan in list(self.pending.items()):
            old_id, new_id, task_id, original_risk = plan
            base, target = jobs.get(jid), jobs.get(task_id)
            old, new = robots.get(old_id), robots.get(new_id)
            if (base is None or target is None or old is None or new is None
                    or base.owner != old_id or old.assignment != jid
                    or not old.operational or not new.operational):
                del self.pending[jid]
            elif base.priority > original_risk and base.priority > target.priority:
                notes.append(f'risk_cancel:{jid}')
                del self.pending[jid]
            else:
                for name in allocation:
                    if name not in (old_id, new_id) and allocation[name] in (jid, task_id):
                        allocation[name] = None
                allocation[old_id], allocation[new_id] = task_id, jid
        for base in sorted((j for j in obs.jobs if j.kind == 'base'), key=lambda j: j.id):
            old = robots.get(base.owner)
            if old is None or not old.operational or old.assignment != base.id:
                continue
            target = jobs.get(allocation.get(old.id))
            new = next((r for r in obs.robots if r.id != old.id and allocation.get(r.id) == base.id), None)
            if target is None or target.kind != 'task' or new is None:
                continue
            self.pending.setdefault(base.id, (old.id, new.id, target.id, base.priority))
            if eta(new, base, obs) == 0:
                notes.append(f'ready_release:{base.id}')
                continue
            finish_if_wait = obs.time + eta(new, base, obs) + eta(old, target, obs) + target.remaining
            if finish_if_wait > target.deadline + 1e-9 and target.priority >= base.priority:
                notes.append(f'deadline_override:{base.id}')
                continue
            allocation[old.id] = base.id
            preparations.append((new.id, base.id))
            notes.append(f'hold_prepare:{base.id}')
        return Action(tuple(sorted(allocation.items())), tuple(preparations), tuple(notes), candidate.allocation_cost)


POLICIES = {'fixed': FixedRole, 'greedy': GreedyETA, 'exact': ExactAssignment, 'handover': RuleBasedHandover}


def make_policy(name, config):
    cls = POLICIES[name]
    p = cls(config.switching_cost, config.unassigned_cost) if name in ('exact', 'handover') else cls()
    p.reset(config.seed)
    return p
