"""Explicit shared training reward, independent of the paper metric evaluator."""
from dataclasses import dataclass, asdict
import math
import itertools


def covered_sites(state):
    """Reward-side capacity predicate, independent of the metric evaluator."""
    bases = {jid: j for jid, j in state['jobs'].items() if j['kind'] == 'base'}
    scale = max([j['priority'] for j in bases.values()] + [1.0])
    choices = []
    for robot in state['robots'].values():
        choices.append((None, *(jid for jid, job in bases.items() if robot['operational']
                                and robot['assignment'] in (None, jid)
                                and math.dist(robot['position'], job['position']) <= state['radius'] + 1e-9)))
    best, score = set(), -1.0
    for vector in itertools.product(*choices):
        ids = [jid for jid in vector if jid is not None]
        if len(ids) != len(set(ids)):
            continue
        candidate = sum(bases[jid]['priority'] / scale for jid in ids)
        if candidate > score:
            best, score = set(ids), candidate
    return best


@dataclass(frozen=True)
class RewardWeights:
    progress: float = 1.0
    completion: float = 10.0
    response: float = 1.0
    failure: float = 10.0
    waiting: float = 1.0
    coverage: float = 2.0
    switching: float = .1
    max_abs_step: float = 100.0
    max_abs_episode: float = 1000.0

    def validate(self):
        if (any(not math.isfinite(v) or v < 0 for v in asdict(self).values())
                or self.max_abs_step <= 0 or self.max_abs_episode <= 0):
            raise ValueError('Reward weights must be finite nonnegative numbers')


def reward_terms(before, after, events, dt, horizon, max_jobs, covered_bases):
    tasks = [j for j in after['jobs'].values() if j['kind'] == 'task']
    # Fixed padding denominator, not the number of future tasks: no dependence on an unrevealed workload size.
    n = max_jobs
    progress = sum(max(0.0, j['work'] - before['jobs'][j['id']]['work']) / j['service'] for j in tasks) / n
    count = lambda name: sum(e.get('event') == name for e in events)
    bases = [j for j in before['jobs'].values() if j['kind'] == 'base']
    scale = max([j['priority'] for j in bases] + [1.0])
    total = sum(j['priority'] / scale for j in bases)
    gap_fraction = sum(j['priority'] / scale for j in bases if j['id'] not in covered_bases) / total if total else 0.0
    wait = sum(min(j['priority'], 10) / 10 for j in before['jobs'].values() if j['kind'] == 'task' and j['status'] == 'active') / n
    switches = sum(e.get('event') == 'assignment_changed' and e.get('reason') == 'policy'
                   and e.get('old') is not None and e.get('old') != e.get('new') for e in events)
    return {'progress': progress, 'completion': count('task_completed') / n,
            'response': count('task_response_started') / n, 'failure': -count('task_expired') / n,
            'waiting': -wait * dt / horizon, 'coverage': -gap_fraction * dt / horizon,
            'switching': -switches / len(before['robots'])}


def combine(terms, weights):
    weights.validate()
    contributions = {key: value * getattr(weights, key) for key, value in terms.items()}
    total = sum(contributions.values())
    if not all(math.isfinite(v) for v in [*terms.values(), *contributions.values(), total]):
        raise ValueError('NaN or infinity in reward')
    if abs(total) > weights.max_abs_step:
        raise ValueError('Reward scale exceeds declared safety threshold')
    return total, contributions
