"""Policy-independent physical cases, classified by deterministic feasibility witnesses."""
from dataclasses import asdict, replace
import random
from ..phase2.model import Config, RobotSpec as R, JobSpec as J, EventSpec as E
from ..phase2.engine import Environment
from ..phase2.model import Action
from ..phase2.policies import FixedRole
from ..phase3a.contracts import Settings
from ..phase3b2a.scenarios import digest

VERSION = 'phase4a-v1'
FAMILIES = ('A', 'B', 'C', 'C_risk', 'R_risk', 'F_incumbent', 'F_successor')


class Witness:
    """Three feasible scripts, not a temporal optimum or a deployed policy."""
    def __init__(self, mode):
        if mode not in ('fixed', 'immediate', 'ready'):
            raise ValueError(mode)
        self.mode = mode

    def reset(self, seed):
        self.fixed = FixedRole(); self.fixed.reset(seed); self.released = False

    def decide(self, obs):
        if self.mode == 'fixed':
            return self.fixed.decide(obs)
        robots = {r.id: r for r in obs.robots}; jobs = {j.id: j for j in obs.jobs}
        worker = 'C' if 'C' in robots else 'B'
        allocation = dict(self.fixed.decide(obs).assignments)
        task = jobs.get('task0')
        if task and robots['A'].operational and robots[worker].operational:
            ready = __import__('math').dist(robots[worker].position, jobs['base0'].position) <= obs.radius + 1e-9
            if self.mode == 'immediate' or ready:
                self.released = True
            allocation[worker] = 'base0'
            allocation['A'] = 'task0' if self.released else 'base0'
            return Action(tuple(sorted(allocation.items())),
                          ((worker, 'base0'),) if not self.released else (), ('feasibility_witness',))
        if self.released:
            allocation['A'] = None
            allocation[worker] = 'base0' if robots[worker].operational else None
        return Action(tuple(sorted(allocation.items())))


def witness(config, mode):
    rows = Environment(config, Witness(mode), 'witness_' + mode).run()
    return rows[-1]['truth']['jobs']['task0']['completed'], rows


def scenario(split, family, team, index):
    if split not in ('validation', 'diagnostic'):
        raise ValueError('Only Phase 4A development banks permitted; holdout rejected')
    if family not in FAMILIES or team not in (2, 4) or not 0 <= index < 10000:
        raise ValueError('Invalid case')
    seed = (41000000 if split == 'validation' else 42000000) + FAMILIES.index(family)*100000 + team*10000 + index
    # Match physical geometry across A/B/C/risk/fault families; vary stated factors only.
    rng = random.Random(f'{VERSION}:{split}:{team}:{index}')
    x_task = rng.uniform(2.5, 3.5); x_worker = rng.uniform(-2.5, -1.5)
    slow = rng.choice((.45, .5, .55)); y = 1.5 if team == 4 else 0.
    worker = 'C' if team == 4 else 'B'
    robots = [R('A', (0., y), initial_job='base0', base_role='guard'),
              R(worker, (x_worker, y), speed=slow)]
    jobs = [J('base0', (0., y), kind='base', spawn=0, deadline=32, service=0, fixed_owner='A'),
            J('task0', (x_task, y), spawn=2, deadline=30, service=2,
              priority=rng.choice((2., 4.)), fixed_owner=worker)]
    if team == 4:
        robots += [R('B', (0., -1.5), initial_job='base1', base_role='guard'), R('D', (-2., -1.5), speed=.5)]
        jobs += [J('base1', (0., -1.5), kind='base', spawn=0, deadline=32, service=0, fixed_owner='B'),
                 J('task1', (-2., -1.5), spawn=0, deadline=16, service=12, priority=2, fixed_owner='D'),
                 J('task2', (-2., -1.5), spawn=16, deadline=30, service=2, priority=2, fixed_owner='D')]
    config = Config(f'{VERSION}-{split}-{family}-{team}-{index}', tuple(sorted(robots, key=lambda r:r.id)),
                    tuple(sorted(jobs, key=lambda j:j.id)), seed=seed, horizon=32)
    times = {mode: witness(config, mode)[0] for mode in ('fixed', 'immediate', 'ready')}
    assert all(t is not None for t in times.values()), times
    assert times['immediate'] < times['ready'] < times['fixed'], times
    # Defined using kinematics before any learned or competing rule is evaluated.
    if family == 'A': deadline = min(30., times['fixed'] + 2)
    elif family in ('B', 'F_incumbent', 'F_successor'):
        deadline = min(times['fixed']-1, times['ready']+rng.choice((0,1)))
    else: deadline = min(times['ready']-1, times['immediate']+rng.choice((0,1)))
    risk = 8. if family == 'C_risk' else 1.
    events = []
    if family == 'R_risk': events = [E(3, 'risk', 'base0', 8)]
    if family == 'F_incumbent': events = [E(2, 'failure', 'A')]
    if family == 'F_successor': events = [E(3, 'failure', worker)]
    # Recovery is not guaranteed. Odd indices recover, even indices stay failed.
    if family.startswith('F_') and index % 2:
        events.append(E(10, 'recovery', 'A' if family == 'F_incumbent' else worker))
    config = replace(config, jobs=tuple(replace(j, deadline=deadline) if j.id=='task0'
                     else replace(j, priority=risk) if j.id=='base0' else j for j in config.jobs), events=tuple(events))
    config.validate()
    payload = dict(generator=VERSION, split=split, family=family, team=team, index=index, event_seed=seed,
                   config=asdict(config), settings=asdict(Settings(sensor_radius=100, sensor_noise=0, radio_delay=0)),
                   feasibility_completion_at=times,
                   response_limit_s={'base0': 2. if family in ('C_risk','R_risk') else 4., **({'base1':4.} if team==4 else {})})
    return {**payload, 'scenario_id':config.name + '-' + digest(payload)[:12], 'sha256':digest(payload)}


def bank(split, count):
    return [scenario(split, family, team, i) for family in FAMILIES for team in (2,4) for i in range(count)]
