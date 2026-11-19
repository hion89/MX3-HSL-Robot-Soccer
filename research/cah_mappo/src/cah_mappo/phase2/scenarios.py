"""Paired deterministic scenarios: policy code never generates external events."""
from .model import Config, RobotSpec as R, JobSpec as J, EventSpec as E

SCENARIOS = ('single', 'simultaneous', 'priorities', 'slow_robot', 'owner_failure',
             'urgent', 'rising_risk', 'communication_delay')


def scenario(name='single', seed=0, jitter=0.0):
    if name not in SCENARIOS:
        raise ValueError(name)
    robots = [R('A', (0, 0), initial_job='base', base_role='guard'), R('B', (-4, 0))]
    jobs = [J('base', (0, 0), kind='base', spawn=0, deadline=24, service=0, fixed_owner='A'),
            J('task1', (4, 0), priority=2, spawn=2, deadline=18, service=2, fixed_owner='B')]
    events = []
    delay = 0.0
    if name in ('simultaneous', 'priorities', 'slow_robot'):
        robots += [R('C', (0, 4)), R('D', (-4, 4))]
        jobs += [J('task2', (4, 4), priority=2, spawn=2, deadline=18, fixed_owner='C'),
                 J('task3', (-4, 4), priority=2, spawn=2, deadline=18, fixed_owner='D')]
    if name == 'priorities':
        jobs = [jobs[0], J('task1', (4, 0), priority=8, spawn=2, deadline=9, fixed_owner='B'),
                J('task2', (4, 4), priority=1, spawn=2, deadline=18, fixed_owner='C'),
                J('task3', (-4, 4), priority=3, spawn=2, deadline=18, fixed_owner='D')]
    if name == 'slow_robot':
        events = [E(3, 'speed', 'C', .25)]
    if name == 'owner_failure':
        events = [E(1, 'failure', 'A')]
    if name == 'urgent':
        jobs[1] = J('task1', (4, 0), priority=8, spawn=2, deadline=8, fixed_owner='B')
    if name == 'rising_risk':
        events = [E(3, 'risk', 'base', 10)]
    if name == 'communication_delay':
        delay = 1.0
    c = Config(name, tuple(robots), tuple(jobs), tuple(events), seed=seed, jitter=jitter, observation_delay=delay)
    c.validate()
    return c
