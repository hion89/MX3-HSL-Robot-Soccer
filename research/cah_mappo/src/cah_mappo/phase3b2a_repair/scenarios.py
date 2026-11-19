"""Fresh split banks; fixed task schedule for the slot/geometry diagnostic ladder."""
from dataclasses import asdict
import random
from ..phase2.model import Config, RobotSpec as R, JobSpec as J
from ..phase3a.contracts import Settings
from ..phase3b2a.scenarios import digest

BANKS = {'train': 400000, 'validation': 500000, 'development': 700000, 'final_holdout': 900000}
FAMILIES = ('fixed', 'positions', 'priority_deadline')


def generate(family, split, index):
    if family not in FAMILIES or split not in BANKS or not 0 <= index < 1000:raise ValueError('Invalid scenario bank')
    seed = BANKS[split] + FAMILIES.index(family) * 10000 + index
    rng = random.Random(f'repair-generator-v1:{seed}')
    jitter = 0. if family == 'fixed' else .25
    robots = []
    for i, pos in enumerate(((0, 1.5), (0, -1.5), (-2, 1.5), (-2, -1.5))):
        scale = min(jitter, .05) if i < 2 else jitter
        xy = tuple(v + rng.uniform(-scale, scale) for v in pos)
        robots.append(R(chr(65+i), xy, initial_job=f'base{i}' if i < 2 else None,
                        base_role='guard' if i < 2 else 'worker'))
    jobs = [J(f'base{i}', pos, kind='base', spawn=0, deadline=32, service=0, fixed_owner=chr(65+i))
            for i, pos in enumerate(((0, 1.5), (0, -1.5)))]
    for i, pos in enumerate(((3, 1), (3, 0), (3, -1))):
        xy = tuple(v + rng.uniform(-jitter, jitter) for v in pos)
        priority = rng.choice((1., 2., 4.)) if family == 'priority_deadline' else 2.
        deadline = rng.choice((20., 24., 28.)) if family == 'priority_deadline' else 24.
        jobs.append(J(f'task{i}', xy, priority=priority, spawn=2+i*2, deadline=deadline,
                      service=2, fixed_owner=chr(67+i%2)))
    config = Config(f'repair-{family}-{split}-{index}', tuple(robots), tuple(jobs), seed=seed, horizon=32)
    config.validate()
    payload = {'generator': 'repair-generator-v1', 'family': family, 'split': split, 'index': index,
               'event_seed': seed, 'config': asdict(config), 'settings': asdict(Settings(sensor_radius=100, sensor_noise=0, radio_delay=0))}
    return {**payload, 'scenario_id': f'repair-{family}-{split}-{index:03d}-{digest(payload)[:10]}', 'sha256': digest(payload)}
