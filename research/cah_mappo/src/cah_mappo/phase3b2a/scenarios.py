"""Policy-independent, split-disjoint scenario manifests (stdlib only)."""
from dataclasses import asdict
import hashlib
import json
import random

from ..phase2.model import Config, RobotSpec as R, JobSpec as J, EventSpec as E, config_from_dict
from ..phase3a.contracts import Settings
from ..phase3a.env import ParallelCore
from ..phase3a.rewards import RewardWeights

STAGES = ('A', 'B2', 'B3', 'C', 'D')
SPLITS = {'train': 10000, 'validation': 20000, 'test': 30000}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def generate(stage, split, index):
    if stage not in STAGES or split not in SPLITS or not isinstance(index, int) or not 0 <= index < 1000:
        raise ValueError('Unknown stage/split or index outside 0..999')
    seed = SPLITS[split] + STAGES.index(stage) * 1000 + index
    rng = random.Random(f'scenario-generator-v1:{seed}')
    n = 2 if stage in ('A', 'B2') else (3 if stage == 'B3' else 4)
    horizon = 32.0
    base_positions = [(0., 0.)] if n < 4 else [(0., 1.5), (0., -1.5)]
    robots = []
    jobs = []
    for i, xy in enumerate(base_positions):
        name = chr(65 + i); jid = f'base{i}'
        robots.append(R(name, tuple(v + rng.uniform(-.08, .08) for v in xy), initial_job=jid, base_role='guard'))
        jobs.append(J(jid, xy, kind='base', spawn=0, deadline=horizon, service=0,
                      priority=rng.choice((1., 2., 4.)), fixed_owner=name))
    for i in range(len(robots), n):
        robots.append(R(chr(65 + i), (rng.uniform(-4., -1.5), rng.uniform(-2., 2.))))
    task_count = 1 if stage == 'A' else (2 if n < 4 else 3)
    for i in range(task_count):
        jobs.append(J(f'task{i}', (rng.uniform(2., 4.5), rng.uniform(-2., 2.)),
                      spawn=rng.choice((1., 2., 3.)) + i * 2,
                      deadline=rng.choice((20., 24., 28.)), service=2,
                      priority=rng.choice((1., 2., 4.)), fixed_owner=chr(65 + (i + len(base_positions)) % n)))
    events = []
    if stage == 'D':
        robots = [R(**{**asdict(r), 'speed': rng.choice((.55, .8, 1.2))}) for r in robots]
        events = [E(5, 'failure', 'C'), E(10, 'recovery', 'C'), E(7, 'speed', 'D', .4),
                  E(6, 'risk', 'base0', 8)]
    config = Config(f'{stage}-{split}-{index:03d}', tuple(robots), tuple(jobs), tuple(events),
                    seed=seed, horizon=horizon, jitter=0, observation_delay=0)
    config.validate()
    settings = Settings(sensor_radius=10 if stage != 'D' else 2.5,
                        sensor_noise=0 if stage != 'D' else .15,
                        radio_delay=0 if stage != 'D' else 2., cache_ttl=4.)
    settings.validate(config)
    payload = {'generator': 'scenario-generator-v1', 'stage': stage, 'split': split, 'index': index,
               'event_seed': seed, 'config': asdict(config), 'settings': asdict(settings),
               'episode_rule': 'fixed_horizon_32_no_early_workload_end'}
    return {**payload, 'scenario_id': f'{stage}-{split}-{index:03d}-{digest(payload)[:12]}', 'sha256': digest(payload)}


def make_env(spec, coverage_weight=2.):
    config = config_from_dict(spec['config'])
    settings_data = {**spec['settings'], 'roster': tuple(spec['settings']['roster'])}
    env = ParallelCore(config, Settings(**settings_data), RewardWeights(coverage=coverage_weight))
    obs, _ = env.reset(spec['event_seed'], {'terminate_on_tasks_resolved': False})
    env.engine.rows[0]['phase3b2a_scenario'] = {k: spec[k] for k in ('scenario_id', 'sha256', 'split', 'stage')}
    return env, obs


def build_manifest(train=32, validation=12, test=20):
    counts = {'train': train, 'validation': validation, 'test': test}
    return {'generator': 'scenario-generator-v1', 'counts': counts,
            'scenarios': {stage: {split: [generate(stage, split, i) for i in range(count)]
                                  for split, count in counts.items()} for stage in STAGES}}
