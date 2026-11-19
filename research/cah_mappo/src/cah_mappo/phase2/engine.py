"""Multi-robot extension, isolated from the frozen Phase 1 simulator."""
from dataclasses import asdict, replace
import copy
import hashlib
import math
from pathlib import Path
import random

from .model import Observation, RobotView, JobView, eta


class Environment:
    def __init__(self, config, policy, policy_name='custom'):
        config.validate()
        self.c, self.policy, self.policy_name = config, policy, policy_name
        self.rows, self.history = [], []
        self.handover_id = 0
        rng = random.Random(f'exogenous:{config.seed}')
        self.robots = {}
        for spec in sorted(config.robots, key=lambda r: r.id):
            r = asdict(spec)
            r['position'] = [round(x + rng.uniform(-config.jitter, config.jitter), 10) for x in spec.position]
            r['assignment'] = spec.initial_job if spec.operational else None
            self.robots[r['id']] = r
        self.jobs = {}
        for spec in sorted(config.jobs, key=lambda j: j.id):
            job = asdict(spec)
            job['position'] = [round(x + rng.uniform(-config.jitter, config.jitter), 10) for x in spec.position] if spec.kind == 'task' else list(spec.position)
            job.update(status='active' if spec.kind == 'base' else 'not_spawned', work=0.0,
                       response=None, completed=None, owner=next((r['id'] for r in self.robots.values() if r['assignment'] == spec.id), None))
            self.jobs[job['id']] = job
        self.plan = {'robots': copy.deepcopy(self.robots), 'jobs': copy.deepcopy(self.jobs),
                     'events': [asdict(e) for e in config.events]}
        policy.reset(config.seed)

    def event(self, t, name, **kwargs):
        self.rows.append({'type': 'event', 'time': t, 'event': name, **kwargs})

    def boundary(self, t):
        for job in self.jobs.values():
            if job['kind'] == 'task' and job['status'] == 'active' and job['work'] >= job['service'] - 1e-8:
                job.update(status='completed', completed=t)
                self.event(t, 'task_completed', job=job['id'])
        for event in self.c.events:
            if abs(event.time - t) > 1e-8:
                continue
            if event.kind == 'risk':
                self.jobs[event.target]['priority'] = event.value
            else:
                robot = self.robots[event.target]
                if event.kind == 'failure':
                    robot['operational'] = False
                elif event.kind == 'recovery':
                    robot['operational'] = True
                elif event.kind == 'speed':
                    robot['speed'] = event.value
            self.event(t, event.kind, target=event.target, value=event.value)
        for job in self.jobs.values():
            if job['kind'] != 'task':
                continue
            if job['status'] == 'not_spawned' and t >= job['spawn']:
                job['status'] = 'active'
                self.event(t, 'task_spawned', job=job['id'], deadline=job['deadline'], priority=job['priority'])
            if job['status'] == 'active' and t >= job['deadline']:
                job['status'] = 'expired'
                self.event(t, 'task_expired', job=job['id'])
        for r in self.robots.values():
            if not r['operational'] or r['assignment'] is not None and self.jobs[r['assignment']]['status'] != 'active':
                if r['assignment'] is not None:
                    old, r['assignment'] = r['assignment'], None
                    self.event(t, 'assignment_changed', robot=r['id'], old=old, new=None, reason='unavailable_or_finished')

    def observe(self, t):
        robots = tuple(RobotView(r['id'], tuple(r['position']), r['heading'], r['speed'], r['turn_speed'],
                                 r['operational'], r['assignment'], r['initial_job'], r['base_role'], r['capacity'])
                       for r in sorted(self.robots.values(), key=lambda r: r['id']))
        jobs = tuple(JobView(j['id'], tuple(j['position']), j['kind'], j['priority'], j['spawn'], j['deadline'],
                             max(0.0, j['service'] - j['work']) if j['kind'] == 'task' else 0.0,
                             j['fixed_owner'], j['owner'])
                     for j in sorted(self.jobs.values(), key=lambda j: j['id']) if j['status'] == 'active')
        now = Observation(t, t, self.c.dt, self.c.radius, robots, jobs)
        self.history.append(now)
        observed = next((h for h in reversed(self.history) if h.observed_at <= t - self.c.observation_delay + 1e-8), self.history[0])
        return replace(observed, time=t)

    def apply(self, t, action):
        raw = dict(action.assignments)
        if len(raw) != len(action.assignments) or any(k not in self.robots for k in raw):
            raise ValueError('Action robot IDs must be unique and known')
        accepted = {}
        rejected = []
        for name, robot in sorted(self.robots.items()):
            jid = raw.get(name)
            if jid is not None and (jid not in self.jobs or self.jobs[jid]['status'] != 'active' or not robot['operational']):
                rejected.append({'robot': name, 'job': jid, 'reason': 'inactive_or_unavailable'})
                jid = None
            accepted[name] = jid
        prep = set()
        for name, jid in action.preparations:
            job = self.jobs.get(jid)
            if job is not None and job['kind'] == 'base' and accepted.get(name) == jid:
                old = job['owner']
                if old != name and old is not None and accepted.get(old) == jid:
                    prep.add((name, jid))
        for jid in self.jobs:
            names = sorted(name for name, target in accepted.items() if target == jid)
            allowed = names[:1]
            if self.jobs[jid]['kind'] == 'base':
                old = self.jobs[jid]['owner']
                if old in names:
                    allowed = [old]
                    successor = [n for n in names if (n, jid) in prep]
                    if successor:
                        allowed.append(successor[0])
            for name in names:
                if name not in allowed:
                    accepted[name] = None
                    rejected.append({'robot': name, 'job': jid, 'reason': 'duplicate_assignment'})
        prep = sorted((name, jid) for name, jid in prep if accepted.get(name) == jid)
        for name, robot in sorted(self.robots.items()):
            if robot['assignment'] != accepted[name]:
                self.event(t, 'assignment_changed', robot=name, old=robot['assignment'], new=accepted[name], reason='policy')
            robot['assignment'] = accepted[name]
        self.update_owners(t)
        for rejection in rejected:
            self.event(t, 'action_rejected', **rejection)
        return {'requested': raw, 'accepted': accepted, 'preparations': [list(p) for p in prep],
                'notes': list(action.notes), 'allocation_cost': action.allocation_cost, 'rejections': rejected}

    def update_owners(self, t):
        for job in self.jobs.values():
            names = sorted(r['id'] for r in self.robots.values() if r['operational'] and r['assignment'] == job['id'])
            if job['kind'] != 'base':
                job['owner'] = names[0] if names else None
                continue
            old = job['owner']
            new = old if old in names else (names[0] if names else None)
            if new != old:
                if old is not None:
                    self.handover_id += 1
                    self.event(t, 'responsibility_released', job=job['id'], old=old, new=new, handover_id=self.handover_id)
                else:
                    self.event(t, 'responsibility_acquired', job=job['id'], new=new)
            job['owner'] = new

    def truth(self):
        return {'robots': copy.deepcopy(self.robots), 'jobs': copy.deepcopy(self.jobs), 'radius': self.c.radius}

    def integrate(self, t):
        for r in self.robots.values():
            job = self.jobs.get(r['assignment'])
            if not r['operational'] or job is None or job['status'] != 'active':
                continue
            distance = math.dist(r['position'], job['position'])
            if distance <= self.c.radius + 1e-9:
                if job['kind'] == 'task':
                    job['work'] = round(job['work'] + self.c.dt, 10)
                continue
            target_heading = math.atan2(job['position'][1] - r['position'][1], job['position'][0] - r['position'][0])
            error = math.remainder(target_heading - r['heading'], 2 * math.pi)
            rotation = min(abs(error), r['turn_speed'] * self.c.dt)
            r['heading'] = math.remainder(r['heading'] + math.copysign(rotation, error), 2 * math.pi)
            time_left = max(0.0, self.c.dt - rotation / r['turn_speed'])
            travel = min(distance, r['speed'] * time_left)
            r['position'] = [round(x + (y - x) * travel / distance, 10) for x, y in zip(r['position'], job['position'])]

    def run(self):
        if self.rows:
            raise RuntimeError('Run each environment only once')
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
        self.rows.append({'type': 'header', 'schema_version': 2, 'policy': self.policy_name,
                          'config': asdict(self.c), 'exogenous_plan': self.plan, 'implementation_sha256': hashes})
        steps = round(self.c.horizon / self.c.dt)
        for tick in range(steps + 1):
            t = round(tick * self.c.dt, 10)
            self.boundary(t)
            if tick == steps:
                self.update_owners(t)
                self.rows.append({'type': 'terminal', 'time': t, 'truth': self.truth()})
                break
            obs = self.observe(t)
            action = self.policy.decide(obs)
            selection = self.apply(t, action)
            self.event(t, 'policy_selection', observation=asdict(obs), action=selection)
            for job in self.jobs.values():
                if job['kind'] == 'task' and job['status'] == 'active' and job['response'] is None:
                    ready = [r for r in self.robots.values() if r['operational'] and r['assignment'] == job['id']
                             and math.dist(r['position'], job['position']) <= self.c.radius + 1e-9]
                    if ready:
                        job['response'] = t
                        self.event(t, 'task_response_started', job=job['id'], robot=ready[0]['id'])
            self.rows.append({'type': 'interval', 'tick': tick, 'start': t, 'end': round(t + self.c.dt, 10),
                              'truth': self.truth(), 'selection': selection})
            self.integrate(t)
        return self.rows
