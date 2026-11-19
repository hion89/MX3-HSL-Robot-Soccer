"""Deterministic discrete-time 2D simulator. All times are seconds.

Boundary order: service completion, fault/recovery, spawn, delivered messages,
local READY detection, zero-delay messages, policy, interval snapshot, integration.
Movement and service are exclusive. Physical arrival is sampled at boundaries.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import random


POLICIES = ('immediate', 'ready', 'late', 'fixed')
SCENARIOS = ('normal', 'urgent', 'successor_failure', 'communication_delay', 'temporary_failure')


@dataclass(frozen=True)
class Config:
    scenario: str = 'normal'
    policy: str = 'ready'
    seed: int = 0
    dt: float = 1.0
    horizon: float = 16.0
    spawn: float = 2.0
    deadline: float = 14.0       # absolute, completion at deadline is allowed
    service: float = 2.0
    speed: float = 1.0
    radius: float = 0.25
    delay: float = 0.0
    failure: float | None = None
    recovery: float | None = None
    late_wait: float = 2.0
    jitter: float = 0.0
    observation_noise: float = 0.0
    base: tuple = (0.0, 0.0)
    task: tuple = (4.0, 0.0)
    initial_a: tuple = (0.0, 0.0)
    initial_b: tuple = (-4.0, 0.0)

    def validate(self):
        if self.scenario not in SCENARIOS or self.policy not in POLICIES:
            raise ValueError('Unknown scenario or policy')
        values = (self.dt, self.horizon, self.spawn, self.deadline, self.service,
                  self.speed, self.radius, self.delay, self.late_wait, self.jitter,
                  self.observation_noise)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite configuration')
        if self.dt <= 0 or self.horizon <= 0 or self.speed <= 0 or self.service <= 0:
            raise ValueError('dt, horizon, speed and service must be positive')
        if min(self.radius, self.delay, self.spawn, self.late_wait, self.jitter, self.observation_noise) < 0:
            raise ValueError('Negative configuration')
        if not self.spawn < self.deadline <= self.horizon:
            raise ValueError('Require spawn < deadline <= horizon')
        times = [self.horizon, self.spawn, self.deadline, self.service, self.delay, self.late_wait]
        times += [v for v in (self.failure, self.recovery) if v is not None]
        if any(not math.isfinite(v) or v < 0 or abs(v / self.dt - round(v / self.dt)) > 1e-8 for v in times):
            raise ValueError('Event times must be finite nonnegative multiples of dt')
        if self.failure is not None and self.failure > self.horizon:
            raise ValueError('Fault must lie within horizon')
        if self.recovery is not None and (self.failure is None or not self.failure < self.recovery <= self.horizon):
            raise ValueError('Invalid recovery')
        for xy in (self.base, self.task, self.initial_a, self.initial_b):
            if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
                raise ValueError('Expected finite 2D coordinates')


def scenario_config(name='normal', policy='ready', seed=0, **overrides):
    options = {'scenario': name, 'policy': policy, 'seed': seed}
    if name == 'urgent':
        options['deadline'] = 9.0
    elif name == 'successor_failure':
        options['failure'] = 4.0
    elif name == 'communication_delay':
        options['delay'] = 1.0
    elif name == 'temporary_failure':
        options.update(failure=4.0, recovery=7.0)
    options.update(overrides)
    c = Config(**options)
    c.validate()
    return c


@dataclass
class Robot:
    name: str
    position: list
    base_role: str
    active_assignment: str | None
    operational: bool = True
    goal: list | None = None
    capacity: int = 1


def can_respond(robot, site, position, radius):
    """Capacity-feasible readiness, independent of declared ownership.

    A robot committed to a different service cannot also cover this site.
    Idle robots may physically cover a nearby base. No assignment grants arrival.
    """
    return (robot['operational'] and robot['capacity'] >= 1
            and robot['active_assignment'] in (None, site)
            and math.dist(robot['position'], position) <= radius + 1e-9)


class Simulator:
    def __init__(self, config):
        config.validate()
        self.c = config
        rng = random.Random(config.seed)
        self.task_position = [v + rng.uniform(-config.jitter, config.jitter) for v in config.task]
        b = [v + rng.uniform(-config.jitter, config.jitter) for v in config.initial_b]
        self.robots = {'A': Robot('A', list(config.initial_a), 'base_guard', 'base'),
                       'B': Robot('B', b, 'reserve', None)}
        self.base_owner = 'A'
        self.task_owner = None
        self.task_status = 'not_spawned'
        self.task_response = None
        self.work = 0.0
        self.requested = False
        self.released = False
        self.ready_sent = False
        self.ack_at = None
        self.inbox = []
        self.sequence = 0
        self.rows = []
        self.observed = {'A': [], 'B': []}

    def event(self, t, name, **data):
        self.rows.append({'type': 'event', 'time': t, 'event': name, **data})

    def send(self, t, kind, sender, recipient):
        self.sequence += 1
        msg = {'id': self.sequence, 'handover_id': 1, 'kind': kind,
               'sender': sender, 'recipient': recipient, 'sent': t,
               'due': round(t + self.c.delay, 10)}
        self.inbox.append(msg)
        self.event(t, 'message_sent', message=copy.deepcopy(msg))

    def deliver(self, t):
        due = [m for m in self.inbox if m['due'] <= t + 1e-9]
        self.inbox = [m for m in self.inbox if m not in due]
        for msg in due:
            recipient = self.robots[msg['recipient']]
            if not recipient.operational:
                self.event(t, 'message_dropped', message=msg, reason='recipient_failed')
                continue
            self.observed[msg['recipient']].append(copy.deepcopy(msg))
            self.event(t, 'message_received', message=msg, age=round(t - msg['sent'], 10))
            if msg['kind'] == 'REQUEST':
                recipient.active_assignment = 'base'
                recipient.goal = list(self.c.base)
                self.event(t, 'prepare_started', robot='B', handover_id=1)
            elif msg['kind'] == 'READY':
                self.ack_at = t

    def boundary(self, t):
        if self.task_status == 'active' and self.work >= self.c.service - 1e-9:
            self.task_status = 'completed'
            self.event(t, 'task_completed', task='dynamic', robot='A')
            self.robots['A'].active_assignment = None
        b = self.robots['B']
        if self.c.failure == t:
            b.operational = False
            self.ready_sent = False
            self.event(t, 'robot_failed', robot='B')
        if self.c.recovery == t:
            b.operational = True
            self.event(t, 'robot_recovered', robot='B')
        if self.c.spawn == t:
            self.task_status = 'active'
            self.event(t, 'task_spawned', task='dynamic', deadline=self.c.deadline,
                       position=list(self.task_position), required_service=self.c.service)
        self.deliver(t)
        local_b = asdict(b)
        local_b['position'] = self.observations(round(t / self.c.dt))['B']['self_position']
        if (self.requested and not self.ready_sent and b.active_assignment == 'base'
                and can_respond(local_b, 'base', self.c.base, self.c.radius)):
            self.ready_sent = True
            self.event(t, 'successor_ready', robot='B', handover_id=1)
            self.send(t, 'READY', 'B', 'A')
        self.deliver(t)

    def decide(self, t):
        """Scripted policies use their own state and delivered READY, not peer GT."""
        if self.task_status != 'active':
            return
        if self.c.policy != 'fixed' and not self.requested:
            self.requested = True
            self.event(t, 'handover_requested', handover_id=1, old='A', new='B', site='base')
            self.send(t, 'REQUEST', 'A', 'B')
            self.deliver(t)
        wait = self.c.late_wait if self.c.policy == 'late' else 0.0
        should_release = (self.c.policy == 'immediate'
                          or (self.c.policy in ('ready', 'late') and self.ack_at is not None
                              and t >= self.ack_at + wait - 1e-9))
        if should_release and not self.released:
            self.released = True
            self.base_owner = 'B'  # declared owner may be far away, delayed or failed
            self.task_owner = 'A'
            self.robots['A'].active_assignment = 'dynamic'
            self.robots['A'].goal = list(self.task_position)
            self.event(t, 'responsibility_released', handover_id=1, old='A', new='B', site='base')
            self.event(t, 'task_assigned', task='dynamic', robot='A')

    def state(self):
        return {'robots': {k: asdict(v) for k, v in self.robots.items()},
                'base': {'position': list(self.c.base), 'owner': self.base_owner},
                'task': {'position': list(self.task_position), 'owner': self.task_owner,
                         'status': self.task_status, 'work': self.work}, 'radius': self.c.radius}

    def observations(self, tick):
        result = {}
        for name, robot in self.robots.items():
            # Separate, tick-indexed noise stream; policy branching cannot alter it.
            rng = random.Random(f'{self.c.seed}:observation:{tick}:{name}')
            result[name] = {'self_position': [v + rng.uniform(-self.c.observation_noise,
                                                             self.c.observation_noise)
                                               for v in robot.position],
                            'base_role': robot.base_role, 'active_assignment': robot.active_assignment,
                            'operational': robot.operational,
                            'received_messages': copy.deepcopy(self.observed[name])}
        return result

    def run(self):
        if self.rows:
            raise RuntimeError('A simulator instance may only be run once')
        source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.rows.append({'type': 'header', 'schema_version': 1, 'config': asdict(self.c),
                          'code_sha256': source_hash,
                          'implementation_sha256': {
                              'core.py': source_hash,
                              'metrics.py': hashlib.sha256(Path(__file__).with_name('metrics.py').read_bytes()).hexdigest(),
                              'run.py': hashlib.sha256((Path(__file__).parents[2] / 'run.py').read_bytes()).hexdigest()},
                          'exogenous_plan': {'task_position': self.task_position,
                                             'initial_b': self.robots['B'].position.copy()}})
        steps = round(self.c.horizon / self.c.dt)
        for tick in range(steps + 1):
            t = round(tick * self.c.dt, 10)
            self.boundary(t)
            self.decide(t)
            self.deliver(t)
            a = self.robots['A']
            if (self.task_status == 'active' and t < self.c.deadline and tick < steps
                    and can_respond(asdict(a), 'dynamic', self.task_position, self.c.radius)
                    and a.active_assignment == 'dynamic'):
                if self.task_response is None:
                    self.task_response = t
                    self.event(t, 'task_response_started', task='dynamic', robot='A')
            # Completion was handled first, so completion exactly at deadline wins.
            if self.task_status == 'active' and t >= self.c.deadline:
                self.task_status = 'expired'
                self.event(t, 'task_expired', task='dynamic')
                a.active_assignment, a.goal = None, None
            if tick == steps:
                self.rows.append({'type': 'terminal', 'time': t, 'truth': self.state()})
                break
            end = round(t + self.c.dt, 10)
            self.rows.append({'type': 'interval', 'tick': tick, 'start': t, 'end': end,
                              'truth': self.state(), 'observations': self.observations(tick)})
            if (self.task_status == 'active' and a.active_assignment == 'dynamic'
                    and can_respond(asdict(a), 'dynamic', self.task_position, self.c.radius)):
                self.work = round(self.work + self.c.dt, 10)
            for robot in self.robots.values():
                if not robot.operational or robot.goal is None:
                    continue
                if (robot.name == 'A' and self.task_status == 'active'
                        and robot.active_assignment == 'dynamic'
                        and can_respond(asdict(robot), 'dynamic', self.task_position, self.c.radius)):
                    continue  # Actual service consumes capacity and stops locomotion.
                distance = math.dist(robot.position, robot.goal)
                travel = min(distance, self.c.speed * self.c.dt)
                if distance:
                    robot.position = [round(x + (y - x) * travel / distance, 10)
                                      for x, y in zip(robot.position, robot.goal)]
        return self.rows


def write_jsonl(rows, path):
    """Exclusive creation protects previous experiments from overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n')
