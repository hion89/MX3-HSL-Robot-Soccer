"""Immutable public policy contracts and scenario specifications (stdlib only)."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class RobotSpec:
    id: str
    position: tuple
    speed: float = 1.0
    turn_speed: float = math.pi
    heading: float = 0.0
    initial_job: str | None = None
    base_role: str = 'worker'
    operational: bool = True
    capacity: int = 1


@dataclass(frozen=True)
class JobSpec:
    id: str
    position: tuple
    kind: str = 'task'  # persistent base or finite task
    priority: float = 1.0
    spawn: float = 2.0
    deadline: float = 18.0
    service: float = 2.0
    fixed_owner: str | None = None


@dataclass(frozen=True)
class EventSpec:
    time: float
    kind: str  # failure, recovery, speed, risk
    target: str
    value: float = 0.0


@dataclass(frozen=True)
class Config:
    name: str
    robots: tuple[RobotSpec, ...]
    jobs: tuple[JobSpec, ...]
    events: tuple[EventSpec, ...] = ()
    dt: float = 1.0
    horizon: float = 24.0
    radius: float = 0.25
    seed: int = 0
    jitter: float = 0.0
    observation_delay: float = 0.0
    switching_cost: float = 2.0
    unassigned_cost: float = 50.0

    def validate(self):
        if not 2 <= len(self.robots) <= 4 or not self.jobs:
            raise ValueError('Require 2..4 robots and at least one job')
        robot_ids, job_ids = [r.id for r in self.robots], [j.id for j in self.jobs]
        if len(set(robot_ids)) != len(robot_ids) or len(set(job_ids)) != len(job_ids):
            raise ValueError('Duplicate IDs')
        numbers = [self.dt, self.horizon, self.radius, self.jitter, self.observation_delay,
                   self.switching_cost, self.unassigned_cost]
        if not all(math.isfinite(v) for v in numbers) or self.dt <= 0 or self.horizon <= 0:
            raise ValueError('Invalid numeric configuration')
        if min(self.radius, self.jitter, self.observation_delay, self.switching_cost) < 0 or self.unassigned_cost <= 0:
            raise ValueError('Negative parameter')
        times = [self.horizon, self.observation_delay]
        initial = []
        for r in self.robots:
            if (r.capacity != 1 or not math.isfinite(r.speed) or r.speed <= 0
                    or not math.isfinite(r.turn_speed) or r.turn_speed <= 0 or not math.isfinite(r.heading)):
                raise ValueError('This minimal engine supports capacity=1 and positive speeds')
            if r.base_role not in ('guard', 'worker'):
                raise ValueError('Base role must be guard or worker')
            if r.initial_job is not None:
                if r.initial_job not in job_ids:
                    raise ValueError('Unknown initial job')
                initial.append(r.initial_job)
        if len(set(initial)) != len(initial):
            raise ValueError('Initial assignments must be unique')
        for j in self.jobs:
            if j.kind not in ('task', 'base') or not math.isfinite(j.priority) or j.priority <= 0:
                raise ValueError('Invalid job')
            if j.fixed_owner is not None and j.fixed_owner not in robot_ids:
                raise ValueError('Unknown fixed owner')
            if j.kind == 'base':
                if j.spawn != 0:
                    raise ValueError('Base must exist at t=0')
            elif not 0 <= j.spawn < j.deadline <= self.horizon or j.service <= 0 or not math.isfinite(j.service):
                raise ValueError('Invalid task time')
            times += [j.spawn, j.deadline, j.service]
        for r in self.robots:
            if r.initial_job is not None and next(j for j in self.jobs if j.id == r.initial_job).kind != 'base':
                raise ValueError('Initial jobs must be persistent bases')
        for e in self.events:
            if e.kind not in ('failure', 'recovery', 'speed', 'risk') or not 0 <= e.time <= self.horizon:
                raise ValueError('Invalid event')
            if e.kind == 'risk':
                if e.target not in job_ids or next(j for j in self.jobs if j.id == e.target).kind != 'base':
                    raise ValueError('Risk target must be a base')
            elif e.target not in robot_ids:
                raise ValueError('Unknown event robot')
            if not math.isfinite(e.value) or e.kind in ('speed', 'risk') and e.value <= 0:
                raise ValueError('Invalid event value')
            times.append(e.time)
        for xy in [r.position for r in self.robots] + [j.position for j in self.jobs]:
            if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
                raise ValueError('Invalid 2D position')
        if any(not math.isfinite(v) or v < 0 or abs(v / self.dt - round(v / self.dt)) > 1e-8 for v in times):
            raise ValueError('Times must be nonnegative multiples of dt')


@dataclass(frozen=True)
class RobotView:
    id: str
    position: tuple
    heading: float
    speed: float
    turn_speed: float
    operational: bool
    assignment: str | None
    initial_job: str | None
    base_role: str
    capacity: int = 1


@dataclass(frozen=True)
class JobView:
    id: str
    position: tuple
    kind: str
    priority: float
    spawn: float
    deadline: float
    remaining: float
    fixed_owner: str | None
    owner: str | None


@dataclass(frozen=True)
class Observation:
    time: float
    observed_at: float
    dt: float
    radius: float
    robots: tuple[RobotView, ...]
    jobs: tuple[JobView, ...]


@dataclass(frozen=True)
class Action:
    assignments: tuple[tuple[str, str | None], ...]
    # Intentional dual occupancy during prepare; never dual task service.
    preparations: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()
    allocation_cost: float | None = None


def eta(robot, job, obs):
    """Translation + turning ETA, rounded to simulation ticks; not a learned model."""
    if not robot.operational:
        return math.inf
    d = math.dist(robot.position, job.position)
    if d <= obs.radius + 1e-9:
        return 0.0
    angle = math.atan2(job.position[1] - robot.position[1], job.position[0] - robot.position[0])
    rotation = abs(math.remainder(angle - robot.heading, 2 * math.pi)) / robot.turn_speed
    duration = rotation + max(0.0, d - obs.radius) / robot.speed
    return math.ceil(duration / obs.dt - 1e-9) * obs.dt


def config_from_dict(data):
    data = dict(data)
    data['robots'] = tuple(RobotSpec(**{**r, 'position': tuple(r['position'])}) for r in data['robots'])
    data['jobs'] = tuple(JobSpec(**{**j, 'position': tuple(j['position'])}) for j in data['jobs'])
    data['events'] = tuple(EventSpec(**e) for e in data.get('events', []))
    c = Config(**data)
    c.validate()
    return c
