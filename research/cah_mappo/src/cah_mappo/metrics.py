"""Independent evaluator: consumes only schema-1 raw logs, never simulator objects."""
import json
import math
from pathlib import Path
from statistics import mean


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate(rows):
    if not rows or rows[0].get('type') != 'header' or rows[0].get('schema_version') != 1:
        raise ValueError('Missing supported header')
    if rows[-1].get('type') != 'terminal':
        raise ValueError('Incomplete log: terminal required')
    config = rows[0]['config']
    horizon = config['horizon']
    intervals = [r for r in rows if r['type'] == 'interval']
    events = [r for r in rows if r['type'] == 'event']
    end = 0.0
    previous_tick = -1
    coverage = []
    for row in intervals:
        if (row['tick'] != previous_tick + 1 or abs(row['start'] - end) > 1e-8
                or row['end'] <= row['start'] or row['end'] > horizon + 1e-8):
            raise ValueError('Intervals must be contiguous and ordered')
        previous_tick, end = row['tick'], row['end']
        state = row['truth']
        ready = []
        # Deliberately recomputed from raw truth, not from policy readiness flags.
        for name, robot in state['robots'].items():
            if robot['capacity'] != 1 or robot['active_assignment'] not in (None, 'base', 'dynamic'):
                raise ValueError('Unsupported capacity or multiple simultaneous assignments')
            if (robot['operational'] and robot['active_assignment'] in (None, 'base')
                    and math.dist(robot['position'], state['base']['position']) <= state['radius'] + 1e-9):
                ready.append(name)
        coverage.append((row['start'], row['end'], ready, state['base']['owner']))
    if not intervals or abs(end - horizon) > 1e-8 or abs(rows[-1]['time'] - horizon) > 1e-8:
        raise ValueError('Wrong evaluation horizon')
    gap = sum(b - a for a, b, ready, _ in coverage if not ready)
    ownership_gap = sum(b - a for a, b, _, owner in coverage if owner is None)
    unsupported_owner = sum(b - a for a, b, ready, owner in coverage if owner not in ready)
    spawned = [e for e in events if e['event'] == 'task_spawned']
    successes = [e for e in events if e['event'] == 'task_completed']
    responses = [e for e in events if e['event'] == 'task_response_started']
    response_times = []
    censored_times = []
    for task in spawned:
        found = [e for e in responses if e['task'] == task['task']]
        if found:
            response_times.append(found[0]['time'] - task['time'])
        # No response is retained as censored; never silently excluded.
        censored_times.append((found[0]['time'] if found else min(task['deadline'], horizon)) - task['time'])
    handovers = []
    for release in (e for e in events if e['event'] == 'responsibility_released'):
        t = release['time']
        future = [row for row in intervals if row['start'] >= t]
        first = next((row['start'] for row in future if _successor_ready(row, release['new'])), None)
        if first is None and _successor_ready({'truth': rows[-1]['truth']}, release['new']):
            first = horizon
        handovers.append({'id': release['handover_id'], 'release': t,
                          'successor_ready_at_or_after_release': first,
                          'initial_gap_s': (horizon if first is None else first) - t,
                          'right_censored': first is None,
                          'physical_gap_union_s': sum(b - max(a, t) for a, b, ready, _ in coverage if b > t and not ready)})
    recoveries = []
    for event in (e for e in events if e['event'] == 'robot_failed'):
        t = event['time']
        restored = next((a for a, _, ready, _ in coverage if a >= t and ready), None)
        if restored is None and _any_base_ready(rows[-1]['truth']):
            restored = horizon
        recoveries.append({'robot': event['robot'], 'failure': t,
                           'coverage_recovery_s': None if restored is None else restored - t,
                           'right_censored': restored is None})
    return {'coverage_gap_ratio': gap / horizon, 'coverage_gap_s': gap,
            'ownership_gap_ratio': ownership_gap / horizon,
            'declared_owner_not_ready_ratio': unsupported_owner / horizon,
            'task_count': len(spawned), 'task_success_count': len(successes),
            'task_success_rate': len(successes) / len(spawned) if spawned else None,
            'response_times_s': response_times,
            'response_time_mean_s': mean(response_times) if response_times else None,
            'unresponded_tasks': len(spawned) - len(response_times),
            'restricted_response_mean_s': mean(censored_times) if censored_times else None,
            'handover_count': len(handovers), 'handovers': handovers,
            'failure_recoveries': recoveries}


def _successor_ready(row, name):
    state = row['truth']
    robot = state['robots'][name]
    return (robot['operational'] and robot['active_assignment'] in (None, 'base')
            and math.dist(robot['position'], state['base']['position']) <= state['radius'] + 1e-9)


def _any_base_ready(state):
    return any(_successor_ready({'truth': state}, name) for name in state['robots'])
