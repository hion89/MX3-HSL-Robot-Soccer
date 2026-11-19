"""Versioned log evaluator; schema 1 delegates to the untouched Phase 1 evaluator."""
import itertools
import math
from statistics import mean

from ..metrics import evaluate as evaluate_phase1, load_jsonl


def physical_coverage(state):
    """Capacity-feasible weighted matching: an idle robot cannot cover many bases."""
    bases = {k: j for k, j in state['jobs'].items() if j['kind'] == 'base'}
    robots = sorted(state['robots'].items())
    choices = []
    for name, r in robots:
        eligible = [jid for jid, j in sorted(bases.items()) if r['operational']
                    and r['assignment'] in (None, jid)
                    and math.dist(r['position'], j['position']) <= state['radius'] + 1e-9]
        choices.append((None, *eligible))
    best_score, best = -1, {}
    for vector in itertools.product(*choices):
        ids = [jid for jid in vector if jid is not None]
        if len(ids) != len(set(ids)):
            continue
        score = sum(bases[jid]['priority'] for jid in ids)
        if score > best_score + 1e-9:
            best_score = score
            best = {jid: name for (name, _), jid in zip(robots, vector) if jid is not None}
    return best


def evaluate(rows):
    if not rows or rows[0].get('type') != 'header':
        raise ValueError('Missing header')
    if rows[0].get('schema_version') == 1:
        return evaluate_phase1(rows)
    if rows[0].get('schema_version') != 2 or rows[-1].get('type') != 'terminal':
        raise ValueError('Unsupported or incomplete log')
    horizon = rows[0]['config']['horizon']
    intervals = [r for r in rows if r['type'] == 'interval']
    terminal = rows[-1]['truth']
    events = [r for r in rows if r['type'] == 'event']
    end, weighted_gap, denominator, raw_gap = 0.0, 0.0, 0.0, 0.0
    duplicate_time = requested_duplicate_time = job_time = 0.0
    roles = {r['id']: r['base_role'] for r in rows[0]['config']['robots']}
    switches = 0
    coverage = []
    for index, row in enumerate(intervals):
        if row['tick'] != index or abs(row['start'] - end) > 1e-8 or not end < row['end'] <= horizon:
            raise ValueError('Noncontiguous interval log')
        duration = row['end'] - row['start']
        state = row['truth']
        if any(r['capacity'] != 1 or r['assignment'] is not None and not isinstance(r['assignment'], str)
               for r in state['robots'].values()):
            raise ValueError('Capacity-one contract violated')
        covered = physical_coverage(state)
        coverage.append((row, covered))
        for jid, job in state['jobs'].items():
            if job['kind'] == 'base':
                denominator += duration * job['priority']
                if jid not in covered:
                    raw_gap += duration
                    weighted_gap += duration * job['priority']
            if job['status'] != 'active':
                continue
            job_time += duration
            actual = [name for name, r in state['robots'].items() if r['assignment'] == jid]
            requested = [name for name, target in row['selection']['requested'].items() if target == jid]
            prep = {name for name, target in row['selection']['preparations'] if target == jid}
            allowed_overlap = job['kind'] == 'base' and job['owner'] is not None
            intended = lambda names: allowed_overlap and job['owner'] in names and all(n == job['owner'] or n in prep for n in names)
            if len(actual) > 1 and not intended(actual):
                duplicate_time += duration
            if len(requested) > 1 and not intended(requested):
                requested_duplicate_time += duration
        for name, robot in state['robots'].items():
            if robot['assignment'] is not None:
                kind = state['jobs'][robot['assignment']]['kind']
                role = 'guard' if kind == 'base' else 'worker'
                if roles[name] != role:
                    switches += 1
                    roles[name] = role
        end = row['end']
    if not intervals or abs(end - horizon) > 1e-8 or abs(rows[-1]['time'] - horizon) > 1e-8:
        raise ValueError('Incomplete horizon')
    tasks = [j for j in terminal['jobs'].values() if j['kind'] == 'task']
    task_details = []
    for task in tasks:
        task_details.append({'id': task['id'], 'priority': task['priority'], 'spawn': task['spawn'],
                             'deadline': task['deadline'], 'response_at': task['response'],
                             'response_s': None if task['response'] is None else task['response'] - task['spawn'],
                             'completed_at': task['completed'], 'success': task['status'] == 'completed',
                             'work': task['work']})
    responses = [j['response_s'] for j in task_details if j['response_s'] is not None]
    completed = [j for j in task_details if j['success']]
    restricted = [(j['response_at'] if j['response_at'] is not None else j['deadline']) - j['spawn'] for j in task_details]
    releases = [e for e in events if e['event'] == 'responsibility_released']
    handovers = []
    for event in releases:
        start, jid, successor = event['time'], event['job'], event['new']
        stop = next((e['time'] for e in releases if e['job'] == jid and e['time'] > start), horizon)
        future = [(r, cv) for r, cv in coverage if start <= r['start'] < stop]
        def ready(state):
            if successor is None:
                return jid in physical_coverage(state)
            r, job = state['robots'][successor], state['jobs'][jid]
            return (r['operational'] and r['assignment'] in (None, jid)
                    and math.dist(r['position'], job['position']) <= state['radius'] + 1e-9)
        first = next((r['start'] for r, _ in future if ready(r['truth'])), None)
        if first is None and stop == horizon and ready(terminal):
            first = horizon
        handovers.append({'id': event['handover_id'], 'base': jid, 'old': event['old'], 'new': successor,
                          'release': start, 'observed_until': stop,
                          'initial_gap_s': (stop if first is None else first) - start,
                          'right_censored': first is None,
                          'physical_gap_union_s': sum(r['end'] - r['start'] for r, cv in future if jid not in cv)})
    base_count = sum(j['kind'] == 'base' for j in terminal['jobs'].values())
    return {'coverage_gap_ratio': weighted_gap / denominator if denominator else None,
            'coverage_gap_s': raw_gap, 'weighted_coverage_gap_s': weighted_gap,
            'coverage_weighted_time_s': denominator,
            'unweighted_coverage_gap_ratio': raw_gap / (horizon * base_count) if base_count else None,
            'handover_count': len(handovers), 'handovers': handovers,
            'handover_gap_mean_s': mean(h['initial_gap_s'] for h in handovers) if handovers else None,
            'task_count': len(tasks), 'task_success_count': len(completed),
            'task_success_rate': len(completed) / len(tasks) if tasks else None,
            'deadline_miss_rate': (len(tasks) - len(completed)) / len(tasks) if tasks else None,
            'response_times_s': responses, 'response_time_mean_s': mean(responses) if responses else None,
            'unresponded_tasks': len(tasks) - len(responses),
            'restricted_response_mean_s': mean(restricted) if restricted else None,
            'all_tasks_makespan_s': max(j['completed_at'] for j in completed) - min(j['spawn'] for j in task_details)
                 if tasks and len(tasks) == len(completed) else None,
            'last_successful_completion_at': max((j['completed_at'] for j in completed), default=None),
            'completed_task_flow_time_sum_s': sum(j['completed_at'] - j['spawn'] for j in completed),
            'role_switch_count': switches,
            'assignment_change_count': sum(e['event'] == 'assignment_changed' for e in events),
            'task_duplicate_rate': duplicate_time / job_time if job_time else 0.0,
            'requested_duplicate_rate': requested_duplicate_time / job_time if job_time else 0.0,
            'task_details': task_details}
