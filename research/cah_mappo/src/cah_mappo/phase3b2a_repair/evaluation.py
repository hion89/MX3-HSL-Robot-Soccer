"""Same physical scenarios, explicit slot traces, and exhaustive failure accounting."""
from collections import Counter
import copy
import math
from pathlib import Path
from statistics import mean
import torch
from ..core import write_jsonl
from ..phase2.metrics import evaluate as legacy_score, physical_coverage
from ..phase3a.contracts import SampledAction
from ..phase3b2a.scenarios import digest
from ..phase3b2a.evaluation import actor_inputs, aggregate as previous_aggregate
from .environment import AuditedEnv


def timelines(rows):
    final = rows[-1]['truth']; result = []
    events = [r for r in rows if r['type'] == 'event']
    adapters = [r['slot_adapter'] for r in rows if r['type'] == 'transition' and 'slot_adapter' in r]
    starts = [r['start'] for r in rows if r['type'] == 'transition' and 'slot_adapter' in r]
    for jid, job in final['jobs'].items():
        if job['kind'] != 'task':continue
        per_robot = {}
        for name in final['robots']:
            def first(event):
                return next((e['time'] for e in events if e['event'] == event and e.get('job') == jid and e.get('robot') == name), None)
            accepted = next((t for t, view in zip(starts, adapters)
                             if view['records'][name]['transition']['executed_assignment'] == jid), None)
            per_robot[name] = {'first_sensor': first('task_sensor_first'), 'first_message_received': first('task_message_first'),
                               'first_actor_known': first('task_actor_first_known'),
                               'first_policy_selected': first('task_policy_first_selected'), 'first_accepted_assignment': accepted}
        known = [r['first_actor_known'] for r in per_robot.values() if r['first_actor_known'] is not None]
        selected = [r['first_policy_selected'] for r in per_robot.values() if r['first_policy_selected'] is not None]
        timely_known = any(t < job['deadline'] for t in known)
        timely_selected = any(t < job['deadline'] for t in selected)
        success = job['status'] == 'completed'
        category = ('success' if success else 'no_timely_information' if not timely_known else
                    'known_but_not_selected' if not timely_selected else
                    'selected_but_no_response' if job['response'] is None else 'responded_but_not_completed')
        result.append({'job': jid, 'spawn': job['spawn'], 'deadline': job['deadline'], 'robots': per_robot,
                       'first_known_any': min(known) if known else None, 'ever_known': bool(known),
                       'timely_known': timely_known, 'timely_selected': timely_selected,
                       'response_started': job['response'], 'completed': job['completed'],
                       'expired': next((e['time'] for e in events if e['event'] == 'task_expired' and e['job'] == jid), None),
                       'success': success, 'category': category})
    return result


def score(rows):
    result = legacy_score(rows)
    intervals = [r for r in rows if r['type'] == 'interval']
    base_seconds = sum((r['end'] - r['start']) * sum(j['kind'] == 'base' for j in r['truth']['jobs'].values()) for r in intervals)
    result.update(evaluation_seconds=rows[-1]['time'], responsibility_region_seconds=base_seconds,
                  actual_gap_region_seconds=result['coverage_gap_s'], risk_weighted_gap_seconds=result['weighted_coverage_gap_s'],
                  risk_weighted_region_seconds=result['coverage_weighted_time_s'])
    return result


def summarize(rows):
    summary = previous_aggregate(rows)
    for key in ('actual_gap_region_seconds','responsibility_region_seconds','risk_weighted_gap_seconds','risk_weighted_region_seconds'):
        summary[key] = mean(r[key] for r in rows) if rows else None
    tasks = [j for r in rows for j in r['task_timelines']]
    counts = Counter(t['category'] for t in tasks)
    total = len(tasks)
    summary.update(task_count=total, failure_categories=dict(counts),
                   never_observed_ratio=sum(not t['ever_known'] for t in tasks) / total if total else None,
                   no_timely_information_ratio=counts['no_timely_information']/total if total else None,
                   known_but_not_selected_ratio=counts['known_but_not_selected']/total if total else None)
    return summary


@torch.no_grad()
def evaluate_one(spec, network, order=(2, 3, 4), mode='greedy', condition='controlled', policy_seed=7):
    if mode not in ('greedy','stochastic'):raise ValueError(mode)
    env = AuditedEnv(spec, order, condition)
    obs = env.obs; generator = torch.Generator(device='cpu').manual_seed(policy_seed*100000 + spec['event_seed'])
    env.engine.rows[0]['evaluation_sampling_seed'] = policy_seed*100000 + spec['event_seed']
    diagnostics = Counter(); probabilities = []; entropies = []; top_prob = []
    job_selection = Counter(); action_selection = Counter()
    while env.agents:
        vectors, masks = actor_inputs(obs, env.settings)
        dist = network.distribution(vectors, masks)
        chosen = dist.probs.argmax(-1) if mode == 'greedy' else torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
        actions = {}
        for name in env.agents:
            i = env.settings.roster.index(name); packet = obs[name]; action = int(chosen[i])
            probs = tuple(dist.probs[i].tolist()) if mode == 'stochastic' else tuple(float(j == action) for j in range(10))
            actions[name] = SampledAction(action, math.log(probs[action]), tuple(packet['action_mask']), probs,
                                          f'repair-eval-{mode}-not-training', packet['observation_token'])
            if packet['active_mask']:
                entropies.append(float(dist.entropy()[i])); top_prob.append(float(dist.probs[i].max()))
                action_selection[str(action)] += 1
                target = None if action == 0 else packet['self_assignment'] if action == 1 else packet['job_ids'][action-2]
                if target is not None:job_selection[target] += 1
                for slot,jid in enumerate(packet['job_ids']):
                    if jid is not None and jid.startswith('task') and packet['action_mask'][slot+2]:
                        probabilities.append(float(dist.probs[i,slot+2]))
        obs, *_ = env.step(actions)
        trace = next(r for r in reversed(env.engine.rows) if r['type'] == 'transition')['slot_adapter']['records']
        for name in trace:
            i = env.settings.roster.index(name)
            trace[name]['model_probabilities'] = dist.probs[i].tolist()
            trace[name]['model_entropy'] = float(dist.entropy()[i])
    log = env.log_rows()
    result = {'scenario_id': spec['scenario_id'], 'split': spec['split'], 'condition': condition, 'order': list(order),
              'mode': mode, 'exogenous_sha256': digest(env.engine.plan), 'return': env.episode_return, **score(log),
              'task_timelines': timelines(log), 'policy_diagnostics': {'mean_entropy': mean(entropies),
                  'mean_top_probability': mean(top_prob), 'mean_valid_task_probability': mean(probabilities) if probabilities else None,
                  'action_selection': dict(action_selection), 'job_selection': dict(job_selection)}}
    return result, log


def evaluate_many(specs, network, orders=((2,3,4),), mode='greedy', condition='controlled', log_dir=None):
    if log_dir is not None:Path(log_dir).mkdir(parents=True, exist_ok=False)
    results = []
    for spec in specs:
        for order in orders:
            row, log = evaluate_one(spec, network, order, mode, condition)
            results.append(row)
            if log_dir is not None:
                write_jsonl(log, Path(log_dir) / f"{spec['scenario_id']}_p{''.join(map(str,order))}.jsonl")
    return {'summary': summarize(results), 'episodes': results}


def coverage_windows(rows, windows=(8,16,32)):
    results = []
    for end in windows:
        weighted_gap=weighted_time=gap=region_seconds=0.
        for r in rows:
            if r['type'] != 'interval' or r['start'] >= end:continue
            duration = min(r['end'],end)-r['start']; state=r['truth']; covered=physical_coverage(state)
            for jid,j in state['jobs'].items():
                if j['kind'] != 'base':continue
                region_seconds += duration; weighted_time += duration*j['priority']
                if jid not in covered:gap += duration; weighted_gap += duration*j['priority']
        results.append({'window_seconds':end,'actual_gap_region_seconds':gap,'responsibility_region_seconds':region_seconds,
                        'risk_weighted_gap_seconds':weighted_gap,'risk_weighted_region_seconds':weighted_time,
                        'coverage_gap_ratio':weighted_gap/weighted_time if weighted_time else None})
    return results
