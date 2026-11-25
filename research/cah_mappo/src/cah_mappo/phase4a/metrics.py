"""Reward-independent ground-truth diagnostics, recomputable from JSONL alone."""
import math
from statistics import mean
from ..phase2.metrics import evaluate, physical_coverage


def distribution(values):
    values=sorted(values)
    if not values:return {'n':0,'mean':None,'p95':None,'max':None,'top5_mean':None}
    n=len(values)
    return dict(n=n,mean=mean(values),p95=values[math.ceil(.95*n)-1],max=values[-1],
                top5_mean=mean(values[-max(1,math.ceil(.05*n)):]))


def segments(rows):
    intervals=[r for r in rows if r['type']=='interval'];opened={};out=[]
    for row in intervals:
        covered=physical_coverage(row['truth'])
        for jid,j in row['truth']['jobs'].items():
            if j['kind']!='base':continue
            if jid not in covered:opened.setdefault(jid,row['start'])
            elif jid in opened:
                start=opened.pop(jid);out.append(dict(base=jid,start=start,end=row['start'],duration=row['start']-start,right_censored=False))
    end=rows[-1]['time'];terminal_covered=physical_coverage(rows[-1]['truth'])
    for jid,start in opened.items():
        out.append(dict(base=jid,start=start,end=end,duration=end-start,right_censored=jid not in terminal_covered))
    return sorted(out,key=lambda s:(s['start'],s['base']))


def score(rows):
    result=evaluate(rows);spans=segments(rows)
    config=rows[0]['config'];base_count=sum(j['kind']=='base' for j in config['jobs'])
    limits=rows[0].get('phase4a',{}).get('response_limit_s',{})
    events=[r for r in rows if r['type']=='event'];intervals=[r for r in rows if r['type']=='interval']
    recovery=[]
    for event in events:
        if event['event']!='failure':continue
        target=event['target'];spec=next(r for r in config['robots'] if r['id']==target)
        base=spec['initial_job']
        # Fault may hit a designated successor rather than an initial guard.
        pending=next((e for e in reversed(events[:events.index(event)]) if e['event']=='controller_requested' and e.get('successor')==target),None)
        base=base or (pending['base'] if pending else None)
        future=[r for r in intervals if r['start']>=event['time']]
        repaired=next((e['time'] for e in events if e['event']=='recovery' and e['target']==target and e['time']>=event['time']),None)
        covered=next((r['start'] for r in future if base in physical_coverage(r['truth'])),None) if base else None
        if base and covered is None and base in physical_coverage(rows[-1]['truth']):covered=rows[-1]['time']
        recovery.append(dict(robot=target,base=base,failure_at=event['time'],device_recovery_at=repaired,
                             device_recovery_s=None if repaired is None else repaired-event['time'],
                             coverage_recovery_s=None if covered is None else covered-event['time'],
                             observed_lower_bound_s=rows[-1]['time']-event['time'] if covered is None else covered-event['time'],
                             right_censored=covered is None))
    result.update(actual_gap_region_seconds=result['coverage_gap_s'],
                  responsibility_region_seconds=base_count*rows[-1]['time'],
                  risk_weighted_region_seconds=result['coverage_weighted_time_s'],
                  gap_segments=spans,episode_has_gap=bool(spans),
                  max_continuous_gap_s=max((s['duration'] for s in spans),default=0.),
                  handover_gap_max_s=max((h['initial_gap_s'] for h in result['handovers']),default=None),
                  threat_limit_violations=sum(s['duration']>limits.get(s['base'],math.inf)+1e-9 for s in spans),
                  controller_requests=sum(e['event']=='controller_requested' for e in events),
                  controller_cancel_count=sum(e['event']=='controller_cancelled' for e in events),
                  controller_failure_count=sum(e['event']=='controller_failed' for e in events),
                  controller_early_release_count=sum(e['event']=='controller_early_release' for e in events),
                  controller_hold_ticks=sum(e['event']=='controller_holding' for e in events),
                  controller_modified_agent_ticks=sum(sum(r['selection'].get('controller_changed',{}).values()) for r in intervals),
                  recovery_details=recovery,
                  coverage_recovery_time_s=mean(r['coverage_recovery_s'] for r in recovery if r['coverage_recovery_s'] is not None)
                     if any(r['coverage_recovery_s'] is not None for r in recovery) else None,
                  recovery_censored_count=sum(r['right_censored'] for r in recovery),
                  task_completion_time_s=mean(j['completed_at']-j['spawn'] for j in result['task_details'] if j['success'])
                    if result['task_success_count'] else None)
    return result


def aggregate(rows):
    keys=('task_success_rate','deadline_miss_rate','coverage_gap_ratio','actual_gap_region_seconds',
          'responsibility_region_seconds','risk_weighted_region_seconds','response_time_mean_s',
          'restricted_response_mean_s','task_completion_time_s','role_switch_count','task_duplicate_rate',
          'requested_duplicate_rate','controller_requests','controller_cancel_count','controller_failure_count',
          'controller_early_release_count','controller_hold_ticks','controller_modified_agent_ticks',
          'coverage_recovery_time_s','recovery_censored_count','threat_limit_violations','episode_return')
    out={key:mean(r[key] for r in rows if r.get(key) is not None) if any(r.get(key) is not None for r in rows) else None for key in keys}
    handovers=[h['initial_gap_s'] for r in rows for h in r['handovers']]
    out.update(episodes=len(rows),episodes_with_gap=sum(r['episode_has_gap'] for r in rows),
               gap_episode_rate=mean(r['episode_has_gap'] for r in rows) if rows else None,
               episode_gap_distribution=distribution([r['actual_gap_region_seconds'] for r in rows]),
               gap_episode_conditional_distribution=distribution([r['actual_gap_region_seconds'] for r in rows if r['episode_has_gap']]),
               continuous_gap_distribution=distribution([s['duration'] for r in rows for s in r['gap_segments']]),
               handover_gap_distribution=distribution(handovers),
               handover_right_censored=sum(h['right_censored'] for r in rows for h in r['handovers']))
    return out
