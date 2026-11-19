"""Ground-truth metrics plus explicitly reconstructed common reference rewards."""
from statistics import mean
from ..phase2.metrics import evaluate
from ..phase3a.rewards import RewardWeights,reward_terms,combine,covered_sites


def reference_rewards(rows):
    intervals=[r for r in rows if r['type']=='interval'];events=[r for r in rows if r['type']=='event']
    total=0.;weights=RewardWeights()
    for i,row in enumerate(intervals):
        before=row['truth'];after=intervals[i+1]['truth'] if i+1<len(intervals) else rows[-1]['truth']
        es=[e for e in events if (e['time']==row['start'] and e['event'] in ('assignment_changed','task_response_started'))
            or (e['time']==row['end'] and e['event'] in ('task_completed','task_expired'))]
        terms=reward_terms(before,after,es,row['end']-row['start'],rows[0]['config']['horizon'],8,covered_sites(before))
        reward,parts=combine(terms,weights)
        row['evaluation_reward']=reward;row['evaluation_reward_terms']=terms;row['evaluation_reward_contributions']=parts
        total+=reward
    rows[0]['reference_reward_reconstruction']='same Phase 3A terms; current policy events/start, completion-expiry/end'
    return total


def score(rows,episode_return=None):
    result=evaluate(rows)
    if episode_return is None:
        transitions=[r for r in rows if r['type']=='transition']
        episode_return=sum(r['reward'] for r in transitions) if transitions else reference_rewards(rows)
    known={};selected={};choices={}
    for row in rows:
        if row.get('event')!='policy_selection':continue
        for jid in row['action']['requested'].values():
            if jid is not None:choices[jid]=choices.get(jid,0)+1;selected.setdefault(jid,row['time'])
        packets=row.get('actor_observations')
        if packets is not None:
            for packet in packets.values():
                for jid in packet['job_ids']:
                    if jid is not None:known.setdefault(jid,row['time'])
        else:
            for j in row['observation']['jobs']:known.setdefault(j['id'],row['time'])
    failures={};tasks=[]
    for task in result['task_details']:
        jid=task['id'];timely=jid in known and known[jid]<task['deadline']
        category='success' if task['success'] else 'unobserved_or_late' if not timely else 'known_not_selected' if jid not in selected else 'selected_no_response' if task['response_at'] is None else 'responded_incomplete'
        failures[category]=failures.get(category,0)+1
        tasks.append({**task,'first_known':known.get(jid),'first_selected':selected.get(jid),
                      'timely_known':timely,'category':category,'selection_count':choices.get(jid,0),
                      'completion_time_s':task['completed_at']-task['spawn'] if task['success'] else None})
    result.update(episode_return=episode_return,evaluation_seconds=rows[-1]['time'],
                  responsibility_region_seconds=2*rows[-1]['time'],actual_gap_region_seconds=result['coverage_gap_s'],
                  risk_weighted_region_seconds=result['coverage_weighted_time_s'],
                  unobserved_task_rate=sum(t['first_known'] is None for t in tasks)/len(tasks),
                  late_or_unobserved_task_rate=sum(not t['timely_known'] for t in tasks)/len(tasks),
                  task_completion_time_s=mean(t['completion_time_s'] for t in tasks if t['success']) if any(t['success'] for t in tasks) else None,
                  detailed_tasks=tasks,failure_categories=failures,job_selection_counts=choices)
    return result


SCALARS=('task_success_rate','deadline_miss_rate','coverage_gap_ratio','actual_gap_region_seconds',
         'responsibility_region_seconds','risk_weighted_region_seconds','handover_gap_mean_s','handover_count',
         'response_time_mean_s','restricted_response_mean_s','task_completion_time_s','all_tasks_makespan_s',
         'unobserved_task_rate','role_switch_count','assignment_change_count','task_duplicate_rate',
         'requested_duplicate_rate','episode_return')


def aggregate(rows):
    result={}
    for key in SCALARS:
        values=[r[key] for r in rows if r.get(key) is not None]
        result[key]=mean(values) if values else None
        if len(values)!=len(rows):result[key+'_observed_episodes']=len(values)
    result['episodes']=len(rows)
    result['unresponded_task_rate']=sum(r['unresponded_tasks'] for r in rows)/sum(r['task_count'] for r in rows) if rows else None
    result['complete_episode_rate']=mean(r['task_success_rate']==1 for r in rows) if rows else None
    counts={}
    for r in rows:
        for k,v in r['failure_categories'].items():counts[k]=counts.get(k,0)+v
    result['failure_categories']=counts
    return result
