"""New metrics version; legacy evaluator and rewards remain untouched."""
from statistics import mean
import math
from ..phase2.metrics import evaluate,physical_coverage
from ..phase4a.metrics import distribution
from .model import arrival,match

VERSION='phase4b-coverage-v1'


def evaluate_new(rows):
    legacy=evaluate(rows);header=rows[0];meta=header['phase4b'];terminal=rows[-1]
    horizon=terminal['time'];bases=[j['id'] for j in header['config']['jobs'] if j['kind']=='base']
    intervals=[r for r in rows if r['type']=='interval'];events=[r for r in rows if r['type']=='event']
    tasks=legacy['task_details'];all_complete=bool(tasks) and all(t['success'] for t in tasks)
    last=max(t['completed_at'] for t in tasks) if all_complete else None
    own_gap=physical_gap=post_legacy=post_ownership=post_physical=0.;exposure=tw_gap=0.
    visible_exposure=visible_tw_gap=0.;per_threat={};potential_at_reveal={};observed={}
    for e in events:
        if e['event']=='threat_observed':observed.setdefault(e['threat'],e['time'])
    for row in intervals:
        raw_state=row['truth'];dt=row['end']-row['start'];owners={}
        period=meta.get('control_period',1.);t=row['start']
        state={**raw_state,'time':t,'control_period':period,
               'decision_delay':max(0.,math.ceil(t/period-1e-9)*period-t)}
        for jid in bases:
            owner=state['jobs'][jid]['owner'];r=state['robots'].get(owner)
            owners[jid]=r is not None and r['operational'] and r['assignment']==jid
        targets={jid:(jid,1.) for jid in bases}
        potential=match(state,targets,meta['response_budgets'])
        legacy_covered=physical_coverage(state)
        own_gap+=dt*sum(not owners[j] for j in bases)
        physical_gap+=dt*sum(j not in potential for j in bases)
        if last is not None:
            overlap=max(0.,row['end']-max(row['start'],last))
            post_legacy+=overlap*sum(j not in legacy_covered for j in bases)
            post_ownership+=overlap*sum(not owners[j] for j in bases)
            post_physical+=overlap*sum(j not in potential for j in bases)
        active={q['id']:(q['target'],q['intensity']) for q in meta['threat_plan']
                if q['start']<=row['start']<arrival(q,state['jobs'])}
        budgets={q['id']:arrival(q,state['jobs'])-row['start'] for q in meta['threat_plan'] if q['id'] in active}
        reachable=match(state,active,budgets,weighted=True)
        for qid,(_,weight) in active.items():
            value=weight*dt;exposure+=value;gap=value if qid not in reachable else 0.;tw_gap+=gap
            account=per_threat.setdefault(qid,{'exposure_s':0.,'unreachable_s':0.})
            account['exposure_s']+=value;account['unreachable_s']+=gap
            if qid in observed and observed[qid]<=row['start']:
                visible_exposure+=value;visible_tw_gap+=gap
                potential_at_reveal.setdefault(qid,qid in reachable)
    outcomes=[e for e in events if e['event'] in ('threat_defended','threat_passed')]
    if len({e['threat'] for e in outcomes})!=len(outcomes):raise ValueError('Duplicate arrival outcome')
    defense=sum(e['event']=='threat_defended' for e in outcomes);passed=len(outcomes)-defense
    for e in outcomes:
        # Recompute explicit outcomes from arrival GT; a named owner alone is not sufficient.
        simultaneous=[x for x in outcomes if x['time']==e['time']]
        query={x['threat']:(x['base'],x['intensity']) for x in simultaneous}
        actual=match(e['arrival_truth'],query,{k:0. for k in query},weighted=True,at_point=True)
        if (e['event']=='threat_defended')!=(e['threat'] in actual):raise ValueError('Arrival GT disagrees with defense event')
    post_den=None if last is None else len(bases)*(horizon-last)
    result={'version':VERSION,'legacy_coverage_gap_ratio':legacy['coverage_gap_ratio'],
        'legacy_gap_region_seconds':legacy['coverage_gap_s'],'legacy_risk_region_seconds':legacy['coverage_weighted_time_s'],
        'responsibility_region_seconds':len(bases)*horizon,'responsibility_active_from':0.,'responsibility_active_until':horizon,
        'ownership_gap_region_seconds':own_gap,'ownership_gap_ratio':own_gap/(len(bases)*horizon),
        'physical_gap_region_seconds':physical_gap,'physical_gap_ratio':physical_gap/(len(bases)*horizon),
        'threat_weighted_gap_seconds':tw_gap,'threat_exposure_seconds':exposure,
        'threat_weighted_gap_ratio':tw_gap/exposure if exposure else None,
        'visible_threat_gap_seconds':visible_tw_gap,'visible_threat_exposure_seconds':visible_exposure,
        'visible_threat_gap_ratio':visible_tw_gap/visible_exposure if visible_exposure else None,
        'post_task_start':last,'post_task_region_seconds':post_den,
        'post_task_legacy_gap_s':post_legacy if last is not None else None,
        'post_task_ownership_gap_s':post_ownership if last is not None else None,
        'post_task_physical_gap_s':post_physical if last is not None else None,
        'post_task_legacy_gap_ratio':post_legacy/post_den if post_den else None,
        'threat_arrivals':len(outcomes),'threat_pass_count':passed,'threat_defended_count':defense,
        'threat_pass_rate':passed/len(outcomes) if outcomes else None,
        'post_task_threat_passes':sum(e['event']=='threat_passed' and last is not None and e['time']>=last for e in outcomes) if last is not None else None,
        'threat_details':[{'id':e['threat'],'arrival':e['time'],'passed':e['event']=='threat_passed',
                          'observed_at':observed.get(e['threat']),'reachable_when_first_observed':potential_at_reveal.get(e['threat']),
                          **per_threat.get(e['threat'],{})} for e in outcomes],
        'task_success_rate':legacy['task_success_rate'],'deadline_miss_rate':legacy['deadline_miss_rate'],
        'task0_success':next((t['success'] for t in tasks if t['id']=='task0'),None),
        'task0_response_s':next((t['response_s'] for t in tasks if t['id']=='task0'),None),
        'task0_completed_at':next((t['completed_at'] for t in tasks if t['id']=='task0'),None),
        'task0_deadline':next((t['deadline'] for t in tasks if t['id']=='task0'),None),
        'task_details':tasks,'response_time_mean_s':legacy['response_time_mean_s'],
        'task_completion_mean_s':mean(t['completed_at']-t['spawn'] for t in tasks if t['success']) if any(t['success'] for t in tasks) else None,
        'handover_details':legacy['handovers'],'handover_gap_mean_s':legacy['handover_gap_mean_s'],
        'handover_ready_at':[h['release']+h['initial_gap_s'] for h in legacy['handovers'] if not h['right_censored']],
        'first_handover_ready_at':next((h['release']+h['initial_gap_s'] for h in legacy['handovers'] if not h['right_censored']),None),
        'role_switch_count':legacy['role_switch_count'],'duplicate_assignment_rate':legacy['task_duplicate_rate'],
        'joint_task_and_defense_success':bool(all_complete and passed==0),
        'policy_decisions':sum(e['event']=='policy_selection' for e in events)}
    return result


SCALARS=('task_success_rate','deadline_miss_rate','legacy_coverage_gap_ratio','legacy_gap_region_seconds',
    'responsibility_region_seconds','ownership_gap_region_seconds','physical_gap_region_seconds',
    'ownership_gap_ratio','physical_gap_ratio','threat_weighted_gap_ratio','threat_weighted_gap_seconds',
    'threat_exposure_seconds','visible_threat_gap_ratio','threat_pass_rate','threat_pass_count',
    'post_task_legacy_gap_s','post_task_physical_gap_s','post_task_region_seconds','post_task_threat_passes',
    'task0_success','response_time_mean_s','task_completion_mean_s','role_switch_count','joint_task_and_defense_success')


def aggregate(rows):
    out={k:mean(r[k] for r in rows if r.get(k) is not None) if any(r.get(k) is not None for r in rows) else None for k in SCALARS}
    out.update(episodes=len(rows),legacy_gap_distribution=distribution([r['legacy_gap_region_seconds'] for r in rows]),
               legacy_gap_episode_rate=mean(r['legacy_gap_region_seconds']>0 for r in rows) if rows else None,
               handover_distribution=distribution([h['initial_gap_s'] for r in rows for h in r['handover_details']]),
               no_threat_episodes=sum(r['threat_arrivals']==0 for r in rows),
               post_task_observed_episodes=sum(r['post_task_start'] is not None for r in rows))
    return out
