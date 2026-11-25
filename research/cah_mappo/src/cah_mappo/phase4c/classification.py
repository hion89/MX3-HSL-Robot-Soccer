"""Conservative, auditable failure attribution; missing witnesses are not proofs."""
from collections import Counter,defaultdict
from statistics import mean
from ..phase3b2b.io import save
from ..phase4b.experiment import read,POLICIES,grouped


def mission_failed(row):
    return row['task_success_rate']<1 or (row['threat_pass_rate'] or 0)>0


def nonempty_witness_ids(witnesses):
    return {sid for sid,entries in witnesses.items() if entries}


def diagnose(row,physical_proof=False,online_joint=False,offline_joint=False,task_counterfactual=False):
    if physical_proof and online_joint:raise ValueError('A physical impossibility certificate contradicts an online success witness')
    confirmed=[]
    if physical_proof:confirmed.append('A')
    if online_joint:confirmed.append('C')
    return {'confirmed_causes':confirmed,'unresolved_full_mission_cause':not confirmed,
            'physical_possibility_witness':bool(online_joint or offline_joint),
            'task_selection_component_confirmed_C':bool(row['task_success_rate']<1 and task_counterfactual),
            'causal_rule':'online witness must use the same information/action interface; offline future-aware witness only proves feasibility'}


def resolution_comparison(records):
    groups=defaultdict(dict)
    for r in records:groups[r['policy'],r['scenario_id']][r['physics_dt']]=r
    pairs=[(d[.125],d[.0625]) for d in groups.values()]
    keys=('legacy_gap_region_seconds','ownership_gap_region_seconds','physical_gap_region_seconds',
          'threat_weighted_gap_seconds','post_task_legacy_gap_s','response_time_mean_s',
          'task_completion_mean_s','first_handover_ready_at','handover_gap_mean_s')
    deltas={}
    for key in keys:
        v=[abs(a[key]-b[key]) for a,b in pairs if a.get(key) is not None and b.get(key) is not None]
        deltas[key]={'paired':len(v),'max_abs_delta':max(v) if v else None,'mean_abs_delta':mean(v) if v else None}
    flips={key:sum(a[key]!=b[key] for a,b in pairs) for key in ('task_success_rate','deadline_miss_rate','threat_pass_rate','joint_task_and_defense_success')}
    return {'pairs':len(pairs),'coarse_dt':.125,'reference_dt':.0625,'continuous_deltas':deltas,'outcome_changes':flips}


def classify_all(root,output):
    target=output/'analysis_final';target.mkdir(exist_ok=False);old=root/'phase4b_outputs/study_v1'
    worlds=read(old/'design/diagnostic_manifest.json');byid={s['scenario_id']:s for s in worlds}
    certificates=read(old/'offline/results.json')['certificates']
    impossible={c['scenario_id'] for c in certificates if c.get('joint_impossible_proven') or c['type']=='required_service_exceeds_total_robot_time'}
    offline={c['scenario_id'] for c in read(old/'offline/results.json')['cases'] if c['has_joint_success']}
    offline.update(c['scenario_id'] for c in read(output/'plans/results.json')['cases'] if c['has_joint_witness'])
    native={name:read(old/'comparison'/name/'results.json')['episodes'] for name in POLICIES}
    diagnostic={p.name:read(p/'results.json')['episodes'] for p in (output/'rules').iterdir() if p.is_dir()}
    models={p.name:read(p/'results.json')['episodes'] for p in (old/'sensor_feedback_transfer').iterdir() if p.is_dir()}
    native_online=defaultdict(list);local_online=defaultdict(list);local_task=defaultdict(list)
    for label,rows in {**native,**{k:v for k,v in diagnostic.items() if not k.startswith('local_')}}.items():
        for r in rows:
            if not mission_failed(r):native_online[r['scenario_id']].append(label)
    for label,rows in diagnostic.items():
        if not label.startswith('local_'):continue
        for r in rows:
            if not mission_failed(r):local_online[r['scenario_id']].append(label)
            if r['task_success_rate']==1:local_task[r['scenario_id']].append(label)
    cases=[];total=0
    for label,rows in {**native,**models}.items():
        actor=label.startswith('seed_');witnesses=local_online if actor else native_online
        for r in rows:
            total+=1
            if not mission_failed(r):continue
            sid=r['scenario_id'];s=byid[sid]
            d=diagnose(r,sid in impossible,bool(witnesses[sid]),sid in offline,bool(local_task[sid]) if actor else bool(native_online[sid]))
            context=[]
            if s['family'] in ('A_near','D_visible'):context.append('B_indistinguishable_future_branch_context_not_individual_causal_proof')
            if actor:context.append('D_transfer_distribution_and_no_threat_preparation_channels_context_not_proven_cause')
            cases.append({'policy':label,'repeat':r.get('repeat',0),'scenario_id':sid,'family':s['family'],'team':s['team'],
                'interface':'146_features_10_actions' if actor else 'current_public_plus_preparation',
                'task_success_rate':r['task_success_rate'],'threat_pass_rate':r['threat_pass_rate'],
                **d,'same_interface_online_joint_witnesses':witnesses[sid],'context_only':context})
    counts=Counter(c for r in cases for c in r['confirmed_causes']);unknown=sum(r['unresolved_full_mission_cause'] for r in cases)
    summary={'primary_policy_episodes':total,'primary_unique_worlds':len(worlds),'mission_failure_episodes':len(cases),
        'mission_success_episodes':total-len(cases),'confirmed_cause_policy_episode_counts':{k:counts[k] for k in 'ABCD'},
        'unresolved_full_mission_cause_episodes':unknown,
        'task_selection_component_confirmed_C_episodes':sum(r['task_selection_component_confirmed_C'] for r in cases),
        'physical_impossibility_certified_worlds':len(impossible),
        'native_information_indistinguishable_history_pairs':108,'frozen_actor_history_pairs':120,
        'B_confirmed_unit':'paired pre-spawn histories; not automatically attributed to every failed policy-episode',
        'D_confirmed_primary_outcome_flips_from_feedback_adapter_fix':0,
        'D_structural_limitations':'uncalibrated speed/service/threat model and metric clock; not a counted causal proof for every failure',
        'denominator_note':'9 native policies x120 +5 frozen models x(120 deterministic +240 stochastic) =2880; excludes duplicate v1 projection and new diagnostic rules',
        'counts_nonadditive':'component/context counts overlap and do not sum to mission failure counts',
        'by_family':{},'by_interface':{}}
    for field in ('family','interface'):
        for value in sorted({r[field] for r in cases}):
            subset=[r for r in cases if r[field]==value]
            summary['by_'+field][value]={'failures':len(subset),'confirmed':dict(Counter(k for r in subset for k in r['confirmed_causes'])),
                'unresolved':sum(r['unresolved_full_mission_cause'] for r in subset)}
    save(target/'failure_cases.json',cases);save(target/'cause_counts.json',summary)
    save(target/'all_method_summaries.json',{label:grouped(rows) for label,rows in {**native,**models,**diagnostic}.items()})
    results=read(output/'sensitivity/parameter_results.json')['episodes'];sens={}
    for profile in sorted({r['profile'] for r in results}):sens[profile]=grouped([r for r in results if r['profile']==profile])
    save(target/'parameter_sensitivity_summary.json',sens)
    fine=read(output/'sensitivity/fine_dt_results.json')['episodes']
    if (output/'commitment_sensitivity/fine_dt_results.json').is_file():
        fine+=read(output/'commitment_sensitivity/fine_dt_results.json')['episodes']
    save(target/'resolution_summary.json',resolution_comparison(fine))
    online_ids=nonempty_witness_ids(native_online)
    save(target/'bounded_feasibility_summary.json',{'certified_impossible_worlds':sorted(impossible),
        'online_joint_witness_worlds':sorted(online_ids),'offline_joint_witness_worlds':sorted(offline),
        'no_online_or_offline_joint_witness_worlds':[s['scenario_id'] for s in worlds if s['scenario_id'] not in online_ids and s['scenario_id'] not in offline],
        'warning':'limited schedule search cannot establish global temporal optimality or impossibility'})
    print('Failure classification',summary['mission_failure_episodes'],'/',total,'confirmed',dict(counts),'unresolved',unknown)
