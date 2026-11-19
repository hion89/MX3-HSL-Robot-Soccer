"""Audit all evaluation trajectories, adapter mappings, timing and sealed holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def verify(run,output):
    from cah_mappo.metrics import load_jsonl
    from cah_mappo.phase3a.contracts import validate_ppo_record
    from cah_mappo.phase3b2a.scenarios import digest
    from cah_mappo.phase3b2a.experiment import load_net
    from cah_mappo.phase3b2a_repair.slots import SlotMap,PERMUTATIONS
    from cah_mappo.phase3b2a_repair.evaluation import score,timelines,evaluate_many
    output.mkdir(parents=True,exist_ok=False)
    preserved=json.loads((ROOT/'phase3b2a_repair_outputs/preservation_before.json').read_text())
    for name,expected in preserved.items():
        assert hashlib.sha256((ROOT.parents[1]/name).read_bytes()).hexdigest()==expected,name
    sealed=json.loads((run/'sealed_final_holdout_manifest.json').read_text())
    assert sealed['status']=='NOT_EVALUATED' and len(sealed['scenarios'])==400
    manifest=json.loads((run/'scenario_manifest.json').read_text())
    active_specs=[*manifest['train'],*manifest['validation'],*manifest['D_development'],*manifest['B3_development'],
                  *[s for group in manifest['development'].values() for s in group]]
    assert not {s['event_seed'] for s in active_specs}&{s['event_seed'] for s in sealed['scenarios']}
    assert not {s['event_seed'] for s in manifest['train']}&{s['event_seed'] for s in manifest['validation']}
    comparisons=json.loads((run/'same_condition_comparisons.json').read_text())
    information=json.loads((run/'information_comparisons.json').read_text())
    b3=json.loads((run/'B3_deterministic_stochastic.json').read_text())
    finals=json.loads((run/'final_checkpoint_comparisons.json').read_text())
    groups=[]
    for label,families in comparisons.items():
        for family,modes in families.items():
            for mode,group in modes.items():
                folder=run/'baseline_logs'/family/mode if label=='old' else run/'comparison_logs'/label/family/mode
                groups.append((folder,group))
    for label,conditions in information.items():
        for condition,modes in conditions.items():
            for mode,group in modes.items():groups.append((run/'information_logs'/label/condition/mode,group))
    for label,modes in b3.items():
        for mode,group in modes.items():groups.append((run/'b3_logs'/label/mode,group))
    for label,modes in finals.items():
        for mode,group in modes.items():groups.append((run/'final_logs'/label/mode,group))
    paired={};logs=0;agent_actions=0;reward_transitions=0
    for folder,group in groups:
        for episode in group['episodes']:
            sid=episode['scenario_id'];suffix=''.join(map(str,episode['order']))
            rows=load_jsonl(folder/f'{sid}_p{suffix}.jsonl')
            assert rows[0]['repair_interface']['split']!='final_holdout'
            assert rows[-1]['type']=='terminal' and rows[-1]['time']==32
            metrics=score(rows);assert metrics=={key:episode[key] for key in metrics}
            assert timelines(rows)==episode['task_timelines']
            expected=paired.setdefault(sid,digest(rows[0]['exogenous_plan']))
            assert expected==digest(rows[0]['exogenous_plan'])==episode['exogenous_sha256']
            selections={r['time']:r for r in rows if r.get('event')=='policy_selection'}
            events=[r for r in rows if r['type']=='event']
            for transition in (r for r in rows if r['type']=='transition'):
                view=transition['slot_adapter'];layout=SlotMap(tuple(view['new_to_old'][2:5]))
                assert view['critic_before']==layout.critic(transition['critic_before'])
                assert view['critic_after']==layout.critic(transition['critic_after'])
                for name,record in view['records'].items():
                    original=selections[transition['start']]['actor_observations'][name]
                    assert layout.packet(original)==view['actor_before'][name]
                    r=record['transition'];validate_ppo_record(r)
                    a=r['sampled_action'];assert record['core_action']==layout.action_to_core(a)
                    if a>=2:assert r['requested_assignment']==record['job_ids'][a-2]
                    assert r['requested_assignment']==transition['records'][name]['requested_assignment']
                    assert record['action_mask'][a]
                    agent_actions+=1
                complete=sum(e['event']=='task_completed' and e['time']==transition['end'] for e in events)
                expired=sum(e['event']=='task_expired' and e['time']==transition['end'] for e in events)
                assert transition['reward_contributions']['completion']==10*complete/8
                assert transition['reward_contributions']['failure']==-10*expired/8
                reward_transitions+=1
            assert abs(sum(r['reward'] for r in rows if r['type']=='transition')-episode['return'])<1e-10
            for task in episode['task_timelines']:
                for name,t in task['robots'].items():
                    for key in ('first_sensor','first_message_received','first_actor_known','first_policy_selected'):
                        assert t[key] is None or t[key]>=task['spawn']
                    receipts=[e for e in events if e['event']=='task_message_first' and e['job']==task['job'] and e['robot']==name]
                    for event in receipts:assert event['time']-event['measured_at']==rows[0]['settings']['radio_delay']
                    sources=[t[k] for k in ('first_sensor','first_message_received') if t[k] is not None]
                    assert t['first_actor_known']==(min(sources) if sources else None)
            logs+=1
    exact=json.loads((run/'exact_feasibility.json').read_text())
    for record in exact:
        rows=load_jsonl(run/'exact_logs'/f"{record['scenario_id']}.jsonl")
        metrics=score(rows);assert metrics=={key:record[key] for key in metrics};assert metrics['task_success_rate']==1
    freeze=json.loads((run/'selection_freeze.json').read_text());replayed=0
    for label,arm in freeze['arms'].items():
        config=json.loads((run/label/'config.json').read_text())
        for name,expected in config['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
        checkpoint=Path(arm['selected_checkpoint'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==freeze['selected_checkpoint_hashes'][label]
        for mode in ('greedy','stochastic'):
            repeated=evaluate_many(manifest['development']['positions'],load_net(checkpoint),PERMUTATIONS,mode)
            assert repeated==comparisons[label]['positions'][mode]
            replayed+=len(repeated['episodes'])
    report={'status':'PASS','preserved_files':len(preserved),'evaluation_logs_recomputed':logs,
            'exact_feasibility_logs_recomputed':len(exact),'agent_action_mappings_checked':agent_actions,
            'reward_timing_transitions_checked':reward_transitions,'paired_scenario_ids':len(paired),
            'checkpoint_replay_episodes':replayed,'sealed_holdout_scenarios':400,'holdout_outcomes_evaluated':0}
    with (output/'audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();verify(args.run,args.output)
