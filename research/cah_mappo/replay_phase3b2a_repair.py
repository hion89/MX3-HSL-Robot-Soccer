"""Reconstruct complete training decisions; verify each saved checkpoint exactly.

This repeats existing computations, never extends training or selects a new model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))


def replay(run,output):
    import torch
    from cah_mappo.phase3b2a_repair.learner import RepairLearner
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((run/'scenario_manifest.json').read_text()); reports={}
    for label in ('identity','augmented'):
        config=json.loads((run/label/'config.json').read_text())
        learner=RepairLearner(manifest['train'],config['parent_checkpoint'],config['augment_episode_slots'])
        verified=[]; start=time.perf_counter();decision_count=0
        with (output/f'{label}_training_decisions.jsonl').open('x',encoding='utf-8') as stream:
            stream.write(json.dumps({'type':'header','provenance':'deterministic training replay, not original-time collection',
                                     'original_config':str(run/label/'config.json'),'target_steps':config['steps']})+'\n')
            while learner.steps<config['steps']:
                batch=learner.collect(min(256,config['steps']-learner.steps))
                for tick,row in enumerate(batch,learner.steps-len(batch)):
                    agents={}
                    for i,name in enumerate(('A','B','C','D')):
                        action=row['actions'][i];executed=row['executed_actions'][i]
                        def target(a):return None if a==0 else row['assignments'][i] if a==1 else row['job_ids'][i][a-2]
                        agents[name]={'job_ids':row['job_ids'][i],'action_mask':row['mask'][i],
                                      'sampled_action':action,'core_action':action if action<2 else row['slot_order'][action-2]+2,
                                      'executed_action':executed,'requested_job':target(action),'executed_job':target(executed),
                                      'probabilities':row['probabilities'][i],'log_probability':row['old_logp'][i],
                                      'active':row['active'][i],'arbitration_changed':row['arbitration_changed'][i]}
                    stream.write(json.dumps({'type':'training_decision','step':tick,'scenario_id':row['scenario_id'],
                                              'episode':tick//32,'time':tick%32,'slot_order':row['slot_order'],
                                              'policy_update':row['policy_update'],'agents':agents},allow_nan=False)+'\n')
                    decision_count+=1
                report=learner.update(batch)
                saved_report=json.loads((run/label/f'update_{learner.steps:06d}.json').read_text())
                for key,value in report.items():
                    assert value==saved_report[key],f'{label} update {learner.steps}: {key}'
                if learner.steps%2048==0:
                    checkpoint=torch.load(run/label/f'checkpoints/step{learner.steps}.pt',weights_only=True,map_location='cpu')
                    assert all(torch.equal(value,checkpoint['network'][key]) for key,value in learner.net.state_dict().items())
                    assert torch.equal(learner.sample_rng.get_state(),checkpoint['sample_rng'])
                    assert torch.equal(learner.shuffle_rng.get_state(),checkpoint['shuffle_rng'])
                    verified.append(learner.steps)
                    print(json.dumps({'replay':label,'verified_step':learner.steps}),flush=True)
        reports[label]={'decisions':decision_count,'agent_decisions':decision_count*4,'checkpoint_steps_exactly_matched':verified,
                        'wall_seconds':time.perf_counter()-start,'new_model_selected':False}
    with (output/'replay_audit.json').open('x',encoding='utf-8') as f:json.dump({'status':'PASS','arms':reports},f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();replay(args.run,args.output)
