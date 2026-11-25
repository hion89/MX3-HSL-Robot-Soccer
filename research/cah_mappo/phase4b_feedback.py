"""Supplement: retain the exact Phase 3A sensor/cache builder for frozen-policy transfer.

The original Phase 4B projection inferred retired status from the active-task
registry and retained the last remaining-work feature. This version supplies
current, actually observable completion feedback. Both results are preserved;
no weights, native-policy parameters or test scenarios are changed.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from cah_mappo.phase2.model import Action
from cah_mappo.phase3a.perception import Perception
from cah_mappo.phase3a.contracts import Settings,SampledAction,validate_sample
from cah_mappo.phase3b2b.io import save,write_raw,read_raw
from cah_mappo.phase4a.evaluation import load_network
from cah_mappo.phase4b.engine import Simulation
from cah_mappo.phase4b.metrics import evaluate_new
from cah_mappo.phase4b.experiment import read,sha,ensure_frozen,grouped


class FeedbackView:
    def __init__(self,base,feedback):self.base=base;self.feedback=feedback
    def __getattr__(self,key):return getattr(self.base,key)
    def record(self):return {**self.base.record(),'current_observed_job_feedback':copy.deepcopy(self.feedback)}


class FeedbackSimulation(Simulation):
    def public(self,t):
        # Explicit sensor model: radius100, no noise/occlusion, current completed
        # jobs stay inspectable as in Phase 3A. Future jobs are filtered BEFORE publishing.
        base=super().public(t)
        feedback={jid:copy.deepcopy(j) for jid,j in self.jobs.items() if j['status']!='not_spawned'}
        return FeedbackView(base,feedback)


class SensorActor:
    information='phase3a_146_sensor_cache_no_threat_features_transfer_reference'
    def __init__(self,network,mode='greedy',sampling_seed=7):
        self.network,self.mode,self.sampling_seed=network,mode,sampling_seed;self.name='sensor_mappo_'+mode
    def reset(self,seed):
        self.rng=torch.Generator(device='cpu').manual_seed(self.sampling_seed);self.perception=None;self.last={}
    def decide(self,view):
        settings=Settings(sensor_radius=100,sensor_noise=0,radio_delay=0)
        obs=view.allocation
        if self.perception is None:
            config=SimpleNamespace(robots=tuple(SimpleNamespace(id=r.id) for r in obs.robots),horizon=view.horizon,seed=0)
            self.perception=Perception(settings,config)
        public_truth={'robots':{r.id:asdict(r) for r in obs.robots},'jobs':copy.deepcopy(view.feedback)}
        self.perception.update(public_truth,obs.time,lambda *a,**kw:None)
        packets={r.id:self.perception.actor(r.id,obs.time) for r in obs.robots};vectors=[];masks=[]
        for name in settings.roster:
            p=packets.get(name);vectors.append(p['observation'] if p else [0.]*146);masks.append(p['action_mask'] if p else [1]+[0]*9)
        with torch.no_grad():
            dist=self.network.distribution(torch.tensor(vectors,dtype=torch.float32),torch.tensor(masks,dtype=torch.bool))
            chosen=dist.probs.argmax(-1) if self.mode=='greedy' else torch.multinomial(dist.probs,1,generator=self.rng).squeeze(-1)
        assignments={};records={}
        for i,name in enumerate(settings.roster):
            if name not in packets:continue
            p=packets[name];a=int(chosen[i]);probs=tuple(dist.probs[i].tolist())
            sample=SampledAction(a,math.log(probs[a]),tuple(p['action_mask']),probs,'phase4b-sensor-feedback-v2',p['observation_token'])
            validate_sample(sample,p['action_mask'],p['observation_token'])
            assignments[name]=None if a==0 else p['self_assignment'] if a==1 else p['job_ids'][a-2]
            records[name]={'sample':asdict(sample),'packet':p}
        self.last={'actor_samples':records,'new_threat_channels_used':False,'completion_feedback':'original_phase3a_Perception'}
        return Action(tuple(sorted(assignments.items())))


def run_all(output):
    torch.set_num_threads(1);frozen=ensure_frozen(ROOT,output);target=output/'sensor_feedback_transfer';target.mkdir(exist_ok=False)
    specs=read(output/'design/diagnostic_manifest.json');summary={}
    save(target/'protocol.json',{'reason':'Active-only projection retained stale remaining-work values after completion. Keep v1 and evaluate original sensor/cache feedback separately.',
        'original_experiments_preserved':True,'native_policies_reselected':False,'weights_changed':False,
        'added_public_data':'only already spawned current completed/expired job feedback; no future jobs/events',
        'source_sha256':sha(Path(__file__))})
    for seed,cp in frozen['checkpoints'].items():
        net=load_network(ROOT/cp['path'],cp['sha256'])
        for mode in ('greedy','stochastic'):
            label=f'seed_{seed}_{mode}';folder=target/label;folder.mkdir();raw=folder/'raw';raw.mkdir();records=[]
            for spec in specs:
                for repeat in range(1 if mode=='greedy' else 2):
                    p=SensorActor(net,mode,int(seed)*100000000+spec['event_seed']+repeat*1000000)
                    engine=FeedbackSimulation(spec,p);rows=engine.run();m=evaluate_new(rows)
                    m.update(scenario_id=spec['scenario_id'],family=spec['family'],team=spec['team'],seed=seed,repeat=repeat)
                    records.append(m);write_raw(raw/f"{spec['scenario_id']}_r{repeat}.jsonl.gz",rows)
            data={'label':label,'summary':grouped(records),'episodes':records};save(folder/'results.json',data);summary[label]=data['summary']
            print('sensor-feedback transfer',label,flush=True)
    save(target/'summary.json',summary)


def audit(output):
    target=output/'sensor_feedback_transfer';frozen=ensure_frozen(ROOT,output);count=samples=0;changes={}
    for folder in target.iterdir():
        if not folder.is_dir():continue
        new=read(folder/'results.json');old=read(output/'comparison'/folder.name/'results.json')
        original={(r['scenario_id'],r['repeat']):r for r in old['episodes']};changed=0
        for entry in new['episodes']:
            rows=read_raw(folder/'raw'/f"{entry['scenario_id']}_r{entry['repeat']}.jsonl.gz");m=evaluate_new(rows)
            for k,v in m.items():assert v==entry[k],(folder.name,k)
            for e in rows:
                if e.get('event')!='policy_selection':continue
                feedback=e['public_observation']['current_observed_job_feedback']
                for j in feedback.values():assert j['spawn']<=e['time']
                for item in e['action']['policy_metadata']['actor_samples'].values():
                    s=item['sample'];p=item['packet']
                    validate_sample(SampledAction(s['action'],s['log_probability'],tuple(s['action_mask']),tuple(s['probabilities']),s['policy_version'],s['observation_token']),p['action_mask'],p['observation_token']);samples+=1
            count+=1;prior=original[entry['scenario_id'],entry['repeat']]
            changed+=any(entry[k]!=prior[k] for k in ('task_success_rate','threat_pass_rate','legacy_gap_region_seconds'))
        changes[folder.name]=changed
    save(target/'audit.json',{'status':'PASS','raw_episodes':count,'sample_records':samples,'changed_outcome_episodes_vs_v1':changes,
        'no_future_feedback':True,'weights_changed':False,'source_sha256':sha(Path(__file__))})
    print('feedback audit PASS',count,'episodes',changes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('run','audit'))
    parser.add_argument('--output',type=Path,default=ROOT/'phase4b_outputs/study_v1');a=parser.parse_args()
    {'run':run_all,'audit':audit}[a.command](a.output)
