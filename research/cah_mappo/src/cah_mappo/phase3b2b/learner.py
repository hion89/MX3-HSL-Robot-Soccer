"""Independent from-scratch runs; frozen PPO update with exact boundary resume."""
import copy
from dataclasses import asdict
from pathlib import Path
import torch
from ..phase3b1.learner import Learner,TrainConfig
from ..phase3a.adapters import padded_batch
from ..phase3a.contracts import SampledAction,validate_ppo_record
from .scenarios import SEEDS,scenario,native_env,VERSION
from .metrics import score


class BaselineLearner(Learner):
    def __init__(self,stream=0,seed=None):
        self.stream=stream
        super().__init__(TrainConfig(seed=SEEDS[stream] if seed is None else seed))
        self.policy_updates=0
        self.env,self.obs=self._new_episode()

    def _new_episode(self):return native_env(scenario('train',self.episodes,self.stream))

    @torch.no_grad()
    def collect(self,n=256):
        batch=[];self.completed=[];self.last_episode_logs=[]
        for _ in range(n):
            packed=padded_batch(self.obs,self.env.state(),self.env.settings)
            vectors=torch.tensor(packed['actor_observations'],dtype=torch.float32)
            masks=torch.tensor(packed['available_actions'],dtype=torch.bool)
            state=self.env.state()['state']
            actions,logp,probabilities=self.net.sample(vectors,masks,self.sample_rng)
            value=self.net.value(torch.tensor([state])).item()
            job_ids=[self.obs[name]['job_ids'] for name in self.env.settings.roster]
            assignments=[self.obs[name]['self_assignment'] for name in self.env.settings.roster]
            samples={name:SampledAction(int(actions[i]),float(logp[i]),tuple(masks[i].int().tolist()),
                                      tuple(probabilities[i].tolist()),f'baseline-update-{self.policy_updates}',self.obs[name]['observation_token'])
                     for i,name in enumerate(self.env.settings.roster)}
            next_obs,rewards,_,_,infos=self.env.step(samples)
            next_value=self.net.value(torch.tensor([self.env.state()['state']])).item()
            executed=[];changed=[]
            for name in self.env.settings.roster:
                record=infos[name]['transition'];validate_ppo_record(record)
                executed.append(record['executed_action']);changed.append(record['arbitration_changed'])
            first=infos['A']['transition']
            batch.append({'obs':packed['actor_observations'],'mask':packed['available_actions'],'state':state,
                          'actions':actions.tolist(),'old_logp':logp.tolist(),'active':[v[0] for v in packed['active_masks']],
                          'value':value,'next_value':next_value,'reward':rewards['A'],
                          'bootstrap':first['bootstrap_mask'],'continuation':first['continuation_mask'],
                          'executed_actions':executed,'arbitration_changed':changed,'job_ids':job_ids,
                          'assignments':assignments,'probabilities':probabilities.tolist(),
                          'scenario_id':self.env.engine.rows[0]['phase3b2b']['scenario_id'],
                          'episode':self.episodes,'time':self.env.time-1,'policy_update':self.policy_updates})
            self.steps+=1
            if self.env.done:
                rows=self.env.log_rows()
                self.completed.append({'episode':self.episodes,'scenario_id':batch[-1]['scenario_id'],**score(rows,self.env.episode_return)})
                self.last_episode_logs.append((self.episodes,rows))
                self.episodes+=1;self.env,self.obs=self._new_episode()
            else:self.obs=next_obs
        return batch

    def update(self,batch):
        result=super().update(batch)
        self.policy_updates+=1
        return result

    def save_exact(self,path):
        if self.env.time!=0 or self.steps%32:raise ValueError('Exact checkpoints require a fresh episode boundary after PPO update')
        payload={'format':'phase3b2b-boundary-v1','generator':VERSION,'config':asdict(self.c),'stream':self.stream,
                 'network':self.net.state_dict(),'actor_optimizer':self.actor_optimizer.state_dict(),
                 'critic_optimizer':self.critic_optimizer.state_dict(),'sample_rng':self.sample_rng.get_state(),
                 'shuffle_rng':self.shuffle_rng.get_state(),'torch_rng':torch.get_rng_state(),
                 'steps':self.steps,'episodes':self.episodes,'policy_updates':self.policy_updates,
                 'boundary_truth':self.env.engine.truth(),'boundary_observations':self.obs,
                 'boundary_critic':self.env.state(),'source_config':'no mid-episode serialization; boundary state reconstructed and verified'}
        with Path(path).open('xb') as f:torch.save(payload,f)

    @classmethod
    def restore_exact(cls,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['format']!='phase3b2b-boundary-v1' or saved['generator']!=VERSION:raise ValueError('Checkpoint protocol mismatch')
        learner=cls(saved['stream'],saved['config']['seed'])
        learner.net.load_state_dict(saved['network']);learner.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        learner.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        learner.sample_rng.set_state(saved['sample_rng']);learner.shuffle_rng.set_state(saved['shuffle_rng']);torch.set_rng_state(saved['torch_rng'])
        learner.steps,learner.episodes,learner.policy_updates=saved['steps'],saved['episodes'],saved['policy_updates']
        learner.env,learner.obs=learner._new_episode()
        if (learner.env.engine.truth()!=saved['boundary_truth'] or learner.obs!=saved['boundary_observations']
                or learner.env.state()!=saved['boundary_critic']):raise ValueError('Boundary reconstruction differs')
        return learner
