"""Fixed network/optimizer/PPO; only episode-stable slot augmentation differs."""
from collections import Counter
import torch
from ..phase3b1.learner import Learner, TrainConfig
from ..phase3a.adapters import padded_batch
from ..phase3a.contracts import SampledAction, validate_ppo_record
from .environment import AuditedEnv
from .slots import PERMUTATIONS
from .evaluation import score


class RepairLearner(Learner):
    def __init__(self, specs, checkpoint, augment=False, seed=7):
        if not specs or any(s['split'] != 'train' for s in specs):raise ValueError('Training split only')
        super().__init__(TrainConfig(seed=seed))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.net.load_state_dict(saved['network'])
        self.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        self.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        self.specs, self.augment = specs, augment
        self.policy_updates = 0
        self.env = self._episode(); self.obs = self.env.obs

    def _episode(self):
        n = len(self.specs); i = self.episodes % n
        layout_index = (self.episodes // n + i) % 6 if self.augment else 0
        return AuditedEnv(self.specs[i], PERMUTATIONS[layout_index])

    @torch.no_grad()
    def collect(self, n=256):
        batch = []; self.completed = []; self.last_episode_logs = []
        for _ in range(n):
            packed = padded_batch(self.obs, self.env.state(), self.env.settings)
            vectors = torch.tensor(packed['actor_observations'], dtype=torch.float32)
            masks = torch.tensor(packed['available_actions'], dtype=torch.bool)
            state = self.env.state()['state']
            selected, logp, probs = self.net.sample(vectors, masks, self.sample_rng)
            value = self.net.value(torch.tensor([state])).item()
            job_ids = [self.obs[name]['job_ids'] if name in self.obs else [None]*8 for name in self.env.settings.roster]
            assignments = [self.obs[name]['self_assignment'] if name in self.obs else None for name in self.env.settings.roster]
            actions = {}
            for name in self.env.agents:
                i = self.env.settings.roster.index(name)
                actions[name] = SampledAction(int(selected[i]), float(logp[i]), tuple(masks[i].int().tolist()),
                                              tuple(probs[i].tolist()), f'repair-update-{self.policy_updates}', self.obs[name]['observation_token'])
            next_obs, rewards, _, _, infos = self.env.step(actions)
            next_value = self.net.value(torch.tensor([self.env.state()['state']])).item()
            first = next(iter(infos.values()))['transition']
            executed, changed = [0]*4,[False]*4
            for name, info in infos.items():
                record=info['transition']; validate_ppo_record(record)
                i=self.env.settings.roster.index(name)
                assert record['sampled_action'] == int(selected[i])
                executed[i],changed[i]=record['executed_action'],record['arbitration_changed']
            batch.append({'obs':packed['actor_observations'],'mask':packed['available_actions'],'state':state,
                          'actions':selected.tolist(),'old_logp':logp.tolist(),'active':[v[0] for v in packed['active_masks']],
                          'value':value,'next_value':next_value,'reward':next(iter(rewards.values())),
                          'bootstrap':first['bootstrap_mask'],'continuation':first['continuation_mask'],
                          'executed_actions':executed,'arbitration_changed':changed,'probabilities':probs.tolist(),
                          'job_ids':job_ids,'assignments':assignments,'scenario_id':self.specs[self.episodes % len(self.specs)]['scenario_id'],
                          'slot_order':list(self.env.layout.new_to_old),'policy_update':self.policy_updates})
            self.steps += 1
            if self.env.done:
                log=self.env.log_rows()
                self.completed.append({'episode':self.episodes,'return':self.env.episode_return,
                                       'scenario_id':batch[-1]['scenario_id'],**score(log)})
                self.last_episode_logs.append((self.episodes,log))
                self.episodes += 1; self.env=self._episode(); self.obs=self.env.obs
            else:self.obs=next_obs
        return batch

    def update(self, batch):
        data=self.tensors(batch)
        with torch.no_grad():
            old=self.net.distribution(data['obs'].reshape(-1,146),data['mask'].reshape(-1,10)).probs
        report=super().update(batch)
        with torch.no_grad():
            new=self.net.distribution(data['obs'].reshape(-1,146),data['mask'].reshape(-1,10)).probs
            live=data['active'].reshape(-1).bool()
            kl=(old*(old.clamp_min(1e-30).log()-new.clamp_min(1e-30).log())).sum(-1)[live]
            change=(new-old).abs().max(-1).values[live]
            report.update(post_update_kl_mean=float(kl.mean()) if kl.numel() else 0.,
                          post_update_kl_max=float(kl.max()) if kl.numel() else 0.,
                          max_action_probability_change=float(change.max()) if change.numel() else 0.)
        jobs=Counter(); actions=Counter()
        for row in batch:
            for i,a in enumerate(row['actions']):
                if not row['active'][i]:continue
                actions[str(a)]+=1
                jid=None if a==0 else row['assignments'][i] if a==1 else row['job_ids'][i][a-2]
                if jid is not None:jobs[jid]+=1
        report['job_selection_counts']=dict(jobs);report['action_selection_counts']=dict(actions)
        self.policy_updates+=1
        return report
