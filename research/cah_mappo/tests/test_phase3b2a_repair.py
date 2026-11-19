"""Slot bijection, physical invariance, delayed information and PPO contracts."""
import copy
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from cah_mappo.phase3b2a_repair.slots import SlotMap,PERMUTATIONS
from cah_mappo.phase3b2a_repair.scenarios import generate
from cah_mappo.phase3b2a_repair.environment import AuditedEnv
from cah_mappo.phase3b2a_repair.evaluation import evaluate_one,timelines,coverage_windows
from cah_mappo.phase3b2a_repair.learner import RepairLearner
from cah_mappo.phase3a.contracts import SampledAction,validate_ppo_record
from cah_mappo.phase3b2a.experiment import load_net
from cah_mappo.phase3b2a.scenarios import generate as old_generate

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'phase3b2a_outputs/pilot_v1/C_weight2/checkpoints/final.pt'


class SlotTests(unittest.TestCase):
    def test_bijection_and_distributions_for_all_permutations(self):
        for order in PERMUTATIONS:
            layout=SlotMap(order)
            for action in range(10):self.assertEqual(layout.action_from_core(layout.action_to_core(action)),action)
            vals=tuple(i/45 for i in range(10)); mapped=layout.to_core_distribution(vals)
            self.assertAlmostEqual(sum(vals),sum(mapped))
            for a in range(10):self.assertEqual(vals[a],mapped[layout.action_to_core(a)])
        with self.assertRaises(ValueError):SlotMap((2,2,3))

    def test_job_mask_features_and_assignment_onehot_are_consistent(self):
        env=AuditedEnv(generate('fixed','development',0),(4,2,3))
        for _ in range(9):
            for name,p in env.obs.items():
                self.assertEqual(len(p['observation']),146)
                for i,jid in enumerate(p['job_ids']):
                    self.assertEqual(p['job_mask'][i],int(jid is not None))
                    self.assertEqual(p['observation'][130+i],p['job_mask'][i])
                    if p['self_assignment']==jid and jid is not None:self.assertEqual(p['observation'][10+i],1)
            self.assertEqual(len(env.state()['state']),141)
            env.step({n:next((i for i in range(2,10) if p['action_mask'][i] and p['job_ids'][i-2].startswith('task')),1 if p['action_mask'][1] else 0) for n,p in env.obs.items()})

    def test_same_physical_actions_produce_identical_truth_and_rewards(self):
        spec=generate('fixed','development',0)
        a,b=AuditedEnv(spec),AuditedEnv(spec,(4,3,2))
        for _ in range(32):
            aa={};bb={}
            for name,p in a.obs.items():
                target=next((jid for i,jid in enumerate(p['job_ids']) if jid and jid.startswith('task') and p['action_mask'][i+2]),None)
                for env,actions in ((a,aa),(b,bb)):
                    packet=env.obs[name]
                    actions[name]=packet['job_ids'].index(target)+2 if target else 1 if packet['action_mask'][1] else 0
            _,ra,*_=a.step(aa);_,rb,*_=b.step(bb)
            self.assertEqual(ra,rb);self.assertEqual(a.engine.truth(),b.engine.truth())
        self.assertEqual(a.metrics(),b.metrics())

    def test_episode_slots_are_stable_and_do_not_reveal_future_ids(self):
        env=AuditedEnv(generate('fixed','development',0),(4,3,2)); old={n:{} for n in env.obs}
        self.assertTrue(all(not any(j and j.startswith('task') for j in p['job_ids']) for p in env.obs.values()))
        for _ in range(12):
            for n,p in env.obs.items():
                for jid,slot in old[n].items():self.assertEqual(p['job_ids'][slot],jid)
                old[n]={jid:i for i,jid in enumerate(p['job_ids']) if jid}
            env.step({n:0 for n in env.obs})

    def test_critic_slot_mapping_has_an_exact_inverse(self):
        spec=generate('fixed','development',0)
        for order in PERMUTATIONS:
            env=AuditedEnv(spec,order)
            for _ in range(8):env.step({n:0 for n in env.obs})
            inverse=SlotMap(tuple(env.layout.old_to_new[i] for i in (2,3,4)))
            self.assertEqual(inverse.critic(env.state()),env.core.state())

    def test_sampled_logp_mask_and_id_survive_translation(self):
        env=AuditedEnv(generate('fixed','development',0),(4,2,3));net=load_net(CHECKPOINT)
        for _ in range(8):
            actions={}
            for n,p in env.obs.items():
                dist=net.distribution(torch.tensor([p['observation']]),torch.tensor([p['action_mask']]))
                action=int(dist.probs.argmax(-1)[0]);probs=tuple(dist.probs[0].tolist())
                actions[n]=SampledAction(action,float(dist.log_prob(torch.tensor([action]))[0]),tuple(p['action_mask']),probs,'test',p['observation_token'])
            before=copy.deepcopy(env.obs);_,_,_,_,infos=env.step(actions)
            for n,info in infos.items():
                r=info['transition'];validate_ppo_record(r)
                self.assertEqual(r['sampled_action'],actions[n].action)
                self.assertEqual(r['log_probability'],actions[n].log_probability)
                if actions[n].action>=2:self.assertEqual(r['requested_assignment'],before[n]['job_ids'][actions[n].action-2])

    def test_stale_view_rejected_atomically(self):
        env=AuditedEnv(generate('fixed','development',0));before=copy.deepcopy(env.engine.rows)
        p=env.obs['A'];bad=SampledAction(0,0.,tuple(p['action_mask']),(1.,)+(0.,)*9,'bad','wrong-token')
        with self.assertRaises(ValueError):env.step({'A':bad,'B':0,'C':0,'D':0})
        self.assertEqual(before,env.engine.rows)


class ObservationAndLearningTests(unittest.TestCase):
    def test_adapter_does_not_leak_hidden_peer_before_message_arrives(self):
        a=old_generate('D','train',53);a['split']='development'
        for i,r in enumerate(a['config']['robots']):r['position']=[i*10.,0.]
        a['settings']['sensor_radius']=1.;a['settings']['radio_delay']=2.
        b=copy.deepcopy(a);b['config']['robots'][1]['position']=[15.,0.]
        ea,eb=AuditedEnv(a,(4,3,2),'realistic'),AuditedEnv(b,(4,3,2),'realistic')
        self.assertNotEqual(ea.state(),eb.state());self.assertEqual(ea.obs['A'],eb.obs['A'])
        ea.step({n:0 for n in ea.obs});eb.step({n:0 for n in eb.obs})
        self.assertEqual(ea.obs['A'],eb.obs['A'])
        ea.step({n:0 for n in ea.obs});eb.step({n:0 for n in eb.obs})
        self.assertNotEqual(ea.obs['A'],eb.obs['A'])

    def test_trace_no_task_before_spawn_and_delayed_message_not_early(self):
        spec=old_generate('D','train',50);spec['split']='development'
        spec['config']['events']=[]
        for i,r in enumerate(spec['config']['robots']):r['position']=[i*10.,0.]
        job=spec['config']['jobs'][2];job['position']=[0.,0.];job['spawn']=2.;job['deadline']=24.
        spec['settings']['sensor_radius']=1.;spec['settings']['radio_delay']=2.
        env=AuditedEnv(spec,condition='realistic')
        for _ in range(6):env.step({n:0 for n in env.obs})
        events=env.engine.rows
        sensor=next(e['time'] for e in events if e.get('event')=='task_sensor_first' and e['robot']=='A' and e['job']=='task0')
        receipt=next(e['time'] for e in events if e.get('event')=='task_message_first' and e['robot']=='B' and e['job']=='task0')
        self.assertEqual(sensor,2.);self.assertEqual(receipt,4.)

    def test_no_information_and_known_unselected_are_distinct(self):
        for condition in ('controlled','realistic'):
            spec=old_generate('D','train',51);spec['split']='development'
            for job in spec['config']['jobs']:
                if job['kind']=='task':job['position']=[60.,60.]
            env=AuditedEnv(spec,condition=condition)
            for _ in range(32):env.step({n:0 for n in env.obs})
            ts=timelines(env.log_rows())
            category='known_but_not_selected' if condition=='controlled' else 'no_timely_information'
            self.assertTrue(all(t['category']==category for t in ts))

    def test_critic_changes_cannot_change_actor_packet(self):
        env=AuditedEnv(generate('fixed','development',0),(4,3,2));obs=copy.deepcopy(env.obs)
        state=env.state();state['state']=[999.]*141
        self.assertEqual(obs,env.obs)
        net=load_net(CHECKPOINT);vectors=torch.tensor([p['observation'] for p in obs.values()]);masks=torch.tensor([p['action_mask'] for p in obs.values()])
        a=net.distribution(vectors,masks).probs
        b=net.distribution(vectors.flip(0),masks.flip(0)).probs.flip(0)
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=0))
        self.assertFalse(torch.equal(a[2],a[3]))

    def test_training_view_logp_and_team_critic_shapes(self):
        learner=RepairLearner([generate('positions','train',i) for i in range(2)],CHECKPOINT,True)
        batch=learner.collect(64);data=learner.tensors(batch)
        self.assertEqual(tuple(data['state'].shape),(64,141))
        self.assertLess(learner.ratio_before_update(data),1e-6)
        report=learner.update(batch)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for k,v in report.items() if not isinstance(v,dict)))
        self.assertTrue(all(not b['continuation'] and b['bootstrap'] for b in (batch[31],batch[63])))

    def test_deterministic_collect_and_augmentation_covers_all_orders(self):
        specs=[generate('positions','train',0)]
        a=RepairLearner(specs,CHECKPOINT,True);b=RepairLearner(specs,CHECKPOINT,True)
        ba,bb=a.collect(32*6),b.collect(32*6)
        self.assertEqual(ba,bb)
        self.assertEqual(len({tuple(ba[i*32]['slot_order']) for i in range(6)}),6)

    def test_holdout_is_rejected_before_environment_or_training(self):
        spec=generate('positions','final_holdout',0)
        with self.assertRaises(ValueError):AuditedEnv(spec)
        with self.assertRaises(ValueError):RepairLearner([spec],CHECKPOINT)

    def test_same_window_different_denominator_is_explicit(self):
        env=AuditedEnv(generate('fixed','development',0))
        for _ in range(32):env.step({n:0 for n in env.obs})
        windows=coverage_windows(env.log_rows())
        self.assertEqual([w['responsibility_region_seconds'] for w in windows],[16.,32.,64.])
        self.assertTrue(all(w['coverage_gap_ratio']==0 for w in windows))


if __name__=='__main__':unittest.main()
