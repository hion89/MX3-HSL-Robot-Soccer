"""Independent seeds, actor/critic separation, common metrics and exact resume."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from cah_mappo.phase3b2b.scenarios import scenario,native_env,SEEDS
from cah_mappo.phase3b2b.learner import BaselineLearner
from cah_mappo.phase3b2b.evaluation import evaluate_one
from cah_mappo.phase3b2b.metrics import score,reference_rewards
from cah_mappo.phase3b2b.training import selection_key,validate
from cah_mappo.phase3b2b.io import write_raw,read_raw
from cah_mappo.phase2.model import config_from_dict
from cah_mappo.phase2.engine import Environment
from cah_mappo.phase2.policies import make_policy
from cah_mappo.phase3b2b.statistics import seed_statistics


class ScenarioTests(unittest.TestCase):
    def test_disjoint_banks_and_seed_streams(self):
        seeds={split:{scenario(split,i,stream)['event_seed'] for i in range(10) for stream in range(5)}
               for split in ('train','validation','benchmark')}
        self.assertFalse(seeds['train']&seeds['validation'] or seeds['train']&seeds['benchmark'] or seeds['validation']&seeds['benchmark'])
        self.assertEqual(len(seeds['train']),50)
        with self.assertRaises(ValueError):scenario('final_holdout',0)
        with self.assertRaises(ValueError):native_env({'split':'final_holdout'})

    def test_scenarios_reproduce_and_task_count_varies(self):
        self.assertEqual(scenario('train',3),scenario('train',3))
        counts={len(scenario('train',i)['config']['jobs'])-2 for i in range(100)}
        self.assertEqual(counts,{1,2,3})
        self.assertTrue(all(len(scenario('benchmark',i)['config']['jobs'])==5 for i in range(10)))

    def test_information_is_observed_at_spawn_not_from_future(self):
        env,obs=native_env(scenario('benchmark',0))
        self.assertTrue(all(not any(j and j.startswith('task') for j in p['job_ids']) for p in obs.values()))
        for _ in range(2):obs,*_=env.step({name:0 for name in obs})
        self.assertTrue(all('task0' in p['job_ids'] for p in obs.values()))
        self.assertTrue(all('task1' not in p['job_ids'] for p in obs.values()))
        self.assertEqual(len(env.state()['state']),141)

    def test_unspawned_hidden_positions_do_not_change_actor_or_critic(self):
        a=scenario('benchmark',5);b=copy.deepcopy(a)
        b['config']['jobs'][2]['position']=[9000.,9000.]
        ea,oa=native_env(a);eb,ob=native_env(b)
        self.assertEqual(oa,ob);self.assertEqual(ea.state(),eb.state())

    def test_common_physical_plan_for_local_central_and_slots(self):
        spec=scenario('benchmark',0);net=BaselineLearner().net
        a,_=evaluate_one(spec,'greedy',net)
        b,_=evaluate_one(spec,'greedy',net,(4,3,2))
        c,_=evaluate_one(spec,'central_fixed')
        self.assertEqual(a['exogenous_sha256'],b['exogenous_sha256'])
        self.assertEqual(a['exogenous_sha256'],c['exogenous_sha256'])


class TrainingTests(unittest.TestCase):
    def test_independent_initializations_and_environment_streams(self):
        a,b=BaselineLearner(0),BaselineLearner(1)
        self.assertTrue(any(not torch.equal(p,b.net.state_dict()[k]) for k,p in a.net.state_dict().items()))
        self.assertNotEqual(a.env.engine.plan,b.env.engine.plan)
        self.assertEqual(a.steps,0);self.assertEqual(b.steps,0)

    def test_same_seed_reproduces_collect_and_update(self):
        a,b=BaselineLearner(0),BaselineLearner(0)
        ba,bb=a.collect(32),b.collect(32);self.assertEqual(ba,bb)
        self.assertEqual(a.update(ba),b.update(bb))

    def test_exact_resume_matches_uninterrupted_training_and_rng(self):
        a=BaselineLearner(0);a.update(a.collect(256))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'boundary.pt';a.save_exact(path);b=BaselineLearner.restore_exact(path)
            ba,bb=a.collect(256),b.collect(256);self.assertEqual(ba,bb)
            self.assertEqual(a.update(ba),b.update(bb))
            self.assertTrue(all(torch.equal(p,b.net.state_dict()[k]) for k,p in a.net.state_dict().items()))
            self.assertTrue(torch.equal(a.sample_rng.get_state(),b.sample_rng.get_state()))
            self.assertTrue(torch.equal(a.shuffle_rng.get_state(),b.shuffle_rng.get_state()))

    def test_mid_episode_checkpoint_rejected(self):
        a=BaselineLearner();a.collect(1)
        with self.assertRaises(ValueError):a.save_exact('must_not_be_created.pt')

    def test_mask_logp_team_value_and_truncation(self):
        a=BaselineLearner();batch=a.collect(64);data=a.tensors(batch)
        self.assertEqual(tuple(data['state'].shape),(64,141))
        self.assertLess(a.ratio_before_update(data),1e-6)
        self.assertEqual([(batch[i]['bootstrap'],batch[i]['continuation']) for i in (31,63)],[(1,0),(1,0)])

    def test_selection_uses_both_modes_not_return(self):
        def reports(g,s,gap=.2):return {m:{'summary':{'task_success_rate':v,'coverage_gap_ratio':gap,'restricted_response_mean_s':5,'episode_return':999}}
                                      for m,v in (('greedy',g),('stochastic',s))}
        self.assertGreater(selection_key(reports(.9,.9)),selection_key(reports(1,.5)))
        with self.assertRaises(ValueError):validate(BaselineLearner().net,[scenario('benchmark',0)])


class EvaluationTests(unittest.TestCase):
    def test_policy_evaluation_does_not_call_network_critic(self):
        net=BaselineLearner().net
        def forbidden(*args,**kwargs):raise AssertionError('Privileged network critic must not be used for policy evaluation')
        net.critic.forward=forbidden
        for mode in ('greedy','stochastic'):
            result,_=evaluate_one(scenario('benchmark',6),mode,net)
            self.assertEqual(result['task_count'],3)
    def test_reference_reward_matches_same_scripted_native_trajectory(self):
        spec=scenario('benchmark',1);config=config_from_dict(spec['config'])
        rows=Environment(config,make_policy('fixed',config),'central_fixed').run()
        env,obs=native_env(spec)
        selections=[r for r in rows if r['type']=='interval']
        for row in selections:
            actions={name:0 if jid is None else obs[name]['job_ids'].index(jid)+2 for name,jid in row['selection']['accepted'].items()}
            obs,*_=env.step(actions)
        self.assertEqual(env.engine.truth(),rows[-1]['truth'])
        self.assertAlmostEqual(reference_rewards(rows),env.episode_return,places=12)
        self.assertEqual(score(rows)['coverage_gap_ratio'],score(env.log_rows())['coverage_gap_ratio'])

    def test_all_metrics_and_fixed_denominator_present(self):
        result,rows=evaluate_one(scenario('benchmark',2),'local_fixed')
        for name in ('task_success_rate','deadline_miss_rate','handover_gap_mean_s','response_time_mean_s','task_completion_time_s',
                     'unobserved_task_rate','role_switch_count','task_duplicate_rate','episode_return'):self.assertIn(name,result)
        self.assertEqual(result['responsibility_region_seconds'],64)
        self.assertEqual(result['risk_weighted_region_seconds'],64)
        self.assertEqual(result['evaluation_seconds'],32)
        self.assertEqual(result['unobserved_task_rate'],0)

    def test_compressed_raw_log_recalculates(self):
        result,rows=evaluate_one(scenario('benchmark',3),'local_fixed')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.jsonl.gz';write_raw(path,rows)
            reloaded=read_raw(path);measured=score(reloaded)
            self.assertTrue(all(result[k]==v for k,v in measured.items()))


class StatisticalTests(unittest.TestCase):
    def test_seed_level_t_interval_has_five_replication_units(self):
        import math
        result=seed_statistics([1.,2.,3.,4.,5.])
        self.assertEqual(result['n_training_seeds'],5)
        self.assertEqual(result['mean'],3.);self.assertEqual(result['median'],3.)
        self.assertAlmostEqual(result['sd'],math.sqrt(2.5),places=12)
        self.assertAlmostEqual(result['t95'][0],3-2.776445105/math.sqrt(2),places=12)

    def test_undefined_handover_is_not_changed_to_zero(self):
        result=seed_statistics([None]*5)
        self.assertEqual(result['n_training_seeds'],0)
        self.assertIsNone(result['mean']);self.assertIsNone(result['t95'])

    def test_identical_seed_means_have_zero_observed_variance(self):
        result=seed_statistics([.75]*5)
        self.assertEqual(result['sd'],0.);self.assertEqual(result['t95'],[.75,.75])


if __name__=='__main__':unittest.main()
