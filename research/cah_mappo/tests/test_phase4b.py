import copy
from dataclasses import asdict,replace
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cah_mappo.phase2.model import Config,RobotSpec as R,JobSpec as J,Action
from cah_mappo.phase2.metrics import evaluate as evaluate_old
from cah_mappo.phase3b2b.io import write_raw,read_raw
from cah_mappo.phase4b.scenarios import scenario,FAMILIES
from cah_mappo.phase4b.engine import Simulation
from cah_mappo.phase4b.model import match,response_time
from cah_mappo.phase4b.metrics import evaluate_new,aggregate
from cah_mappo.phase4b.policies import Native,FrozenActor
from cah_mappo.phase4b.planning import OfflineSchedule,forecast


class Script:
    name='unit_script'
    def __init__(self,fn):self.fn=fn
    def reset(self,seed):self.last={}
    def decide(self,v):return self.fn(v)


def unit_spec(task=True,threat=True,radius=0):
    jobs=[J('base0',(0.,0.),kind='base',spawn=0,deadline=32,service=0,fixed_owner='A')]
    if task:jobs.append(J('task0',(1.,0.),spawn=0,deadline=10,service=2,fixed_owner='B'))
    c=Config('unit',tuple([R('A',(0.,0.),initial_job='base0',base_role='guard'),R('B',(-1.,0.))]),tuple(jobs),
             dt=.25,horizon=32,radius=radius)
    return {'split':'unit','scenario_id':'unit','family':'unit','team':2,'config':asdict(c),'control_period':1.,
            'threats':[{'id':'q0','target':'base0','origin':[4.,0.],'speed':1.,'start':0.,'reveal':0.,'intensity':1.}] if threat else [],
            'response_budgets':{'base0':2.}}


def idle(view):return Action(tuple((r.id,None) for r in view.allocation.robots))
def task_A(view):return Action((('A','task0'),('B',None)))


class MetricDefinitions(unittest.TestCase):
    def test_ownership_absent_can_still_be_physically_covered(self):
        rows=Simulation(unit_spec(False),Script(idle)).run();m=evaluate_new(rows)
        self.assertEqual(m['ownership_gap_region_seconds'],32)
        self.assertEqual(m['physical_gap_region_seconds'],0)
        self.assertEqual(m['legacy_gap_region_seconds'],0)
        self.assertEqual(m['threat_pass_rate'],0)

    def test_named_owner_far_away_is_not_physical_coverage(self):
        s=unit_spec(False);s['config']['robots'][0]['position']=[5.,0.];s['config']['robots'][0]['speed']=.1
        s['config']['robots'][1]['operational']=False
        rows=Simulation(s,Native('fixed')).run();m=evaluate_new(rows)
        self.assertEqual(m['ownership_gap_region_seconds'],0)
        self.assertGreater(m['physical_gap_region_seconds'],0)
        self.assertEqual(m['threat_pass_rate'],1)

    def test_potential_eta_coverage_is_not_a_defense_guarantee(self):
        s=unit_spec(False);s['config']['robots'][0]['position']=[1.,0.]
        rows=Simulation(s,Script(idle)).run();m=evaluate_new(rows)
        self.assertEqual(m['physical_gap_region_seconds'],0)
        self.assertEqual(m['threat_pass_rate'],1)
        self.assertGreater(m['threat_weighted_gap_seconds'],0)

    def test_finish_then_respond_respects_service_capacity(self):
        s=unit_spec();engine=Simulation(s,Script(task_A));engine.boundary(0)
        engine.apply(0,Action((('A','task0'),('B',None))))
        state=engine.truth();state['time']=0;state['control_period']=1
        robot=state['robots']['A']
        # Travel 1, service 2, rotate pi radians (1s), return 1: five seconds.
        self.assertEqual(response_time(robot,'base0',state),5)
        query={'x':('base0',1)}
        state['robots']['B']['operational']=False
        self.assertEqual(match(state,query,{'x':2}),{})
        self.assertIn('x',match(state,query,{'x':5}))

    def test_no_threat_has_NA_denominator_not_a_perfect_defense_rate(self):
        m=evaluate_new(Simulation(unit_spec(False,False),Script(idle)).run())
        self.assertIsNone(m['threat_pass_rate']);self.assertIsNone(m['threat_weighted_gap_ratio'])
        self.assertEqual(m['threat_exposure_seconds'],0)

    def test_post_task_duty_remains_active_and_can_fail_later(self):
        s=unit_spec();s['threats'][0].update(start=16,reveal=16)
        rows=Simulation(s,Script(task_A)).run();m=evaluate_new(rows)
        self.assertEqual(m['post_task_start'],3)
        self.assertEqual(m['responsibility_region_seconds'],32)
        self.assertEqual(m['post_task_region_seconds'],29)
        self.assertEqual(m['post_task_legacy_gap_s'],29)
        self.assertEqual(m['post_task_threat_passes'],1)

    def test_post_task_metric_is_NA_if_some_task_failed(self):
        s=unit_spec();s['config']['jobs'][1]['deadline']=1
        m=evaluate_new(Simulation(s,Script(task_A)).run())
        self.assertIsNone(m['post_task_start']);self.assertIsNone(m['post_task_legacy_gap_s'])

    def test_old_CGR_is_unchanged(self):
        rows=Simulation(scenario('validation','C_reactive',4,0),Native('exact')).run()
        self.assertEqual(evaluate_new(rows)['legacy_coverage_gap_ratio'],evaluate_old(rows)['coverage_gap_ratio'])

    def test_one_robot_cannot_block_two_simultaneous_attacks(self):
        s=unit_spec(False);s['config']['robots'][1]['operational']=False
        q=copy.deepcopy(s['threats'][0]);q['id']='q1';s['threats'].append(q)
        m=evaluate_new(Simulation(s,Native('fixed')).run())
        self.assertEqual(m['threat_arrivals'],2);self.assertEqual(m['threat_defended_count'],1);self.assertEqual(m['threat_pass_count'],1)

    def test_one_robot_cannot_cover_two_response_regions(self):
        s=unit_spec(False);s['config']['robots'][1]['operational']=False
        s['config']['jobs']=[*s['config']['jobs'],asdict(J('base1',(0.,1.),kind='base',spawn=0,deadline=32,service=0))]
        s['response_budgets']['base1']=2
        m=evaluate_new(Simulation(s,Native('fixed')).run())
        self.assertEqual(m['physical_gap_region_seconds'],32)
        self.assertEqual(m['responsibility_region_seconds'],64)

    def test_raw_jsonl_alone_recomputes_every_metric(self):
        rows=Simulation(scenario('validation','D_visible',2,0),Native('observed_reserve')).run()
        expected=evaluate_new(rows)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.jsonl.gz';write_raw(p,rows)
            self.assertEqual(expected,evaluate_new(read_raw(p)))
            with self.assertRaises(FileExistsError):write_raw(p,rows)

    def test_corrupt_outcome_is_detected(self):
        rows=Simulation(unit_spec(False),Native('fixed')).run()
        next(r for r in rows if r.get('event')=='threat_defended')['event']='threat_passed'
        with self.assertRaises(ValueError):evaluate_new(rows)


class VisibilityAndPreposition(unittest.TestCase):
    def test_future_job_position_and_time_are_not_in_public_view(self):
        a=scenario('validation','D_visible',2,0);b=copy.deepcopy(a)
        next(j for j in b['config']['jobs'] if j['id']=='task0').update(position=[-9.,4.],spawn=5,deadline=15)
        p=Simulation(a,Native('exact'));q=Simulation(b,Native('exact'))
        p.boundary(0);q.boundary(0)
        self.assertEqual(p.public(0).record(),q.public(0).record())
        serialized=json.dumps(p.public(0).record())
        for key in ('scenario_id','event_seed','not_spawned','task0','config'):self.assertNotIn(key,serialized)

    def test_future_failure_is_not_a_planner_input(self):
        a=scenario('validation','D_visible',2,0);b=copy.deepcopy(a)
        b['config']['events']=[{'time':10,'kind':'failure','target':'A','value':0}]
        p=Simulation(a,Native('lookahead'));q=Simulation(b,Native('lookahead'))
        p.boundary(0);q.boundary(0)
        self.assertEqual(p.policy.decide(p.public(0)),q.policy.decide(q.public(0)))

    def test_visible_and_reactive_cases_have_same_physics_different_information(self):
        a=scenario('validation','C_reactive',2,0);b=scenario('validation','D_visible',2,0)
        self.assertEqual(a['config']['robots'],b['config']['robots'])
        self.assertEqual(a['config']['jobs'],b['config']['jobs'])
        p=Simulation(a,Native('fixed'));q=Simulation(b,Native('fixed'));p.boundary(0);q.boundary(0)
        self.assertEqual(len(p.public(0).threats),0);self.assertEqual(len(q.public(0).threats),1)

    def test_observed_preposition_uses_actual_warning(self):
        a=Simulation(scenario('validation','D_visible',2,0),Native('observed_reserve'))
        b=Simulation(scenario('validation','C_reactive',2,0),Native('observed_reserve'))
        a.boundary(0);b.boundary(0)
        self.assertTrue(a.policy.decide(a.public(0)).preparations)
        self.assertFalse(b.policy.decide(b.public(0)).preparations)

    def test_standing_reserve_is_allowed_even_without_warning(self):
        e=Simulation(scenario('validation','C_reactive',2,0),Native('standing_reserve'));e.boundary(0)
        self.assertTrue(e.policy.decide(e.public(0)).preparations)

    def test_near_task_and_far_task_have_indistinguishable_prefixes(self):
        a=Simulation(scenario('validation','A_near',2,0),Native('observed_reserve'))
        b=Simulation(scenario('validation','D_visible',2,0),Native('observed_reserve'))
        a.boundary(0);b.boundary(0)
        self.assertEqual(a.public(0).record(),b.public(0).record())

    def test_preposition_can_hurt_a_near_job_and_is_not_free(self):
        s=scenario('validation','A_near',2,0)
        fixed=evaluate_new(Simulation(s,Native('fixed')).run())
        standing=evaluate_new(Simulation(s,Native('standing_reserve')).run())
        self.assertEqual(fixed['task0_success'],True);self.assertEqual(standing['task0_success'],False)

    def test_oracle_is_explicitly_labeled_not_a_real_policy(self):
        self.assertIn('privileged',OfflineSchedule(0,2).information)
        self.assertNotIn('privileged',Native('observed_reserve').information)

    def test_frozen_actor_does_not_call_critic_or_receive_hidden_jobs(self):
        import torch
        class Actor:
            def distribution(self,x,m):
                self.last=x.clone();logits=torch.full((len(x),10),-torch.inf,dtype=torch.float64);logits[:,0]=0
                return torch.distributions.Categorical(logits=logits)
            def value(self,*args):raise AssertionError('Critic forbidden')
        net=Actor();s=scenario('validation','D_visible',2,0)
        e=Simulation(s,FrozenActor(net));e.boundary(0);e.policy.decide(e.public(0));first=net.last.clone()
        other=copy.deepcopy(s);next(j for j in other['config']['jobs'] if j['id']=='task0')['position']=[9.,9.]
        f=Simulation(other,FrozenActor(net));f.boundary(0);f.policy.decide(f.public(0))
        self.assertTrue(torch.equal(first,net.last));self.assertEqual(tuple(first.shape),(4,146))


class TimeAndReproducibility(unittest.TestCase):
    def test_exact_grid_physics_does_not_change_speed_or_service(self):
        for dt in (1.,.5,.25,.125):
            m=evaluate_new(Simulation(unit_spec(),Script(task_A),dt).run())
            self.assertEqual(m['task0_response_s'],1);self.assertEqual(m['task0_completed_at'],3)

    def test_control_clock_is_fixed_when_physics_dt_changes(self):
        for dt in (1.,.5,.25,.125):
            rows=Simulation(unit_spec(False),Native('fixed'),dt).run()
            self.assertEqual([r['time'] for r in rows if r.get('event')=='policy_selection'],list(range(32)))
            self.assertEqual(len([r for r in rows if r['type']=='interval']),32/dt)

    def test_arrival_is_exactly_once_on_all_grids(self):
        for dt in (1.,.5,.25,.125):
            rows=Simulation(unit_spec(False),Native('fixed'),dt).run()
            impacts=[r for r in rows if r.get('event') in ('threat_defended','threat_passed')]
            self.assertEqual([r['time'] for r in impacts],[4])

    def test_arrival_precedes_new_command_at_same_time(self):
        fn=lambda v:Action((('A','base0' if v.allocation.time<4 else 'task0'),('B',None)))
        s=unit_spec();m=evaluate_new(Simulation(s,Script(fn)).run())
        self.assertEqual(m['threat_pass_count'],0)

    def test_same_seed_and_policy_reproduce_entire_log(self):
        s=scenario('validation','D_visible',4,0)
        self.assertEqual(Simulation(s,Native('observed_reserve')).run(),Simulation(s,Native('observed_reserve')).run())

    def test_hidden_or_invalid_split_rejected_before_setup(self):
        with self.assertRaises(ValueError):Simulation({'split':'final_holdout'},Native('fixed'))
        with self.assertRaises(ValueError):scenario('final_holdout','A_quiet',2,0)

    def test_physics_and_clock_alignment_is_validated(self):
        with self.assertRaises(ValueError):Simulation(unit_spec(False),Native('fixed'),.3)

    def test_capacity_impossibility_certificate_is_a_real_lower_bound(self):
        for team in (2,4):
            s=scenario('validation','I_capacity',team,0);proof=s['infeasibility_certificate']
            self.assertGreater(proof['required_robot_seconds'],proof['available_robot_seconds'])
            for p in ('fixed','exact','ready','observed_reserve'):
                self.assertLess(evaluate_new(Simulation(s,Native(p)).run())['task_success_rate'],1)

    def test_reported_numbers_are_finite_or_explicitly_NA(self):
        m=evaluate_new(Simulation(scenario('validation','E_incumbent',4,0),Native('lookahead')).run())
        for key,value in m.items():
            if isinstance(value,(float,int)):self.assertTrue(math.isfinite(value),key)

    def test_reward_annotations_cannot_change_new_metrics(self):
        rows=Simulation(unit_spec(),Script(task_A)).run();original=evaluate_new(rows)
        for row in rows:row['reward']=1e100
        self.assertEqual(original,evaluate_new(rows))


if __name__=='__main__':unittest.main()
