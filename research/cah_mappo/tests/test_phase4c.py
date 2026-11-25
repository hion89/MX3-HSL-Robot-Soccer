import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from cah_mappo.phase4b.scenarios import scenario
from cah_mappo.phase4b.engine import Simulation
from cah_mappo.phase4b.policies import Native
from cah_mappo.phase4c.runs import execute,TeamSchedule
from cah_mappo.phase4c.policies import RiskDirectedSwap,BranchHintPolicy,LocalDistribution,DelayedExternalInformation
from cah_mappo.phase4c.evidence import validate_item
from cah_mappo.phase4c.classification import mission_failed,diagnose,resolution_comparison,nonempty_witness_ids
from cah_mappo.phase4c.information import compare_histories
from cah_mappo.phase4c.evaluation import failure_service_recovery,coverage_denominators


class EvidenceAndCauses(unittest.TestCase):
    def test_unmeasured_must_stay_null(self):
        with self.assertRaises(ValueError):validate_item({'kind':'unmeasured','value':.2})
    def test_setting_is_not_measurement(self):
        for k in ('software_limit','training_setting','timer_setting'):
            with self.assertRaises(ValueError):validate_item({'kind':k,'value':1,'actual_performance':True})
    def test_null_is_valid_unmeasured(self):
        self.assertIsNone(validate_item({'kind':'unmeasured','value':None})['value'])
    def test_no_threat_not_failure_if_tasks_complete(self):
        self.assertFalse(mission_failed({'task_success_rate':1.,'threat_pass_rate':None}))
    def test_task_failure_cannot_hide_behind_coverage(self):
        self.assertTrue(mission_failed({'task_success_rate':0.,'threat_pass_rate':0.}))
    def test_defense_failure_cannot_hide_behind_task_success(self):
        self.assertTrue(mission_failed({'task_success_rate':1.,'threat_pass_rate':1.}))
    def test_no_witness_means_unknown_not_impossible(self):
        self.assertTrue(diagnose({'task_success_rate':0})['unresolved_full_mission_cause'])
    def test_empty_witness_lookup_is_not_success(self):
        self.assertEqual(nonempty_witness_ids({'success':['ready'],'missing':[]}),{'success'})
    def test_offline_witness_not_online_policy_limitation_proof(self):
        r=diagnose({'task_success_rate':0},offline_joint=True)
        self.assertEqual(r['confirmed_causes'],[]);self.assertTrue(r['physical_possibility_witness'])
    def test_same_interface_success_is_policy_counterfactual(self):
        self.assertEqual(diagnose({'task_success_rate':0},online_joint=True)['confirmed_causes'],['C'])
    def test_infeasibility_proof_cannot_contradict_success(self):
        with self.assertRaises(ValueError):diagnose({'task_success_rate':0},True,True)
    def test_task_component_is_not_whole_mission_cause(self):
        r=diagnose({'task_success_rate':0},task_counterfactual=True)
        self.assertTrue(r['task_selection_component_confirmed_C']);self.assertTrue(r['unresolved_full_mission_cause'])


class InformationAndPolicies(unittest.TestCase):
    def spec(self,f='D_visible',team=4,i=0):return scenario('diagnostic',f,team,i)
    def test_commitment_rule_solves_all_six_four_robot_visible_cases(self):
        for i in range(6):
            m,_=execute(self.spec(i=i),RiskDirectedSwap(True));self.assertTrue(m['joint_task_and_defense_success'])
    def test_first_rule_failure_is_retained_as_negative_control(self):
        m,_=execute(self.spec(),RiskDirectedSwap());self.assertFalse(m['joint_task_and_defense_success'])
    def test_commitment_rule_does_not_remove_hidden_future_tradeoff(self):
        m,_=execute(self.spec('A_near',2),RiskDirectedSwap(True));self.assertFalse(m['task0_success'])
    def test_rule_reset_clears_planned_swap(self):
        p=RiskDirectedSwap(True);execute(self.spec(),p);p.reset(1);self.assertEqual(p.plans,{})
    def test_original_native_seed_api_caveat_is_removed_in_replay(self):
        from phase4c_reset_audit import CommonResetNative
        a=CommonResetNative('fixed');b=CommonResetNative('fixed');a.reset(52020000);b.reset(52024000)
        self.assertEqual(a.fixed.rng.getstate(),b.fixed.rng.getstate())
        self.assertEqual(a.effective_reset_seed,b.effective_reset_seed)
    def test_identical_past_includes_actor_packets_and_samples(self):
        from phase4b_feedback import SensorActor
        a=execute(self.spec('A_near',2),SensorActor(LocalDistribution('local_eta')),feedback=True)[1]
        b=execute(self.spec('D_visible',2),SensorActor(LocalDistribution('local_eta')),feedback=True)[1]
        self.assertTrue(compare_histories(a,b)['identical'])
        altered=copy.deepcopy(b);r=next(r for r in altered if r.get('event')=='policy_selection')
        r['action']['policy_metadata']['history_change']='test'
        self.assertFalse(compare_histories(a,altered)['identical'])
    def test_future_hint_explicitly_privileged(self):
        s=self.spec('D_visible',2);task=next(j for j in s['config']['jobs'] if j['id']=='task0')
        self.assertIn('privileged',BranchHintPolicy(task).information)
        with_hint=execute(s,BranchHintPolicy(task))[0];without=execute(s,BranchHintPolicy())[0]
        self.assertTrue(with_hint['joint_task_and_defense_success']);self.assertFalse(without['joint_task_and_defense_success'])
    def test_local_policy_only_selects_masked_valid_actions(self):
        import torch
        v=torch.zeros((4,146));mask=torch.zeros((4,10),dtype=torch.bool);mask[:,0]=True
        d=LocalDistribution('local_eta').distribution(v,mask)
        self.assertTrue(torch.equal(d.probs[:,0],torch.ones(4,dtype=torch.float64)))
        self.assertEqual(float(d.probs[:,1:].sum()),0)
    def test_delivery_delay_hides_current_task_until_due(self):
        s=self.spec('D_visible',2);p=DelayedExternalInformation(Native('fixed'),1.)
        _,rows=execute(s,p)
        inputs={r['time']:r['action']['policy_metadata']['delivered_public'] for r in rows if r.get('event')=='policy_selection'}
        self.assertNotIn('task0',[j['id'] for j in inputs[2]['allocation']['jobs']])
        self.assertIn('task0',[j['id'] for j in inputs[3]['allocation']['jobs']])
        self.assertEqual(inputs[0]['threats'],[]);self.assertEqual(len(inputs[1]['threats']),1)
    def test_offline_schedule_cannot_select_task_before_spawn(self):
        p=TeamSchedule('A','base0','B','base0',0,2)
        _,rows=execute(self.spec(),p)
        for r in rows:
            if r.get('event')=='policy_selection' and r['time']<2:
                self.assertNotIn('task0',r['action']['requested'].values())
    def test_fine_step_preserves_events_and_control_clock(self):
        a,ra=execute(self.spec('B_normal',2),Native('ready'),.125)
        b,rb=execute(self.spec('B_normal',2),Native('ready'),.0625)
        self.assertEqual(a['exogenous_sha256'],b['exogenous_sha256'])
        self.assertEqual([r['time'] for r in ra if r.get('event')=='policy_selection'],
                         [r['time'] for r in rb if r.get('event')=='policy_selection'])
        self.assertEqual(resolution_comparison([a,b])['pairs'],1)
    def test_new_diagnostics_reject_holdout(self):
        s=self.spec();s['split']='holdout'
        with self.assertRaises(ValueError):Simulation(s,RiskDirectedSwap(True))
    def test_recovery_metric_counts_service_not_repair_timeout(self):
        _,rows=execute(self.spec('E_incumbent',2,1),Native('fixed'))
        r=failure_service_recovery(rows)[0]
        self.assertEqual(r['failure_at'],2.);self.assertEqual(r['service_recovery_delay_s'],8.)
        self.assertFalse(r['right_censored'])
    def test_recovery_metric_keeps_failure_censored(self):
        _,rows=execute(self.spec('E_incumbent',2,0),Native('fixed'))
        r=failure_service_recovery(rows)[0]
        self.assertIsNone(r['service_recovery_delay_s']);self.assertTrue(r['right_censored'])
    def test_legacy_weighted_denominator_not_raw_time_denominator(self):
        _,rows=execute(self.spec('R_changed',4),RiskDirectedSwap(True));d=coverage_denominators(rows)
        self.assertEqual(d['unweighted_denominator_region_seconds'],64)
        self.assertEqual(d['legacy_weighted_denominator_region_seconds'],136)
        self.assertNotEqual(d['unweighted_gap_region_seconds']/136,d['legacy_weighted_cgr'])


if __name__=='__main__':unittest.main()
