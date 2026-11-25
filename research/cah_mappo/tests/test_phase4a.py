import copy
from dataclasses import replace
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from cah_mappo.phase2.model import config_from_dict
from cah_mappo.phase2.metrics import physical_coverage
from cah_mappo.phase3a.contracts import validate_ppo_record
from cah_mappo.phase3b2b.io import write_raw,read_raw
from cah_mappo.phase4a.scenarios import scenario,witness,bank
from cah_mappo.phase4a.evaluation import run
from cah_mappo.phase4a.rules import Gate
from cah_mappo.phase4a.metrics import score,aggregate,distribution


class PhysicalCases(unittest.TestCase):
    def test_same_seed_exact_case_and_independent_split(self):
        a=scenario('validation','B',4,0)
        self.assertEqual(a,scenario('validation','B',4,0))
        self.assertNotEqual(a['event_seed'],scenario('diagnostic','B',4,0)['event_seed'])

    def test_matched_geometry_across_deadline_families(self):
        cases=[scenario('validation',f,4,1) for f in ('A','B','C')]
        self.assertTrue(all(c['config']['robots']==cases[0]['config']['robots'] for c in cases))
        self.assertEqual([j['position'] for j in cases[0]['config']['jobs']],
                         [j['position'] for j in cases[2]['config']['jobs']])

    def test_A_fixed_is_constructive_success_zero_gap(self):
        for team in (2,4):
            s=scenario('validation','A',team,0);m,_=run(s,'witness_fixed')
            self.assertEqual(m['task_success_rate'],1);self.assertEqual(m['actual_gap_region_seconds'],0)

    def test_B_ready_succeeds_fixed_misses_and_no_gap(self):
        for team in (2,4):
            s=scenario('validation','B',team,0)
            fixed,_=run(s,'witness_fixed');ready,_=run(s,'witness_ready')
            self.assertLess(fixed['task_success_rate'],1);self.assertEqual(ready['task_success_rate'],1)
            self.assertEqual(ready['actual_gap_region_seconds'],0)

    def test_C_immediate_succeeds_ready_misses_with_tradeoff(self):
        for team in (2,4):
            s=scenario('validation','C',team,0)
            immediate,_=run(s,'witness_immediate');ready,_=run(s,'witness_ready')
            self.assertEqual(immediate['task_success_rate'],1);self.assertLess(ready['task_success_rate'],1)
            self.assertGreater(immediate['actual_gap_region_seconds'],0)
            self.assertEqual(ready['actual_gap_region_seconds'],0)

    def test_actual_completion_matches_preclassified_times(self):
        for s in bank('validation',1):
            if s['family']!='A':continue
            for mode in ('fixed','immediate','ready'):
                done,_=witness(config_from_dict(s['config']),mode)
                self.assertEqual(done,s['feasibility_completion_at'][mode])

    def test_capacity_and_fixed_32_second_horizon(self):
        for team in (2,4):
            s=scenario('validation','B',team,2)
            self.assertEqual(s['config']['horizon'],32)
            self.assertTrue(all(r['capacity']==1 for r in s['config']['robots']))

    def test_holdout_rejected_before_generation_or_evaluation(self):
        with self.assertRaises(ValueError):scenario('final_holdout','A',4,0)
        with self.assertRaises(ValueError):run({'split':'final_holdout'},'central_fixed')


class HandoverRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_ready_gate_waits_for_physical_arrival_not_assignment(self):
        s=scenario('validation','B',2,0);m,rows=run(s,'witness_immediate',gate_mode='ready')
        self.assertEqual(m['task_success_rate'],1);self.assertEqual(m['actual_gap_region_seconds'],0)
        holding=[r for r in rows if r.get('event')=='controller_holding']
        self.assertTrue(holding);self.assertGreater(holding[0]['successor_eta'],0)

    def test_adaptive_C_releases_early_ready_misses(self):
        s=scenario('validation','C',2,0)
        ready,_=run(s,'witness_immediate',gate_mode='ready')
        adaptive,_=run(s,'witness_immediate',gate_mode='adaptive',parameters={'eta_budget':4})
        self.assertLess(ready['task_success_rate'],1);self.assertEqual(adaptive['task_success_rate'],1)
        self.assertGreater(adaptive['controller_early_release_count'],0)

    def test_high_risk_blocks_urgent_override(self):
        s=scenario('validation','C_risk',2,0)
        m,_=run(s,'witness_immediate',gate_mode='adaptive',parameters={'risk_ratio':1.,'eta_budget':4})
        self.assertEqual(m['controller_early_release_count'],0)
        self.assertEqual(m['actual_gap_region_seconds'],0)

    def test_risk_increase_cancels_pending_handover(self):
        _,rows=run(scenario('validation','R_risk',2,0),'witness_immediate',gate_mode='ready')
        self.assertTrue(any(r.get('event')=='controller_cancelled' and r['reason']=='risk_increased' for r in rows))

    def test_successor_failure_is_not_READY(self):
        m,rows=run(scenario('validation','F_successor',2,0),'witness_immediate',gate_mode='ready')
        self.assertGreaterEqual(m['controller_failure_count'],1)
        self.assertTrue(any(r.get('event')=='controller_failed' and r['reason']=='successor_failed' for r in rows))
        self.assertEqual(m['actual_gap_region_seconds'],0)

    def test_gate_does_not_invent_new_task_choice(self):
        _,rows=run(scenario('validation','B',4,0),'central_exact',gate_mode='ready')
        for row in rows:
            if row['type']!='interval':continue
            selection=row['selection']
            for name,target in selection['controller_requested'].items():
                if target and row['truth']['jobs'][target]['kind']=='task':
                    self.assertEqual(target,selection['policy_requested'][name])

    def test_gate_preserves_exogenous_world(self):
        s=scenario('validation','B',4,0)
        a,_=run(s,'central_exact');b,_=run(s,'central_exact',gate_mode='ready')
        self.assertEqual(a['exogenous_sha256'],b['exogenous_sha256'])

    def test_future_failure_is_not_used_by_gate(self):
        s=scenario('validation','F_successor',2,0);control=copy.deepcopy(s)
        control['config']['events']=[]
        _,a=run(s,'witness_immediate',gate_mode='ready')
        _,b=run(control,'witness_immediate',gate_mode='ready')
        before=lambda rows:[r['selection']['controller_requested'] for r in rows if r['type']=='interval' and r['start']<3]
        self.assertEqual(before(a),before(b))

    def test_incumbent_fault_rescue_restores_physical_coverage(self):
        s=scenario('validation','F_incumbent',2,0)
        original,_=run(s,'central_fixed');rescue,_=run(s,'central_fixed',gate_mode='ready')
        self.assertTrue(original['recovery_details'][0]['right_censored'])
        self.assertFalse(rescue['recovery_details'][0]['right_censored'])
        times=s['feasibility_completion_at']
        self.assertEqual(rescue['coverage_recovery_time_s'],times['ready']-times['immediate'])

    def test_invalid_thresholds_rejected(self):
        for v in (0,-1,math.inf,math.nan):
            with self.assertRaises(ValueError):Gate('adaptive',risk_ratio=v)

    def test_mask_logp_requested_and_executed_stay_separate_without_critic(self):
        class NoopActor:
            def distribution(self,x,masks):
                logits=torch.full((len(x),10),-torch.inf,dtype=torch.float64);logits[:,0]=0
                return torch.distributions.Categorical(logits=logits)
            def value(self,*args):raise AssertionError('Critic forbidden at inference')
        m,rows=run(scenario('validation','A',4,0),'mappo_greedy',NoopActor(),gate_mode='ready')
        self.assertEqual(m['actual_gap_region_seconds'],0)
        changed=0
        for row in rows:
            if row['type']!='transition':continue
            for record in row['records'].values():
                validate_ppo_record(record)
                self.assertEqual(record['sampled_action'],0);self.assertEqual(record['log_probability'],0)
                self.assertIsNone(record['requested_assignment'])
                changed+=record['arbitration_changed']
        self.assertGreater(changed,0)


class GroundTruthMetrics(unittest.TestCase):
    def test_reactive_C_is_not_a_global_impossibility_claim(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from phase4a_schedule_probe import Scheduled
        from cah_mappo.phase2.engine import Environment
        s=scenario('diagnostic','C',2,5)
        rows=Environment(config_from_dict(s['config']),Scheduled(0,3),'proactive_witness').run()
        result=score(rows)
        self.assertEqual(result['task_success_rate'],1)
        self.assertEqual(result['actual_gap_region_seconds'],0)
        reactive,_=run(s,'witness_ready')
        self.assertEqual(reactive['task_success_rate'],0)

    def test_denominator_uses_actual_base_count(self):
        for team,expected in ((2,32),(4,64)):
            m,_=run(scenario('validation','A',team,0),'central_fixed')
            self.assertEqual(m['responsibility_region_seconds'],expected)

    def test_jsonl_only_reconstructs_all_metrics(self):
        m,rows=run(scenario('validation','C',2,0),'witness_immediate')
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'log.jsonl.gz';write_raw(p,rows);loaded=read_raw(p)
            actual=score(loaded)
            for k,v in actual.items():self.assertEqual(m[k],v,k)
            with self.assertRaises(FileExistsError):write_raw(p,rows)

    def test_no_handover_is_NA_not_zero_duration(self):
        m,_=run(scenario('validation','A',2,0),'central_fixed');a=aggregate([m])
        self.assertEqual(a['handover_gap_distribution']['n'],0)
        self.assertIsNone(a['handover_gap_distribution']['mean'])

    def test_full_and_conditional_top5_are_distinct(self):
        self.assertEqual(distribution([0]*99+[10])['top5_mean'],2)
        self.assertEqual(distribution([10])['top5_mean'],10)
        self.assertIsNone(distribution([])['mean'])

    def test_fault_recovery_and_unrecovered_censor_are_distinct(self):
        recovered,_=run(scenario('validation','F_incumbent',2,1),'central_fixed')
        permanent,_=run(scenario('validation','F_incumbent',2,0),'central_fixed')
        self.assertEqual(recovered['recovery_details'][0]['coverage_recovery_s'],8)
        self.assertFalse(recovered['recovery_details'][0]['right_censored'])
        self.assertIsNone(permanent['recovery_details'][0]['coverage_recovery_s'])
        self.assertTrue(permanent['recovery_details'][0]['right_censored'])
        self.assertEqual(permanent['recovery_details'][0]['observed_lower_bound_s'],30)

    def test_metrics_do_not_depend_on_reward_annotations(self):
        _,rows=run(scenario('validation','C',2,0),'witness_immediate')
        before=score(rows)
        for row in rows:row['reward']=9999;row['evaluation_reward']=-1234
        self.assertEqual(before,score(rows))

    def test_all_reported_numeric_metrics_are_finite(self):
        m,_=run(scenario('validation','F_successor',4,0),'central_exact',gate_mode='adaptive')
        for k,v in m.items():
            if isinstance(v,(int,float)):self.assertTrue(math.isfinite(v),k)


if __name__=='__main__':unittest.main()
