"""Synthetic offline tests. No safety certification or physical measurements."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cah_mappo.measurement.collection import session_template, consume_message
from cah_mappo.measurement.storage import read, save, sha
from cah_mappo.measurement.schema import time_value, quantity
from cah_mappo.measurement_p0.audit import audit_streams, timestamp_audit, readiness, raw_image_budget, P1_EVIDENCE
from cah_mappo.measurement_p0.static_review import review


class P0Audit(unittest.TestCase):
    def setUp(self):
        self.m = session_template()
        self.m.update(experiment_id='synthetic_p0', robot_id='fake', trial_id='t0', data_origin='synthetic')
        self.profile = {'id': 'synthetic', 'streams': {'/imu/data': {'type': 'sensor_msgs/msg/Imu', 'min_messages': 2}},
                        'frame_required_topics': ['/imu/data']}

    def msg(self, idx=0, ns=100, topic='/imu/data', type_name='sensor_msgs/msg/Imu', decoded=None):
        decoded = decoded if decoded is not None else {'header': {'stamp': {'sec': 1, 'nanosec': ns}, 'frame_id': 'imu_link'}}
        return consume_message(self.m, topic, type_name, decoded, ns, idx)

    def audit(self, rows, profile=None):
        return audit_streams(rows, self.m, self.profile if profile is None else profile)

    def reasons(self, rows, profile=None):
        return {e['reason'] for e in self.audit(rows, profile)['errors']}

    def test_expected_stream_absent(self):
        self.assertIn('required_stream_missing_or_short', self.reasons([]))

    def test_one_message_not_enough(self):
        self.assertIn('required_stream_missing_or_short', self.reasons([self.msg()]))

    def test_valid_stream_two_messages(self):
        self.assertEqual(self.reasons([self.msg(), self.msg(1, 200)]), set())

    def test_event_stream_allowed_to_be_silent(self):
        p = {'streams': {'/kick_done': {'type': 'std_msgs/msg/Bool', 'min_messages': 0}}}
        self.assertFalse(self.reasons([], p))

    def test_wrong_type_detected(self):
        r = self.msg(); r['source']['message_type'] = 'std_msgs/msg/String'
        self.assertIn('message_type_mismatch', self.reasons([r]))

    def test_missing_sensor_frame(self):
        r = self.msg(); r['decoded']['header']['frame_id'] = ''
        self.assertIn('missing_or_undeclared_frame', self.reasons([r]))

    def test_undeclared_frame(self):
        r = self.msg(); r['decoded']['header']['frame_id'] = 'unregistered'
        self.assertIn('missing_or_undeclared_frame', self.reasons([r]))

    def test_time_unit_rejected(self):
        r = self.msg(); r['timestamp']['unit'] = 'ms'
        self.assertIn('schema', self.reasons([r]))

    def test_receive_reversal(self):
        self.assertIn('receive_timestamp_reversal', self.reasons([self.msg(0, 200), self.msg(1, 100)]))

    def test_header_reversal(self):
        a, b = self.msg(), self.msg(1, 200)
        b['decoded']['header']['stamp']['sec'] = 0
        self.assertIn('header_timestamp_reversal', self.reasons([a, b]))

    def test_invalid_stamp_nanosecond_range(self):
        r = self.msg(); r['decoded']['header']['stamp']['nanosec'] = 1_000_000_000
        self.assertIn('invalid_stamp', self.reasons([r]))

    def test_zero_dynamic_stamp_warns(self):
        r = self.msg(); r['decoded']['header']['stamp'] = {'sec': 0, 'nanosec': 0}
        self.assertIn('zero_or_negative_dynamic_stamp', {w['reason'] for w in self.audit([r])['warnings']})

    def tf(self, ns, idx=0, topic='/tf'):
        return self.msg(idx, ns, topic, 'tf2_msgs/msg/TFMessage', {'transforms': [
            {'header': {'frame_id': 'odom', 'stamp': {'sec': 0, 'nanosec': ns}}, 'child_frame_id': 'base_link'}]})

    def test_nested_tf_reversal(self):
        self.assertIn('nested_tf_timestamp_reversal', self.reasons([self.tf(200), self.tf(100, 1)], {}))

    def test_nested_tf_missing_child(self):
        r = self.tf(100); r['decoded']['transforms'][0]['child_frame_id'] = ''
        self.assertIn('invalid_tf_frames', self.reasons([r], {}))

    def test_static_tf_zero_is_valid(self):
        self.assertFalse(self.audit([self.tf(0, topic='/tf_static')], {})['warnings'])

    def test_unknown_tf_frame(self):
        r = self.tf(100); r['decoded']['transforms'][0]['child_frame_id'] = 'unknown'
        self.assertIn('undeclared_tf_frame', self.reasons([r], {}))

    def test_GT_missing_does_not_become_zero(self):
        r = self.audit([self.msg(), self.msg(1, 200)])
        self.assertEqual(r['GT_missing_records'], 2)
        self.assertFalse(r['GT_accuracy_validated'])

    def test_position_unit_error(self):
        r = self.msg()
        r['estimated_position'] = quantity([0., 0., 0.], 'map', ['x', 'y', 'yaw'], ['cm', 'm', 'rad'], r['timestamp'])
        self.assertIn('schema', self.reasons([r]))

    def test_long_interval_not_claimed_packet_loss(self):
        r = self.audit([self.msg(i, t) for i, t in enumerate((100, 200, 300, 400, 10000))])
        stream = r['receive_streams'][0]
        self.assertEqual(len(stream['long_interval_candidates']), 1)
        self.assertIn('not proven', stream['gap_interpretation'])

    def test_signature_mixing_and_original_order(self):
        rows = []
        for i, (name, stamp) in enumerate((('a', 300), ('b', 100), ('a', 400), ('b', 200))):
            d = {'header': {'stamp': {'sec': 0, 'nanosec': stamp}, 'frame_id': ''}, 'name': [name], 'position': [0.]}
            rows.append(self.msg(i, 1000 + i, '/joint_goals', 'sensor_msgs/msg/JointState', d))
        result = timestamp_audit(rows)
        self.assertEqual(result['cross_signature_reversals'], 2)
        self.assertEqual(result['recorder']['negative_intervals'], 0)
        self.assertTrue(all(g['header']['negative_intervals'] == 0 for g in result['signatures']))
        self.assertEqual(result['clock_synchronization'], 'NOT_VERIFIED')
        self.assertIsNone(result['one_way_latency_s'])

    def test_empty_evidence_blocks_P1(self):
        self.assertEqual(readiness({})['P1_evidence_status'], 'BLOCKED')

    def evidence(self):
        e = {k: {'confirmed': True, 'source': 'synthetic fixture', 'signed_by': 'fake', 'checked_at': 'fixture', 'origin': 'onsite'} for k in P1_EVIDENCE}
        e['clock_alignment'].update(offset_ns=0, residual_error_ns=10, valid_from_ns=0, valid_to_ns=100)
        e['independent_GT_calibration'].update(position_uncertainty_m=.01, yaw_uncertainty_rad=.01)
        e['equipment_stop_limits'].update(temperature_stop_c=1, current_stop_ma=1, voltage_stop_v=1)
        return e

    def test_even_complete_form_never_authorizes_motion(self):
        r = readiness(self.evidence()); self.assertFalse(r['motion_authorized'])
        self.assertEqual(r['P1_evidence_status'], 'REQUIRES_HUMAN_REVIEW')

    def test_synthetic_evidence_cannot_pass_onsite_gate(self):
        e = self.evidence(); e['onsite_P0_recording']['origin'] = 'synthetic'
        self.assertEqual(readiness(e)['P1_evidence_status'], 'BLOCKED')

    def test_unmeasured_clock_error_blocks(self):
        e = self.evidence(); e['clock_alignment']['residual_error_ns'] = None
        self.assertIn('clock_alignment:unmeasured_values', readiness(e)['missing'])

    def test_invalid_clock_validity_blocks(self):
        e = self.evidence(); e['clock_alignment']['valid_to_ns'] = 0
        self.assertIn('clock_alignment:invalid_range', readiness(e)['missing'])

    def test_negative_GT_uncertainty_blocks(self):
        e = self.evidence(); e['independent_GT_calibration']['position_uncertainty_m'] = -.1
        self.assertIn('independent_GT_calibration:negative_uncertainty', readiness(e)['missing'])

    def test_image_budget_conditional_not_measured(self):
        self.assertEqual(raw_image_budget()['payload_bytes'], 1_831_680_000)

    def test_image_budget_invalid_dimensions(self):
        with self.assertRaises(ValueError): raw_image_budget(width=-1)

    def test_file_collision_preserves_content(self):
        # All new temporary files remain inside the research directory.
        base = ROOT / 'phase5b1_p0_outputs' / 'test_tmp'; base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as d:
            p = Path(d) / 'test.json'; save(p, {'v': 1}); before = sha(p)
            with self.assertRaises(FileExistsError): save(p, {'v': 2})
            self.assertEqual(sha(p), before)

    def test_audit_deterministic_and_does_not_mutate(self):
        rows = [self.msg(), self.msg(1, 200)]; before = copy.deepcopy(rows)
        self.assertEqual(self.audit(rows), self.audit(rows)); self.assertEqual(rows, before)

    def test_review_text_does_not_execute_python(self):
        # Source parse against actual checked-in source, never imports it.
        r = review(ROOT.parents[1] / 'MX3_Match_Sim' / 'src')
        self.assertFalse(r['hardware_accessed']); self.assertFalse(r['ROS_imported'])
        self.assertFalse([f for f in r['files'] if f['classification'] == 'UNRESOLVED_PATH'])
        self.assertTrue(any('call_async' in c['call'] for f in r['files'] for c in f.get('python_AST_calls', [])))


if __name__ == '__main__': unittest.main()
