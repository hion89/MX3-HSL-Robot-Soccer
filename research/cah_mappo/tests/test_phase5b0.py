"""Synthetic tests only. They never represent physical measurements."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.measurement.schema import validate_manifest,validate_record,time_value,blank_record,quantity,aligned_delay
from cah_mappo.measurement.storage import create_session,append_records,seal_session,archive_source,read,save,sha,iter_records,JournalWriter,write_dataset
from cah_mappo.measurement.collection import session_template,consume_message,import_gt_csv,passive_ros_capture
from cah_mappo.measurement.validation import validate_dataset
from cah_mappo.measurement.analysis import intervals,command_segments,analyze_dataset


class Measurements(unittest.TestCase):
    def setUp(self):
        base=ROOT/'phase5b0_outputs/test_tmp';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.path=Path(self.temp.name)
        assert self.path.resolve().is_relative_to(base.resolve())
        self.addCleanup(self.temp.cleanup)
        self.m=session_template();self.m.update(experiment_id='synthetic_test',robot_id='fake_robot',trial_id='trial_0',data_origin='synthetic')
        self.r=blank_record(self.m,'r0',time_value(100,'receive_ros','synthetic event','synthetic'),{'locator':'synthetic:0','topic':'test'},'synthetic_test')
    def session(self):return create_session(self.path/'session',self.m)
    def test_null_fields_remain_explicit(self):self.assertIsNone(validate_record(self.r,self.m)['estimated_position'])
    def test_unknown_historical_repeat_stays_null(self):
        self.m.update(mode='historical_forensic',repeat_index=None);self.assertIsNone(validate_manifest(self.m)['repeat_index'])
    def test_unfilled_template_identity_rejected(self):
        with self.assertRaises(ValueError):validate_manifest(session_template())
    def test_missing_field_is_detected(self):
        del self.r['ground_truth_position']
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def test_unknown_time_unit_rejected(self):
        self.r['timestamp']['unit']='ms'
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def test_float_epoch_ns_is_not_accepted(self):
        self.r['timestamp']['value']=1790000000000000000.
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def test_nonfinite_input_rejected(self):
        self.r['decoded']={'x':float('nan')}
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def velocity(self):return quantity([1.,0.,0.],'base_link',['vx','vy','wz'],['m/s','m/s','rad/s'],self.r['timestamp'])
    def test_quantity_unit_mismatch_rejected(self):
        self.r['command_velocity']=self.velocity();self.r['command_velocity']['units'][0]='mm/s'
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def test_unknown_frame_rejected(self):
        self.r['command_velocity']=self.velocity();self.r['command_velocity']['frame_id']='mystery'
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def gt(self,method='external_camera'):
        return quantity([0.,0.,0.],'map',['x','y','yaw'],['m','m','rad'],self.r['timestamp'],independent=True,
                        device_id='fake_device',calibration_id='fake_calibration',method=method)
    def test_odometry_cannot_be_declared_GT(self):
        self.r['ground_truth_position']=self.gt('odometry')
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def test_synthetic_truth_cannot_be_labeled_real(self):
        self.m['data_origin']='real_robot';self.r['ground_truth_position']=self.gt('synthetic_truth')
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def test_GT_provenance_required(self):
        self.r['ground_truth_position']=self.gt();self.r['ground_truth_position']['independent']=False
        with self.assertRaises(ValueError):validate_record(self.r,self.m)
    def test_sync_link_missing_returns_null(self):
        a=time_value(100,'external_gt','a');b=time_value(200,'receive_ros','b')
        self.assertIsNone(aligned_delay(a,b,self.m)['value_s'])
    def test_sync_uncertainty_and_offset(self):
        self.m['synchronization'][0].update(offset_ns=20,uncertainty_ns=5,valid_from_ns=0,valid_to_ns=1000)
        a=time_value(100,'external_gt','a',uncertainty_ns=2);b=time_value(150,'receive_ros','b',uncertainty_ns=3)
        d=aligned_delay(a,b,self.m);self.assertEqual(d['value_s'],30/1e9);self.assertEqual(d['uncertainty_s'],10/1e9)
    def test_expired_sync_returns_null(self):
        self.m['synchronization'][0].update(offset_ns=0,uncertainty_ns=1,valid_from_ns=0,valid_to_ns=50)
        self.assertEqual(aligned_delay(time_value(100,'external_gt','a'),time_value(200,'receive_ros','b'),self.m)['reason'],'outside_sync_validity')
    def test_negative_delay_not_clamped_to_zero(self):
        d=aligned_delay(time_value(200,'receive_ros','a'),time_value(100,'receive_ros','b'),self.m)
        self.assertIsNone(d['value_s']);self.assertEqual(d['reason'],'negative_aligned_delay')
    def test_single_message_has_no_rate(self):self.assertIsNone(intervals([(5,'a')])['observed_hz'])
    def test_gap_candidates_are_not_proven_losses(self):
        d=intervals([(0,'a'),(10**9,'b'),(2*10**9,'c'),(6*10**9,'d')]);self.assertEqual(len(d['long_interval_candidates']),1)
        self.assertIn('not proven',d['gap_interpretation'])
    def test_clock_reversal_does_not_become_positive_rate(self):
        d=intervals([(10,'a'),(9,'b')]);self.assertEqual(d['negative_intervals'],1);self.assertIsNone(d['observed_hz'])
    def test_last_command_duration_right_censored(self):
        a=[{'id':'a','ns':0,'clock_id':'c','value':[1,0,0]},{'id':'b','ns':10**9,'clock_id':'c','value':[0,0,0]}]
        d=command_segments(a);self.assertEqual(d[0]['duration_s'],1);self.assertIsNone(d[1]['duration_s']);self.assertTrue(d[1]['right_censored'])
    def test_command_reversal_rejected(self):
        a=[{'id':'a','ns':1,'clock_id':'c','value':[1,0,0]},{'id':'b','ns':0,'clock_id':'c','value':[1,0,0]}]
        with self.assertRaises(ValueError):command_segments(a)
    def test_existing_session_never_overwritten(self):
        p=self.session()
        with self.assertRaises(FileExistsError):create_session(p,self.m)
    def test_duplicate_record_never_overwrites(self):
        p=self.session();append_records(p,[self.r]);h=sha(p/'records.jsonl')
        with self.assertRaises(ValueError):append_records(p,[self.r])
        self.assertEqual(sha(p/'records.jsonl'),h)
    def test_invalid_batch_writes_nothing(self):
        p=self.session();bad=copy.deepcopy(self.r);bad['record_id']='r1';bad['success']='yes'
        with self.assertRaises(ValueError):append_records(p,[self.r,bad])
        self.assertEqual((p/'records.jsonl').stat().st_size,0)
    def test_sealed_journal_immutable(self):
        p=self.session();append_records(p,[self.r]);seal_session(p)
        with self.assertRaises(ValueError):append_records(p,[self.r])
    def test_seal_detects_tampering(self):
        p=self.session();append_records(p,[self.r]);seal_session(p)
        with (p/'records.jsonl').open('a') as f:f.write('{}\n')
        self.assertEqual(validate_dataset(p)['status'],'FAIL')
    def test_original_archive_is_exclusive_and_same_hash(self):
        p=self.session();source=self.path/'source.txt';source.write_text('raw original')
        d=archive_source(p,source);self.assertEqual(sha(d['archive']),sha(source))
        with self.assertRaises(FileExistsError):archive_source(p,source)
    def test_sealed_session_cannot_add_raw_archive(self):
        p=self.session();seal_session(p);f=self.path/'source';f.write_text('raw')
        with self.assertRaises(ValueError):archive_source(p,f)
        self.assertFalse((p/'raw_sources').exists())
    def test_archived_raw_tamper_detected(self):
        p=self.session();f=self.path/'source';f.write_text('raw');d=archive_source(p,f);seal_session(p)
        Path(d['archive']).write_text('changed synthetic data')
        self.assertTrue(any(x['reason']=='archived_source_changed' for x in validate_dataset(p)['errors']))
    def test_truncated_line_detected(self):
        p=self.session();(p/'records.jsonl').write_text('{"broken":',encoding='utf-8')
        self.assertEqual(validate_dataset(p)['status'],'FAIL')
    def test_validator_reports_timestamp_reversal(self):
        p=self.session();b=copy.deepcopy(self.r);b['record_id']='r1';b['timestamp']['value']=99;append_records(p,[self.r,b])
        self.assertTrue(any(x['reason']=='timestamp_reversal' for x in validate_dataset(p)['errors']))
    def test_unknown_safety_does_not_become_ready(self):
        p=self.session();d=validate_dataset(p);self.assertTrue(d['preflight_blockers']);self.assertEqual(d['motion_readiness'],'NOT_AUTHORIZED_BY_THIS_TOOL')
    def test_writer_checks_duplicate_without_reopening(self):
        p=self.session()
        with JournalWriter(p) as w:
            w.append([self.r])
            with self.assertRaises(ValueError):w.append([self.r])
        self.assertEqual(len(list(iter_records(p/'records.jsonl'))),1)
    def test_missing_GT_does_not_generate_speed_error(self):
        p=self.session();append_records(p,[self.r]);d=analyze_dataset(p,self.path/'analysis')
        self.assertIsNone(d['absolute_position_error']['value']);self.assertIsNone(d['actual_ground_speed_accuracy']['value']);self.assertEqual(d['data_origin'],'synthetic')
    def test_sender_only_cannot_generate_network_latency(self):
        p=self.session();self.r['communication_send_time']=self.r['timestamp'];append_records(p,[self.r])
        self.assertIsNone(analyze_dataset(p,self.path/'analysis')['one_way_network_delay']['value'])
    def test_unmatched_pair_does_not_generate_network_latency(self):
        p=self.session();self.r['communication_send_time']=self.r['timestamp'];self.r['communication_receive_time']=time_value(200,'receive_ros','rx')
        append_records(p,[self.r]);d=analyze_dataset(p,self.path/'analysis')
        self.assertEqual(d['declared_timestamp_delays'][0]['reason'],'no_matched_packet_provenance')
    def test_passive_opt_in_checked_before_ROS_import(self):
        with self.assertRaises(ValueError):passive_ros_capture(self.path/'absent',1,False)
    def test_static_TF_requests_transient_local(self):
        self.assertEqual(next(x for x in self.m['subscriptions'] if x['topic']=='/tf_static')['qos'],'transient_local')
    def test_camera_and_detection_metadata_requested(self):
        names={x['topic'] for x in self.m['subscriptions']}
        self.assertIn('/perception/detections_3d',names);self.assertIn('/camera/camera/color/camera_info',names)
    def test_callback_records_commands_without_real_speed(self):
        d={'linear':{'x':.2,'y':0.,'z':0.},'angular':{'x':0.,'y':0.,'z':0.}}
        r=consume_message(self.m,'/cmd_vel','geometry_msgs/msg/Twist',d,100,1)
        self.assertEqual(r['command_velocity']['value'],[.2,0.,0.]);self.assertIsNone(r['estimated_velocity'])
    def test_GT_csv_import_keeps_clock_and_explicit_missing_estimate(self):
        p=self.session();f=self.path/'gt.csv';f.write_text('timestamp_ns,x_m,y_m,yaw_rad\n100,1,2,0\n')
        import_gt_csv(p,f,'external_gt','map','fake_device','fake_calibration');r=next(iter_records(p/'records.jsonl'))
        self.assertEqual(r['ground_truth_position']['timestamp']['clock_id'],'external_gt');self.assertIsNone(r['estimated_position'])
    def test_normalized_gzip_is_deterministic(self):
        write_dataset(self.path/'one',self.m,[self.r]);write_dataset(self.path/'two',self.m,[self.r])
        self.assertEqual(sha(self.path/'one/records.jsonl.gz'),sha(self.path/'two/records.jsonl.gz'))


if __name__=='__main__':unittest.main()
