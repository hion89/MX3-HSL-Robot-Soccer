"""Read-only source/log ledger. Configured limits are never labeled measurements."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from ..phase3b2b.io import save


def validate_item(item):
    if item['kind']=='unmeasured' and item['value'] is not None:raise ValueError('Unmeasured values must be null')
    if item['kind'] in ('software_limit','training_setting','timer_setting') and item.get('actual_performance',False):
        raise ValueError('A setting cannot establish actual performance')
    return item


def collect(root,output):
    repo=root.parents[1];workspace=repo.parent/'isaac_ws';target=output/'evidence';target.mkdir(exist_ok=False)
    hashes={};items=[]
    def text(path):
        p=Path(path);data=p.read_bytes();hashes[str(p.resolve())]=hashlib.sha256(data).hexdigest()
        return data.decode('utf-8-sig')
    def quote(path,pattern):
        lines=text(path).splitlines();hits=[i for i,line in enumerate(lines) if re.search(pattern,line)]
        if not hits:raise ValueError(f'Evidence pattern missing: {path}: {pattern}')
        return {'path':str(Path(path).resolve()),'sha256':hashes[str(Path(path).resolve())],
                'lines':[{'line':i+1,'text':lines[i]} for i in hits[:8]]}
    def add(key,kind,value,unit,note,sources):
        items.append(validate_item({'key':key,'kind':kind,'value':value,'unit':unit,'note':note,'sources':sources,
                                   'actual_performance':kind in ('recorded_measurement','operator_console')}))
    code=repo/'MX3_Match_Sim/src';hsl=code/'hsl_src';mx=code/'mx3_src'
    role=hsl/'hsl_bringup/launch/real_role_trial.launch.py';params=hsl/'hsl_bringup/config/hsl.yaml'
    skills=hsl/'hsl_behavior/hsl_behavior/skills.py';team=hsl/'hsl_team_comm/hsl_team_comm/team_comm_node.py'
    add('autonomous_trial_command_limits','software_limit',{'vx':.20,'vy':.15,'wz':.50},'m/s,rad/s',
        'Configured one-robot launch override, not measured maximum or installed-runtime verification.',[quote(role,r"'vx_max': 0.20")])
    add('general_behavior_command_limits','software_limit',{'vx':.40,'vy':.25,'wz':1.},'m/s,rad/s',
        'General YAML is overridden by the role-trial launch; teleop/other launches can differ.',[quote(params,r'(vx_max|vy_max|wz_max):')])
    add('goal_arrival_latch','software_limit',{'position_tolerance_m':.15,'yaw_tolerance_deg':15,'leave_position_m':.35,'leave_yaw_deg':35},
        'm,deg','Controller acceptance/hysteresis thresholds, not measured endpoint error.',[quote(skills,r'pos_tol=0.15|leave_yaw=math.radians\(35')])
    add('command_shaping','software_limit',{'linear_acceleration':.6,'angular_acceleration':2.5,'min_command_norm':.12},'command units',
        'Software command shaping, not measured robot acceleration.',[quote(skills,r'accel_lin:|accel_ang:|min_norm:')])
    add('time_to_ball_speed_assumption','software_limit',.30,'m/s','Strategy ETA denominator, not observed walking speed.',
        [quote(hsl/'hsl_behavior/hsl_behavior/behavior_node.py',r'time_to_ball_walk_speed')])
    deploy=repo/'XM_Walking_Policy/policy/deploy.yaml';ranges={}
    for name in ('lin_vel_x','lin_vel_y','ang_vel_z'):
        matched=re.search(rf'{name}:\s*(\[[^\]]+\])',text(deploy));ranges[name]=ast.literal_eval(matched.group(1))
    add('walking_training_command_ranges','training_setting',ranges,'m/s,rad/s','Training commands do not establish hardware tracking limits.',
        [quote(deploy,r'lin_vel_x:|lin_vel_y:|ang_vel_z:|step_dt:')])
    add('isaac_timing','training_setting',{'physics_dt_s':.005,'decimation':4,'nominal_policy_dt_s':.02},'s',
        'Serialized local Isaac training metadata; not a server query or hardware measurement.',
        [quote(repo/'XM_Walking_Policy/training/env.yaml',r'^  dt: 0.005|^decimation: 4')])
    add('team_timers','timer_setting',{'send_hz':2.,'event_rate_hz':.25,'event_burst':3.,'receive_poll_s':.02,'publish_s':.1,'teammate_timeout_s':2.},
        'Hz,s','0.5s nominal send period is NOT measured one-way latency. Source comments about 4v4 are simulation reports.',
        [quote(params,r'send_rate_hz:|teammate_timeout:|event_rate_hz:|event_burst:'),quote(team,r'create_timer\(0.02|create_timer\(0.1')])
    add('team_packet_observability','software_limit',{'sender_timestamp_present':False,'receiver_timestamp_present':True},'schema',
        'Payload has sequence but no sender timestamp. Receiving stamp alone cannot establish one-way latency.',
        [quote(team,r'^FMT =|t.received =|_seq\)')])
    add('world_model_timers','timer_setting',{'localizer_hz':30.},'Hz','Configured update cadence, not absolute localization accuracy.',
        [quote(params,r'update_rate: 30.0')])
    tracker=hsl/'hsl_world_model/src/ball_tracker_node.cpp'
    timer_lines=quote(tracker,r'create_wall_timer|milliseconds\(|publish_rate|update_rate')
    add('ball_tracker_timer_source','timer_setting',None,'source','See exact timer statement; per-detection callback and publication cadence differ.',[timer_lines])
    fall=mx/'humanoid_serial_bridge/humanoid_serial_bridge/fall_manager_node.py'
    add('fall_recovery_settings','timer_setting',{'fall_deg':45,'recover_deg':20,'settle_s':1,'standup_wait_s':10,
        'post_standup_settle_s':2,'retries':3,'fsm_hz':50},'deg,s,Hz',
        'FSM thresholds/timeouts, not measured successful recovery duration.',[quote(fall,r'declare_parameter\("(fall_tilt_deg|recover_tilt_deg|settle_duration_sec|standup_wait_sec|post_standup_settle_sec|max_standup_retries|update_rate_hz)"')])
    field=hsl/'hsl_world_model/include/hsl_world_model/field_model.hpp'
    add('field_geometry','software_limit',{'practice_m':[3.9,2.6],'standard_m':[9.,6.],
        'practice_penalty_m':[.87,2.17],'practice_goal_area_m':[.43,1.30],'practice_goal_width_m':1.13},'m',
        'Source geometry matching the user-provided drawing; not a new tape survey or a check of the robot installation.',
        [quote(field,r'length = 9.0|f.length = 3.9|f.penalty_area_depth|f.goal_area_depth|f.goal_width')])
    evidence=workspace/'sim2real_evidence'
    analysis_path=evidence/'analysis_walk_20261007_161217/analysis.json';a=json.loads(text(analysis_path))
    shadow_path=evidence/'observations_20261105/walk_shadow_20261105_162155_qeh2uxxq/summary.json';s=json.loads(text(shadow_path))
    add('historical_ros_feedback_cadence','recorded_measurement',{'goals_ms':a['goal_interval_ms'],'joints_ms':a['joint_feedback_interval_ms']},'ms',
        '2026-10-07 old zero-velocity failed-policy trial, not a walking speed experiment.',[{'path':str(analysis_path),'sha256':hashes[str(analysis_path.resolve())]}])
    add('shadow_runtime','recorded_measurement',{'goals_hz':s['goal_receive_hz'],'inference_ms':s['inference_ms']},'Hz,ms',
        'Real sensor/inference shadow trial with outputs remapped; inference time is not command-to-motion latency.',
        [{'path':str(shadow_path),'sha256':hashes[str(shadow_path.resolve())]}])
    lag_path=evidence/'analysis_walk_20261007_161217/exploratory_lag_fit.json';lag=json.loads(text(lag_path))
    add('exploratory_joint_trajectory_shift','recorded_measurement',{k:v['best_grid_lag_ms'] for k,v in lag.items()},'ms',
        'Trajectory-alignment fit, explicitly not isolated transport or walking onset delay.',
        [{'path':str(lag_path),'sha256':hashes[str(lag_path.resolve())]}])
    add('failed_old_recovery_trial','recorded_measurement',{'recovery_enter_s':2.239244805,'fault_s':2.637695766,
        'successful_recovery_observed':False},'s after goal start',
        'Old dual-policy failure; 0.39845s to FAULT is NOT a successful recovery time.',
        [{'path':str(analysis_path),'sha256':hashes[str(analysis_path.resolve())]}])
    kick_path=evidence/'kick_balance/kick_balance_20261110_151804/analysis/summary.json';k=json.loads(text(kick_path))
    add('physical_kick_failure','recorded_measurement',{'protect_s':next(x['t'] for x in k['/robot_state'] if x['data']=='PROTECT'),
        'kick_done_count':len(k['/kick_done']),'record_end_s':k['topics']['/robot_state']['last']},'s after trigger',
        'Failure/censored continuation; no successful recovery sample in this bag.',[{'path':str(kick_path),'sha256':hashes[str(kick_path.resolve())]}])
    reported={'source':'User console pasted in this conversation; no independent raw pose ground truth',
        'duration_s':20,'messages':601,'receive_hz':30.,'x_min_max_cm':[-173.32,-172.64],
        'y_min_max_cm':[60.29,61.79],'theta_min_max_deg':[-11.31,-10.09]}
    save(target/'user_reported_stationary_console.json',reported)
    add('stationary_pose_report','operator_console',{'x_span_m':.0068,'y_span_m':.015,'yaw_span_deg':1.22,'receive_hz':30.},'m,deg,Hz',
        'Reported stationary min-max spread; NOT stddev or absolute localization error.',[{'path':'user_reported_stationary_console.json'}])
    unmeasured=('actual_max_translation_speed','actual_operating_translation_speed','actual_turn_rate_and_turn_time',
        'goal_arrival_time_and_absolute_error','command_to_motion_delay','absolute_localization_error',
        'ball_detection_end_to_end_latency_and_tracking_hz','multi_robot_one_way_network_delay',
        'successful_fall_to_service_recovery_duration','real_opponent_arrival_prediction_accuracy')
    for key in unmeasured:add(key,'unmeasured',None,None,'No qualifying synchronized ground-truth experiment located in reviewed local evidence.',[])
    # Read-only SQLite metadata confirms the physical bags lack trajectory ground truth.
    bags=[]
    for path in (evidence/'walk_20261007_161217/walk_20261007_161217_0.db3',
                 evidence/'kick_balance/kick_balance_20261110_151804/kick_balance_20261110_151804_0.db3'):
        if not path.is_file():bags.append({'path':str(path),'present':False});continue
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
            topics=db.execute('SELECT name,type FROM topics ORDER BY name').fetchall()
            count,first,last=db.execute('SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages').fetchone()
        bags.append({'path':str(path),'present':True,'topics':topics,'messages':count,'record_duration_s':(last-first)/1e9,
                     'read_only':True,'independent_position_ground_truth_topic_present':any('ground_truth' in n for n,_ in topics)})
    save(target/'physical_bag_metadata.json',bags);save(target/'parameter_ledger.json',items)
    save(target/'read_sources_sha256.json',hashes)
    print('Evidence ledger',len(items),'items;',len(unmeasured),'explicitly unmeasured;',len(hashes),'read-only source files')
