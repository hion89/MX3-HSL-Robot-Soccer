"""Prepared collection paths: manual records/GT and opt-in passive ROS subscribers."""
import csv
from pathlib import Path
import time
from .schema import VERSION,blank_record,time_value,quantity
from .storage import read,sha,append_records,JournalWriter
from .importers import source_info,from_decoded,plain

TOPICS={'/cmd_vel':'geometry_msgs/msg/Twist','/cmd_vel_raw':'geometry_msgs/msg/Twist',
    '/cmd_vel_behavior':'geometry_msgs/msg/Twist','/cmd_vel_teleop':'geometry_msgs/msg/Twist',
    '/joint_states':'sensor_msgs/msg/JointState','/joint_goals':'sensor_msgs/msg/JointState',
    '/imu/data':'sensor_msgs/msg/Imu','/imu/data_50hz':'sensor_msgs/msg/Imu',
    '/robot_state':'std_msgs/msg/String','/motor_diagnostics':'std_msgs/msg/String',
    '/u2d2/torque_enabled':'std_msgs/msg/Bool','/u2d2/behavior_status':'std_msgs/msg/String',
    '/buttons/stop_pressed':'std_msgs/msg/Bool','/kick_trigger':'std_msgs/msg/Bool','/kick_done':'std_msgs/msg/Bool',
    '/hsl/odom':'nav_msgs/msg/Odometry','/hsl/robot_pose':'hsl_interfaces/msg/RobotPoseEstimate',
    '/hsl/ball':'hsl_interfaces/msg/BallEstimate','/hsl/team_state':'hsl_interfaces/msg/TeamState',
    '/hsl/strategy_state':'hsl_interfaces/msg/StrategyState','/hsl/behavior_debug':'std_msgs/msg/String',
    '/perception/detections_3d':'vision_msgs/msg/Detection3DArray',
    '/camera/camera/color/camera_info':'sensor_msgs/msg/CameraInfo',
    '/tf':'tf2_msgs/msg/TFMessage','/tf_static':'tf2_msgs/msg/TFMessage'}


def session_template():
    clocks={'receive_ros':{'unit':'ns','basis':'ROS host clock, verify use_sim_time','host':None},
            'external_gt':{'unit':'ns','basis':'external measurement clock, unverified','host':None}}
    clocks.update({'header:'+topic:{'unit':'ns','basis':'message header; semantics unverified','host':None} for topic in TOPICS})
    return {'schema_version':VERSION,'experiment_id':'SET_EXPERIMENT_ID','robot_id':'SET_ROBOT_ID','trial_id':'SET_TRIAL_ID',
        'data_origin':'real_robot','repeat_index':0,'mode':'planned_collection',
        'trial_condition':{'stage':'P0','floor':None,'battery':None,'model_sha256':None,'runner_variant':None,
            'command_caps':None,'initial_pose':None,'GT_calibration_id':None,'notes':'Not measured; no actuation authorized'},
        'clocks':clocks,'synchronization':[{'source_clock':'external_gt','target_clock':'receive_ros','method':'UNMEASURED',
            'offset_ns':None,'uncertainty_ns':None,'valid_from_ns':None,'valid_to_ns':None}],
        'frames':{'map':'field center +x opponent goal/+y left; verify registration','odom':'local estimate',
                  'base_link':'robot forward/left/up','imu_link':'sensor frame'},'sources':[],
        'safety':{k:{'value':None,'source':None} for k in ('temperature_stop_c','voltage_stop_v','current_stop_ma','balance_stop_definition')},
        'subscriptions':[{'topic':t,'type':v,'qos':'transient_local' if t=='/tf_static' else 'sensor_data'} for t,v in TOPICS.items()],
        'recording_scope':'State/command messages only. Raw images, TF, CDR and packet capture require a separately prepared original recorder.'}


def import_gt_csv(session,csv_path,clock_id,frame,device_id,calibration_id,method='external_camera'):
    m=read(Path(session)/'manifest.json');source=source_info(csv_path);rows=[]
    with Path(csv_path).open(encoding='utf-8-sig',newline='') as f:
        for i,row in enumerate(csv.DictReader(f),2):
            stamp=time_value(int(row['timestamp_ns']),clock_id,'external GT CSV.timestamp_ns','external_measurement')
            r=blank_record(m,f'gt_{source["sha256"][:12]}_{i}',stamp,{**source,'locator':f'CSV line {i}','topic':'independent_GT'},'ground_truth_sample')
            r['ground_truth_position']=quantity([float(row[k]) for k in ('x_m','y_m','yaw_rad')],frame,['x','y','yaw'],['m','m','rad'],stamp,
                independent=True,device_id=device_id,calibration_id=calibration_id,method=method)
            rows.append(r)
    append_records(session,rows)
    if sha(csv_path)!=source['sha256']:raise ValueError('GT source changed during import')
    return len(rows)


def consume_message(manifest,topic,type_name,message,received_ns,index,clock='receive_ros'):
    # Used by the future ROS callback and offline tests. Never publishes a message.
    stamp=time_value(received_ns,clock,'passive subscriber ROS callback','callback_receive')
    return from_decoded(manifest,f'callback_{index}',stamp,{'locator':f'passive callback {index}','path':None,'sha256':None},
                        topic,type_name,plain(message))


def passive_ros_capture(session,duration_s,explicit_opt_in=False):
    """Not executed in Phase5B-0. Does not launch nodes, publish or call services."""
    if not explicit_opt_in:raise ValueError('Passive subscriptions require explicit opt-in; no robot control exists')
    if not 0<duration_s<=3600:raise ValueError('Choose a bounded recording duration')
    m=read(Path(session)/'manifest.json')
    if m.get('mode')!='planned_collection':raise ValueError('Historical evidence is not a live session')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy,ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    rclpy.init();node=None;subscriptions=[];pending=[];sequence=0;writer=None
    try:
        writer=JournalWriter(session);writer.__enter__();sequence=len(writer.ids)
        node=rclpy.create_node('measurement_passive_recorder')
        def callback(topic,type_name,message):
            nonlocal sequence
            sequence+=1
            pending.append(consume_message(m,topic,type_name,message,node.get_clock().now().nanoseconds,sequence))
        # Fail visibly on absent custom interfaces; do not silently omit requested topics.
        for cfg in m['subscriptions']:
            message_type=get_message(cfg['type'])
            if cfg['qos']=='sensor_data':qos=qos_profile_sensor_data
            elif cfg['qos']=='transient_local':qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
            else:raise ValueError('Unknown subscription QoS profile')
            subscriptions.append(node.create_subscription(message_type,cfg['topic'],
                lambda message,t=cfg['topic'],typ=cfg['type']:callback(t,typ,message),qos))
        deadline=time.monotonic()+duration_s
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
            if len(pending)>=100:writer.append(pending);pending.clear()
    finally:
        try:
            if pending and writer is not None:writer.append(pending)
        finally:
            if writer is not None and hasattr(writer,'file'):writer.__exit__(None,None,None)
            if node is not None:node.destroy_node()
            rclpy.shutdown()
