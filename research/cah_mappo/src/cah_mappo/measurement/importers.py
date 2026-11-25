"""Optional existing rosbags decoder; SQLite/source logs are opened read-only."""
import csv
from dataclasses import fields,is_dataclass
from decimal import Decimal
import hashlib
import math
from pathlib import Path
import sqlite3
from .schema import VERSION,blank_record,time_value,quantity
from .storage import sha,write_dataset


def plain(v):
    if is_dataclass(v):return {f.name:plain(getattr(v,f.name)) for f in fields(v) if not f.name.startswith('__')}
    if hasattr(v,'get_fields_and_field_types'):return {k:plain(getattr(v,k)) for k in v.get_fields_and_field_types()}
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if hasattr(v,'tolist'):return plain(v.tolist())
    if isinstance(v,(str,int,float,bool)) or v is None:return v
    raise TypeError('Unsupported message value '+str(type(v)))


def manifest(experiment,robot,trial,origin,condition,clocks,sources):
    return {'schema_version':VERSION,'experiment_id':experiment,'robot_id':robot,'trial_id':trial,
        'data_origin':origin,'repeat_index':None,'trial_condition':condition,'mode':'historical_forensic',
        'clocks':clocks,'synchronization':[],
        'frames':{'base_link':'robot forward/left/up; command frame contract, not measured GT',
                  'map':'field center +x opponent goal, +y left; not surveyed in these records',
                  'odom':'locally integrated frame, not independent GT','imu_link':'IMU message frame'},
        'sources':sources,'safety':None,
        'limitations':['Missing quantities stay null. Recorder/header clocks are not assumed synchronized.',
                       'Historical robot/software state is not evidence of independent trajectory GT.']}


def source_info(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}


def header_time(decoded,topic):
    h=decoded.get('header')
    if not h:return None
    t=h['stamp'];sec=int(t['sec']);nano=int(t['nanosec'])
    if not 0<=nano<10**9:raise ValueError('Invalid ROS stamp nanosecond field')
    return time_value(sec*10**9+nano,'header:'+topic,'message.header.stamp','header_semantics_not_verified')


def from_decoded(m,rid,stamp,source,topic,type_name,decoded):
    r=blank_record(m,rid,stamp,{**source,'topic':topic,'message_type':type_name},'ros_message')
    r['decoded']=decoded;r['header_timestamp']=header_time(decoded,topic)
    qtime=r['header_timestamp'] or stamp
    if type_name=='geometry_msgs/msg/Twist':
        r['command_velocity']=quantity([decoded['linear']['x'],decoded['linear']['y'],decoded['angular']['z']],
            'base_link',['vx','vy','wz'],['m/s','m/s','rad/s'],stamp,classification='A',
            interpretation='recorded command, not actual ground speed; frame follows topic contract')
        r['event_type']='velocity_command'
    elif type_name=='nav_msgs/msg/Odometry':
        frame=decoded['header']['frame_id'];child=decoded['child_frame_id']
        if frame not in m['frames'] or child not in m['frames']:raise ValueError('Odometry frame undeclared; do not assume map')
        p=decoded['pose']['pose'];q=p['orientation'];yaw=math.atan2(2*(q['w']*q['z']+q['x']*q['y']),1-2*(q['y']**2+q['z']**2))
        r['estimated_position']=quantity([p['position']['x'],p['position']['y'],yaw],frame,['x','y','yaw'],['m','m','rad'],qtime,classification='B')
        t=decoded['twist']['twist'];r['estimated_velocity']=quantity([t['linear']['x'],t['linear']['y'],t['angular']['z']],child,['vx','vy','wz'],['m/s','m/s','rad/s'],qtime,classification='B')
    elif type_name=='hsl_interfaces/msg/RobotPoseEstimate':
        frame=decoded['header']['frame_id']
        r['estimated_position']=quantity([decoded['x'],decoded['y'],decoded['theta']],frame,['x','y','yaw'],['m','m','rad'],qtime,classification='B')
    return r


def bag_metadata(path):
    path=Path(path).resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity failure')
        topics=[{'id':i,'topic':n,'type':t,'serialization_format':s} for i,n,t,s in db.execute('SELECT id,name,type,serialization_format FROM topics ORDER BY id')]
        for t in topics:
            t['count'],t['first_record_ns'],t['last_record_ns']=db.execute('SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages WHERE topic_id=?',(t['id'],)).fetchone()
        count,first,last=db.execute('SELECT COUNT(*),MIN(timestamp),MAX(timestamp) FROM messages').fetchone()
    return {'topics':topics,'count':count,'first_record_ns':first,'last_record_ns':last,
            'duration_s':(last-first)/1e9 if count else None,'read_only':True}


def extract_bag(path,output,robot='robo04'):
    # Do not import legacy scripts: some of them write into original analysis directories.
    from rosbags.typesys import Stores,get_typestore
    source=source_info(path);meta=bag_metadata(path);store=get_typestore(Stores.ROS2_HUMBLE)
    topics={t['id']:t for t in meta['topics']};clocks={'recorder':{'unit':'ns','basis':'rosbag_record_time_unverified_epoch','host':robot}}
    for t in meta['topics']:clocks['header:'+t['topic']]={'unit':'ns','basis':'message_header_unverified','host':None}
    m=manifest('historical_bag',robot,Path(path).stem,'real_robot',{'description':'existing bag; physical conditions not reconstructed','source_metadata':meta},clocks,[source])
    def records():
        with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) as db:
            db.execute('PRAGMA query_only=ON')
            # Original insertion order is retained: sorting by timestamp would hide clock reversals.
            for mid,tid,t,raw in db.execute('SELECT id,topic_id,timestamp,data FROM messages ORDER BY id'):
                topic=topics[tid];stamp=time_value(int(t),'recorder','rosbag.messages.timestamp')
                provenance={**source,'locator':f'messages.id={mid}','payload_sha256':hashlib.sha256(raw).hexdigest()}
                try:
                    d=plain(store.deserialize_cdr(raw,topic['type']))
                    r=from_decoded(m,f'message_{mid}',stamp,provenance,topic['topic'],topic['type'],d)
                except (KeyError,ValueError,TypeError) as error:
                    r=blank_record(m,f'message_{mid}',stamp,{**provenance,'topic':topic['topic'],'message_type':topic['type']},'decode_error')
                    r['failure_reason']=f'{type(error).__name__}: {error}'
                yield r
    count=write_dataset(output,m,records())
    if sha(path)!=source['sha256']:raise ValueError('Original bag changed during extraction')
    return count


def extract_shadow(csv_path,output):
    source=source_info(csv_path);clock='shadow_csv';m=manifest('historical_shadow','robo04',Path(csv_path).stem,'real_robot_shadow',
        {'description':'real sensor shadow; outputs remapped; no measured body trajectory'},
        {clock:{'unit':'ns','basis':'CSV time_sec semantics not independently synchronized','host':'robo04'}},[source])
    def records():
        with Path(csv_path).open(encoding='utf-8-sig',newline='') as f:
            for line,row in enumerate(csv.DictReader(f),2):
                ns=int(Decimal(row['time_sec'])*10**9);stamp=time_value(ns,clock,'CSV.time_sec','csv_time_unverified')
                r=blank_record(m,f'csv_{line}',stamp,{**source,'locator':f'CSV line {line}','topic':'shadow_csv'},'shadow_policy_row')
                r['command_velocity']=quantity([float(row[k]) for k in ('cmd_vx','cmd_vy','cmd_wz')],'base_link',['vx','vy','wz'],['m/s','m/s','rad/s'],stamp,classification='A',interpretation='shadow command only')
                r['decoded']={k:(float(v) if k not in ('run_state','gait_state','robot_state') and v!='' else v) for k,v in row.items()}
                yield r
    count=write_dataset(output,m,records())
    if sha(csv_path)!=source['sha256']:raise ValueError('Original CSV changed')
    return count
