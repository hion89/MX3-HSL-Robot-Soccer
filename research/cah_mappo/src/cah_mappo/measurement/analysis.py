"""Reproducible statistics from normalized raw records. No absolute-GT claims."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from statistics import mean,pstdev
from .schema import aligned_delay
from .storage import read,iter_records,save,sha
from .validation import validate_dataset


def quantile(values,p):
    a=sorted(values)
    if not a:return None
    i=(len(a)-1)*p;l=int(i);f=i-l
    return a[l]*(1-f)+a[min(l+1,len(a)-1)]*f


def stats(values):
    a=list(values)
    return None if not a else {'n':len(a),'min':min(a),'mean':mean(a),'median':quantile(a,.5),
        'p95':quantile(a,.95),'p99':quantile(a,.99),'max':max(a),'population_std':pstdev(a)}


def intervals(samples,gap_factor=2.5):
    # sample = (integer ns, original record ID), in original sequence, not time-sorted.
    deltas=[(b[0]-a[0],a[1],b[1],a[0],b[0]) for a,b in zip(samples,samples[1:])]
    positive=[d[0]/1e9 for d in deltas if d[0]>0];typical=quantile(positive,.5)
    span=(samples[-1][0]-samples[0][0])/1e9 if len(samples)>1 else None
    reversed_count=sum(d[0]<0 for d in deltas)
    gaps=[{'from_record':a,'to_record':b,'start_ns':start,'end_ns':end,'duration_s':d/1e9}
          for d,a,b,start,end in deltas if typical is not None and d/1e9>gap_factor*typical]
    return {'samples':len(samples),'positive_interval_s':stats(positive),'zero_intervals':sum(d[0]==0 for d in deltas),
        'negative_intervals':reversed_count,'observed_hz':(len(samples)-1)/span if span and span>0 and not reversed_count else None,
        'gap_factor':gap_factor,'long_interval_candidates':gaps,
        'gap_interpretation':'gaps > factor x median positive interval; not proven packet/message loss without source sequence'}


def command_segments(commands):
    if not commands:return []
    out=[];start=commands[0];last=start
    for row in commands[1:]:
        if row['clock_id']!=last['clock_id'] or row['ns']<last['ns']:
            raise ValueError('Commands cannot be integrated across clock changes/reversals')
        if row['value']!=last['value']:
            out.append({'start_record':start['id'],'end_record':row['id'],'start_ns':start['ns'],'end_ns':row['ns'],
                'duration_s':(row['ns']-start['ns'])/1e9,'value':start['value'],'right_censored':False,
                'assumption':'sample-held command interval, not a measured motion duration'})
            start=row
        last=row
    out.append({'start_record':start['id'],'end_record':last['id'],'start_ns':start['ns'],'end_ns':None,
        'duration_s':None,'observed_lower_bound_s':(last['ns']-start['ns'])/1e9,'value':start['value'],
        'right_censored':True,'assumption':'no extrapolation after last recorded command'})
    return out


def state_name(decoded):
    raw=decoded.get('data')
    if not isinstance(raw,str):return None
    try:d=json.loads(raw)
    except ValueError:return raw
    if isinstance(d,dict):return d.get('state',d.get('mode'))
    return raw


def attitude(q):
    x,y,z,w=(q[k] for k in ('x','y','z','w'));norm=math.sqrt(x*x+y*y+z*z+w*w)
    if norm<1e-12:return None
    x,y,z,w=(x/norm,y/norm,z/norm,w/norm)
    return {'tilt_deg':math.degrees(math.acos(max(-1.,min(1.,1-2*(x*x+y*y))))),
            'yaw_rad':math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))}


def unavailable(reason):return {'classification':'D','value':None,'reason':reason}


def analyze_dataset(path,output):
    from pathlib import Path
    path,output=Path(path),Path(output);output.mkdir(parents=True,exist_ok=False)
    m=read(path/'manifest.json');file=path/('records.jsonl.gz' if (path/'records.jsonl.gz').is_file() else 'records.jsonl')
    times=defaultdict(list);headers=defaultdict(list);kinds={};commands=defaultdict(list);states=defaultdict(list)
    estimated=defaultdict(list);joint_groups=defaultdict(list);triggers=[];imu=[];health=defaultdict(list);inference=[];event_delays=[];count=0
    conditions=m['trial_condition'];declared=conditions.get('source_metadata',{}).get('topics',[])
    for t in declared:times[t['topic']];kinds[t['topic']]=t['type']
    for r in iter_records(file):
        count+=1;source=r['source'];topic=source.get('topic','manual_events');t=r['timestamp'];rid=r['record_id'];ns=t['value'];d=r.get('decoded') or {}
        times[topic].append((ns,rid));kinds.setdefault(topic,source.get('message_type','CSV/measurement event'))
        if r['header_timestamp']:headers[topic].append((r['header_timestamp']['value'],rid))
        if source.get('message_type')=='sensor_msgs/msg/JointState':joint_groups[(topic,tuple(d.get('name',[])))].append((ns,rid,r['header_timestamp']['value'] if r['header_timestamp'] else None))
        if r['command_velocity']:
            commands[topic].append({'id':rid,'ns':ns,'clock_id':t['clock_id'],'value':r['command_velocity']['value']})
        if r['estimated_velocity']:
            q=r['estimated_velocity'];estimated[(q['frame_id'],tuple(q['units']))].append(q['value'])
        name=state_name(d)
        if topic in ('/robot_state','/xm540_dual_policy/state') and name is not None:
            if not states[topic] or states[topic][-1]['state']!=name:states[topic].append({'ns':ns,'record_id':rid,'state':name})
        if topic=='/kick_trigger' and d.get('data') is True:triggers.append({'ns':ns,'record_id':rid})
        if source.get('message_type')=='sensor_msgs/msg/Imu' and 'orientation' in d:
            a=attitude(d['orientation'])
            if a is not None:imu.append({'ns':ns,'record_id':rid,**a})
        if topic=='/motor_diagnostics':
            try:diagnostic=json.loads(d['data'])
            except (ValueError,KeyError,TypeError):diagnostic={}
            for motor in diagnostic.get('motors',[]):
                for key in ('voltage_v','temp_c','current_ma','hw_error'):
                    if key in motor:health[key].append(motor[key])
        if isinstance(d.get('inference_ms'),(int,float)):inference.append(d['inference_ms'])
        for start_key,end_key,label in [('command_start_time','motion_start_time','command_to_motion'),
            ('command_start_time','target_reached_time','command_to_target_reached'),
            ('communication_send_time','communication_receive_time','app_send_to_receive')]:
            if r[start_key] and r[end_key]:
                packet=r['source'].get('packet_match') or {}
                if label=='app_send_to_receive' and not all(packet.get(k) for k in ('packet_id','sender','receiver','tx_locator','rx_locator')):
                    value={'value_s':None,'reason':'no_matched_packet_provenance'}
                else:value=aligned_delay(r[start_key],r[end_key],m)
                event_delays.append({'record_id':rid,'metric':label,**value})
    provenance={'normalized_file':str(file),'normalized_sha256':sha(file),'manifest_sha256':sha(path/'manifest.json'),
                'selection':'original record order; each event references raw message ID/CSV line via source.locator'}
    topics={topic:{'message_type':kinds.get(topic),'classification':'A','first_record_ns':v[0][0] if v else None,
        'last_record_ns':v[-1][0] if v else None,'recorder_intervals':intervals(v),'header_intervals':intervals(headers[topic]) if headers[topic] else None,
        'header_interpretation':'source/header clock semantics not verified; never substituted for recorder time'} for topic,v in times.items()}
    cmd={};all_segments=[]
    for topic,v in commands.items():
        try:segments=command_segments(v);error=None
        except ValueError as e:segments=[];error=str(e)
        cmd[topic]={'classification':'A','samples':len(v),'translation_norm_m_s':stats(math.hypot(*r['value'][:2]) for r in v),
            'abs_yaw_command_rad_s':stats(abs(r['value'][2]) for r in v),'max_abs_components':[max(abs(r['value'][i]) for r in v) for i in range(3)],
            'segment_count':len(segments),'duration_error':error,'interpretation':'commanded/shadow values, not actual ground speed'}
        all_segments.extend({'topic':topic,**s} for s in segments)
    recovery=[]
    for topic,entries in states.items():
        for i,e in enumerate(entries):
            if str(e['state']).upper() not in ('RECOVERY','RECOVERING','STANDUP','GETUP','PROTECT'):continue
            end=entries[i+1] if i+1<len(entries) else None
            recovery.append({'classification':'B','topic':topic,'start_record':e['record_id'],'end_record':end['record_id'] if end else None,
                'state':e['state'],'next_state':end['state'] if end else None,
                'recorded_state_dwell_s':(end['ns']-e['ns'])/1e9 if end else None,'right_censored':end is None,
                'interpretation':'software-state dwell, not confirmed physical successful recovery'})
    comparisons=[]
    for trigger in triggers:
        before=[e for e in states['/robot_state'] if e['ns']<=trigger['ns']]
        after=next((e for e in states['/robot_state'] if e['ns']>trigger['ns'] and (not before or e['state']!=before[-1]['state'])),None)
        if after:comparisons.append({'classification':'B','kind':'kick_trigger_to_changed_robot_state','start_record':trigger['record_id'],
            'end_record':after['record_id'],'delta_s':(after['ns']-trigger['ns'])/1e9,'end_state':after['state'],
            'interpretation':'recorded event interval; no causal motor-onset or success claim'})
        crossing=next((v for v in imu if v['ns']>=trigger['ns'] and v['tilt_deg']>=45),None)
        protect=next((e for e in states['/robot_state'] if e['ns']>=trigger['ns'] and e['state']=='PROTECT'),None)
        if crossing and protect:comparisons.append({'classification':'B','kind':'IMU_45deg_to_PROTECT_recorded_proxy',
            'start_record':crossing['record_id'],'end_record':protect['record_id'],'delta_s':(protect['ns']-crossing['ns'])/1e9,
            'threshold_source':'offline 45deg diagnostic, not a proposed safety threshold','interpretation':'sensor-estimate/recorder proxy, not physical fall onset latency'})
    velocity=[{'classification':'B','frame_id':frame,'units':list(units),'components':[stats(x[i] for x in rows) for i in range(3)]}
              for (frame,units),rows in estimated.items()]
    report={'schema_version':m['schema_version'],'data_origin':m['data_origin'],'records':count,'provenance':provenance,
        'topics':topics,'commands':cmd,'estimated_velocity':velocity or unavailable('No odometry/body-velocity estimate recorded; joint velocity is not body velocity'),
        'state_transitions':dict(states),'software_recovery_intervals':recovery,'recorded_event_comparisons':comparisons,
        'health_values':{k:stats(v) for k,v in health.items()},'shadow_inference_ms':stats(inference),'declared_timestamp_delays':event_delays,
        'actual_ground_speed_accuracy':unavailable('No independent synchronized trajectory GT in existing records'),
        'absolute_position_error':unavailable('Odometry/self-estimate alone is not independent GT'),
        'target_to_motion_delay':unavailable('No linked target/start and independent physical motion onset events') if not any(x['metric']=='command_to_motion' and x.get('value_s') is not None for x in event_delays) else {'classification':'B','source':'declared_timestamp_delays'},
        'one_way_network_delay':unavailable('No matched sender/receiver events with measured clock offset/error') if not any(x['metric']=='app_send_to_receive' and x.get('value_s') is not None for x in event_delays) else {'classification':'B','source':'declared_timestamp_delays'},
        'successful_physical_recovery_time':unavailable('No independently confirmed upright/stable/service-resumed sequence'),
        'no_position_or_GT_fabrication':True}
    report['joint_message_signatures']=[{'topic':topic,'names':list(names),'recorder_intervals':intervals([(t,i) for t,i,h in values]),
        'header_intervals':intervals([(h,i) for t,i,h in values if h is not None]),
        'meaning':'grouped by joint-name signature; not proof of publisher identity or motor/policy frequency'} for (topic,names),values in joint_groups.items()]
    save(output/'analysis.json',report);save(output/'command_segments.json',all_segments)
    save(output/'validation.json',validate_dataset(path))
    return report
