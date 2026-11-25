"""Explicit missingness, units, clocks and provenance for measurement-v1."""
import math

VERSION='measurement-v1'
TIME_FIELDS=('command_start_time','motion_start_time','target_reached_time',
             'communication_send_time','communication_receive_time')
QUANTITIES=('command_velocity','estimated_velocity','ground_truth_position','estimated_position','target_position')
ORIGINS=('real_robot','real_robot_shadow','synthetic')


def time_value(ns,clock,source,semantic='record_receive',uncertainty_ns=None):
    return {'value':ns,'unit':'ns','clock_id':clock,'source':source,'semantic':semantic,'uncertainty_ns':uncertainty_ns}


def quantity(values,frame,components,units,timestamp,**extra):
    return {'value':list(values),'frame_id':frame,'components':list(components),'units':list(units),
            'timestamp':timestamp,**extra}


def blank_record(manifest,record_id,timestamp,source,event_type):
    return {'schema_version':VERSION,'experiment_id':manifest['experiment_id'],'robot_id':manifest['robot_id'],
            'trial_id':manifest['trial_id'],'trial_condition_ref':'manifest.trial_condition','record_id':record_id,
            'timestamp':timestamp,'header_timestamp':None,'source':source,'event_type':event_type,
            **{k:None for k in TIME_FIELDS+QUANTITIES},'success':None,'failure_reason':None,'decoded':None}


def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite value')
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    if isinstance(value,(list,tuple)):
        for v in value:finite_tree(v)


def validate_time(t,manifest):
    if not isinstance(t,dict):raise ValueError('Timestamp must be an object')
    if isinstance(t.get('value'),bool) or not isinstance(t.get('value'),int):raise ValueError('Timestamp needs integer ns')
    if t.get('unit')!='ns' or t.get('clock_id') not in manifest['clocks']:raise ValueError('Unknown clock or time unit')
    if not t.get('source') or not t.get('semantic'):raise ValueError('Timestamp provenance missing')
    u=t.get('uncertainty_ns')
    if u is not None and (isinstance(u,bool) or not isinstance(u,(int,float)) or u<0 or not math.isfinite(u)):raise ValueError('Invalid time uncertainty')


def validate_manifest(m):
    finite_tree(m)
    for k in ('experiment_id','robot_id','trial_id'):
        if not isinstance(m.get(k),str) or not m[k] or m[k].startswith('SET_'):raise ValueError('Missing identity: '+k)
    if m.get('schema_version')!=VERSION or m.get('data_origin') not in ORIGINS:raise ValueError('Unknown schema/origin')
    if not isinstance(m.get('trial_condition'),dict) or not m['trial_condition']:raise ValueError('Trial conditions required')
    repeat=m.get('repeat_index')
    if not (repeat is None and m.get('mode')=='historical_forensic') and (not isinstance(repeat,int) or isinstance(repeat,bool) or repeat<0):raise ValueError('Repeat index required; historical unknown may be null')
    if not m.get('clocks') or not m.get('frames'):raise ValueError('Clocks and frames required')
    for c in m['clocks'].values():
        if c.get('unit')!='ns' or not c.get('basis') or 'host' not in c:raise ValueError('Clock declaration incomplete')
    for s in m.get('synchronization',[]):
        if s.get('source_clock') not in m['clocks'] or s.get('target_clock') not in m['clocks']:raise ValueError('Sync clock missing')
        if not s.get('method'):raise ValueError('Sync method missing')
        for k in ('offset_ns','uncertainty_ns'):
            if k not in s:raise ValueError('Explicit null required for unavailable synchronization')
            if s[k] is not None and (isinstance(s[k],bool) or not isinstance(s[k],(int,float))):raise ValueError('Sync values must be numeric or null')
        if s['uncertainty_ns'] is not None and s['uncertainty_ns']<0:raise ValueError('Negative sync uncertainty')
    return m


def validate_record(r,m):
    finite_tree(r)
    for key in ('schema_version','experiment_id','robot_id','trial_id'):
        if r.get(key)!=m[key]:raise ValueError('Identity mismatch: '+key)
    for k in TIME_FIELDS+QUANTITIES+('success','failure_reason','header_timestamp'):
        if k not in r:raise ValueError('Missing field; use explicit null: '+k)
    if not isinstance(r.get('record_id'),str) or not r['record_id'] or not r.get('event_type'):raise ValueError('Record identity/type missing')
    if not isinstance(r.get('source'),dict) or not r['source'].get('locator'):raise ValueError('Source locator required')
    validate_time(r['timestamp'],m)
    for k in TIME_FIELDS+('header_timestamp',):
        if r[k] is not None:validate_time(r[k],m)
    for k in QUANTITIES:
        q=r[k]
        if q is None:continue
        if q.get('frame_id') not in m['frames']:raise ValueError('Unknown coordinate frame')
        v,u,c=q.get('value'),q.get('units'),q.get('components')
        expected=(['vx','vy','wz'],['m/s','m/s','rad/s']) if 'velocity' in k else (['x','y','yaw'],['m','m','rad'])
        if c!=expected[0] or u!=expected[1] or not isinstance(v,list) or len(v)!=3:raise ValueError('Quantity units/components/shape mismatch')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in v):raise ValueError('Quantity needs numeric values')
        validate_time(q['timestamp'],m)
        if k=='ground_truth_position':
            if q.get('independent') is not True or not q.get('device_id') or not q.get('calibration_id'):raise ValueError('Independent GT provenance required')
            if q.get('method') not in ('external_camera','motion_capture','survey','synthetic_truth'):raise ValueError('Odometry is not independent GT')
            if q['method']=='synthetic_truth' and m['data_origin']!='synthetic':raise ValueError('Synthetic truth cannot be real evidence')
    if r['success'] is not None and type(r['success']) is not bool:raise ValueError('Success must be bool or null')
    if r['failure_reason'] is not None and not isinstance(r['failure_reason'],str):raise ValueError('Failure reason must be string/null')
    return r


def aligned_delay(start,end,m):
    """Only declared clock links; null uncertainty never becomes a measured bound."""
    validate_time(start,m);validate_time(end,m)
    if start['clock_id']==end['clock_id']:
        offset=0;sync_u=0
    else:
        candidates=[s for s in m.get('synchronization',[]) if s['source_clock']==start['clock_id'] and s['target_clock']==end['clock_id']]
        if not candidates:return {'value_s':None,'reason':'no_clock_link'}
        s=candidates[0];offset=s['offset_ns'];sync_u=s['uncertainty_ns']
        if offset is None or sync_u is None:return {'value_s':None,'reason':'sync_offset_or_error_unmeasured'}
        if s.get('valid_from_ns') is None or s.get('valid_to_ns') is None or not s['valid_from_ns']<=start['value']<=s['valid_to_ns']:
            return {'value_s':None,'reason':'outside_sync_validity'}
    delta=end['value']-start['value']-offset
    if delta<0:return {'value_s':None,'reason':'negative_aligned_delay','raw_difference_ns':delta}
    u=[sync_u,start['uncertainty_ns'],end['uncertainty_ns']]
    return {'value_s':delta/1e9,'uncertainty_s':None if any(v is None for v in u) else sum(u)/1e9,
            'reason':None,'interpretation':'declared event timestamp difference, not automatically physical latency'}
