"""Offline ROS2 Humble bag analysis. SQLite is opened read-only; no ROS node exists."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sqlite3

import numpy as np
from rosbags.typesys import Stores, get_typestore


def stamp_ns(msg, fallback):
    if not hasattr(msg, 'header'):
        return fallback
    stamp = msg.header.stamp
    result = int(stamp.sec) * 10**9 + int(stamp.nanosec)
    return result if result else fallback


def asof_indices(reference, query):
    """Latest source-time sample <= query; -1 denotes no available past sample."""
    return np.searchsorted(reference, query, side='right') - 1


def attitude(q):
    v = np.array([q.x, q.y, q.z, q.w], dtype=float)
    length = np.linalg.norm(v)
    if not np.isfinite(v).all() or length < 1e-9:
        raise ValueError('Invalid IMU orientation')
    x, y, z, w = v / length
    gx, gy, gz = 2*(w*y-x*z), -2*(w*x+y*z), -(1-2*(x*x+y*y))
    tilt = math.degrees(math.acos(float(np.clip(-gz, -1, 1))))
    roll = math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = math.degrees(math.asin(float(np.clip(2*(w*y-z*x), -1, 1))))
    return tilt, roll, pitch


def stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(zip(['min', 'p50', 'p99', 'max'], map(float, np.percentile(a, [0, 50, 99, 100]))))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    database = next(args.bag.glob('*.db3')) if args.bag.is_dir() else args.bag
    database = database.resolve()
    input_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
        integrity = db.execute('PRAGMA quick_check').fetchone()[0]
        if integrity != 'ok':
            raise ValueError(integrity)
        topics = {name: (idx, typ) for idx, name, typ in db.execute('SELECT id,name,type FROM topics')}
        goal_id = topics['/joint_goals'][0]
        first, last, count = db.execute('SELECT MIN(timestamp),MAX(timestamp),COUNT(*) FROM messages WHERE topic_id=?', (goal_id,)).fetchone()
        if not count:
            raise ValueError('No goals in bag')
        all_first, all_last, all_count = db.execute('SELECT MIN(timestamp),MAX(timestamp),COUNT(*) FROM messages').fetchone()

        def read(name, whole=False):
            idx, typ = topics[name]
            query = 'SELECT timestamp,data FROM messages WHERE topic_id=?'
            params = [idx]
            if not whole:
                query += ' AND timestamp>=? AND timestamp<=?'
                params += [first-2*10**9, last+2*10**9]
            query += ' ORDER BY timestamp,id'
            return [(int(t), typestore.deserialize_cdr(raw, typ)) for t, raw in db.execute(query, params)]

        raw_goals = read('/joint_goals', True)
        raw_preview = read('/xm540_dual_policy/preview_joint_goals', True)
        raw_joints = read('/joint_states')
        raw_imus = read('/imu/data')
        raw_states = read('/xm540_dual_policy/state', True)
        raw_diag = read('/motor_diagnostics')
        raw_cmd = read('/cmd_vel')
        raw_torque = read('/u2d2/torque_enabled', True)
        raw_stop = read('/buttons/stop_pressed', True)
        raw_behavior = read('/u2d2/behavior_status', True)

    goal_ns = np.array([stamp_ns(m, t) for t, m in raw_goals], dtype=np.int64)
    origin = int(goal_ns[0])
    relative = lambda n: (int(n)-origin)/1e9
    names = list(raw_goals[0][1].name)
    legs = [n for n in names if any(k in n for k in ('pelvis', 'knee', 'ankle'))]
    g = np.array([[dict(zip(m.name, m.position))[n] for n in names] for _, m in raw_goals])
    # Source timestamps and recorder timestamps are retained independently.
    raw_joints.sort(key=lambda pair: stamp_ns(pair[1], pair[0]))
    joint_ns = np.array([stamp_ns(m, t) for t, m in raw_joints], dtype=np.int64)
    q = np.array([[dict(zip(m.name, m.position))[n] for n in names] for _, m in raw_joints])
    qd = np.array([[dict(zip(m.name, m.velocity))[n] for n in names] for _, m in raw_joints])
    if not np.isfinite(g).all() or not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError('Non-finite joint values')
    raw_imus.sort(key=lambda pair: stamp_ns(pair[1], pair[0]))
    imu_ns = np.array([stamp_ns(m, t) for t, m in raw_imus], dtype=np.int64)
    angles = np.array([attitude(m.orientation) for _, m in raw_imus])
    gyro = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for _, m in raw_imus])
    jt, gt, it = np.array(list(map(relative, joint_ns))), np.array(list(map(relative, goal_ns))), np.array(list(map(relative, imu_ns)))
    qix = asof_indices(joint_ns, goal_ns)
    iix = asof_indices(imu_ns, goal_ns)
    if (qix < 0).any() or (iix < 0).any():
        raise ValueError('Missing pre-goal state')
    baseline = (jt >= -1) & (jt < 0)
    baseline_q = np.median(q[baseline], axis=0)

    state_events = []
    previous = None
    for t, m in raw_states:
        value = json.loads(m.data)
        if value != previous:
            state_events.append({'t': relative(t), **value})
            previous = value

    def changes(rows, parse):
        result, previous = [], None
        for t, m in rows:
            value = parse(m)
            if value != previous:
                result.append({'t': relative(t), 'value': value})
                previous = value
        return result

    rows = []
    for index, time in enumerate(gt):
        row = {'t': float(time), 'goal_record_delay_ms': (raw_goals[index][0]-int(goal_ns[index]))/1e6,
               'feedback_age_ms': (int(goal_ns[index])-int(joint_ns[qix[index]]))/1e6,
               'imu_age_ms': (int(goal_ns[index])-int(imu_ns[iix[index]]))/1e6,
               'tilt_deg': float(angles[iix[index], 0])}
        for j, name in enumerate(names):
            row['goal_'+name] = float(g[index,j])
            row['real_'+name] = float(q[qix[index],j])
            row['error_'+name] = float(q[qix[index],j]-g[index,j])
        rows.append(row)
    write_csv(output/'goals_with_past_feedback.csv', rows)
    write_csv(output/'joint_feedback.csv', [
        {'t': float(time), **{n: float(q[k,j]) for j,n in enumerate(names)}} for k,time in enumerate(jt)])
    write_csv(output/'imu.csv', [
        {'t': float(time), 'tilt_deg': float(angles[k,0]), 'roll_deg': float(angles[k,1]), 'pitch_deg': float(angles[k,2]),
         'wx': float(gyro[k,0]), 'wy': float(gyro[k,1]), 'wz': float(gyro[k,2])} for k,time in enumerate(it)])
    diagnostics = []
    for t, m in raw_diag:
        d = json.loads(m.data)
        for motor in d['motors']:
            diagnostics.append({'t': relative(t), **motor})
    write_csv(output/'motor_diagnostics.csv', diagnostics)
    active_diag = [d for d in diagnostics if 0 <= d['t'] <= gt[-1] and d['name'] in legs]
    cmd = np.array([[m.linear.x, m.linear.y, m.angular.z] for _, m in raw_cmd])

    threshold_events = {}
    for threshold in (5,10,15,20,30,45,55):
        hits = np.flatnonzero((it >= 0) & (angles[:,0] >= threshold))
        threshold_events[str(threshold)] = float(it[hits[0]]) if len(hits) else None

    phase_ranges = [('early_0_1',0,1),('before_2s',0,2),
                    ('before15deg',0,threshold_events['15'] or float(gt[-1])),
                    ('before20deg',0,threshold_events['20'] or float(gt[-1])),('all_goals',0,float(gt[-1]))]
    joint_metrics = {}
    for name in legs:
        j = names.index(name)
        metrics = {'prestart_median_rad':float(baseline_q[j]),'first_goal_rad':float(g[0,j]),'last_goal_rad':float(g[-1,j]),'max_goal_step_rad':float(np.max(np.abs(np.diff(g[:,j])))),
                   'first_feedback_rad':float(q[qix[0],j]), 'last_feedback_rad':float(q[qix[-1],j])}
        for label,start,end in phase_ranges:
            mask=(gt>=start)&(gt<=end)
            errors=q[qix[mask],j]-g[mask,j]
            metrics[label]={'max_abs_error_rad':float(np.max(np.abs(errors))), 'rmse_rad':float(np.sqrt(np.mean(errors**2))),
                            'max_abs_error_t':float(gt[mask][np.argmax(np.abs(errors))]),
                            'max_abs_goal_change_from_start_rad':float(np.max(np.abs(g[mask,j]-g[0,j])))}
        joint_metrics[name]=metrics

    snapshots = []
    for t in (0,.25,.5,.75,1,1.25,1.5,1.75,2,2.1,2.25,float(gt[-1]),3,4):
        gi=int(asof_indices(gt,t)); qi=int(asof_indices(jt,t)); ii=int(asof_indices(it,t))
        if min(gi,qi,ii)<0: continue
        snapshot={'t':t,'tilt_deg':float(angles[ii,0]), 'roll_deg':float(angles[ii,1]),'pitch_deg':float(angles[ii,2])}
        for name in ('l_knee_pitch_joint','r_knee_pitch_joint','l_ankle_pitch_joint','r_ankle_pitch_joint'):
            j=names.index(name)
            snapshot[name]={'goal':float(g[gi,j]),'real':float(q[qi,j])}
        snapshots.append(snapshot)

    header_delays={}
    for name, raw in [('/joint_goals',raw_goals),('/joint_states',raw_joints),('/imu/data',raw_imus)]:
        header_delays[name]=stats([(t-stamp_ns(m,t))/1e6 for t,m in raw])
    leg_ids=[names.index(n) for n in legs]
    step_max=np.max(np.abs(np.diff(g[:,leg_ids],axis=0)),axis=1)
    high_steps=np.flatnonzero(step_max>.041)
    report={'database':str(database),'input_sha256':input_hash,'sqlite_quick_check':integrity,
            'bag_duration_s':(all_last-all_first)/1e9,'bag_message_count':all_count,'goal_start_record_ns':first,
            'goal_start_header_ns':origin,'goal_offset_from_bag_start_s':(origin-all_first)/1e9,
            'goal_count':count,'goal_duration_s':float(gt[-1]),'goal_interval_ms':stats(np.diff(goal_ns)/1e6),
            'goal_gaps_over_40ms':int(np.sum(np.diff(goal_ns)>40e6)),
            'first_goal_step_over_0041_rad_s':float(gt[high_steps[0]+1]) if len(high_steps) else None,
            'joint_feedback_interval_ms':stats(np.diff(joint_ns)/1e6),
            'largest_feedback_gap_start_s':float(jt[np.argmax(np.diff(joint_ns))]),
            'preview_equal_to_goals_by_sequence':len(raw_preview)==len(raw_goals) and all(list(a.name)==list(b.name) and np.array_equal(a.position,b.position) for (_,a),(_,b) in zip(raw_goals,raw_preview)),
            'prestart_tilt_stats_deg':stats(angles[(it>=-1)&(it<0),0]),
            'threshold_first_crossings_s':threshold_events,'state_events':state_events,
            'torque_events':changes(raw_torque,lambda m:bool(m.data)),
            'stop_events':changes(raw_stop,lambda m:bool(m.data)),
            'behavior_events':changes(raw_behavior,lambda m:json.loads(m.data)),
            'command_max_abs_in_window':np.max(np.abs(cmd),axis=0).tolist(),
            'header_to_recorder_ms':header_delays,'feedback_age_at_goal_ms':stats([r['feedback_age_ms'] for r in rows]),
            'imu_age_at_goal_ms':stats([r['imu_age_ms'] for r in rows]),
            'joint_metrics':joint_metrics,'snapshots':snapshots,
            'active_diag_voltage_v':stats([d['voltage_v'] for d in active_diag]),
            'active_diag_temperature_c':stats([d['temp_c'] for d in active_diag]),
            'active_diag_max_abs_current_ma':max(abs(d['current_ma']) for d in active_diag),
            'active_diag_nonzero_hw_error':[d for d in active_diag if d['hw_error']],
            'diagnostics':diagnostics,
            'limitations':['Joint errors are simultaneous comparisons, not transport-delay-corrected actuator errors.',
                           '5Hz state messages bound but do not timestamp internal policy transitions exactly.',
                           '2Hz diagnostics cannot rule out short transients or infer motor torque directly.',
                           'Raw actor output, pre-limiter target, motor internal trajectory, and model identity at recording are absent.',
                           'Recorder/header differences do not measure sensor acquisition-to-policy delay in full.']}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(4,2,figsize=(14,13),sharex=True)
    for ax, name in zip(axes[:3,:].flat, ['l_knee_pitch_joint','r_knee_pitch_joint','l_ankle_pitch_joint','r_ankle_pitch_joint','l_pelvis_pitch_joint','r_pelvis_pitch_joint']):
        j=names.index(name)
        ax.plot(jt,q[:,j],label='Measured',lw=1.8)
        ax.step(gt,g[:,j],where='post',label='Published target',lw=1.4)
        ax.set_title(name); ax.set_ylabel('rad'); ax.grid(alpha=.25); ax.legend(loc='best')
    ax=axes[3,0]
    for j,label in enumerate(['Tilt','Roll','Pitch']): ax.plot(it,angles[:,j],label=label)
    ax.set_ylabel('deg'); ax.legend(); ax.grid(alpha=.25)
    ax=axes[3,1]
    for name in ['l_knee_pitch_joint','r_knee_pitch_joint']:
        ds=[d for d in diagnostics if d['name']==name]
        ax.plot([d['t'] for d in ds],[d['current_ma'] for d in ds],marker='o',label=name)
    ax.set_ylabel('mA (2Hz diagnostic samples)'); ax.legend(fontsize=8); ax.grid(alpha=.25)
    for ax in axes.flat:
        ax.axvline(0,color='black',lw=.8)
        for event in state_events:
            if event['mode'] in ('RECOVERY','FAULT') and -1<event['t']<5:
                ax.axvline(event['t'],color='red' if event['mode']=='FAULT' else 'orange',ls=':',alpha=.8)
        ax.set_xlim(-1,4)
    axes[3,0].set_xlabel('Seconds from first published goal (header time)')
    axes[3,1].set_xlabel('Orange: first RECOVERY report; red: first FAULT report')
    fig.suptitle('XM540 recorded failure: published targets, measured joints, body tilt and current\nState reports are sampled at 5Hz; target traces end when publishing stops',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(output/'failure_timeline.png',dpi=160); plt.close(fig)
    # Validate output bookkeeping and no accidental write to evidence.
    assert report['preview_equal_to_goals_by_sequence']
    assert np.array_equal(asof_indices([1,3],[0,1,2,3,4]),[-1,0,0,1,1])
    assert hashlib.sha256(database.read_bytes()).hexdigest()==input_hash
    report['source_unchanged_after_analysis']=True
    (output/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    brief={k:v for k,v in report.items() if k not in ('diagnostics','snapshots','joint_metrics')}
    brief['knee_metrics']={n:joint_metrics[n] for n in ['l_knee_pitch_joint','r_knee_pitch_joint']}
    print(json.dumps(brief,indent=2,ensure_ascii=False))
    print('Artifacts:',output)


if __name__=='__main__':
    main()
