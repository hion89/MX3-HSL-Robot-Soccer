"""Read-only bag analysis; timestamps retain recorder and message clocks."""
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore
from analyze_walk_failure import attitude, stamp_ns

root = Path(__file__).resolve().parent / 'kick_balance'
bag = sorted(root.glob('kick_balance_*'))[-1]
database = next(bag.glob('*.db3'))
out = bag / 'analysis'
out.mkdir(exist_ok=True)
store = get_typestore(Stores.ROS2_HUMBLE)
data = {}
with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
    assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    for idx, name, typ in db.execute('SELECT id,name,type FROM topics'):
        data[name] = [(t, store.deserialize_cdr(raw, typ)) for t, raw in
                      db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id', (idx,))]
triggers = data.get('/kick_trigger', [])
origin = triggers[0][0] if triggers else min(v[0][0] for v in data.values() if v)
sec = lambda t: (t-origin)/1e9
summary = {'database': str(database), 'sha256': hashlib.sha256(database.read_bytes()).hexdigest(),
           'topics': {k: {'count':len(v), 'first':sec(v[0][0]) if v else None,
                          'last':sec(v[-1][0]) if v else None} for k,v in data.items()}}
for topic in ['/kick_trigger', '/kick_done', '/robot_state', '/motor_diagnostics']:
    rows = data.get(topic, [])
    events = []
    last = None
    for t,m in rows:
        value = m.data
        if value != last:
            events.append({'t':sec(t),'data':value})
            last = value
    summary[topic] = events

imu = []
for t,m in data.get('/imu/data_50hz', []):
    tilt, roll, pitch = attitude(m.orientation)
    imu.append({'t':sec(t),'source_t':sec(stamp_ns(m,t)), 'tilt_deg':tilt,'roll_deg':roll,'pitch_deg':pitch,
                'wx':m.angular_velocity.x,'wy':m.angular_velocity.y,'wz':m.angular_velocity.z})
for topic, filename in [('/joint_goals','goals.csv'),('/joint_states','joints.csv')]:
    rows=[]
    for t,m in data.get(topic,[]):
        rows.append({'t':sec(t),'source_t':sec(stamp_ns(m,t)), **dict(zip(m.name,m.position))})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out/filename).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
with (out/'imu.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(imu[0]));writer.writeheader();writer.writerows(imu)
summary['tilt_crossings']={str(th):next((r for r in imu if r['t']>=0 and r['tilt_deg']>=th),None) for th in [10,20,30,45,60]}
summary['imu_bins']=[]
for start in np.arange(-1,5,.25):
    samples=[r for r in imu if start<=r['t']<start+.25]
    if samples:
        summary['imu_bins'].append({'t':float(start),**{k:round(float(np.median([r[k] for r in samples])),2) for k in ['tilt_deg','roll_deg','pitch_deg']}})
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k != '/motor_diagnostics'},indent=2))
