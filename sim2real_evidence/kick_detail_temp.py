import sys, json, csv, math
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path('sim2real_evidence').resolve()))
import analyze_kick_balance as a
print('GOAL SCHEMAS',Counter(tuple(m.name) for t,m in a.data['/joint_goals']))
for lo,hi in [(-1,0),(0,.3),(.3,.7),(.7,1),(1,1.5),(1.5,1.9),(1.9,3.1)]:
    goals=[(t,m) for t,m in a.data['/joint_goals'] if lo<=a.sec(t)<hi and 'l_knee_pitch_joint' in m.name]
    print('BIN',lo,hi,'LEG GOALS',len(goals))
    for n in ['l_pelvis_roll_joint','r_pelvis_roll_joint','l_knee_pitch_joint','r_knee_pitch_joint']:
        v=[dict(zip(m.name,m.position))[n] for t,m in goals]
        if v:print(n,'deg range',np.round(np.degrees([min(v),max(v)]),1).tolist())
for target in [0,.3,.5,.7,.85,1,1.25,1.5,1.7,1.85]:
    print('SNAPSHOT',target)
    for topic in ['/joint_goals','/joint_states']:
        ts=[(t,m) for t,m in a.data[topic] if 'l_knee_pitch_joint' in m.name]
        t,m=min(ts,key=lambda v:abs(a.sec(v[0])-target));d=dict(zip(m.name,m.position))
        print(topic,round(a.sec(t),3),{k:round(math.degrees(d[k]),1) for k in ['l_pelvis_roll_joint','r_pelvis_roll_joint','l_knee_pitch_joint','r_knee_pitch_joint','l_ankle_roll_joint','r_ankle_roll_joint']})
for t,m in a.data['/motor_diagnostics']:
    if -.3<=a.sec(t)<=2:
        motors=json.loads(m.data)['motors']
        print('MOTOR',round(a.sec(t),3),'voltage_min',min(x['voltage_v'] for x in motors),'peak_current',max(abs(x['current_ma']) for x in motors),'errors',[(x['id'],x['hw_error']) for x in motors if x['hw_error']])
