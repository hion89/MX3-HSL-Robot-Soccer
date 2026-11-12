import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=sorted((Path(__file__).resolve().parent/'kick_balance').glob('kick_balance_*'))[-1]/'analysis'
def read(name):
    with (root/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))
imu=read('imu.csv');goals=read('goals.csv');joints=read('joints.csv')
fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
axes[0].plot([float(r['t']) for r in imu],[float(r['tilt_deg']) for r in imu],label='IMU total tilt')
axes[0].plot([float(r['t']) for r in imu],[float(r['roll_deg']) for r in imu],label='IMU roll',alpha=.65)
axes[0].set_ylabel('Body angle (deg)');axes[0].set_ylim(-25,110)
for ax,joint in zip(axes[1:],['l_ankle_roll_joint','l_pelvis_roll_joint']):
    for rows,label in [(goals,'Target'),(joints,'Measured')]:
        subset=[r for r in rows if r.get(joint,'')!='']
        ax.plot([float(r['t']) for r in subset],np.degrees([float(r[joint]) for r in subset]),label=label)
    ax.set_ylabel(joint.replace('_joint','')+' (deg)')
for ax in axes:
    ax.axvline(0,color='gray',ls=':',label='Kick trigger')
    ax.axvline(1.931258497,color='red',ls='--',label='PROTECT')
    ax.axvline(2,color='black',ls=':',alpha=.45,label='2 s reference (not observed done)')
    ax.grid(alpha=.25);ax.legend(loc='upper left',fontsize=8);ax.set_xlim(-.3,2.3)
axes[-1].set_xlabel('Seconds relative to recorded kick trigger')
fig.suptitle('Recorded kick: tilt grows before any kick_done message\nRecorder time; joint traces do not identify publisher or foot contact')
fig.tight_layout();fig.savefig(root/'kick_timeline.png',dpi=160)
print(root/'kick_timeline.png')
