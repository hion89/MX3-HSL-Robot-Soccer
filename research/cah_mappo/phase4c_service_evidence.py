"""Supplementary read-only timer/service evidence; no ROS imports or launches."""
from pathlib import Path
import sys
import re
import hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from cah_mappo.phase3b2b.io import save


def collect(output):
    code=ROOT.parents[1]/'MX3_Match_Sim/src';rows=[];hashes={}
    def item(key,relative,pattern,value,note):
        p=code/relative;data=p.read_bytes();h=hashlib.sha256(data).hexdigest();hashes[str(p.resolve())]=h
        hits=[{'line':i+1,'text':line} for i,line in enumerate(data.decode('utf-8-sig').splitlines()) if re.search(pattern,line)]
        assert hits,key
        rows.append({'key':key,'kind':'timer_setting','value':value,'note':note,
                     'source':str(p.resolve()),'sha256':h,'lines':hits[:10],'actual_performance':False})
    item('ball_tracker_default_hz','hsl_src/hsl_world_model/src/ball_tracker_node.cpp',r'update_rate", 50.0|create_wall_timer',50.,
         'Default periodic publishing timer, not measured new-detection frequency or end-to-end latency.')
    item('kick_default_occupancy_settings','mx3_src/humanoid_bringup/launch/control.launch.py',
         r'DeclareLaunchArgument\("kick_duration"|DeclareLaunchArgument\("kick_start_hold"|"kick_duration_s":|"resume_hold_seconds": kick_start_hold',
         {'kick_duration_s':2.,'pre_kick_hold_s':.6},
         'Current source defaults; actual launch overrides, approach, post-kick settling and failed attempts alter physical occupancy. No successful timing measurement.')
    item('behavior_kick_timeout','hsl_src/hsl_behavior/hsl_behavior/behavior_node.py',r'kick_timeout =|now - self.kick_t >',6.,
         'A timeout for the skill completion signal, not a measured 6-second kick.')
    save(output/'evidence/service_timer_supplement.json',rows)
    save(output/'evidence/service_timer_sources_sha256.json',hashes)
    print('Read-only service/timer supplement',len(rows))

if __name__=='__main__':collect(ROOT/'phase4c_outputs/study_v1')
