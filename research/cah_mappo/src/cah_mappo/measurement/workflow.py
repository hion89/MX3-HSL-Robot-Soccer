"""Concrete historical study with file hashes and line/message provenance."""
import json
from pathlib import Path
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from .importers import extract_bag,extract_shadow,source_info
from .analysis import analyze_dataset
from .collection import session_template
from .storage import read,save,sha


def console_evidence(path):
    source=source_info(path);text=Path(path).read_text(encoding='utf-8-sig');rates=[];tf=[]
    for line,text in enumerate(text.splitlines(),1):
        hit=re.search(r'average rate:\s*([\d.]+)',text)
        if hit:rates.append({'line':line,'printed_average_hz':float(hit[1])})
        if 'TF_WAIT' in text or 'TF_READY' in text:
            age=re.search(r'age=([\d.]+)s',text)
            tf.append({'line':line,'text':text,'printed_age_s':float(age[1]) if age else None})
    return {'source':source,'classification':'B','printed_rates':rates,'TF_events':tf,
        'limitations':['Console summaries are not per-message raw timestamps or positive-ball detection rates.',
                       'No physical motion/absolute error/one-way delay inferred from console output.']}


def safety_manual_evidence(repo):
    files=[];terms=('\uc628\ub3c4','\uc804\uc555','\uc804\ub958','\uacfc\uc5f4','\ubc30\ud130\ub9ac','\ube44\uc0c1','\uc9c0\uc9c0')
    for path in (repo/'Manual').glob('*.docx'):
        with zipfile.ZipFile(path) as z:root=ET.fromstring(z.read('word/document.xml'))
        paragraphs=[]
        for e in root.iter():
            if e.tag.rsplit('}',1)[-1]=='p':
                s=''.join(t.text or '' for t in e.iter() if t.tag.rsplit('}',1)[-1]=='t')
                if any(word in s.lower() for word in terms):paragraphs.append(s)
        files.append({'source':source_info(path),'matching_paragraphs':paragraphs,
            'verified_numeric_motor_battery_balance_stop_limits':None})
    return {'manuals':files,'classification':'D','numeric_safe_limits':None,
        'reason':'No verified exact hardware/battery stop envelope established; generic model names and observed diagnostic ranges are not safe limits.',
        'operator_records':['docs/MX3_HSL_DEPLOYMENT_PROGRESS.md: 2026-11-04 overheat concern and supported torque-off guidance',
                            'MX3_Match_Sim/src/mx3_src/humanoid_serial_bridge/config/motor_gains.yaml: current/heat cautions, not safe envelope'],
        'configured_command_limits_class_C':{'real_role_trial_vx_m_s':.20,'vy_m_s':.15,'wz_rad_s':.50},
        'not_permission_to_use_these_values_as_hardware_safe_limits':True}


def study(root,output,sources):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);sources=Path(sources);repo=root.parents[1]
    specs=[('walk','walk_20261007_161217/walk_20261007_161217_0.db3'),
           ('kick','kick_balance/kick_balance_20261110_151804/kick_balance_20261110_151804_0.db3')]
    shadow=sources/'observations_20261105/walk_shadow_20261105_162155_qeh2uxxq/v0_2_log_20261105_162156.csv'
    extra=[sources/'observations_20261105/walk_shadow_20261105_162155_qeh2uxxq/summary.json',
           sources/'observations_20261105/walk_shadow_20261105_162155_qeh2uxxq/runner.log',
           sources/'observations_20261104/detections_3d.txt',sources/'observations_20261104/ball_tracker_tf_v3_runtime.txt',
           sources/'kick_video_20261110/metadata.json']
    raw=[sources/n for _,n in specs]+[shadow]+extra
    before={str(p.resolve()):sha(p) for p in raw};save(output/'original_hashes_before.json',before)
    results={}
    for name,relative in specs:
        p=sources/relative;dataset=output/(name+'_dataset');extract_bag(p,dataset)
        results[name]=analyze_dataset(dataset,output/(name+'_analysis'))
        print('Imported/analyzed',name,results[name]['records'],flush=True)
    extract_shadow(shadow,output/'shadow_dataset');results['shadow']=analyze_dataset(output/'shadow_dataset',output/'shadow_analysis')
    save(output/'console_evidence.json',[console_evidence(extra[2]),console_evidence(extra[3])])
    save(output/'video_evidence.json',{'source':source_info(extra[4]),'classification':'B','metadata':read(extra[4]),
        'original_video_candidates':[{'path':str(repo.parent/'KakaoTalk_20261110_155326490.mp4'),
                                      'exists':(repo.parent/'KakaoTalk_20261110_155326490.mp4').is_file()}],
        'frames_directory':str(sources/'kick_video_20261110'),'physical_GT_calibration':None,
        'note':'Existing extracted frames/metadata only; no invented trajectory or synchronization.'})
    save(output/'safety_source_review.json',safety_manual_evidence(repo));save(output/'session_template.json',session_template())
    summary={name:{'records':v['records'],'origin':v['data_origin'],'topics':{t:details['recorder_intervals']['samples'] for t,details in v['topics'].items()},
                   'commands':v['commands'],'event_comparisons':v['recorded_event_comparisons'],
                   'unmeasured':[k for k in ('absolute_position_error','actual_ground_speed_accuracy','one_way_network_delay','successful_physical_recovery_time') if v[k].get('value') is None]}
             for name,v in results.items()}
    save(output/'summary.json',summary)
    after={p:sha(p) for p in before};assert before==after
    save(output/'original_hashes_after.json',after)
    save(output/'execution.json',{'python':sys.version,'no_installation':True,'decoder':'existing rosbags',
        'original_files_preserved':len(before),'actuated':False,'ROS_launched':False,'remote_connected':False,
        'retrained':False,'git_commit_push':False,'protected_holdout_opened':False})
    print('Historical offline study complete; originals unchanged',len(before),flush=True)
