"""Analyze the saved 2026-11-05 trial and replay the same stationary feedback.

No ROS process, topic, service, or hardware interface is instantiated.
Counterfactual replay changes gait observation and recomputes action history only;
joint and IMU feedback remain recorded, so it cannot validate physical balance.
"""
import ast
from collections import Counter, deque
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

import numpy as np
import onnxruntime as ort

BASE = Path(__file__).resolve().parent
LOG = BASE / 'observations_20261105/walk_shadow_20261105_162155_qeh2uxxq'
SOURCE = Path('C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim/src')
OUT = BASE / 'analysis_walk_shadow_20261105'


def main():
    OUT.mkdir(exist_ok=True)
    source = SOURCE / 'mx3_src/humanoid_policy_runner/humanoid_policy_runner/actor_onnx_walking_runner.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in {'_make_frame_spec', 'FrameHistory', 'build_obs_from_frames'}]
    ns = {'np': np, 'deque': deque, 'HISTORY_LEN': 5}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    slices, width = ns['_make_frame_spec'](3)
    assert width == 48
    path = LOG / 'v0_2_log_20261105_162156.csv'
    all_rows = list(csv.DictReader(path.open(newline='')))
    rows = [r for r in all_rows if r['run_state'] == 'RUN' and r['inference_ms']]
    joints = [k[2:] for k in rows[0] if k.startswith('q_') and
              not k.startswith(('q_rel_', 'q_target_', 'q_default_'))]
    vector = lambda r, p: np.array([float(r[p + j]) for j in joints], dtype=np.float32)
    times = np.array([float(r['time_sec']) for r in rows])
    observed = np.stack([vector(r, 'q_') for r in rows])
    targets = np.stack([vector(r, 'q_target_') for r in rows])
    defaults = np.stack([vector(r, 'q_default_') for r in rows])
    recorded_actions = np.stack([vector(r, 'a_raw_') for r in rows])
    frequencies = np.array([float(r['gait_freq']) for r in rows])
    f32 = lambda values: np.array(values, dtype=np.float32)

    def frame(r, actions=None, stop=False):
        gyro = f32([float(r[k]) for k in ('imu_wx', 'imu_wy', 'imu_wz')]) * np.float32(0.2)
        gravity = f32([float(r[k]) for k in ('grav_x', 'grav_y', 'grav_z')])
        command = f32([float(r[k]) for k in ('cmd_vx', 'cmd_vy', 'cmd_wz')])
        gait = f32([1, 0, 0]) if stop else f32([
            float(r['gait_cos']), float(r['gait_sin']), float(r['gait_freq']) / 2.0])
        previous = vector(r, 'a_clip_') if actions is None else actions
        return np.concatenate([gyro, gravity, command, gait, vector(r, 'q_rel_'),
                               vector(r, 'qd_scaled_'), previous]).astype(np.float32)

    model = SOURCE / 'hsl_src/hsl_bringup/models/walk_v0_6_13500.onnx'
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    assert model_hash == '72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1'
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    infer = lambda history: session.run(['actions'], {'obs': ns['build_obs_from_frames'](
        history, slices, 240).reshape(1, -1)})[0].reshape(12)

    # Validate layout, ordering and the one-cycle lag against measured actions.
    original_frames = [frame(r) for r in rows]
    reconstructed = np.stack([infer(np.stack(original_frames[i-5:i])) for i in range(5, len(rows))])
    errors = np.abs(reconstructed - recorded_actions[5:])
    reproduction = {'compared_rows': len(reconstructed),
                    'max_action_error': float(errors.max()),
                    'p99_action_error': float(np.quantile(errors, 0.99))}
    print('REPRODUCTION', json.dumps(reproduction), flush=True)
    if errors.max() > 0.002:
        raise RuntimeError('Recorded action reconstruction failed; do not interpret counterfactual')

    # Explicit stop from the start, with newly computed action history.
    history = ns['FrameHistory'](5, width)
    history.reset(frame(rows[0], np.zeros(12, dtype=np.float32), stop=True))
    stop_targets = []
    for r in rows:
        action = np.clip(infer(history.get_frames()), -10, 10).astype(np.float32)
        target = vector(r, 'q_default_') + np.float32(0.25) * action
        target = np.clip(target, vector(r, 'q_default_') - 10.80, vector(r, 'q_default_') + 10.80)
        target = np.clip(target, vector(r, 'q_') - 10.35, vector(r, 'q_') + 10.35)
        stop_targets.append(target)
        history.push(frame(r, action, stop=True))
    stop_targets = np.stack(stop_targets)
    assert np.isfinite(stop_targets).all()

    urdf = ET.parse(SOURCE / 'hsl_src/hsl_world_model/urdf/XM_robot04.urdf').getroot()
    limits = {j.attrib['name']: (float(j.find('limit').attrib['lower']), float(j.find('limit').attrib['upper']))
              for j in urdf.findall('joint') if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
    last = times >= times[-1] - 5
    per_joint = []
    for i, joint in enumerate(joints):
        difference = targets[:, i] - observed[:, i]
        peak = int(np.argmax(np.abs(difference)))
        lower, upper = limits[joint]
        per_joint.append({
            'joint': joint, 'default_rad': float(np.median(defaults[:, i])),
            'current_last5_median_rad': float(np.median(observed[last, i])),
            'target_last5_median_rad': float(np.median(targets[last, i])),
            'delta_last5_median_deg': float(np.rad2deg(np.median(difference[last]))),
            'peak_delta_deg': float(np.rad2deg(difference[peak])),
            'peak_time_seconds': float(times[peak]), 'peak_gait_state': rows[peak]['gait_state'],
            'stopped_peak_abs_delta_deg': max(abs(float(np.rad2deg(difference[k]))) for k in range(len(rows))
                                              if rows[k]['gait_state'] == 'STOPPED'),
            'stop_replay_peak_abs_delta_deg': float(np.rad2deg(np.abs(stop_targets[:, i] - observed[:, i]).max())),
            'stop_replay_last5_median_delta_deg': float(np.rad2deg(np.median(stop_targets[last, i] - observed[last, i]))),
            'urdf_limits_rad': [lower, upper],
            'recorded_targets_outside_urdf': int(((targets[:, i] < lower) | (targets[:, i] > upper)).sum()),
        })
    transitions = []
    previous = None
    for r in all_rows:
        state = (r['run_state'], r['gait_state'])
        if state != previous:
            transitions.append({'time_seconds': float(r['time_sec']), 'run_state': state[0],
                                'gait_state': state[1], 'gait_frequency': float(r['gait_freq'])})
            previous = state
    intervals = np.diff(times) * 1000
    report = {
        'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'model_sha256': model_hash,
        'runtime': {'onnxruntime': ort.__version__, 'numpy': np.__version__, 'platform': 'Windows CPU offline'},
        'all_rows': len(all_rows), 'inference_rows': len(rows),
        'run_states': dict(Counter(r['run_state'] for r in all_rows)),
        'gait_states_inference': dict(Counter(r['gait_state'] for r in rows)),
        'transitions': transitions, 'action_reproduction': reproduction,
        'csv_inference_hz': (len(times)-1)/(times[-1]-times[0]),
        'csv_interval_ms': {'median': float(np.median(intervals)), 'p99': float(np.quantile(intervals, 0.99)),
                            'max': float(intervals.max()), 'count_over_30ms': int((intervals > 30).sum())},
        'interpretation': 'Recorded stationary feedback replay; no physical feedback to replayed targets; no balance claim',
        'stop_replay_initialization': 'Five copies of first RUN measurement with zero previous action and zero gait; exact hold sensor history unavailable',
        'joints': per_joint,
    }
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (OUT / 'joint_summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_joint[0]))
        writer.writeheader(); writer.writerows(per_joint)
    with (OUT / 'replayed_stop_targets.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_sec'] + ['q_target_' + j for j in joints])
        writer.writerows([[float(t)] + list(map(float, q)) for t, q in zip(times, stop_targets)])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    axes[0].step(times, frequencies, where='post', label='기록: auto 모드', color='#b35b16')
    axes[0].plot(times, np.zeros(len(times)), label='재생: 처음부터 stop 모드', color='#27659c', linestyle='--')
    axes[0].set_ylabel('위상 주파수 [Hz]'); axes[0].legend(loc='upper right')
    for side, color, label in [('l', '#27659c', '왼쪽'), ('r', '#b35b16', '오른쪽')]:
        for index, suffix in [(1, 'knee_pitch_joint'), (2, 'pelvis_pitch_joint')]:
            j = joints.index(side + '_' + suffix)
            axes[index].plot(times, np.rad2deg(targets[:, j] - observed[:, j]), color=color, label=label + ' 기록')
            axes[index].plot(times, np.rad2deg(stop_targets[:, j] - observed[:, j]), color=color,
                             linestyle='--', label=label + ' stop 재생')
    axes[1].set_ylabel('무릎 목표 - 현재 [°]'); axes[2].set_ylabel('골반 pitch 목표 - 현재 [°]')
    for ax in axes:
        ax.axvspan(1.0121, 1.9896, color='#ffdb9b', alpha=0.3)
        ax.axhline(0, color='#666666', linewidth=0.6); ax.grid(alpha=0.2)
    axes[1].legend(ncol=2); axes[2].legend(ncol=2)
    axes[2].set_xlabel('러너 시작 후 시간 [초]')
    fig.suptitle('속도 명령 0: 저장된 센서 입력과 모델 출력 대조\n실제 모터 구동 없이 수행한 분석 — 균형 유지 검증 아님', fontsize=13)
    fig.savefig(OUT / 'zero_command_analysis.png', dpi=160)
    plt.close(fig)
    print('joint / recorded peak / recorded steady / stop-replay peak / stop-replay steady (degrees)')
    for row in per_joint:
        print(row['joint'], *(round(row[k], 3) for k in ['peak_delta_deg', 'delta_last5_median_deg',
              'stop_replay_peak_abs_delta_deg', 'stop_replay_last5_median_delta_deg']))
    print('REPORT=' + str(OUT / 'analysis.json'))


if __name__ == '__main__':
    main()
