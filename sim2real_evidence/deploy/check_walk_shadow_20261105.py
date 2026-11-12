"""Run the installed walking policy for 20 seconds with zero command and isolated outputs.

Real joint/IMU feedback is read. No hardware command topic or service is used.
This verifies runtime wiring and finite outputs, not physical balance or tracking.
"""
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tempfile
import time

MODEL_HASH = '72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1'
RUNNER_HASH = '477dcff62aa2572b12d7cad58dc71c4a1628871d0c8695a869450ffc5bdf52aa'
PREFIX = '/hsl/shadow/walk'
OUTPUT_REMAPS = {
    '/joint_goals': PREFIX + '/joint_goals',
    '/kick_done': PREFIX + '/kick_done',
    '/gait/phase': PREFIX + '/gait/phase',
    '/gait/effective_frequency': PREFIX + '/gait/effective_frequency',
    '/gait/state': PREFIX + '/gait/state',
}
INPUT_REMAPS = {
    '/cmd_vel': PREFIX + '/cmd_vel',
    '/kick_trigger': PREFIX + '/kick_trigger',
    '/gait_mode': PREFIX + '/gait_mode',
    '/gait_frequency': PREFIX + '/gait_frequency',
}


def percentile(values, fraction):
    values = sorted(values)
    return values[int((len(values) - 1) * fraction)] if values else None


def review_runner(path):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != RUNNER_HASH:
        raise RuntimeError('Installed walking runner differs from the reviewed source')
    tree = ast.parse(payload.decode('utf-8'))
    topics = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr == 'create_client':
            raise RuntimeError('Unexpected service client in walking runner')
        if call.func.attr == 'create_publisher':
            if len(call.args) < 2 or not isinstance(call.args[1], ast.Constant):
                raise RuntimeError('Nonliteral publisher requires a new review')
            topics.append(call.args[1].value)
    if set(topics) != set(OUTPUT_REMAPS):
        raise RuntimeError('Output remaps do not cover the installed publishers')
    return topics


def ensure_no_runner():
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            words = (process / 'cmdline').read_bytes().split(b'\0')[:2]
        except OSError:
            continue
        if any(Path(os.fsdecode(word)).name == 'actor_onnx_walking_runner' for word in words if word):
            raise RuntimeError('A walking runner is already running (PID ' + process.name + ')')


def summarize_csv(paths):
    inferences, rows = [], 0
    for path in paths:
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('run_state') != 'RUN' or not row.get('inference_ms'):
                    continue
                for name, value in row.items():
                    numeric = name == 'inference_ms' or name.startswith(
                        ('imu_', 'grav_', 'cmd_', 'q_', 'qd_', 'q_rel_', 'qd_scaled_',
                         'a_prev_', 'a_raw_', 'a_clip_', 'q_target_', 'q_default_'))
                    if numeric and (not value or not math.isfinite(float(value))):
                        raise RuntimeError('Non-finite or missing CSV value: ' + name)
                for name in ('cmd_vx', 'cmd_vy', 'cmd_wz'):
                    if abs(float(row[name])) > 1e-8:
                        raise RuntimeError('Nonzero command in the zero-command trial')
                inferences.append(float(row['inference_ms']))
                rows += 1
    if not rows:
        raise RuntimeError('No actual inference rows in CSV (hold output alone is insufficient)')
    return {'inference_rows': rows, 'inference_ms': {
        'median': statistics.median(inferences),
        'p99': percentile(inferences, 0.99), 'max': max(inferences)}}


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    for action in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        process.send_signal(action)
        try:
            process.wait(timeout=3)
            return
        except subprocess.TimeoutExpired:
            continue


def main():
    os.environ['ROS_DOMAIN_ID'] = '42'
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    import rclpy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Imu, JointState

    spec = importlib.util.find_spec('humanoid_policy_runner.actor_onnx_walking_runner')
    if spec is None or spec.origin is None:
        raise RuntimeError('Walking runner module is missing')
    reviewed = review_runner(Path(spec.origin))
    ensure_no_runner()
    model = Path(get_package_share_directory('hsl_bringup')) / 'models/walk_v0_6_13500.onnx'
    if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_HASH:
        raise RuntimeError('Model differs from the reviewed v0_6 13500 file')
    binary = Path(get_package_prefix('humanoid_policy_runner')) / 'lib/humanoid_policy_runner/actor_onnx_walking_runner'
    root = Path('/home/robo04/sim2real_ws/runtime_logs')
    root.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='walk_shadow_' + time.strftime('%Y%m%d_%H%M%S') + '_', dir=root))
    log_path = run_dir / 'runner.log'
    print('LOG_DIR=' + str(run_dir), flush=True)
    print('20-second zero-command trial; all five outputs are remapped.', flush=True)
    parameters = {
        'actor_path': str(model), 'policy_variant': 'v0_2', 'onnx_provider': 'cpu',
        'policy_rate_hz': 50.0, 'action_scale': 0.25, 'gait_frequency': 2.0,
        'gait_frequency_scale': 2.0, 'gait_stop_rate': 2.0, 'standing_threshold': 0.1,
        'gait_mode': 'auto', 'stop_delay_s': 1.0, 'startup_hold_seconds': 1.0,
        'resume_hold_seconds': 1.0, 'gravity_source': 'derive',
        'calibrate_on_startup': False, 'kick_yield_enable': False,
        'enable_csv_log': True, 'csv_output_dir': str(run_dir), 'debug_print_every': 100,
    }
    command = [str(binary), '--ros-args', '-r', '__node:=hsl_walk_shadow']
    for name, value in parameters.items():
        literal = str(value).lower() if isinstance(value, bool) else str(value)
        command.extend(['-p', name + ':=' + literal])
    for old, new in {**INPUT_REMAPS, **OUTPUT_REMAPS}.items():
        command.extend(['-r', old + ':=' + new])
    (run_dir / 'launch.json').write_text(json.dumps({
        'argv': command, 'model_sha256': MODEL_HASH, 'runner_sha256': RUNNER_HASH,
        'output_remaps': OUTPUT_REMAPS, 'duration_seconds': 20,
        'interpretation': 'Runtime/finite-output check only; no physical balance test'}, indent=2))

    rclpy.init(args=[])
    node = rclpy.create_node('walk_shadow_observer')
    publisher = node.create_publisher(Twist, PREFIX + '/cmd_vel', 10)
    zero = Twist()
    state = {'joint_count': 0, 'imu_count': 0, 'joints': {}, 'times': [],
             'max_delta_rad': 0.0, 'max_delta_joint': None, 'errors': []}

    def joints(message):
        state['joint_count'] += 1
        state['joints'] = dict(zip(message.name, message.position))

    def imu(message):
        state['imu_count'] += 1

    def goals(message):
        state['times'].append(time.monotonic())
        if len(message.name) != 12 or len(set(message.name)) != 12 or len(message.position) != 12:
            state['errors'].append('Expected 12 distinct joint names/positions')
            return
        for name, target in zip(message.name, message.position):
            current = state['joints'].get(name)
            if not math.isfinite(target) or current is None or not math.isfinite(current):
                state['errors'].append('Non-finite or unmatched joint: ' + name)
                continue
            delta = abs(target - current)
            if delta > state['max_delta_rad']:
                state['max_delta_rad'], state['max_delta_joint'] = delta, name

    subscriptions = [
        node.create_subscription(JointState, '/joint_states', joints, 20),
        node.create_subscription(Imu, '/imu/data', imu, 20),
        node.create_subscription(JointState, PREFIX + '/joint_goals', goals, 20),
    ]
    process, failure, actual_outputs = None, None, []
    with log_path.open('wb') as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline, next_command = time.monotonic() + 20, 0.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Walking runner exited: ' + str(process.returncode))
                if time.monotonic() >= next_command:
                    publisher.publish(zero)
                    next_command = time.monotonic() + 0.05
                rclpy.spin_once(node, timeout_sec=0.005)
                if state['errors']:
                    raise RuntimeError(state['errors'][0])
            actual_outputs = node.get_publisher_names_and_types_by_node('hsl_walk_shadow', '/')
            output_names = {name for name, _ in actual_outputs}
            if not set(OUTPUT_REMAPS.values()).issubset(output_names):
                raise RuntimeError('Runtime publisher graph did not confirm all five output remaps')
            if set(OUTPUT_REMAPS).intersection(output_names):
                raise RuntimeError('Unexpected original output topic on the test runner')
            if state['joint_count'] == 0 or state['imu_count'] == 0 or len(state['times']) < 2:
                raise RuntimeError('Required real sensor feedback or policy outputs were not received')
        except Exception as error:
            failure = str(error)
        finally:
            stop_child(process)
            for subscription in subscriptions:
                node.destroy_subscription(subscription)
            node.destroy_node()
            rclpy.shutdown()
    times = state['times']
    intervals = [1000 * (b - a) for a, b in zip(times, times[1:])]
    summary = {
        'status': 'RUNTIME_CHECK_PASSED' if failure is None else 'FAILED',
        'failure': failure, 'real_joint_messages': state['joint_count'],
        'real_imu_messages': state['imu_count'], 'goal_messages': len(times),
        'goal_receive_hz': (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else None,
        'goal_interval_ms': {'p99': percentile(intervals, 0.99), 'max': max(intervals) if intervals else None},
        'max_target_minus_current_rad': state['max_delta_rad'],
        'max_delta_joint': state['max_delta_joint'],
        'runtime_outputs': actual_outputs, 'csv_files': [str(p) for p in run_dir.glob('*.csv')],
        'log_directory': str(run_dir),
    }
    try:
        summary.update(summarize_csv(sorted(run_dir.glob('*.csv'))))
    except Exception as error:
        summary['status'] = 'FAILED'
        summary['csv_failure'] = str(error)
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print('--- runner log (last 20 lines) ---', flush=True)
    print('\n'.join(log_path.read_text(errors='replace').splitlines()[-20:]), flush=True)
    return 0 if summary['status'] == 'RUNTIME_CHECK_PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
