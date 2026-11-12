"""Probe the installed twist_mux in isolated ROS domain 77; no hardware or policy process.

The walking runner's actual velocity callback is replayed as a three-value cache
to distinguish emitting zero from becoming silent. This is not a running policy.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace


def main():
    os.environ['ROS_DOMAIN_ID'] = '77'
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    import rclpy
    import yaml
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool

    prefix = Path(get_package_prefix('twist_mux'))
    binary = prefix / 'lib/twist_mux/twist_mux'
    config = Path(get_package_share_directory('humanoid_bringup')) / 'config/twist_mux_topics.yaml'
    settings = yaml.safe_load(config.read_text())['twist_mux']['ros__parameters']
    timeout = float(settings['topics']['behavior']['timeout'])
    topic = lambda value: '/' + value.lstrip('/')
    spec = importlib.util.find_spec('humanoid_policy_runner.actor_onnx_walking_runner')
    if spec is None or spec.origin is None:
        raise RuntimeError('Walking runner module missing')
    source = Path(spec.origin).read_bytes()
    if hashlib.sha256(source).hexdigest() != '477dcff62aa2572b12d7cad58dc71c4a1628871d0c8695a869450ffc5bdf52aa':
        raise RuntimeError('Walking runner differs from the reviewed source')
    tree = ast.parse(source.decode('utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActorOnnxWalkingNode')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'cb_cmd_vel')
    ns = {'Twist': Twist}
    exec(compile(ast.Module(body=[method], type_ignores=[]), spec.origin, 'exec'), ns)
    cache = SimpleNamespace(cmd=[0.0, 0.0, 0.0])

    root = Path('/home/robo04/sim2real_ws/runtime_logs')
    root.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='mux_stop_' + time.strftime('%Y%m%d_%H%M%S') + '_', dir=root))
    log_path = run_dir / 'mux.log'
    print('LOG_DIR=' + str(run_dir), flush=True)
    print('Isolated domain 77; no hardware/policy nodes started.', flush=True)
    rclpy.init(args=[])
    node = rclpy.create_node('mux_stop_probe')
    behavior = node.create_publisher(Twist, topic(settings['topics']['behavior']['topic']), 10)
    teleop = node.create_publisher(Twist, topic(settings['topics']['teleop']['topic']), 10)
    game_lock = node.create_publisher(Bool, topic(settings['locks']['game_stop']['topic']), 10)
    estop = node.create_publisher(Bool, topic(settings['locks']['estop']['topic']), 10)
    output = '/mux_stop_test/output'
    events, phases = [], []

    def receive(message):
        ns['cb_cmd_vel'](cache, message)
        events.append({'time': time.monotonic(), 'vx': message.linear.x,
                       'vy': message.linear.y, 'wz': message.angular.z})

    subscription = node.create_subscription(Twist, output, receive, 20)
    zero, forward = Twist(), Twist()
    forward.linear.x = 0.2
    process, failure = None, None
    command = [str(binary), '--ros-args', '--params-file', str(config),
               '-p', 'use_stamped:=false', '-r', 'cmd_vel_out:=' + output]
    with log_path.open('wb') as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

            def phase(name, seconds, send_behavior, lock=False, send_stop=False):
                start, next_send = time.monotonic(), 0.0
                initial_cache = list(cache.cmd)
                while time.monotonic() - start < seconds:
                    if process.poll() is not None:
                        raise RuntimeError('twist_mux exited: ' + str(process.returncode))
                    if time.monotonic() >= next_send:
                        game_lock.publish(Bool(data=lock))
                        estop.publish(Bool(data=False))
                        if send_behavior:
                            behavior.publish(forward)
                        if send_stop:
                            teleop.publish(zero)
                        next_send = time.monotonic() + 0.05
                    rclpy.spin_once(node, timeout_sec=0.005)
                received = [e for e in events if e['time'] >= start]
                cutoff = timeout + 0.2 if name == 'behavior_timeout' else 0.3
                late = [e for e in received if e['time'] - start >= cutoff]
                result = {'name': name, 'seconds': seconds, 'output_count': len(received),
                          'late_output_count': len(late),
                          'late_zero_count': sum(max(abs(e[k]) for k in ('vx', 'vy', 'wz')) < 1e-6 for e in late),
                          'late_forward_count': sum(e['vx'] > 0.19 for e in late),
                          'cache_before': initial_cache, 'cache_after': list(cache.cmd)}
                phases.append(result)
                print(json.dumps(result), flush=True)

            phase('discovery', 2.5, False)
            phase('behavior_forward', 1.2, True)
            phase('behavior_timeout', max(1.3, timeout + 0.7), False)
            phase('resume_forward', 1.0, True)
            phase('game_stop_lock', 1.2, True, lock=True)
            phase('explicit_zero_via_teleop', 1.2, True, lock=True, send_stop=True)
        except Exception as error:
            failure = str(error)
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            node.destroy_subscription(subscription)
            node.destroy_node()
            rclpy.shutdown()
    by_name = {p['name']: p for p in phases}
    status = 'FAILED'
    findings = {}
    if failure is None:
        active = by_name['behavior_forward']
        quiet = by_name['behavior_timeout']
        locked = by_name['game_stop_lock']
        stopped = by_name['explicit_zero_via_teleop']
        if active['late_forward_count'] == 0 or stopped['late_zero_count'] == 0:
            failure = 'Expected forward/explicit-zero routing was not observed'
        else:
            findings = {
                'timeout_emitted_zero': quiet['late_zero_count'] > 0,
                'timeout_cached_vx': quiet['cache_after'][0],
                'game_lock_emitted_zero': locked['late_zero_count'] > 0,
                'game_lock_cached_vx': locked['cache_after'][0],
                'explicit_zero_stopped_cache': max(map(abs, stopped['cache_after'])) < 1e-6,
                'cache_interpretation': 'Source cb_cmd_vel replay only; neural walking policy not started',
            }
            status = ('COMMAND_TIMEOUT_GUARD_REQUIRED' if abs(quiet['cache_after'][0]) > 0.01
                      or abs(locked['cache_after'][0]) > 0.01 else 'MUX_ZERO_STOP_OBSERVED')
    report = {'status': status, 'failure': failure, 'domain': 77, 'config': str(config),
              'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
              'behavior_timeout_seconds': timeout,
              'priorities': {'behavior': settings['topics']['behavior']['priority'],
                             'teleop': settings['topics']['teleop']['priority'],
                             'game_stop': settings['locks']['game_stop']['priority']},
              'findings': findings, 'phases': phases, 'log_directory': str(run_dir)}
    (run_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print('--- mux log (last 12 lines) ---', flush=True)
    print('\n'.join(log_path.read_text(errors='replace').splitlines()[-12:]), flush=True)
    return 1 if failure is not None else 0


if __name__ == '__main__':
    raise SystemExit(main())
