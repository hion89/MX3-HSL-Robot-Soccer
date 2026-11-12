"""Start/stop the assigned role on robo04 through existing ROS parameter services."""
import argparse
import json
import signal
import time

import rclpy
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['attacker', 'defender', 'goalkeeper', 'stop', 'status'])
    parser.add_argument('--seconds', type=float, default=0.0,
                        help='Automatically stop after this many seconds; 0 runs until stop')
    args = parser.parse_args()
    command = args.command
    rclpy.init(args=[])
    # Keep this client's ROS context usable long enough to send stop on Ctrl+C.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    node = rclpy.create_node('robo04_role_control')
    clients = {}

    def set_parameter(target, name, value, retry_seconds=0.0):
        client = clients.get(target)
        if client is None:
            client = node.create_client(SetParameters, '/' + target + '/set_parameters')
            clients[target] = client
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError(target + ' is not running')
        deadline = time.monotonic() + retry_seconds
        while True:
            req = SetParameters.Request()
            req.parameters = [Parameter(name, value=value).to_parameter_msg()]
            future = client.call_async(req)
            rclpy.spin_until_future_complete(node, future, timeout_sec=4)
            if not future.done() or future.result() is None:
                raise RuntimeError('Parameter call timed out: ' + target + '/' + name)
            results = future.result().results
            if results and all(r.successful for r in results):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError('; '.join(r.reason for r in results))
            rclpy.spin_once(node, timeout_sec=0.15)

    def stop_motion():
        errors = []
        for target, name, value in (
            ('hsl_behavior', 'default_game_state', 'initial'),
            ('actor_onnx_walking_node', 'gait_mode', 'stop'),
            ('actor_onnx_walkkick_v1_node', 'motion_enabled', False),
        ):
            try:
                set_parameter(target, name, value)
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('; '.join(errors))

    started = None
    try:
        if command != 'status':
            # Stop strategy before changing role. Cancel any kick and hold gait
            # stopped before enabling a new role; no torque service is called.
            stop_motion()
            if command != 'stop':
                set_parameter('hsl_behavior', 'assigned_role', command, retry_seconds=3)
                set_parameter('actor_onnx_walkkick_v1_node', 'motion_enabled', True)
                set_parameter('actor_onnx_walking_node', 'gait_mode', 'auto')
                set_parameter('hsl_behavior', 'default_game_state', 'playing')
                started = time.monotonic()
            print('ROLE_CONTROL:', command, flush=True)
        received = []
        subscription = node.create_subscription(String, '/hsl/behavior_debug',
                                               lambda m: received.append(m.data), 10)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if received:
                data = json.loads(received[-1])
                expected = 'INITIAL' if command == 'stop' else 'PLAYING'
                if command == 'status' or (data.get('state') == expected and
                        (command == 'stop' or data.get('assigned_role') == command)):
                    print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
                    break
        else:
            print('No fresh behavior debug received; inspect the role launch terminal.', flush=True)
        node.destroy_subscription(subscription)
        if started is not None and args.seconds > 0:
            while time.monotonic() - started < args.seconds:
                rclpy.spin_once(node, timeout_sec=0.1)
            stop_motion()
            started = None
            print('AUTO_STOPPED: role trial complete; standing control remains on.', flush=True)
    except Exception as error:
        if started is not None:
            try:
                stop_motion()
            except Exception:
                pass
        print('ROLE_CONTROL_FAILED:', str(error), flush=True)
        raise SystemExit(1)
    except KeyboardInterrupt:
        if started is not None and rclpy.ok():
            stop_motion()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
