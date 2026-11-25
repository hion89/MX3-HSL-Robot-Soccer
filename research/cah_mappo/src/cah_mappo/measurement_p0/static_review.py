"""Scoped source evidence extraction. Reads text/AST; never imports ROS code."""
import ast
from pathlib import Path
from ..measurement.storage import sha


TARGETS = {
    'hsl_src/hsl_bringup/launch/real_base.launch.py': ('EXCLUDE_P0', 'Hardware include starts motor/button/neck control; not a sensor-only launch.'),
    'hsl_src/hsl_bringup/launch/real_role_trial.launch.py': ('EXCLUDE_P0', 'Starts behavior and walking/kick control.'),
    'mx3_src/humanoid_serial_bridge/launch/dynamixel_system.launch.py': ('EXCLUDE_P0', 'Starts live MCU, motor controller, buttons and neck controllers.'),
    'mx3_src/humanoid_serial_bridge/humanoid_serial_bridge/body_motor_controller_node.py': ('EXCLUDE_P0', 'Opens U2D2; applies gains at startup; command/services available. Software torque false is not hardware readback.'),
    'mx3_src/humanoid_serial_bridge/humanoid_serial_bridge/hardware/u2d2_hardware.py': ('EXCLUDE_P0', 'Live serial bus and read/write SDK. open itself constructs writes and reads PWM limits; caller applies gains.'),
    'mx3_src/humanoid_serial_bridge/humanoid_serial_bridge/mcu_bridge_node.py': ('CONDITIONAL_DEVICE_WRITES', 'Serial open; IMU enable, ping and watchdog transmit; firmware/power partition not verified.'),
    'mx3_src/humanoid_serial_bridge/humanoid_serial_bridge/hardware/mcu_serial.py': ('CONDITIONAL_DEVICE_WRITES', 'Serial transport can send arbitrary packet; not a passive recording path.'),
    'mx3_src/humanoid_serial_bridge/humanoid_serial_bridge/imu_parser_node.py': ('OFFLINE_SEPARABLE_CANDIDATE', 'Parser consumes raw_rx and publishes estimates. Live IMU still depends on MCU and firmware.'),
    'mx3_src/humanoid_serial_bridge/humanoid_serial_bridge/button_input_node.py': ('EXCLUDE_P0', 'START can call robot_init and torque services; STOP path can alter torque.'),
    'mx3_src/humanoid_serial_bridge/humanoid_serial_bridge/neck_control_node.py': ('EXCLUDE_P0', 'Publishes joint goals/profile velocity and calls torque services.'),
    'mx3_src/humanoid_serial_bridge/humanoid_serial_bridge/neck_scan_node.py': ('EXCLUDE_P0', 'Enable path publishes neck torque true and angle commands.'),
    'mx3_src/humanoid_bringup/launch/control.launch.py': ('EXCLUDE_P0', 'Walking/kick/fall/mux and control chain; stopped gait is not isolation.'),
    'mx3_src/humanoid_perception/launch/perception_bringup.launch.py': ('SPLIT_REQUIRED', 'External RealSense process and URDF/optional GUI/detectors. Driver internals and USB reset behavior not audited.'),
    'mx3_src/humanoid_perception/humanoid_perception/yolo_detector_3d_node.py': ('OFFLINE_SEPARABLE_CANDIDATE', 'Inference/outputs depend on camera/depth/info. External provider dependencies remain unaudited.'),
    'hsl_src/hsl_world_model/src/leg_odometry_node.cpp': ('ESTIMATE_ONLY_CANDIDATE', 'Needs real joint feedback and IMU; publishes odom/TF, not independent GT.'),
    'hsl_src/hsl_world_model/src/self_localizer_node.cpp': ('ESTIMATE_ONLY_CANDIDATE', 'Pose depends on detections, TF, IMU and field config; timer output does not prove fresh observations.'),
    'hsl_src/hsl_world_model/src/ball_tracker_node.cpp': ('ESTIMATE_ONLY_CANDIDATE', 'Prediction timer and detection timestamp differ; requires camera-to-base/odom TF.'),
}
NEEDLES = ('IncludeLaunchDescription', 'PythonLaunchDescriptionSource', 'executable=', 'ExecuteProcess',
           'self.hardware.open(', '_apply_gains(', 'self._torque_enabled = False',
           'set_torque', 'create_client(', 'call_async(', 'call_robot_init_on_start', 'auto_torque_off',
           'create_publisher', 'create_subscription', 'SensorDataQoS', 'BEST_EFFORT',
           'TRANSIENT_LOCAL', 'header.stamp', 'stamp_at_read_start', 'self.mcu.open(',
           '_enable_imu_stream(', 'send_ping', 'self.mcu.send(', 'read_pwm_limits(',
           'GroupSyncWrite(', 'pub_torque.publish', 'last_seen', 'actor_path')


def review(source_root):
    root = Path(source_root)
    out = []
    for relative, (classification, finding) in TARGETS.items():
        path = root / relative
        if not path.is_file():
            out.append({'path': relative, 'classification': 'UNRESOLVED_PATH', 'finding': finding})
            continue
        text = path.read_text(encoding='utf-8-sig')
        calls = []
        if path.suffix == '.py':
            # AST omits comments/docstring examples, but does not resolve dynamic behavior.
            tree = ast.parse(text, filename=relative)
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    func = ast.unparse(n.func)
                    if any(k in func for k in ('create_publisher', 'create_subscription', 'create_client', 'call_async', 'Node', 'ExecuteProcess', 'IncludeLaunchDescription')):
                        calls.append({'line': n.lineno, 'call': ast.unparse(n)[:600]})
        evidence = [{'line': i, 'text': line.strip()} for i, line in enumerate(text.splitlines(), 1)
                    if any(k in line for k in NEEDLES) and not line.lstrip().startswith('#')]
        out.append({'path': relative, 'sha256': sha(path), 'classification': classification,
                    'finding': finding, 'line_evidence': evidence, 'python_AST_calls': sorted(calls, key=lambda x: x['line'])})
    return {'scope': 'Selected checked-in sources only; not installed robot runtime or external driver/firmware certification.',
            'hardware_accessed': False, 'ROS_imported': False, 'files': out,
            'dependency_edges': [
                ['real_base', 'dynamixel_system', 'body/button/neck/MCU; exclude'],
                ['real_role_trial', 'control', 'walking/kick/mux; exclude'],
                ['imu_parser', 'mcu/raw_rx', 'live source needs audited MCU firmware and motor power isolation'],
                ['leg_odometry', 'joint_states + IMU', 'joint feedback comes from non-passive motor driver'],
                ['ball_tracker/self_localizer', 'camera/detections/TF', 'dynamic neck TF needs real joint feedback'],
                ['passive recorder', 'already approved publishers', 'does not start or isolate their producers']],
            'unknown': ['Installed package hashes/processes, motor/battery identities, physical power isolation.',
                        'External RealSense driver, Dynamixel SDK, MCU firmware and kernel USB reset effects.',
                        'Live DDS QoS negotiation, load, disk errors and sustained fsync throughput.']}
