"""Exercise the installed ball tracker in isolated ROS domain 77, without motors.

Static TF is published before the C++ node starts. Future-stamped detections
must not initialize the filter; matching timestamped detections must produce
a valid ball at base coordinates (1.1, 0.0). No robot-control topics are used.
"""
import os
os.environ['ROS_DOMAIN_ID'] = '77'
os.environ['ROS_LOCALHOST_ONLY'] = '1'

import math
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from ament_index_python.packages import get_package_prefix
import rclpy
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from hsl_interfaces.msg import BallEstimate
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose


def transform(parent, child, x=0.0, z=0.0, optical=False):
    result = TransformStamped()
    result.header.frame_id = parent
    result.child_frame_id = child
    result.transform.translation.x = x
    result.transform.translation.z = z
    result.transform.rotation.w = 1.0
    if optical:
        q = result.transform.rotation
        q.x, q.y, q.z, q.w = -0.5, 0.5, -0.5, 0.5
    return result


def main():
    rclpy.init(args=[])
    node = rclpy.create_node('isolated_ball_tracker_tf_test')
    dynamic = node.create_publisher(TFMessage, '/tf', 100)
    static_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    static = node.create_publisher(TFMessage, '/tf_static', static_qos)
    detections = node.create_publisher(Detection3DArray, '/perception/detections_3d', 10)
    state = {'last': None, 'count': 0, 'valid_seen': False}

    def receive(message):
        state['last'] = message
        state['count'] += 1
        state['valid_seen'] |= message.valid

    subscription = node.create_subscription(BallEstimate, '/hsl/ball', receive, 10)
    fixed = [transform('map', 'odom'),
             transform('base_link', 'camera_color_optical_frame', x=0.1, z=0.5, optical=True)]
    for item in fixed:
        item.header.stamp = node.get_clock().now().to_msg()
    static.publish(TFMessage(transforms=fixed))
    binary = Path(get_package_prefix('hsl_world_model')) / 'lib/hsl_world_model/ball_tracker'
    process = None
    log = tempfile.TemporaryFile(mode='w+b')
    success = False
    try:
        environment = dict(os.environ, ROS_DOMAIN_ID='77', ROS_LOCALHOST_ONLY='1')
        process = subprocess.Popen([str(binary)], env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)

        def run_phase(seconds, future):
            deadline = time.monotonic() + seconds
            next_tf, next_detection = 0.0, 0.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('C++ ball tracker exited: ' + str(process.returncode))
                clock = node.get_clock().now()
                monotonic = time.monotonic()
                if monotonic >= next_tf:
                    current = transform('odom', 'base_link')
                    current.header.stamp = clock.to_msg()
                    dynamic.publish(TFMessage(transforms=[current]))
                    next_tf = monotonic + 0.02
                if monotonic >= next_detection:
                    message = Detection3DArray()
                    message.header.frame_id = 'camera_color_optical_frame'
                    stamp = clock + Duration(seconds=60) if future else clock - Duration(nanoseconds=60000000)
                    message.header.stamp = stamp.to_msg()
                    detection = Detection3D()
                    detection.header = message.header
                    detection.bbox.center.position.z = 2.0 if future else 1.0
                    detection.bbox.center.orientation.w = 1.0
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = 'ball'
                    hypothesis.hypothesis.score = 0.95
                    detection.results = [hypothesis]
                    message.detections = [detection]
                    detections.publish(message)
                    next_detection = monotonic + 0.05
                rclpy.spin_once(node, timeout_sec=0.005)

        run_phase(8.0, future=True)
        if state['count'] == 0:
            raise RuntimeError('No BallEstimate output in isolated domain 77')
        if state['valid_seen']:
            raise RuntimeError('Future timestamp accepted without a matching transform')
        snapshot = os.pread(log.fileno(), os.fstat(log.fileno()).st_size, 0).decode('utf-8', errors='replace')
        if 'latest_odom=yes latest_base=yes' not in snapshot:
            raise RuntimeError('Future rejection is unproven: latest TF was not available in the tested buffer')
        print('PASS: future timestamp rejected while latest TF was available', flush=True)
        run_phase(6.0, future=False)
        ball = state['last']
        if not ball or not ball.valid or not ball.cam_valid:
            raise RuntimeError('Late-joined static TF did not produce a valid ball')
        if not all(math.isfinite(value) for value in (ball.rel_x, ball.rel_y, ball.cam_z)):
            raise RuntimeError('Non-finite ball position')
        if abs(ball.rel_x - 1.1) > 0.02 or abs(ball.rel_y) > 0.02:
            raise RuntimeError('Wrong base position: ' + str((ball.rel_x, ball.rel_y)))
        print('PASS: static TF published before node start was received', flush=True)
        print('PASS: base ball position =', ball.rel_x, ball.rel_y, flush=True)
        success = True
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        log.seek(0)
        output = log.read().decode('utf-8', errors='replace')
        if success and 'TF_CONFIG listener=parent-node-dedicated' not in output:
            success = False
            print('FAIL: installed executable does not contain the reviewed patch', flush=True)
        if 'Do not call canTransform or lookupTransform with a timeout' in output:
            success = False
            print('FAIL: Buffer dedicated-thread requirement was not satisfied', flush=True)
        print('--- C++ test log (last 30 lines) ---')
        print('\n'.join(output.splitlines()[-30:]))
        log.close()
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
    if not success:
        raise SystemExit(1)
    print('ISOLATED_TF_TEST_PASSED (domain 77; no motor commands)')


if __name__ == '__main__':
    main()
