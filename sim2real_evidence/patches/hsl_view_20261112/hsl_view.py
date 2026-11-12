"""Observe existing HSL estimates in RViz. No control publishers or TF broadcasters."""
from collections import deque
import math
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import Point
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray
from hsl_interfaces.msg import BallEstimate, RobotPoseEstimate
from hsl_behavior.geometry import Field
from view_math import behavior_data, finite, robot_name, transform_xyz


def point(x, y, z=0.02):
    return Point(x=float(x), y=float(y), z=float(z))


def field_segments(f):
    h, w = f.hl, f.hw
    segments = [(-h, -w, h, -w), (-h, w, h, w),
                (-h, -w, -h, w), (h, -w, h, w), (0, -w, 0, w)]
    for side in (-1, 1):
        x = side * h
        for depth, width in ((f.penalty_area_depth, f.penalty_area_width),
                             (f.goal_area_depth, f.goal_area_width)):
            inner, half = side * (h - depth), width / 2
            segments.extend([(x, -half, inner, -half), (x, half, inner, half),
                             (inner, -half, inner, half)])
        mark = side * (h - f.penalty_mark_dist)
        segments.extend([(mark - .025, 0, mark + .025, 0),
                         (mark, -.025, mark, .025)])
    return segments


class HslView(Node):
    def __init__(self):
        super().__init__('hsl_estimate_view')
        profile = str(self.declare_parameter('field_profile', 'practice').value)
        if profile not in ('practice', 's_field'):
            raise ValueError('Unknown field profile: ' + profile)
        self.field = Field.for_profile(profile)
        self.robot_name = robot_name(self.declare_parameter(
            'robot_name', socket.gethostname().split('-')[0]).value)
        self.pose = self.ball = None
        self.pose_t = self.ball_t = 0.0
        self.behavior = {}
        self.behavior_t = 0.0
        self.detections = deque(maxlen=15)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        # Non-blocking lookups below allow this node's normal executor to receive TF.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self.trail = deque(maxlen=1200)
        self.pub = self.create_publisher(MarkerArray, '/hsl/view/markers', 2)
        self.create_subscription(RobotPoseEstimate, '/hsl/robot_pose', self.on_pose, 10)
        self.create_subscription(BallEstimate, '/hsl/ball', self.on_ball, 10)
        self.create_subscription(String, '/hsl/behavior_debug', self.on_behavior, 10)
        self.create_subscription(Detection3DArray, '/perception/detections_3d',
                                 self.on_detections, qos_profile_sensor_data)
        self.create_timer(.2, self.render)
        f = self.field
        self.get_logger().info(
            'VIEW ONLY: %s %.2fx%.2fm, goal width %.3fm, penalty width %.3fm'
            % (profile, f.length, f.width, f.goal_width, f.penalty_area_width))
        self.get_logger().info('Robot label: %s; all observations use capture-time TF' % self.robot_name)

    def on_pose(self, msg):
        if msg.header.frame_id != 'map' or not all(
                math.isfinite(v) for v in (msg.x, msg.y, msg.theta)):
            return
        self.pose, self.pose_t = msg, time.monotonic()
        if not self.trail or math.hypot(msg.x - self.trail[-1][0], msg.y - self.trail[-1][1]) > .015:
            self.trail.append((msg.x, msg.y))

    def on_ball(self, msg):
        if msg.header.frame_id == 'map':
            self.ball, self.ball_t = msg, time.monotonic()

    def on_behavior(self, msg):
        try:
            data = behavior_data(msg.data)
        except (ValueError, TypeError):
            return
        self.behavior, self.behavior_t = data, time.monotonic()

    def on_detections(self, msg):
        self.detections.append((time.monotonic(), msg))

    def observed_markers(self, now):
        """Latest transformable observation; never substitute identity/latest TF."""
        latest_missing = 0
        for received, array in reversed(self.detections):
            if now-received > .8:
                break
            markers, missing, transforms = [], 0, {}
            for detection in array.detections[:80]:
                if not detection.results:
                    continue
                result = max(detection.results, key=lambda r: r.hypothesis.score)
                label = str(result.hypothesis.class_id)[:32]
                if not finite((result.hypothesis.score,)) or result.hypothesis.score < .3:
                    continue
                pos = detection.bbox.center.position
                xyz = (pos.x, pos.y, pos.z)
                if not finite(xyz) or sum(v*v for v in xyz) < 1e-8:
                    continue
                frame = detection.header.frame_id or array.header.frame_id
                stamp = detection.header.stamp
                if not (stamp.sec or stamp.nanosec):
                    stamp = array.header.stamp
                if frame != 'map':
                    if not frame or not (stamp.sec or stamp.nanosec):
                        missing += 1
                        continue
                    key = (frame, stamp.sec, stamp.nanosec)
                    try:
                        if key not in transforms:
                            transforms[key] = self.tf_buffer.lookup_transform(
                                'map', frame, Time.from_msg(stamp), timeout=Duration(seconds=0.0))
                        tr = transforms[key].transform
                        xyz = transform_xyz(xyz, (tr.translation.x, tr.translation.y, tr.translation.z),
                                            (tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w))
                    except (TransformException, ValueError):
                        missing += 1
                        continue
                normalized = label.lower()
                if normalized == 'ball':
                    color = (1., .65, .15, .85)
                elif 'goal' in normalized:
                    color = (1., 1., .05, 1.)
                elif normalized.startswith('l'):
                    color = (.1, 1., 1., 1.)
                elif normalized.startswith('t'):
                    color = (.1, 1., .2, 1.)
                elif normalized.startswith('x'):
                    color = (1., .2, 1., 1.)
                else:
                    color = (.8, .8, .8, 1.)
                idx = len(markers)
                observed = self.marker('observed', idx, Marker.CUBE, color, (.075, .075, .035))
                observed.pose.position = point(xyz[0], xyz[1], .055)
                label_marker = self.text(100+idx, xyz[0], xyz[1]+.10,
                                         '%s %.2f' % (label, result.hypothesis.score), color)
                label_marker.scale.z = .065
                markers.extend((observed, label_marker))
            if not latest_missing:
                latest_missing = missing
            if markers or not array.detections:
                return markers, latest_missing, now-received
        return [], latest_missing, None

    def marker(self, ns, idx, kind, color, size=(.03, .03, .03)):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = self.robot_name + '/' + ns, idx, kind
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = size
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime.nanosec = 800000000
        return m

    def text(self, idx, x, y, text, color=(1., 1., 1., 1.)):
        m = self.marker('labels', idx, Marker.TEXT_VIEW_FACING, color, (.0, .0, .10))
        m.pose.position = point(x, y, .12)
        m.text = text
        return m

    def render(self):
        f, now = self.field, time.monotonic()
        markers = []
        behavior_fresh = now-self.behavior_t < 1.0
        info = self.behavior if behavior_fresh else {}
        role = str(info.get('role', 'unknown'))[:32]
        assigned = str(info.get('assigned_role', 'unknown'))[:32]
        state = str(info.get('state', 'unknown'))[:16]
        skill = str(info.get('skill', 'unknown'))[:64]
        markers.append(self.text(10, 0, f.hw+.72,
            '%s | role=%s | state=%s\nassigned=%s | active=%s | skill=%s%s' %
            (self.robot_name, role, state, assigned, info.get('active', '?'), skill,
             '' if behavior_fresh else ' | NO FRESH BEHAVIOR')))
        lines = self.marker('field', 0, Marker.LINE_LIST, (.9, .9, .9, 1.))
        for x0, y0, x1, y1 in field_segments(f):
            lines.points.extend([point(x0, y0), point(x1, y1)])
        markers.append(lines)
        circle = self.marker('field', 1, Marker.LINE_STRIP, (.9, .9, .9, 1.))
        circle.points = [point(f.center_circle_radius * math.cos(i * math.tau / 96),
                               f.center_circle_radius * math.sin(i * math.tau / 96)) for i in range(97)]
        markers.append(circle)
        posts = self.marker('field', 2, Marker.SPHERE_LIST, (1., .8, .1, 1.), (.09, .09, .09))
        posts.points = [point(x, y, .05) for x in (-f.hl, f.hl)
                        for y in (-f.goal_width / 2, f.goal_width / 2)]
        markers.extend([posts, self.text(0, -f.hl, -f.hw - .16, 'OUR GOAL (-X)'),
                        self.text(1, f.hl, -f.hw - .16, 'OPP GOAL (+X)')])
        if self.pose is None:
            markers.append(self.text(2, 0, f.hw + .2, 'WAITING: /hsl/robot_pose', (1., .6, .0, 1.)))
        else:
            p = self.pose
            age = now - self.pose_t
            color = (.1, .7, 1., 1.) if age < 1.0 else (1., .5, .0, 1.)
            arrow = self.marker('robot', 0, Marker.ARROW, color, (.35, .07, .07))
            arrow.pose.position = point(p.x, p.y, .06)
            arrow.pose.orientation.z = math.sin(p.theta / 2)
            arrow.pose.orientation.w = math.cos(p.theta / 2)
            markers.append(arrow)
            label = self.text(11, p.x, p.y+.22, '%s | %s' % (self.robot_name, role), color)
            label.pose.position.z = .25
            markers.append(label)
            markers.append(self.text(2, 0, f.hw + .35,
                'x=%+.2fm  y=%+.2fm  yaw=%+.1fdeg  quality=%d  validity=%.2f\n'
                'landmarks/cycle=%d  receive_age=%.1fs%s' %
                (p.x, p.y, math.degrees(p.theta), p.quality, p.validity,
                 p.num_landmarks_used, age, ' STALE' if age > 1.0 else ''), color))
        if len(self.trail) > 1:
            trail = self.marker('robot', 1, Marker.LINE_STRIP, (.0, .8, .8, .8), (.018, .0, .0))
            trail.points = [point(x, y, .035) for x, y in self.trail]
            markers.append(trail)
        if self.ball is not None and self.ball.valid and now - self.ball_t < 1.0:
            b = self.ball
            if all(math.isfinite(v) for v in (b.field_x, b.field_y)):
                ball = self.marker('ball', 0, Marker.SPHERE, (1., .35, .0, 1.), (.12, .12, .12))
                ball.pose.position = point(b.field_x, b.field_y, .07)
                markers.extend([ball, self.text(3, b.field_x, b.field_y - .17,
                    'ball seen %.2fs ago' % b.time_since_seen)])
                markers[-1].text = 'BALL (%+.2f, %+.2f)m | seen %.2fs ago' % (
                    b.field_x, b.field_y, b.time_since_seen)
        target = info.get('target')
        if target is not None:
            tx, ty, heading_deg = target
            target_arrow = self.marker('target', 0, Marker.ARROW, (1., .1, .7, 1.), (.30, .06, .06))
            target_arrow.pose.position = point(tx, ty, .08)
            angle = math.radians(heading_deg)
            target_arrow.pose.orientation.z = math.sin(angle/2)
            target_arrow.pose.orientation.w = math.cos(angle/2)
            markers.append(target_arrow)
            markers.append(self.text(12, tx, ty-.20,
                                      'TARGET (%+.2f, %+.2f)m %+.0fdeg' % (tx, ty, heading_deg),
                                      (1., .3, .8, 1.)))
            if self.pose is not None and now-self.pose_t < 1.0:
                path = self.marker('target', 1, Marker.LINE_STRIP, (1., .2, .7, .8), (.015, .0, .0))
                path.points = [point(self.pose.x, self.pose.y, .04), point(tx, ty, .04)]
                markers.append(path)
        elif behavior_fresh:
            reason = 'NO WALK TARGET: ' + skill
            if 'target' not in info:
                reason = 'TARGET TELEMETRY MISSING: rebuild/restart hsl_behavior'
            markers.append(self.text(12, 0, -f.hw-.30, reason, (1., .5, .1, 1.)))
        observed, missing, observed_age = self.observed_markers(now)
        markers.extend(observed)
        status = 'OBSERVED: %d | TF unavailable: %d' % (len(observed)//2, missing)
        if observed_age is not None:
            status += ' | age %.2fs' % observed_age
        else:
            status += ' | no fresh transformed detections'
        markers.append(self.text(13, 0, -f.hw-.50, status, (.5, 1., .8, 1.)))
        self.pub.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    node = HslView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
