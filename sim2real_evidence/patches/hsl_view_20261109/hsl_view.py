"""Observe existing HSL estimates in RViz. No control publishers or TF broadcasters."""
from collections import deque
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from hsl_interfaces.msg import BallEstimate, RobotPoseEstimate
from hsl_behavior.geometry import Field


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
        self.pose = self.ball = None
        self.pose_t = self.ball_t = 0.0
        self.trail = deque(maxlen=1200)
        self.pub = self.create_publisher(MarkerArray, '/hsl/view/markers', 2)
        self.create_subscription(RobotPoseEstimate, '/hsl/robot_pose', self.on_pose, 10)
        self.create_subscription(BallEstimate, '/hsl/ball', self.on_ball, 10)
        self.create_timer(.2, self.render)
        f = self.field
        self.get_logger().info(
            'VIEW ONLY: %s %.2fx%.2fm, goal width %.3fm, penalty width %.3fm'
            % (profile, f.length, f.width, f.goal_width, f.penalty_area_width))

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

    def marker(self, ns, idx, kind, color, size=(.03, .03, .03)):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type = ns, idx, kind
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = size
        m.color.r, m.color.g, m.color.b, m.color.a = color
        m.lifetime.sec = 1
        return m

    def text(self, idx, x, y, text, color=(1., 1., 1., 1.)):
        m = self.marker('labels', idx, Marker.TEXT_VIEW_FACING, color, (.0, .0, .10))
        m.pose.position = point(x, y, .12)
        m.text = text
        return m

    def render(self):
        f, now = self.field, time.monotonic()
        markers = []
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
