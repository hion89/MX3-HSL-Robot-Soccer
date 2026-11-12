"""Offline regression checks; ROS message stubs are not runtime verification."""
import ast
from collections import deque
import importlib.util
import math
from pathlib import Path
import random
import json
import hashlib
import tarfile
import tempfile
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'patches/hsl_view_20261112'
CANONICAL = ROOT.parent / 'MX3_Match_Sim'
if not (CANONICAL / 'src').is_dir():
    CANONICAL = Path('C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim')
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(CANONICAL / 'src/hsl_src/hsl_behavior'))
from view_math import behavior_data, transform_xyz
from hsl_behavior import skills


def load_module(name, text):
    module = ModuleType(name)
    module.__package__ = 'hsl_behavior'
    exec(compile(text, name, 'exec'), module.__dict__)
    return module


def old_skills_text():
    text = (CANONICAL / 'src/hsl_src/hsl_behavior/hsl_behavior/skills.py').read_text(encoding='utf-8-sig')
    return text.replace('        self.target_local = None  # Display telemetry; never used to compute motion.\n', '')\
        .replace('        self.target_local = None\n', '')\
        .replace('        self.target_local = (tx, ty, tth)\n', '')


def stubs():
    def module(name, **values):
        mod = ModuleType(name)
        mod.__dict__.update(values)
        sys.modules[name] = mod
        return mod
    module('rclpy')
    module('rclpy.node', Node=object)
    module('rclpy.duration', Duration=lambda **kw: NS(**kw))
    module('rclpy.qos', qos_profile_sensor_data=object())
    module('rclpy.time', Time=NS(from_msg=lambda x: x))
    module('geometry_msgs')
    module('geometry_msgs.msg', Point=lambda **kw: NS(**kw))
    class Marker:
        ADD, ARROW, CUBE, SPHERE, LINE_LIST, LINE_STRIP, SPHERE_LIST, TEXT_VIEW_FACING = range(8)
        def __init__(self):
            self.header = NS(frame_id='', stamp=None)
            self.pose = NS(position=NS(x=0., y=0., z=0.), orientation=NS(x=0., y=0., z=0., w=0.))
            self.scale = NS(x=0., y=0., z=0.)
            self.color = NS(r=0., g=0., b=0., a=0.)
            self.lifetime = NS(sec=0, nanosec=0)
            self.points, self.text = [], ''
    module('visualization_msgs')
    module('visualization_msgs.msg', Marker=Marker, MarkerArray=lambda **kw: NS(**kw))
    module('std_msgs')
    module('std_msgs.msg', String=object)
    module('vision_msgs')
    module('vision_msgs.msg', Detection3DArray=object)
    module('hsl_interfaces')
    module('hsl_interfaces.msg', BallEstimate=object, RobotPoseEstimate=object)
    class TransformException(Exception):
        pass
    module('tf2_ros', Buffer=object, TransformListener=object, TransformException=TransformException)
    spec = importlib.util.spec_from_file_location('viewer_under_test', BUNDLE / 'hsl_view.py')
    view = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(view)
    return view, TransformException


class ReviewTests(unittest.TestCase):
    def test_coordinate_rotation_and_normalization(self):
        q = (0, 0, math.sqrt(.5), math.sqrt(.5))
        xyz = transform_xyz((1, 0, 0), (2, 3, 4), q)
        for actual, expected in zip(xyz, (2, 4, 4)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(transform_xyz((1, 2, 3), (0, 0, 0), (0, 0, 0, 2)), (1, 2, 3))
        with self.assertRaises(ValueError):
            transform_xyz((1, 0, 0), (0, 0, 0), (0, 0, 0, 0))

    def test_behavior_parser_rejects_invalid_targets(self):
        self.assertEqual(behavior_data('{"target":[1,2,90]}')['target'], [1., 2., 90.])
        self.assertIsNone(behavior_data('{"target":null}')['target'])
        for text in ('[]', '{"target":[1,2]}', '{"target":[NaN,0,0]}'):
            with self.assertRaises(ValueError):
                behavior_data(text)

    def test_motion_outputs_unchanged(self):
        old = load_module('hsl_behavior.skills_before_view', old_skills_text())
        limits = skills.WalkLimits(vx_max=.2, vy_max=.15, wz_max=.5)
        a, b = old.WalkToPose(limits), skills.WalkToPose(limits)
        rng = random.Random(20261112)
        for _ in range(1500):
            if rng.random() < .08:
                a.reset(); b.reset()
            args = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi))
            kw = dict(final_heading=rng.choice((True, False)), avoid=(rng.uniform(-.1, .1), rng.uniform(-.1, .1)))
            self.assertEqual(a(*args, **kw), b(*args, **kw))
            self.assertEqual(a.arrived, b.arrived)
        a, b = old.GoToBallAndKick(limits, old.KickEnvelope()), skills.GoToBallAndKick(limits, skills.KickEnvelope())
        for step in range(1500):
            if step % 50 == 0:
                a.reset(); b.reset()
            args = (rng.uniform(-1, 3), rng.uniform(-1, 1), rng.uniform(-math.pi, math.pi),
                    rng.uniform(0, .8), rng.uniform(0, .6), step*.05)
            self.assertEqual(a(*args), b(*args))
            self.assertEqual(a.phase, b.phase)

    def test_target_telemetry_only_behavior_change(self):
        path = CANONICAL / 'src/hsl_src/hsl_behavior/hsl_behavior/behavior_node.py'
        text = path.read_text(encoding='utf-8-sig')
        text = text.replace('        # Clear display telemetry only; preserve all arrival and motion state.\n'
                            '        for walker in (self.walk, self.play.walk, self.search.walk):\n'
                            '            walker.target_local = None\n', '')
        text = text.replace('            target = None\n'
                            '            for walker in (self.walk, self.play.walk, self.search.walk):\n'
                            '                if walker.target_local is not None:\n'
                            '                    lx, ly, heading = walker.target_local\n'
                            '                    tx, ty = me_pose.to_field(lx, ly)\n'
                            '                    target = [round(tx, 3), round(ty, 3),\n'
                            '                              round(math.degrees(norm_angle(me_pose.th + heading)), 1)]\n', '')
        text = text.replace('                target=target, target_kind="walk" if target is not None else None,\n', '')
        before = (ROOT / 'patches/real_robot_20261109/behavior_node.py').read_text(encoding='utf-8-sig')
        self.assertEqual(ast.dump(ast.parse(text)), ast.dump(ast.parse(before)))

    def test_view_render_and_capture_time_tf(self):
        view, TransformException = stubs()
        from hsl_behavior.geometry import Field
        node = view.HslView.__new__(view.HslView)
        node.robot_name, node.field = 'robo04', Field.for_profile('practice')
        now = view.time.monotonic()
        node.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: NS(sec=1, nanosec=0)))
        node.pose = NS(x=-1., y=.2, theta=0., quality=2, validity=.98, num_landmarks_used=3)
        node.pose_t = node.ball_t = node.behavior_t = now
        node.ball = NS(valid=True, field_x=.2, field_y=.3, time_since_seen=.05)
        node.behavior = {'role':'attacker', 'assigned_role':'attacker', 'state':'PLAYING',
                         'active':4, 'skill':'play_ball_align', 'target':[.1, .2, 90.]}
        node.trail = deque([(-1.1, .2), (-1., .2)])
        node.detections = deque()
        received = []
        node.pub = NS(publish=lambda msg: received.append(msg))
        node.render()
        markers = received[-1].markers
        identities = [(m.ns, m.id) for m in markers]
        self.assertEqual(len(identities), len(set(identities)))
        target = next(m for m in markers if m.ns.endswith('/target') and m.id == 0)
        self.assertEqual((target.pose.position.x, target.pose.position.y), (.1, .2))
        self.assertTrue(any('robo04 | attacker' in m.text for m in markers))
        self.assertTrue(any('BALL (' in m.text for m in markers))
        node.behavior_t = now-2
        node.render()
        self.assertFalse(any(m.ns.endswith('/target') for m in received[-1].markers))
        stamp = NS(sec=123, nanosec=456)
        detection = NS(header=NS(frame_id='camera_color_optical_frame', stamp=stamp),
                       bbox=NS(center=NS(position=NS(x=1., y=0., z=1.))),
                       results=[NS(hypothesis=NS(class_id='T', score=.9))])
        node.detections.append((now, NS(header=detection.header, detections=[detection])))
        requests = []
        def lookup(target, frame, when, **kwargs):
            requests.append((target, frame, when.sec, when.nanosec))
            return NS(transform=NS(translation=NS(x=2., y=3., z=0.),
                                   rotation=NS(x=0., y=0., z=math.sqrt(.5), w=math.sqrt(.5))))
        node.tf_buffer = NS(lookup_transform=lookup)
        observed, missing, age = node.observed_markers(now)
        self.assertEqual(missing, 0)
        self.assertAlmostEqual(observed[0].pose.position.x, 2.)
        self.assertAlmostEqual(observed[0].pose.position.y, 4.)
        self.assertEqual(requests[0], ('map', 'camera_color_optical_frame', 123, 456))
        def unavailable(*args, **kwargs):
            raise TransformException('missing')
        node.tf_buffer.lookup_transform = unavailable
        observed, missing, age = node.observed_markers(now)
        self.assertEqual(observed, [])
        self.assertEqual(missing, 1)
        self.assertEqual(node.observed_markers(now+2)[0], [])

    def test_bundle_apply_and_guards(self):
        manifest = json.loads((BUNDLE / 'manifest.json').read_text())
        for name, digest in manifest['bundle_files'].items():
            self.assertEqual(hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest(), digest)
        originals = {'behavior_node.py': (ROOT/'patches/real_robot_20261109/behavior_node.py').read_bytes()}
        with tarfile.open(ROOT/'deploy/mx3_hsl_robot_20261103.tar.gz') as tar:
            originals['skills.py'] = tar.extractfile('src/hsl_src/hsl_behavior/hsl_behavior/skills.py').read()
        installer = (BUNDLE/'apply_behavior.py').read_text()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root/'proc'; proc.mkdir()
            for entry in manifest['sources']:
                path=root/entry['path'];path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(originals[entry['payload']])
            code=installer.replace("Path('/home/robo04/sim2real_ws')", 'Path('+repr(str(root))+')')\
                .replace("Path('/proc')", 'Path('+repr(str(proc))+')')
            def run():
                try:
                    exec(compile(code,'installer-under-test','exec'), {'__file__':str(BUNDLE/'apply_behavior.py')})
                except SystemExit as error:
                    return error.code
            run()
            for entry in manifest['sources']:
                self.assertEqual((root/entry['path']).read_bytes(), (BUNDLE/entry['payload']).read_bytes())
            self.assertTrue((root/'runtime_logs/COLCON_IGNORE').exists())
            self.assertEqual(run(), 0)
            path=root/manifest['sources'][1]['path'];path.write_bytes(b'unknown=1\n')
            first=root/manifest['sources'][0]['path'];before=first.read_bytes()
            self.assertIn('Source differs', run())
            self.assertEqual(first.read_bytes(),before)
            active=proc/'123';active.mkdir();(active/'cmdline').write_bytes(
                b'/usr/bin/python3\0/x/hsl_behavior/lib/hsl_behavior/behavior\0')
            self.assertIn('STOPPED',run())


if __name__ == '__main__':
    unittest.main()
