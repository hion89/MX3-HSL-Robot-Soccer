"""Role and start/stop regressions; uses the deployed pure strategy and actual callbacks."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent
SRC = Path('C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim/src')
sys.path.insert(0, str(SRC / 'hsl_src/hsl_behavior'))
from hsl_behavior import strategy as S
from hsl_behavior.geometry import Field, Pose


def callback(filename, classname, method):
    tree = ast.parse((SRC / filename).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    namespace = {'S': S, 'STATE_NAMES': {0:'INITIAL',1:'READY',2:'SET',3:'PLAYING',4:'FINISHED'},
                 'SetParametersResult': lambda **kw: SimpleNamespace(**kw)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_callback', 'exec'), namespace)
    return namespace[method]


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.f = Field.for_profile('practice')
        self.me = S.Agent(4, Pose(-1.7, 0.6, 0))

    def test_single_robot_attacker(self):
        d = S.decide_assigned_role('attacker', self.me, (0, 0), self.f)
        self.assertTrue(d.active)
        self.assertEqual(d.active_player, 4)
        self.assertEqual(d.role, S.ROLE_FORWARD)

    def test_single_robot_defender_uses_original_position(self):
        d = S.decide_assigned_role('defender', self.me, (0, 0.3), self.f)
        self.assertFalse(d.active)
        self.assertEqual(d.role, S.ROLE_DEFENDER)
        self.assertEqual(d.target, S.defender_pose((0, 0.3), self.f))

    def test_robo04_goalkeeper_holds_goal_for_far_ball(self):
        d = S.decide_assigned_role('goalkeeper', self.me, (0.8, 0), self.f)
        self.assertFalse(d.active)
        self.assertEqual(d.role, S.ROLE_GOALKEEPER)
        self.assertEqual(d.target, S.goalkeeper_pose((0.8, 0), self.f))

    def test_robo04_goalkeeper_clears_own_penalty_ball(self):
        d = S.decide_assigned_role('goalkeeper', self.me, (-1.6, 0.1), self.f)
        self.assertTrue(d.active)
        self.assertEqual(d.active_player, 4)

    def test_auto_role_preserved(self):
        self.assertIsNone(S.decide_assigned_role('auto', self.me, (0, 0), self.f))

    def test_practice_defender_remains_between_own_goal_and_observed_ball(self):
        ball = (-0.65, 0.40)
        target = S.defender_pose(ball, self.f)
        self.assertLess(-self.f.hl, target.x)
        self.assertLess(target.x, ball[0])
        self.assertLess(abs(target.y), self.f.hw)

    def test_no_role_falls_back_to_forward_when_active(self):
        d = S.decide_playing(self.me, [], (0, 0), self.f, 0, {4:S.ROLE_NONE})
        self.assertTrue(d.active)
        self.assertEqual(d.role, S.ROLE_FORWARD)

    def test_existing_strategy_regressions(self):
        path = SRC / 'hsl_src/hsl_behavior/test/test_strategy.py'
        spec = importlib.util.spec_from_file_location('existing_strategy_tests', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, value in vars(module).items():
            if name.startswith('test_') and callable(value):
                with self.subTest(name=name):
                    value()

    def test_role_parameter_changes_live_state(self):
        fn = callback('hsl_src/hsl_behavior/hsl_behavior/behavior_node.py', 'Behavior', '_on_parameters')
        actor = SimpleNamespace(kick_phase=None, assigned_role='auto', active_player=4, my_role=0,
                    play=SimpleNamespace(reset=lambda:None), walk=SimpleNamespace(reset=lambda:None),
                    get_logger=lambda:SimpleNamespace(info=lambda message:None))
        result = fn(actor, [SimpleNamespace(name='assigned_role', value='defender')])
        self.assertTrue(result.successful)
        self.assertEqual(actor.assigned_role, 'defender')
        actor.kick_phase = 'arming'
        result = fn(actor, [SimpleNamespace(name='default_game_state', value='initial')])
        self.assertTrue(result.successful)
        self.assertIsNone(actor.kick_phase)
        self.assertEqual(actor.default_state, 'INITIAL')

    def test_stop_disables_and_finishes_active_kick(self):
        fn = callback('mx3_src/humanoid_policy_runner/humanoid_policy_runner/actor_onnx_walkkick_v1_runner.py',
                      'ActorOnnxWalkKickV1Node', '_on_set_parameters')
        finished = []
        actor = SimpleNamespace(kick_active=True, motion_enabled=True, history_initialized=True,
                                _finish_kick=lambda:finished.append(True))
        result = fn(actor, [SimpleNamespace(name='motion_enabled', value=False)])
        self.assertTrue(result.successful)
        self.assertEqual(finished, [True])
        self.assertFalse(actor.motion_enabled)
        self.assertFalse(actor.history_initialized)


if __name__ == '__main__':
    unittest.main()
