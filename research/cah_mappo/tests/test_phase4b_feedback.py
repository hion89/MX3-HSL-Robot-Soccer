import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase4b_feedback import FeedbackSimulation,SensorActor
from test_phase4b import unit_spec,Script,idle
from cah_mappo.phase4b.policies import Native
from cah_mappo.phase4b.engine import Simulation
from cah_mappo.phase4b.metrics import evaluate_new


class FeedbackContract(unittest.TestCase):
    def test_future_job_is_filtered_before_feedback_publication(self):
        s=unit_spec();s['config']['jobs'][1]['spawn']=2
        e=FeedbackSimulation(s,Native('fixed'));e.boundary(0);v=e.public(0)
        self.assertNotIn('task0',v.feedback)
        self.assertNotIn('task0',str(v.record()))

    def test_adding_completed_feedback_does_not_change_native_physics(self):
        s=unit_spec();a=Simulation(s,Native('exact')).run();b=FeedbackSimulation(s,Native('exact')).run()
        self.assertEqual(evaluate_new(a),evaluate_new(b))

    def test_original_perception_gets_zero_remaining_after_completed_job(self):
        import torch
        class Noop:
            def distribution(self,x,m):
                logits=torch.full((len(x),10),-torch.inf,dtype=torch.float64);logits[:,0]=0
                return torch.distributions.Categorical(logits=logits)
            def value(self,*a):raise AssertionError('No critic at inference')
        actor=SensorActor(Noop());e=FeedbackSimulation(unit_spec(),actor);e.boundary(0)
        actor.decide(e.public(0));e.jobs['task0'].update(status='completed',work=2,completed=3)
        actor.decide(e.public(3));packet=actor.last['actor_samples']['A']['packet']
        slot=packet['job_ids'].index('task0')
        self.assertEqual(packet['observation'][50+slot*9+7],0)
        self.assertEqual(packet['action_mask'][slot+2],0)

    def test_canonical_actor_inputs_are_independent_of_unseen_future_position(self):
        import torch
        class Noop:
            def distribution(self,x,m):
                self.x=x.clone();logits=torch.full((len(x),10),-torch.inf,dtype=torch.float64);logits[:,0]=0
                return torch.distributions.Categorical(logits=logits)
        s=unit_spec();s['config']['jobs'][1]['spawn']=2;other=copy.deepcopy(s);other['config']['jobs'][1]['position']=[9.,4.]
        n=Noop();a=FeedbackSimulation(s,SensorActor(n));a.boundary(0);a.policy.decide(a.public(0));first=n.x.clone()
        b=FeedbackSimulation(other,SensorActor(n));b.boundary(0);b.policy.decide(b.public(0))
        self.assertTrue(torch.equal(first,n.x))
