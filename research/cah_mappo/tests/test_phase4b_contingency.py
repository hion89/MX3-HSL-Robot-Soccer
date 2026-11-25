from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase4b_contingency import FaultAwareRelease
from cah_mappo.phase4b.scenarios import scenario
from cah_mappo.phase4b.engine import Simulation
from cah_mappo.phase4b.policies import Native
from cah_mappo.phase4b.metrics import evaluate_new


class FaultContingency(unittest.TestCase):
    def test_no_fault_keeps_original_ready_actions(self):
        s=scenario('validation','B_normal',2,0)
        a=Simulation(s,Native('ready')).run();b=Simulation(s,FaultAwareRelease()).run()
        self.assertEqual(evaluate_new(a),evaluate_new(b))

    def test_failure_does_not_terminate_defense_responsibility(self):
        s=scenario('validation','E_successor',2,0);rows=Simulation(s,FaultAwareRelease()).run();m=evaluate_new(rows)
        self.assertEqual(m['responsibility_active_until'],32)
        # A release is a deliberate cost, not removal of the base from metrics.
        changes=[r for r in rows if r.get('event')=='policy_selection' and
                 r['action']['policy_metadata'].get('fault_release_without_known_active_threat')]
        self.assertTrue(all(r['time']>=6 for r in changes))
