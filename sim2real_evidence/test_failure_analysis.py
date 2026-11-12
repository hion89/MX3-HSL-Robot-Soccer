"""Numerical checks for alignment and quaternion conventions in the offline report."""
import math
from types import SimpleNamespace
import unittest
import numpy as np
from analyze_walk_failure import asof_indices, attitude, stamp_ns


class AnalysisTests(unittest.TestCase):
    def test_alignment_never_uses_future_sample_and_handles_duplicates(self):
        np.testing.assert_array_equal(asof_indices([10,20,20,40],[5,10,19,20,39,41]),[-1,0,0,2,2,3])

    def test_tilt_does_not_depend_on_yaw_and_quaternion_scale(self):
        yaw90=SimpleNamespace(x=0.,y=0.,z=math.sqrt(.5)*2,w=math.sqrt(.5)*2)
        self.assertAlmostEqual(attitude(yaw90)[0],0.)
        roll90=SimpleNamespace(x=math.sqrt(.5),y=0.,z=0.,w=math.sqrt(.5))
        self.assertAlmostEqual(attitude(roll90)[0],90.)
        with self.assertRaises(ValueError):
            attitude(SimpleNamespace(x=0.,y=0.,z=0.,w=0.))

    def test_time_is_integer_nanoseconds_and_zero_stamp_falls_back(self):
        msg=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1791357549,nanosec=265054924)))
        self.assertEqual(stamp_ns(msg,0),1791357549265054924)
        msg.header.stamp.sec=msg.header.stamp.nanosec=0
        self.assertEqual(stamp_ns(msg,42),42)


if __name__=='__main__':
    unittest.main()
