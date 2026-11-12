import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'deploy/check_walk_shadow_20261105.py'
spec = importlib.util.spec_from_file_location('walk_shadow', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'sample.csv'

    def write(self, **changes):
        row = dict(run_state='RUN', inference_ms='0.5', cmd_vx='0', cmd_vy='0',
                   cmd_wz='0', a_raw_test='0.1', q_target_test='0.6')
        row.update(changes)
        with self.path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_real_inference_is_required(self):
        self.write(run_state='HOLD_START', inference_ms='')
        with self.assertRaisesRegex(RuntimeError, 'No actual inference'):
            module.summarize_csv([self.path])

    def test_nan_raw_action_rejected(self):
        self.write(a_raw_test='nan')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            module.summarize_csv([self.path])

    def test_infinite_target_rejected(self):
        self.write(q_target_test='inf')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
            module.summarize_csv([self.path])

    def test_nonzero_command_rejected(self):
        self.write(cmd_vx='0.2')
        with self.assertRaisesRegex(RuntimeError, 'Nonzero command'):
            module.summarize_csv([self.path])

    def test_finite_zero_command_trial_reported(self):
        self.write()
        report = module.summarize_csv([self.path])
        self.assertEqual(report['inference_rows'], 1)
        self.assertEqual(report['inference_ms']['median'], 0.5)


if __name__ == '__main__':
    unittest.main()
