import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


BUNDLE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tf_patch_apply', BUNDLE / 'apply.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        self.target = self.workspace / module.RELATIVE
        self.target.parent.mkdir(parents=True)
        self.original = (BUNDLE / 'ball_tracker_node.original.cpp').read_bytes()
        self.target.write_bytes(self.original)

    def test_inspect_does_not_write(self):
        result = module.apply(self.workspace, BUNDLE)
        self.assertEqual(result['status'], 'INSPECTED_NO_FILES_CHANGED')
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertFalse((self.workspace / 'patch_backups').exists())

    def test_backup_and_idempotence(self):
        with patch.object(module, 'ensure_stopped'):
            first = module.apply(self.workspace, BUNDLE, True)
            second = module.apply(self.workspace, BUNDLE, True)
        self.assertEqual(first['status'], 'APPLIED_SOURCE_ONLY')
        self.assertEqual(second['status'], 'ALREADY_APPLIED')
        backup = Path(first['backup']) / self.target.name
        self.assertEqual(backup.read_bytes(), self.original)
        self.assertEqual(self.target.read_bytes(), (BUNDLE / 'ball_tracker_node.cpp').read_bytes())
        self.assertEqual(len(list((self.workspace / 'patch_backups').iterdir())), 1)

    def test_upgrade_installed_v2_preserves_v2_backup(self):
        previous = (BUNDLE / 'ball_tracker_node.v2.cpp').read_bytes()
        self.target.write_bytes(previous)
        with patch.object(module, 'ensure_stopped'):
            result = module.apply(self.workspace, BUNDLE, True)
        self.assertEqual(result['status'], 'APPLIED_SOURCE_ONLY')
        self.assertEqual((Path(result['backup']) / self.target.name).read_bytes(), previous)
        self.assertEqual(self.target.read_bytes(), (BUNDLE / 'ball_tracker_node.cpp').read_bytes())

    def test_reject_unreviewed_source(self):
        self.target.write_bytes(self.original + b'// user change\n')
        before = self.target.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Source differs'):
            module.apply(self.workspace, BUNDLE, True)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertFalse((self.workspace / 'patch_backups').exists())

    def test_active_process_prevents_apply(self):
        with patch.object(module, 'ensure_stopped', side_effect=RuntimeError('active process')):
            with self.assertRaisesRegex(RuntimeError, 'active process'):
                module.apply(self.workspace, BUNDLE, True)
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertFalse((self.workspace / 'patch_backups').exists())
        self.target.write_bytes((BUNDLE / 'ball_tracker_node.cpp').read_bytes())
        with patch.object(module, 'ensure_stopped', side_effect=RuntimeError('active process')):
            with self.assertRaisesRegex(RuntimeError, 'active process'):
                module.apply(self.workspace, BUNDLE, True)

    def test_failed_readback_restores_source(self):
        original_write = module.atomic_write
        calls = []

        def corrupt_first_write(path, data):
            calls.append(data)
            original_write(path, b'BROKEN' if len(calls) == 1 else data)

        with patch.object(module, 'ensure_stopped'), patch.object(module, 'atomic_write', corrupt_first_write):
            with self.assertRaisesRegex(RuntimeError, 'readback mismatch'):
                module.apply(self.workspace, BUNDLE, True)
        self.assertEqual(self.target.read_bytes(), self.original)
        self.assertEqual(len(calls), 2)

    def test_proc_guard_finds_cpp_process(self):
        proc = self.workspace / 'proc'
        process = proc / '123'
        process.mkdir(parents=True)
        (process / 'cmdline').write_bytes(b'/ws/install/pkg/lib/pkg/ball_tracker\0--ros-args\0')
        with self.assertRaisesRegex(RuntimeError, 'PID 123'):
            module.ensure_stopped(proc)


if __name__ == '__main__':
    unittest.main()
