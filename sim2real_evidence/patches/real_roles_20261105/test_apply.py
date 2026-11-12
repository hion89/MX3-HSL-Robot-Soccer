import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('role_apply', ROOT / 'apply.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        self.entries = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))['files']
        for entry in self.entries:
            if entry['before_sha256'] is not None:
                target = self.ws / entry['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / (entry['payload'] + '.original')).read_bytes())

    def test_inspection_does_not_change_source(self):
        result = module.apply(self.ws, ROOT)
        self.assertEqual(result['changed_files'], len(self.entries))
        self.assertFalse((self.ws / 'patch_backups').exists())

    def test_backup_and_idempotent_application(self):
        with patch.object(module, 'ensure_stopped'):
            result = module.apply(self.ws, ROOT, True)
            again = module.apply(self.ws, ROOT, True)
        self.assertEqual(again['status'], 'ALREADY_APPLIED')
        for entry in self.entries:
            self.assertEqual((self.ws / entry['path']).read_bytes(), (ROOT / entry['payload']).read_bytes())
            if entry['before_sha256'] is not None:
                self.assertEqual((Path(result['backup']) / entry['path']).read_bytes(),
                                 (ROOT / (entry['payload'] + '.original')).read_bytes())

    def test_unreviewed_source_prevents_all_changes(self):
        target = self.ws / self.entries[-2]['path']
        target.write_bytes(b'user edited source')
        first = self.ws / self.entries[0]['path']
        before = first.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Unreviewed source'):
            module.apply(self.ws, ROOT, True)
        self.assertEqual(first.read_bytes(), before)
        self.assertFalse((self.ws / 'patch_backups').exists())

    def test_active_motion_prevents_apply(self):
        with patch.object(module, 'ensure_stopped', side_effect=RuntimeError('active')):
            with self.assertRaisesRegex(RuntimeError, 'active'):
                module.apply(self.ws, ROOT, True)
        self.assertFalse((self.ws / 'patch_backups').exists())

    def test_failed_second_write_restores_first(self):
        original = module.replace
        calls = []
        def fail_second(path, data):
            calls.append(path)
            if len(calls) == 2:
                raise OSError('injected write failure')
            original(path, data)
        with patch.object(module, 'ensure_stopped'), patch.object(module, 'replace', fail_second):
            with self.assertRaisesRegex(OSError, 'injected'):
                module.apply(self.ws, ROOT, True)
        for entry in self.entries:
            path = self.ws / entry['path']
            if entry['before_sha256'] is None:
                self.assertFalse(path.exists())
            else:
                self.assertEqual(path.read_bytes(), (ROOT / (entry['payload'] + '.original')).read_bytes())


if __name__ == '__main__':
    unittest.main()
