"""Install display-only target telemetry with source hash guards and rollback."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile

root = Path('/home/robo04/sim2real_ws').resolve(strict=True)
bundle = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
for process in Path('/proc').iterdir():
    if not process.name.isdigit():
        continue
    try:
        words = (process / 'cmdline').read_bytes().split(b'\0')[:2]
    except OSError:
        continue
    names = [os.fsdecode(word) for word in words if word]
    if any(Path(name).name in ('actor_onnx_walking_runner', 'actor_onnx_walkkick_v1_runner',
                              'keyboard_teleop_node')
           or name.endswith('/hsl_behavior/lib/hsl_behavior/behavior') for name in names):
        raise SystemExit('STOPPED: close teleop and start.sh/control launch first. PID=' + process.name)
for name, digest in manifest['bundle_files'].items():
    path = (bundle / name).resolve(strict=True)
    if not path.is_relative_to(bundle) or sha(path.read_bytes()) != digest:
        raise SystemExit('Bundle integrity check failed: ' + name)
pending = []
for entry in manifest['sources']:
    path = (root / entry['path']).resolve(strict=True)
    if not path.is_relative_to(root / 'src'):
        raise SystemExit('Unexpected source path')
    old = path.read_bytes()
    if sha(old) not in (entry['before_sha256'], entry['after_sha256']):
        raise SystemExit('Source differs from reviewed versions; no files changed: ' + str(path))
    new = (bundle / entry['payload']).read_bytes()
    if sha(new) != entry['after_sha256']:
        raise SystemExit('Source payload integrity failed')
    ast.parse(new.decode('utf-8-sig'))
    if old != new:
        pending.append((path, old, new))
if not pending:
    print('ALREADY_APPLIED: behavior target display telemetry')
    raise SystemExit(0)
logs = root / 'runtime_logs'
logs.mkdir(exist_ok=True)
(logs / 'COLCON_IGNORE').touch(exist_ok=True)
backup = Path(tempfile.mkdtemp(prefix='hsl_view_targets_', dir=logs))
def replace(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.view_target_', delete=False) as stream:
            stream.write(data)
            temporary = Path(stream.name)
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
changed = []
try:
    for path, old, new in pending:
        (backup / path.name).write_bytes(old)
        replace(path, new)
        changed.append((path, old))
except BaseException:
    for path, old in reversed(changed):
        replace(path, old)
    raise
print('BACKUP=' + str(backup))
print('APPLIED: actual walking target appended to behavior debug; motion calculations unchanged')
