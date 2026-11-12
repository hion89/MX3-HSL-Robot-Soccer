"""Install reviewed practice field + viewer telemetry sources with backups and rollback."""
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
    if any(Path(name).name in ('actor_onnx_walking_runner','actor_onnx_walkkick_v1_runner',
                              'keyboard_teleop_node','self_localizer','hsl_view.py')
           or name.endswith('/hsl_behavior/lib/hsl_behavior/behavior') for name in names):
        raise SystemExit('STOPPED: close teleop, role/control, base and old viewer launches. PID=' + process.name)
for name,digest in manifest['bundle_files'].items():
    path = (bundle / name).resolve(strict=True)
    if not path.is_relative_to(bundle) or sha(path.read_bytes()) != digest:
        raise SystemExit('Bundle integrity failed: ' + name)
pending = []
for entry in manifest['sources']:
    path = (root / entry['path']).resolve(strict=True)
    if not path.is_relative_to(root / 'src'):
        raise SystemExit('Unexpected source path')
    old = path.read_bytes()
    if sha(old) not in entry['accepted_sha256']:
        raise SystemExit('Source differs from reviewed versions; no files changed: ' + str(path))
    new = (bundle / entry['payload']).read_bytes()
    if sha(new) != entry['after_sha256']:
        raise SystemExit('Payload integrity failed')
    if path.suffix == '.py':
        ast.parse(new.decode('utf-8-sig'))
    if old != new:
        pending.append((path,old,new))
logs = root / 'runtime_logs'
logs.mkdir(exist_ok=True)
(logs / 'COLCON_IGNORE').touch(exist_ok=True)
# This CMake package installs this intentionally empty directory.
(root / 'src/hsl_src/hsl_world_model/config').mkdir(exist_ok=True)
if not pending:
    print('ALREADY_APPLIED: photo practice field and target telemetry')
    raise SystemExit(0)
backup = Path(tempfile.mkdtemp(prefix='practice_photo_',dir=logs))
def replace(path,data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.practice_',delete=False) as stream:
            stream.write(data)
            temporary = Path(stream.name)
        os.chmod(temporary,path.stat().st_mode & 0o777)
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
changed = []
try:
    for path,old,new in pending:
        (backup / path.name).write_bytes(old)
        replace(path,new)
        changed.append((path,old))
except BaseException:
    for path,old in reversed(changed):
        replace(path,old)
    raise
print('BACKUP=' + str(backup))
print('APPLIED: photo practice geometry in C++ localizer + Python strategy + viewer')
print('START temporary seed: practice outside margin default 0.10 m; manual initialpose unchanged')
print('Model files, motor calibration, walk/kick tuning and S-Field dimensions unchanged')
