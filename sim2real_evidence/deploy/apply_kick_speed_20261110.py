"""Change only the kick_cmd_vx launch default; never start robot motion."""
import ast
import hashlib
import os
from pathlib import Path
import tempfile

root = Path('/home/robo04/sim2real_ws').resolve(strict=True)
target = root / 'src/mx3_src/humanoid_bringup/launch/control.launch.py'
if not target.resolve(strict=True).is_relative_to(root / 'src'):
    raise SystemExit('Unexpected source path')
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        args = (entry / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    executables = {Path(os.fsdecode(arg)).name for arg in args if arg}
    if executables & {'actor_onnx_walking_runner', 'actor_onnx_walkkick_v1_runner',
                      'actor_onnx_walking_node', 'actor_onnx_walkkick_v1_node',
                      'keyboard_teleop_node'}:
        raise SystemExit('STOPPED: close the keyboard and start.sh/control launch terminals first. PID=' + entry.name)
before = target.read_bytes()
old = b'DeclareLaunchArgument("kick_cmd_vx", default_value="0.30",'
new = b'DeclareLaunchArgument("kick_cmd_vx", default_value="0.20",'
if before.count(new) == 1 and before.count(old) == 0:
    print('ALREADY_APPLIED: kick_cmd_vx=0.20')
    raise SystemExit(0)
if before.count(old) != 1 or before.count(new):
    raise SystemExit('Unexpected source content; no files changed')
after = before.replace(old, new, 1)
ast.parse(after.decode('utf-8-sig'))
backup_root = root / 'runtime_logs/kick_speed_backups'
backup_root.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(prefix='control_', suffix='.launch.py', dir=backup_root, delete=False) as stream:
    stream.write(before)
    backup = Path(stream.name)
temp_path = None
try:
    with tempfile.NamedTemporaryFile(prefix='.kick_speed_', dir=target.parent, delete=False) as stream:
        stream.write(after)
        temp_path = Path(stream.name)
    os.chmod(temp_path, target.stat().st_mode & 0o777)
    os.replace(temp_path, target)
finally:
    if temp_path is not None and temp_path.exists():
        temp_path.unlink()
assert target.read_bytes() == after
print('BACKUP=' + str(backup))
print('APPLIED: kick_cmd_vx 0.30 -> 0.20; only this literal changed')
print('SOURCE_SHA256=' + hashlib.sha256(after).hexdigest())
