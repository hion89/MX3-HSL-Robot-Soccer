"""Change only kick_start_hold 0.3 -> 0.6; no motor commands or ROS nodes."""
import ast
import os
from pathlib import Path
import tempfile

root = Path('/home/robo04/sim2real_ws').resolve(strict=True)
target = root / 'src/mx3_src/humanoid_bringup/launch/control.launch.py'
if not target.resolve(strict=True).is_relative_to(root / 'src'):
    raise SystemExit('Unexpected source path; no files changed')
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
before = target.read_bytes()
old = b'DeclareLaunchArgument("kick_start_hold", default_value="0.3",'
new = b'DeclareLaunchArgument("kick_start_hold", default_value="0.6",'
if before.count(new) == 1 and before.count(old) == 0:
    print('ALREADY_APPLIED: kick_start_hold=0.6')
    raise SystemExit(0)
if before.count(old) != 1 or before.count(new):
    raise SystemExit('Unexpected source content; no files changed')
after = before.replace(old, new, 1)
ast.parse(after.decode('utf-8-sig'))
logs = root / 'runtime_logs'
logs.mkdir(exist_ok=True)
(logs / 'COLCON_IGNORE').touch(exist_ok=True)
backups = logs / 'kick_hold_backups'
backups.mkdir(exist_ok=True)
with tempfile.NamedTemporaryFile(prefix='control_', suffix='.launch.py', dir=backups, delete=False) as stream:
    stream.write(before)
    backup = Path(stream.name)
temporary = None
try:
    with tempfile.NamedTemporaryFile(prefix='.kick_hold_', dir=target.parent, delete=False) as stream:
        stream.write(after)
        temporary = Path(stream.name)
    os.chmod(temporary, target.stat().st_mode & 0o777)
    os.replace(temporary, target)
finally:
    if temporary is not None and temporary.exists():
        temporary.unlink()
assert target.read_bytes() == after
print('BACKUP=' + str(backup))
print('APPLIED: kick_start_hold 0.3 -> 0.6; only this literal changed')
