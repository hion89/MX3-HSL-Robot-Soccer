"""Change only the enabled kick filter time constant from 0.03 to 0.02 s."""
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
text = before.decode('utf-8-sig')
tree = ast.parse(text)
matches = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == 'kick_params' for t in n.targets)]
if len(matches) != 1 or not isinstance(matches[0].value, ast.Dict):
    raise SystemExit('Unexpected kick_params structure; no files changed')
pairs = list(zip(matches[0].value.keys, matches[0].value.values))
def value_node(name):
    nodes = [v for k,v in pairs if isinstance(k, ast.Constant) and k.value == name]
    if len(nodes) != 1 or not isinstance(nodes[0], ast.Constant):
        raise SystemExit('Expected existing kick filter setting: ' + name + '; no files changed')
    return nodes[0]
if value_node('enable_action_filter').value is not True:
    raise SystemExit('Kick filter is not explicitly enabled; no files changed')
tau = value_node('tau_action_filter')
if tau.value == 0.02:
    print('ALREADY_APPLIED: kick tau_action_filter=0.02 s')
    raise SystemExit(0)
if tau.value != 0.03:
    raise SystemExit('Expected kick tau=0.03; no files changed')
lines = text.splitlines(keepends=True)
line = lines[tau.lineno - 1]
if line.count('0.03') != 1:
    raise SystemExit('Unexpected tau line; no files changed')
lines[tau.lineno - 1] = line.replace('0.03', '0.02', 1)
after_text = ''.join(lines)
tau.value = 0.02
if ast.dump(tree) != ast.dump(ast.parse(after_text)):
    raise SystemExit('Unrelated code change detected; no files changed')
after = (b'\xef\xbb\xbf' if before.startswith(b'\xef\xbb\xbf') else b'') + after_text.encode('utf-8')
logs = root / 'runtime_logs'
logs.mkdir(exist_ok=True)
(logs / 'COLCON_IGNORE').touch(exist_ok=True)
backups = logs / 'kick_filter_backups'
backups.mkdir(exist_ok=True)
with tempfile.NamedTemporaryFile(prefix='control_tau003_', suffix='.launch.py', dir=backups, delete=False) as stream:
    stream.write(before)
    backup = Path(stream.name)
temporary = None
try:
    with tempfile.NamedTemporaryFile(prefix='.kick_tau_', dir=target.parent, delete=False) as stream:
        stream.write(after)
        temporary = Path(stream.name)
    os.chmod(temporary, target.stat().st_mode & 0o777)
    os.replace(temporary, target)
finally:
    if temporary is not None and temporary.exists():
        temporary.unlink()
assert target.read_bytes() == after
print('BACKUP=' + str(backup))
print('APPLIED: kick tau_action_filter 0.03 -> 0.02 s')
print('All other settings and model files are unchanged')
