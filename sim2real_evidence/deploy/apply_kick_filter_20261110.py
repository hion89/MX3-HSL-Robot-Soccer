"""Enable the existing kick action filter only; never start motion."""
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
def kick_assignment(parsed):
    matches = [n for n in ast.walk(parsed) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'kick_params' for t in n.targets)]
    if len(matches) != 1 or not isinstance(matches[0].value, ast.Dict):
        raise SystemExit('Unexpected kick_params structure; no files changed')
    return matches[0]
node = kick_assignment(tree)
settings = {k.value: v for k, v in zip(node.value.keys, node.value.values)
            if isinstance(k, ast.Constant)}
expected = {'enable_action_filter': True, 'tau_action_filter': 0.03}
present = set(expected) & set(settings)
if present:
    if present == set(expected) and all(isinstance(settings[k], ast.Constant)
            and type(settings[k].value) is type(v) and settings[k].value == v
            for k, v in expected.items()):
        print('ALREADY_APPLIED: kick action filter enabled, tau=0.03 s')
        raise SystemExit(0)
    raise SystemExit('Existing kick filter settings differ; no files changed')
lines = text.splitlines(keepends=True)
header = lines[node.lineno - 1]
if header.strip() != 'kick_params = {':
    raise SystemExit('Unexpected kick_params header; no files changed')
newline = '\r\n' if header.endswith('\r\n') else '\n'
indent = header[:len(header) - len(header.lstrip())] + '    '
insertion = newline.join([
    indent + '# Smooth kick actions only; leave the walking policy unchanged.',
    indent + '"enable_action_filter": True,',
    indent + '"tau_action_filter": 0.03,', ''])
lines.insert(node.lineno, insertion)
after_text = ''.join(lines)
after_tree = ast.parse(after_text)
after_dict = kick_assignment(after_tree).value
keep = [(k,v) for k,v in zip(after_dict.keys, after_dict.values)
        if not (isinstance(k, ast.Constant) and k.value in expected)]
after_dict.keys = [k for k,v in keep]
after_dict.values = [v for k,v in keep]
if ast.dump(tree) != ast.dump(after_tree):
    raise SystemExit('Unexpected unrelated code change; no files changed')
after = (b'\xef\xbb\xbf' if before.startswith(b'\xef\xbb\xbf') else b'') + after_text.encode('utf-8')
logs = root / 'runtime_logs'
logs.mkdir(exist_ok=True)
(logs / 'COLCON_IGNORE').touch(exist_ok=True)
backups = logs / 'kick_filter_backups'
backups.mkdir(exist_ok=True)
with tempfile.NamedTemporaryFile(prefix='control_', suffix='.launch.py', dir=backups, delete=False) as stream:
    stream.write(before)
    backup = Path(stream.name)
temporary = None
try:
    with tempfile.NamedTemporaryFile(prefix='.kick_filter_', dir=target.parent, delete=False) as stream:
        stream.write(after)
        temporary = Path(stream.name)
    os.chmod(temporary, target.stat().st_mode & 0o777)
    os.replace(temporary, target)
finally:
    if temporary is not None and temporary.exists():
        temporary.unlink()
assert target.read_bytes() == after
print('BACKUP=' + str(backup))
print('APPLIED: kick enable_action_filter=True, tau_action_filter=0.03 s')
print('Walking settings, kick speed/hold/duration and model files are unchanged')
