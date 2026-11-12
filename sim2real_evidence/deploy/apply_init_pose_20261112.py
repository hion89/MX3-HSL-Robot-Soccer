"""Update the existing manual initial-pose helper; do not publish a pose or start motion."""
import os
from pathlib import Path
import tempfile

incoming = Path('/home/robo04/incoming').resolve(strict=True)
root = Path('/home/robo04/sim2real_ws').resolve(strict=True)
target = incoming / 'real_robot_20261109/init_pose.sh'
if not target.resolve(strict=True).is_relative_to(incoming):
    raise SystemExit('Unexpected initial-pose script path')
payload = incoming / 'init_practice_pose_20261112.sh'
before, after = target.read_bytes(), payload.read_bytes()
if after.count(b'x: -1.95, y: 0.8675, z: 0.0') != 1 or b'y: 0.7583333' in after:
    raise SystemExit('Unexpected initial-pose payload; no files changed')
if b'/initialpose geometry_msgs/msg/PoseWithCovarianceStamped' not in after:
    raise SystemExit('Unexpected initial-pose command; no files changed')
if before == after:
    print('ALREADY_APPLIED: initial pose (-1.95, +0.8675, 0 rad)')
    raise SystemExit(0)
if b'y: 0.7583333' not in before and b'y: 0.8675' not in before:
    raise SystemExit('Initial pose was customized; no files changed')
logs = root / 'runtime_logs'
logs.mkdir(exist_ok=True)
(logs / 'COLCON_IGNORE').touch(exist_ok=True)
backup = Path(tempfile.mkdtemp(prefix='initial_pose_',dir=logs)) / 'init_pose.sh'
backup.write_bytes(before)
temporary = None
try:
    with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.init_pose_',delete=False) as stream:
        stream.write(after)
        temporary = Path(stream.name)
    os.chmod(temporary,target.stat().st_mode & 0o777)
    os.replace(temporary,target)
finally:
    if temporary is not None and temporary.exists():
        temporary.unlink()
assert target.read_bytes() == after
print('BACKUP=' + str(backup))
print('APPLIED: init_pose.sh -> x=-1.95, y=+0.8675, yaw=0')
print('No pose was published. Run init_pose.sh only after placing the robot and pressing START.')
print('The 0.10 m temporary START seed requires the prior hsl_field_20261112 map update/build/restart.')
