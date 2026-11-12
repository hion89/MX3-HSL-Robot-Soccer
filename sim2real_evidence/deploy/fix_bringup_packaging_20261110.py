"""Exclude launch caches/backups; preserve the old package build without deleting it."""
import ast
import os
from pathlib import Path
import shutil
import tempfile

root = Path('/home/robo04/sim2real_ws').resolve(strict=True)
source = root / 'src/mx3_src/humanoid_bringup/setup.py'
if not source.resolve(strict=True).is_relative_to(root / 'src'):
    raise SystemExit('Unexpected setup.py location')
before = source.read_bytes()
after = before
for old, new in [
    (b"glob('launch/*')", b"sorted(glob('launch/*.launch.py'))"),
    (b"glob('config/*')", b"sorted(path for path in glob('config/*') if os.path.isfile(path))"),
]:
    if after.count(new) == 1:
        continue
    if after.count(old) != 1:
        raise SystemExit('Unexpected setup.py content; no files changed')
    after = after.replace(old, new, 1)
ast.parse(after.decode('utf-8-sig'))
backup_parent = root / 'runtime_logs/packaging_backups'
backup_parent.mkdir(parents=True, exist_ok=True)
# Colcon can identify a Python package from a backed-up setup.py alone.
# Exclude this entire runtime-only tree before creating package backups.
(root / 'runtime_logs/COLCON_IGNORE').touch(exist_ok=True)
backup = Path(tempfile.mkdtemp(prefix='humanoid_bringup_', dir=backup_parent))
(backup / 'setup.py').write_bytes(before)
if after != before:
    temporary = source.with_name('.setup_packaging_fix.tmp')
    if temporary.exists():
        raise SystemExit('Temporary file already exists; no source files changed')
    with temporary.open('xb') as stream:
        stream.write(after)
    os.chmod(temporary, source.stat().st_mode & 0o777)
    os.replace(temporary, source)
package_build = root / 'build/humanoid_bringup'
if package_build.exists() or package_build.is_symlink():
    expected = root / 'build/humanoid_bringup'
    if package_build.is_symlink() or package_build.resolve() != expected:
        raise SystemExit('Unexpected build path; source fixed, build not moved')
    shutil.move(str(package_build), str(backup / 'build_humanoid_bringup'))
print('PACKAGING_FIXED: install only *.launch.py and regular config files')
print('BACKUP=' + str(backup))
print('Only humanoid_bringup build cache was moved; no directories deleted')
