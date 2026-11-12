"""Apply the reviewed walking-command timeout patch, with identity checks and backup."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

RELATIVE = Path('src/mx3_src/humanoid_policy_runner/humanoid_policy_runner/actor_onnx_walking_runner.py')
ORIGINAL_SHA256 = '477dcff62aa2572b12d7cad58dc71c4a1628871d0c8695a869450ffc5bdf52aa'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ensure_stopped(proc_root=Path('/proc')):
    if not proc_root.is_dir():
        raise RuntimeError('Apply on Linux; /proc is required')
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            words = (process / 'cmdline').read_bytes().split(b'\0')[:2]
        except (OSError, ProcessLookupError):
            continue
        if any(Path(os.fsdecode(word)).name == 'actor_onnx_walking_runner' for word in words if word):
            raise RuntimeError('Stop the walking runner first (PID ' + process.name + ')')


def atomic_write(path, data):
    permissions = stat.S_IMODE(path.stat().st_mode)
    descriptor, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, permissions)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def apply(workspace, bundle, write=False):
    workspace, bundle = Path(workspace).resolve(), Path(bundle).resolve()
    target = (workspace / RELATIVE).resolve()
    if not target.is_relative_to(workspace):
        raise RuntimeError('Source resolves outside the selected workspace')
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    candidate = (bundle / 'actor_onnx_walking_runner.py').read_bytes()
    if manifest['original_sha256'] != ORIGINAL_SHA256:
        raise RuntimeError('Unexpected original source identity in manifest')
    if digest(candidate) != manifest['patched_sha256']:
        raise RuntimeError('Patch payload hash mismatch')
    original = target.read_bytes()
    original_hash = digest(original)
    if original_hash not in (ORIGINAL_SHA256, manifest['patched_sha256']):
        raise RuntimeError('Source differs from the reviewed version; no files changed')
    result = {'target': str(target), 'before_sha256': original_hash,
              'patched_sha256': manifest['patched_sha256'], 'backup': None}
    if not write:
        return dict(result, status='INSPECTED_NO_FILES_CHANGED')
    ensure_stopped()
    if original_hash == manifest['patched_sha256']:
        return dict(result, status='ALREADY_APPLIED')
    if target.read_bytes() != original:
        raise RuntimeError('Source changed during inspection; no patch applied')
    archive = workspace / 'patch_backups'
    archive.mkdir(exist_ok=True)
    label = datetime.now(timezone.utc).strftime('walk_cmd_timeout_%Y%m%dT%H%M%SZ_')
    backup = Path(tempfile.mkdtemp(prefix=label, dir=archive))
    (backup / target.name).write_bytes(original)
    (backup / 'manifest.json').write_bytes((bundle / 'manifest.json').read_bytes())
    try:
        atomic_write(target, candidate)
        if target.read_bytes() != candidate:
            raise RuntimeError('Patch readback mismatch')
    except Exception:
        if target.read_bytes() != original:
            atomic_write(target, original)
        raise
    return dict(result, status='APPLIED_SOURCE_ONLY', backup=str(backup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/home/robo04/sim2real_ws'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        result = apply(args.workspace, Path(__file__).resolve().parent, args.apply)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, 'PATCH FAILED: ' + str(error) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
