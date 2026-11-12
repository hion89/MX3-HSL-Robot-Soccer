"""Apply only the reviewed role-integration source files, with a recoverable backup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ensure_stopped():
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            words = (process / 'cmdline').read_bytes().split(b'\0')[:2]
        except OSError:
            continue
        for word in words:
            name = os.fsdecode(word)
            if (Path(name).name in ('actor_onnx_walking_runner', 'actor_onnx_walkkick_v1_runner')
                    or name.endswith('/hsl_behavior/lib/hsl_behavior/behavior')):
                raise RuntimeError('Stop only the behavior/motion launch first; active PID ' + process.name)


def replace(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def apply(workspace, bundle, write=False):
    workspace, bundle = Path(workspace).resolve(), Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    prepared = []
    for entry in manifest['files']:
        target = (workspace / entry['path']).resolve()
        if not target.is_relative_to(workspace / 'src'):
            raise RuntimeError('Target outside workspace/src: ' + str(target))
        payload = (bundle / entry['payload']).read_bytes()
        if digest(payload) != entry['after_sha256']:
            raise RuntimeError('Bundle hash mismatch: ' + entry['payload'])
        old = target.read_bytes() if target.exists() else None
        old_hash = digest(old) if old is not None else None
        if old_hash not in (entry['before_sha256'], entry['after_sha256']):
            raise RuntimeError('Unreviewed source; no files changed: ' + str(target))
        prepared.append((entry, target, old, payload))
    changes = [(e, p, old, new) for e, p, old, new in prepared if old != new]
    result = {'status': 'INSPECTED', 'changed_files': len(changes), 'backup': None}
    if not write:
        return result
    ensure_stopped()
    if not changes:
        return dict(result, status='ALREADY_APPLIED')
    archive = workspace / 'patch_backups'
    archive.mkdir(exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix='real_robot_20261109_', dir=archive))
    for entry, target, old, new in changes:
        if (target.read_bytes() if target.exists() else None) != old:
            raise RuntimeError('Source changed during inspection')
        if old is not None:
            saved = backup / entry['path']
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(old)
    (backup / 'manifest.json').write_bytes((bundle / 'manifest.json').read_bytes())
    changed = []
    try:
        for entry, target, old, new in changes:
            changed.append((target, old))
            replace(target, new)
            if target.read_bytes() != new:
                raise RuntimeError('Source readback mismatch')
    except Exception:
        for target, old in reversed(changed):
            if old is None:
                target.unlink(missing_ok=True)
            else:
                replace(target, old)
        raise
    return dict(result, status='APPLIED_AND_VERIFIED', backup=str(backup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/home/robo04/sim2real_ws'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        report = apply(args.workspace, Path(__file__).resolve().parent, args.apply)
    except Exception as error:
        parser.exit(1, 'APPLY FAILED: ' + str(error) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
