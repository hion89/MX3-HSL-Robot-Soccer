"""Package the already reviewed role changes plus a unified base launch."""
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'patches' / 'real_roles_20261105'
BUNDLE = ROOT / 'patches' / 'real_robot_20261109'
SOURCE = Path('C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ORIGINAL / 'manifest.json').read_text(encoding='utf-8'))
    names = [entry['payload'] for entry in manifest['files']]
    names += ['apply.py', 'apply_and_build.sh', 'start.sh', 'role.sh', 'role_control.py']
    for name in names:
        data = (ORIGINAL / name).read_bytes()
        if name == 'apply.py':
            data = data.replace(b'real_roles_20261105_', b'real_robot_20261109_')
        if name.endswith('.sh'):
            data = data.replace(b'\r\n', b'\n')
        (BUNDLE / name).write_bytes(data)

    relative = 'src/hsl_src/hsl_bringup/launch/real_base.launch.py'
    data = (SOURCE / relative).read_bytes()
    (BUNDLE / 'real_base.launch.py').write_bytes(data)
    manifest['files'].append({
        'path': relative, 'payload': 'real_base.launch.py',
        'before_sha256': None, 'after_sha256': sha(data),
    })
    manifest['deployment_mode'] = 'Unified base launch; separate real motion/role launch; one robo04'
    (BUNDLE / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    (BUNDLE / 'README.md').write_bytes((ROOT / 'RESTART_3_TERMINALS_20261109.md').read_bytes())

    for entry in manifest['files']:
        payload = (BUNDLE / entry['payload']).read_bytes()
        assert sha(payload) == entry['after_sha256'], entry['payload']
        assert (SOURCE / entry['path']).read_bytes() == payload, entry['path']
    for path in BUNDLE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 10))
    for path in BUNDLE.glob('*.sh'):
        assert b'\r' not in path.read_bytes(), path
        subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(path)], check=True)

    deploy = ROOT / 'deploy'
    archive = deploy / 'real_robot_20261109.tar.gz'
    files = sorted(p for p in BUNDLE.iterdir() if p.is_file())
    with tarfile.open(archive, 'w:gz') as output:
        for path in files:
            payload = path.read_bytes()
            info = tarfile.TarInfo('real_robot_20261109/' + path.name)
            info.size = len(payload)
            info.mode = 0o755 if path.suffix == '.sh' else 0o644
            output.addfile(info, io.BytesIO(payload))
    with tarfile.open(archive, 'r:gz') as output:
        assert len(output.getmembers()) == len(files)
        for path in files:
            assert output.extractfile('real_robot_20261109/' + path.name).read() == path.read_bytes()
    digest = sha(archive.read_bytes())
    archive.with_name(archive.name + '.sha256').write_bytes(
        (digest + '  ' + archive.name + '\n').encode('ascii'))
    print('SOURCE IDENTITIES: %d verified' % len(manifest['files']))
    print('Python 3.10 AST / Bash syntax / LF checksum / archive readback: PASS')
    print('FILES:', len(files))
    print('ARCHIVE:', archive)
    print('SHA256:', digest)


if __name__ == '__main__':
    main()
