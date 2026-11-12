"""Build reviewed viewer/telemetry bundle; do not connect to a robot."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
bundle = root / 'patches/hsl_view_20261112'
canonical = root.parent / 'MX3_Match_Sim'
if not (canonical / 'src').is_dir():
    canonical = Path('C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim')
sha = lambda data: hashlib.sha256(data).hexdigest()
sources = []
with tarfile.open(root / 'deploy/mx3_hsl_robot_20261103.tar.gz') as old_archive:
    for name in ('behavior_node.py', 'skills.py'):
        relative = 'src/hsl_src/hsl_behavior/hsl_behavior/' + name
        after = (canonical / relative).read_bytes()
        if name == 'behavior_node.py':
            before = (root / 'patches/real_robot_20261109/behavior_node.py').read_bytes()
        else:
            before = old_archive.extractfile(relative).read()
        ast.parse(after.decode('utf-8-sig'))
        (bundle / name).write_bytes(after)
        sources.append({'path': relative, 'payload': name,
                        'before_sha256': sha(before), 'after_sha256': sha(after)})
manifest = {'version': '20261112', 'sources': sources,
            'bundle_files': {f.name: sha(f.read_bytes()) for f in sorted(bundle.iterdir())
                             if f.is_file() and f.name != 'manifest.json'}}
(bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
archive = root / 'deploy/hsl_view_20261112.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for file in sorted(bundle.iterdir()):
        if not file.is_file():
            continue
        data = file.read_bytes()
        info = tarfile.TarInfo('hsl_view_20261112/' + file.name)
        info.size = len(data)
        info.mode = 0o755 if file.suffix == '.sh' else 0o644
        info.mtime = 0
        tar.addfile(info, io.BytesIO(data))
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        assert tar.extractfile(member).read() == (bundle / Path(member.name).name).read_bytes()
digest = sha(archive.read_bytes())
archive.with_name(archive.name+'.sha256').write_bytes((digest+'  '+archive.name+'\n').encode('ascii'))
print(json.dumps({'archive': str(archive), 'sha256': digest,
                  'files': len(manifest['bundle_files'])+1}, indent=2))
