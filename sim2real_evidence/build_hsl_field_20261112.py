"""Package a photo-matched map and verify its inputs; no robot connection."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
bundle=root/'patches/hsl_field_20261112'
canonical = root.parent / 'MX3_Match_Sim'
if not (canonical / 'src').is_dir():
    canonical = Path('C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim')
sha=lambda data:hashlib.sha256(data).hexdigest()
telemetry=json.loads((root/'patches/hsl_view_20261112/manifest.json').read_text())
telemetry_by_name={Path(e['path']).name:e for e in telemetry['sources']}
paths=[
    'src/hsl_src/hsl_behavior/hsl_behavior/geometry.py',
    'src/hsl_src/hsl_world_model/include/hsl_world_model/field_model.hpp',
    'src/hsl_src/hsl_world_model/src/self_localizer_node.cpp',
    'src/hsl_src/hsl_behavior/hsl_behavior/behavior_node.py',
    'src/hsl_src/hsl_behavior/hsl_behavior/skills.py',
]
entries=[]
for relative in paths:
    name=Path(relative).name
    after=(canonical/relative).read_bytes()
    if name in telemetry_by_name:
        prior=telemetry_by_name[name]
        accepted={prior['before_sha256'],prior['after_sha256'],sha(after)}
    else:
        accepted={sha((bundle/'before'/name).read_bytes()),sha(after)}
    if name.endswith('.py'):
        ast.parse(after.decode('utf-8-sig'),feature_version=(3,10))
    (bundle/name).write_bytes(after)
    entries.append({'path':relative,'payload':name,'accepted_sha256':sorted(accepted),'after_sha256':sha(after)})
manifest={'version':'20261112-photo','sources':entries,
          'bundle_files':{f.name:sha(f.read_bytes()) for f in sorted(bundle.iterdir())
                          if f.is_file() and f.name!='manifest.json'}}
(bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
archive=root/'deploy/hsl_field_20261112.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for f in sorted(bundle.iterdir()):
        if not f.is_file():continue
        data=f.read_bytes();info=tarfile.TarInfo('hsl_field_20261112/'+f.name)
        info.size=len(data);info.mode=0o755 if f.suffix=='.sh' else 0o644;info.mtime=0
        tar.addfile(info,io.BytesIO(data))
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        assert tar.extractfile(member).read()==(bundle/Path(member.name).name).read_bytes()
digest=sha(archive.read_bytes())
archive.with_name(archive.name+'.sha256').write_bytes((digest+'  '+archive.name+'\n').encode('ascii'))
print(json.dumps({'archive':str(archive),'sha256':digest,'files':len(manifest['bundle_files'])+1},indent=2))
