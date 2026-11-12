"""Validate and package the standalone observer without changing robot sources."""
import ast
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile

import yaml


ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'patches/hsl_view_20261109'
SOURCE = Path('C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim/src/hsl_src')


def main():
    for path in BUNDLE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 10))
    for path in BUNDLE.glob('*.sh'):
        data = path.read_bytes().replace(b'\r\n', b'\n')
        path.write_bytes(data)
        subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(path)], check=True)
    config = yaml.safe_load((BUNDLE / 'hsl_view.rviz').read_text(encoding='utf-8'))
    assert config['Visualization Manager']['Global Options']['Fixed Frame'] == 'map'

    # Check the drawn coordinates against the current estimator's documented map dimensions.
    source = ast.parse((BUNDLE / 'hsl_view.py').read_text(encoding='utf-8'))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'field_segments')
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'field_segments', 'exec'), scope)
    import sys
    spec = importlib.util.spec_from_file_location('checked_hsl_geometry',
        SOURCE / 'hsl_behavior/hsl_behavior/geometry.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    field = module.Field.for_profile('practice')
    assert field.length == 3.9 and field.width == 2.6
    assert abs(field.goal_width - .78) < 1e-12
    assert abs(field.penalty_area_width - 4 * 3.9 / 9) < 1e-12
    segments = scope['field_segments'](field)
    assert len(segments) == 21
    assert (-1.95, -1.3, 1.95, -1.3) in segments
    assert any(abs(a + field.hl) < 1e-12 and abs(b - field.penalty_area_width / 2) < 1e-12
               for a, b, c, d in segments)

    archive = ROOT / 'deploy/hsl_view_20261109.tar.gz'
    paths = sorted(p for p in BUNDLE.iterdir() if p.is_file())
    with tarfile.open(archive, 'w:gz') as stream:
        for path in paths:
            data = path.read_bytes()
            member = tarfile.TarInfo('hsl_view_20261109/' + path.name)
            member.size = len(data)
            member.mode = 0o755 if path.suffix == '.sh' else 0o644
            stream.addfile(member, io.BytesIO(data))
    with tarfile.open(archive) as stream:
        for path in paths:
            assert stream.extractfile('hsl_view_20261109/' + path.name).read() == path.read_bytes()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + '.sha256').write_bytes(
        (digest + '  ' + archive.name + '\n').encode('ascii'))
    print('PASS: Python3.10 AST, Bash syntax, YAML, HSL field coordinates, archive readback')
    print('FILES:', len(paths))
    print('SHA256:', digest)


if __name__ == '__main__':
    main()
