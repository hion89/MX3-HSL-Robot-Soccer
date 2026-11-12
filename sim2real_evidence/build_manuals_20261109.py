"""Check manual command syntax and package documentation for sharing."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parent
MANUALS = ROOT / 'manuals'
NAMES = ['ROBOT_OPERATION_MANUAL_KO_20261109.md', 'BHUMAN_PORT_AUDIT_KO_20261109.md',
         'ROBOT_FLEET_TEMPLATE_20261109.csv', 'BHUMAN_OFFICIAL_REVIEW_20261109.json']


def main():
    commands = 0
    for name in NAMES[:2]:
        text = (MANUALS / name).read_text(encoding='utf-8')
        assert '\ufffd' not in text
        for block in re.findall(r'```bash\n(.*?)```', text, flags=re.S):
            subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n'],
                           input=block, text=True, encoding='utf-8', check=True)
            commands += 1
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if not target.startswith(('http://', 'https://')):
                assert (MANUALS / target).is_file(), target
    fleet = list(csv.DictReader(io.StringIO((MANUALS / NAMES[2]).read_text(encoding='utf-8'))))
    assert len(fleet) == 8
    assert len({(r['team_number'], r['player_number']) for r in fleet}) == 8
    assert all(None not in r for r in fleet)
    assert all(r['ros_localhost_only'] == '1' for r in fleet)
    review = json.loads((MANUALS / NAMES[3]).read_text(encoding='utf-8'))
    assert all(f['fetched'] for f in review['files'])
    checksums = {name: hashlib.sha256((MANUALS / name).read_bytes()).hexdigest() for name in NAMES}
    (MANUALS / 'MANIFEST_20261109.json').write_text(
        json.dumps({'files': checksums, 'bash_blocks_checked': commands,
                    'runtime_source_changed': False}, indent=2) + '\n',
        encoding='utf-8', newline='\n')
    archive = ROOT / 'deploy/robot_manuals_20261109.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for name in NAMES + ['MANIFEST_20261109.json']:
            output.write(MANUALS / name, arcname='robot_manuals_20261109/' + name)
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
        for name in NAMES:
            assert output.read('robot_manuals_20261109/' + name) == (MANUALS / name).read_bytes()
    print('PASS: UTF-8, %d Bash blocks, document links, 8 robot identities, frozen upstream review, ZIP readback' % commands)
    print('ARTIFACT:', archive)


if __name__ == '__main__':
    main()
