"""Copy deployment evidence and progress from the companion workspace, without deletion."""
import argparse
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '.analysis_env', '.venv', 'venv', '__pycache__',
             'build', 'install', 'log', 'runtime_logs', 'chrome_profile', '.pytest_cache'}
SKIP_SUFFIXES = {'.pyc', '.pyo', '.db3', '.mcap', '.mp4', '.avi', '.pem', '.key'}


def copy_tree(source, destination):
    changed = 0
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if not path.is_file() or path.is_symlink() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.name == '.env' or path.name.startswith('.env.'):
            continue
        if 'credential' in path.name.lower() or path.name.startswith(('id_rsa', 'id_ed25519')):
            continue
        target = destination / relative
        if target.exists() and target.read_bytes() == path.read_bytes():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        changed += 1
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=REPO.parent / 'isaac_ws')
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    evidence = workspace / 'sim2real_evidence'
    if not evidence.is_dir():
        raise SystemExit('Missing companion sim2real_evidence directory')
    if evidence.resolve() == (REPO / 'sim2real_evidence').resolve():
        raise SystemExit('Source and destination must differ')
    changed = copy_tree(evidence, REPO / 'sim2real_evidence')
    progress = workspace / 'isaac_ws/sim2real_ws/docs'
    for name in ('MX3_HSL_DEPLOYMENT_PROGRESS.md', 'MX3_HSL_REAL_DEPLOYMENT.md'):
        source = progress / name
        if source.is_file():
            target = REPO / 'docs' / name
            target.parent.mkdir(exist_ok=True)
            if not target.exists() or target.read_bytes() != source.read_bytes():
                shutil.copy2(source, target)
                changed += 1
    print('SYNCED_CHANGED_FILES:', changed)
    print('No files deleted; review git status before staging and publishing.')


if __name__ == '__main__':
    main()
