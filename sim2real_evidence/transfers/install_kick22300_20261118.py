"""Copy a hash-verified v1_5b checkpoint into an isolated server log directory."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def install(bundle, target):
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    paths = []
    for relative, digest in manifest['files'].items():
        path = bundle / relative
        if not path.resolve().is_relative_to(bundle.resolve()):
            raise RuntimeError('Invalid manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Payload checksum mismatch: ' + relative)
        paths.append(relative)
    if target.exists():
        for relative in paths:
            path = target / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][relative]:
                raise RuntimeError('Existing import differs; nothing overwritten: ' + str(target))
        print('ALREADY_IMPORTED:', target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.import_22300_', dir=target.parent))
    for relative in paths:
        out = stage / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / relative, out)
        if hashlib.sha256(out.read_bytes()).hexdigest() != manifest['files'][relative]:
            raise RuntimeError('Copy checksum mismatch')
    shutil.copy2(bundle / 'manifest.json', stage / 'manifest.json')
    stage.rename(target)
    print('IMPORTED_AND_HASH_VERIFIED:', target)
    print('CHECKPOINT:', target / 'model_22300.pt')
    print('ONNX:', target / 'walkkick_v1_5b_22300.onnx')
    print('Original v1_5b weights; no training started and no robot files changed.')


if __name__ == '__main__':
    install(Path(__file__).resolve().parent,
            Path('/home/bami122/isaac_ws/unitree_rl_lab_kick16/logs/rsl_rl/'
                 'unitree_mx3_walkkick_v1_6c/import_v1_5b_22300_20261118'))
