"""Build/refresh `unitree_rl_lab_kick16/` inside the local Isaac-work-space clone (PC side, stdlib only).

Flow used with the training server:  PC -> push hion89/Isaac-work-space -> server `git pull` -> train.

  python scripts/assemble_kick16_isaac_ws.py --isaac-repo C:/Users/jes/Desktop/isaac_ws/isaac_ws

The folder = committed files of the existing `unitree_rl_lab/` (git ls-files, so local junk is not copied)
+ XM_Kick_Policy/unitree_rl_lab_overlay/ (kick V1-5B + V1-6) + kick16_init/model_22300.pt + rsl_rl patch
+ kick16_setup.sh + KICK16_README_KO.md. The existing `unitree_rl_lab/` is not modified.
It never commits or pushes; it prints the exact git add commands (binaries need -f because
unitree_rl_lab/.gitignore ignores *.usd / *.pt).
"""
import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KICK = REPO / "XM_Kick_Policy"
OVL = KICK / "unitree_rl_lab_overlay"
NAME = "unitree_rl_lab_kick16"

SETUP = r"""#!/usr/bin/env bash
# Server-side setup after `git pull` (run from ~/isaac_ws/unitree_rl_lab_kick16, Isaac Lab env active).
#   bash kick16_setup.sh
# Places the 22300 resume checkpoint for V1-6A/B/C, checks the rsl_rl patch (dry-run only), prints the
# install command. It does not pip install, does not edit site-packages, does not start training.
set -euo pipefail
cd "$(dirname "$0")"
CK=kick16_init/model_22300.pt
[ -f "$CK" ] || { echo "missing $CK (git pull incomplete?)"; exit 2; }
echo "checkpoint sha256: $(sha256sum "$CK" | cut -d' ' -f1)  (expected $(cat kick16_init/model_22300.pt.sha256))"
[ "$(sha256sum "$CK" | cut -d' ' -f1)" = "$(cat kick16_init/model_22300.pt.sha256)" ] || { echo "SHA MISMATCH"; exit 3; }
for e in a b c d e; do
  d="logs/rsl_rl/unitree_mx3_walkkick_v1_6$e/init_22300"
  mkdir -p "$d" && cp -n "$CK" "$d/model_22300.pt"
done
echo "resume checkpoints placed: logs/rsl_rl/unitree_mx3_walkkick_v1_6{a,b,c,d,e}/init_22300/"
RSL=$(python -c "import rsl_rl, os; print(os.path.dirname(rsl_rl.__file__))" 2>/dev/null || true)
if [ -z "$RSL" ]; then
  echo "rsl_rl not importable: activate the Isaac Lab conda env first"
elif patch --dry-run -R -p1 -d "$(dirname "$RSL")" < kick16_init/ppo_huber_nan_guard.diff >/dev/null 2>&1; then
  echo "rsl_rl patch: already applied ($RSL)"
elif patch --dry-run -p1 -d "$(dirname "$RSL")" < kick16_init/ppo_huber_nan_guard.diff >/dev/null 2>&1; then
  echo "rsl_rl patch: NOT applied. After review: patch -p1 -d $(dirname "$RSL") < kick16_init/ppo_huber_nan_guard.diff"
else
  echo "rsl_rl patch: dry-run failed (different rsl_rl layout) - check by hand"
fi
PKG=$(python -c "import unitree_rl_lab, os; print(os.path.dirname(unitree_rl_lab.__file__))" 2>/dev/null || true)
echo "unitree_rl_lab imported from: ${PKG:-<not installed>}"
case "$PKG" in
  */unitree_rl_lab_kick16/*) echo "OK: this env uses unitree_rl_lab_kick16" ;;
  *) echo "NOTE: this env does not use unitree_rl_lab_kick16 yet. In a cloned env:"
     echo "      pip install -e $(pwd)/source/unitree_rl_lab" ;;
esac
echo "KICK16_SETUP_DONE"
"""

README = """# unitree_rl_lab_kick16 — MX3 킥 V1-6 학습용 (서버에서 git pull 후 사용)

PC에서 `MX3-HSL-Robot-Soccer/scripts/assemble_kick16_isaac_ws.py`로 만든 폴더다. 직접 고치지 말고,
원본(MX3-HSL-Robot-Soccer의 XM_Kick_Policy)을 고친 뒤 다시 조립해 push 한다.

- 구성: 기존 `unitree_rl_lab/`의 커밋된 파일 + 킥 overlay(V1-5B·V1-6) + `kick16_init/`(22300 체크포인트·sha256·rsl_rl 패치).
- 기존 `../unitree_rl_lab/`과 기존 conda 환경은 그대로 둔다. 같은 conda 환경에 `pip install -e`를 하면
  그 환경의 unitree_rl_lab이 이 폴더로 바뀌므로 복제 환경을 쓴다.
- 태스크·근거·체크포인트 선택: MX3-HSL-Robot-Soccer `XM_Kick_Policy/KICK_V1_6_KO.md`,
  `sim2real_evidence/kick_robust_20261116/FINDINGS_KO.md`.

```bash
cd ~/isaac_ws && git pull --ff-only
conda create -n env_kick16 --clone env_isaaclab     # 처음 한 번
conda activate env_kick16
pip install -e ~/isaac_ws/unitree_rl_lab_kick16/source/unitree_rl_lab   # 처음 한 번
cd ~/isaac_ws/unitree_rl_lab_kick16 && bash kick16_setup.sh             # KICK16_SETUP_DONE 확인
SMOKE=1 bash scripts/kick_v1_6/train_v16.sh C                          # 64 env x 5 iter 확인
tmux new -s kick16
bash scripts/kick_v1_6/train_v16.sh C                                  # 본 학습
```
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_lf(p: Path, text: str):
    p.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isaac-repo", type=Path, required=True)
    a = ap.parse_args()
    isa = a.isaac_repo.resolve(strict=True)
    remote = subprocess.run(["git", "-C", str(isa), "remote", "get-url", "origin"], capture_output=True,
                            text=True, check=True).stdout.strip()
    if "Isaac-work-space" not in remote:
        raise SystemExit(f"origin is {remote}, expected hion89/Isaac-work-space")
    files = subprocess.run(["git", "-C", str(isa), "ls-files", "unitree_rl_lab"], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    if not files:
        raise SystemExit("no tracked files under unitree_rl_lab/")
    dst = isa / NAME
    if dst.exists():
        shutil.rmtree(dst)          # generated folder only (tracked content is recreated below)
    for rel in files:
        src = isa / rel
        out = dst / Path(rel).relative_to("unitree_rl_lab")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    n_base = len(files)
    n_ovl = 0
    for src in OVL.rglob("*"):
        if src.is_file() and "__pycache__" not in src.parts and src.suffix != ".pyc":
            out = dst / src.relative_to(OVL)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            n_ovl += 1
    # Isaac-work-space has never used Git LFS (server may lack git-lfs): store binaries as plain git blobs.
    ga = dst / ".gitattributes"
    kept = [ln for ln in ga.read_text(encoding="utf-8").splitlines() if "filter=lfs" not in ln] if ga.exists() else []
    kept += ["# kick16: no Git LFS (Isaac-work-space does not use it); keep binaries byte-exact",
             "*.usd -text", "*.usda -text", "*.usdc -text", "*.pt -text", "*.onnx -text", "*.diff -text",
             "*.sh text eol=lf"]
    write_lf(ga, "\n".join(kept) + "\n")
    init = dst / "kick16_init"
    init.mkdir()
    shutil.copy2(KICK / "policy" / "model_22300.pt", init / "model_22300.pt")
    write_lf(init / "model_22300.pt.sha256", sha256(KICK / "policy" / "model_22300.pt") + "\n")
    shutil.copy2(KICK / "rsl_rl_patch" / "ppo_huber_nan_guard.diff", init / "ppo_huber_nan_guard.diff")
    write_lf(dst / "kick16_setup.sh", SETUP)
    write_lf(dst / "KICK16_README_KO.md", README)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    write_lf(dst / "kick16_init" / "SOURCE.txt",
             f"assembled from hion89/MX3-HSL-Robot-Soccer {commit} (XM_Kick_Policy/unitree_rl_lab_overlay)\n"
             f"base: Isaac-work-space unitree_rl_lab tracked files ({n_base})\n")
    forced = [p for p in dst.rglob("*") if p.is_file() and p.suffix.lower() in (".usd", ".usda", ".usdc", ".pt")]
    print(f"ASSEMBLED {dst}: base {n_base} files + overlay {n_ovl} files + kick16_init")
    print("next (PC, Isaac-work-space): stage ONLY this folder, then commit/push:")
    print(f"  git -C {isa} add {NAME}")
    for p in forced:
        print(f"  git -C {isa} add -f {p.relative_to(isa).as_posix()}")


if __name__ == "__main__":
    main()
