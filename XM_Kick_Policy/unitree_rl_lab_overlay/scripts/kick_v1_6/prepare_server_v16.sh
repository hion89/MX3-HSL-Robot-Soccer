#!/usr/bin/env bash
# Prepare a SEPARATE unitree_rl_lab copy on the training server for WalkKick V1-6 (A/B/C).
#
#   bash prepare_server_v16.sh <XM_Kick_Policy dir> <target unitree_rl_lab dir>
#
# What it does (and does not do):
#   * refuses to touch the target if it has uncommitted git changes (set ALLOW_DIRTY=1 to override)
#   * records git remote / HEAD / status of the target into <target>/logs/kick_v16/prepare_<ts>.log
#   * tar-backs-up every target file the overlay would overwrite -> <target>/logs/kick_v16/backup_<ts>.tar.gz
#   * copies XM_Kick_Policy/unitree_rl_lab_overlay/. into the target (MX3/__init__.py: see below)
#   * places model_22300.pt as the resume point of the three experiments
#   * checks (dry-run only) whether the rsl_rl Huber/NaN patch is applied; it never edits site-packages
#   * does NOT start training, does NOT pip install, does NOT touch the robot
set -euo pipefail

PKG=${1:?usage: prepare_server_v16.sh <XM_Kick_Policy dir> <target unitree_rl_lab dir>}
TGT=${2:?usage: prepare_server_v16.sh <XM_Kick_Policy dir> <target unitree_rl_lab dir>}
PKG=$(cd "$PKG" && pwd)
TGT=$(cd "$TGT" && pwd)
OVL="$PKG/unitree_rl_lab_overlay"
TS=$(date +%Y%m%d_%H%M%S)

[ -d "$OVL/source/unitree_rl_lab" ] || { echo "ERROR: $OVL is not the kick overlay"; exit 2; }
[ -d "$TGT/source/unitree_rl_lab" ] || { echo "ERROR: $TGT does not look like unitree_rl_lab"; exit 2; }
[ -f "$PKG/policy/model_22300.pt" ] || { echo "ERROR: $PKG/policy/model_22300.pt missing"; exit 2; }

# dirty check BEFORE this script writes anything into the target
if git -C "$TGT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ -n "$(git -C "$TGT" status --porcelain --untracked-files=normal -- . ':!logs')" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "ERROR: target has uncommitted changes. Use a separate copy (see KICK_V1_6_KO.md) or ALLOW_DIRTY=1."
    git -C "$TGT" status --short | head -20
    exit 3
  fi
fi
META="$TGT/logs/kick_v16"
mkdir -p "$META"
LOG="$META/prepare_$TS.log"

{
  echo "== kick v1_6 prepare $TS"
  echo "package: $PKG"
  echo "target : $TGT"
  if git -C "$TGT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$TGT" remote -v || true
    git -C "$TGT" log -1 --oneline || true
    git -C "$TGT" status --short | head -50 || true
  else
    echo "(target is not a git work tree)"
  fi
  sha256sum "$PKG/policy/model_22300.pt" "$PKG/policy/policy.onnx"
} | tee "$LOG"

INIT_REL="source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/MX3/__init__.py"
if [ -f "$TGT/$INIT_REL" ] && ! cmp -s "$TGT/$INIT_REL" "$OVL/$INIT_REL" && [ "${OVERWRITE_MX3_INIT:-0}" != "1" ]; then
  echo "ERROR: target already has a different robots/MX3/__init__.py (other MX3 tasks registered?)." | tee -a "$LOG"
  echo "       Merge the V1-5/V1-5B/V1-6 blocks by hand, or rerun with OVERWRITE_MX3_INIT=1 after review." | tee -a "$LOG"
  exit 4
fi

# backup files that will be overwritten
LIST=$(mktemp)
(cd "$OVL" && find . -type f) | while read -r f; do
  if [ -f "$TGT/$f" ]; then echo "$f"; fi
done > "$LIST"
N=$(wc -l < "$LIST")
if [ "$N" -gt 0 ]; then
  tar -C "$TGT" -czf "$META/backup_$TS.tar.gz" -T "$LIST"
  echo "backup: $N files -> logs/kick_v16/backup_$TS.tar.gz (restore: tar -C <target> -xzf ...)" | tee -a "$LOG"
fi
rm -f "$LIST"

cp -r "$OVL/." "$TGT/"
echo "overlay copied" | tee -a "$LOG"

for e in a b c; do
  d="$TGT/logs/rsl_rl/unitree_mx3_walkkick_v1_6$e/init_22300"
  mkdir -p "$d"
  cp -n "$PKG/policy/model_22300.pt" "$d/model_22300.pt"
done
echo "resume checkpoints: logs/rsl_rl/unitree_mx3_walkkick_v1_6{a,b,c}/init_22300/model_22300.pt" | tee -a "$LOG"

# rsl_rl patch check (dry-run only)
RSL=$(python -c "import rsl_rl, os; print(os.path.dirname(rsl_rl.__file__))" 2>/dev/null || true)
if [ -n "$RSL" ] && command -v patch >/dev/null; then
  if patch --dry-run -R -p1 -d "$(dirname "$RSL")" < "$PKG/rsl_rl_patch/ppo_huber_nan_guard.diff" >/dev/null 2>&1; then
    echo "rsl_rl patch: already applied ($RSL)" | tee -a "$LOG"
  elif patch --dry-run -p1 -d "$(dirname "$RSL")" < "$PKG/rsl_rl_patch/ppo_huber_nan_guard.diff" >/dev/null 2>&1; then
    echo "rsl_rl patch: NOT applied, applies cleanly. Review, then: patch -p1 -d $(dirname "$RSL") < $PKG/rsl_rl_patch/ppo_huber_nan_guard.diff" | tee -a "$LOG"
  else
    echo "rsl_rl patch: dry-run failed (different rsl_rl version or path layout) — check by hand" | tee -a "$LOG"
  fi
else
  echo "rsl_rl patch: could not locate rsl_rl in this python (activate the Isaac Lab env first)" | tee -a "$LOG"
fi
python -c "import isaaclab, importlib.metadata as m; print('isaaclab', getattr(isaaclab,'__version__','?'))" 2>/dev/null | tee -a "$LOG" || true
echo "PREPARE_V16_DONE log=$LOG" | tee -a "$LOG"
