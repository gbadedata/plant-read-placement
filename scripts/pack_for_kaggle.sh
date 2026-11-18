#!/usr/bin/env bash
# Zip work/windows (the only input the GPU steps need) for upload as a private Kaggle dataset.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f work/windows/train.npz ]] || { echo "work/windows/train.npz missing: run 'prp windows' first"; exit 1; }
OUT=${1:-$HOME/prp_windows.zip}
rm -f "$OUT"
(cd work && python -m zipfile -c "$OUT" windows)
ls -lh "$OUT"
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "Open it from Windows at: \\\\wsl.localhost\\${WSL_DISTRO_NAME:-Ubuntu}$(echo "$OUT" | tr '/' '\\')"
fi
