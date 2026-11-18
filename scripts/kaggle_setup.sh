#!/usr/bin/env bash
# Build an isolated environment on a Kaggle (or Colab) GPU notebook that matches PlantCaduceus's tested
# setup: Python 3.11, torch 2.5.1 (CUDA 12.4), transformers 4.49.0, mamba-ssm 2.2.4, causal-conv1d
# 1.5.0.post8. The notebook's own Python (3.13 on Kaggle as of Nov 2026) cannot be used: torch 2.5.1
# ships no Triton for 3.13, and mamba-ssm imports Triton. uv downloads a standalone Python 3.11.
# Run from the repository root.
set -euo pipefail
# Kaggle puts its own sitecustomize on PYTHONPATH; it must not leak into a different Python version.
unset PYTHONPATH
VENV=${VENV:-/tmp/prp-venv}
PYVER=${PYVER:-3.11}

if [[ -x "$VENV/bin/python" ]]; then
  have=$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  if [[ "$have" != "$PYVER" ]]; then
    echo "[setup] removing old environment built on Python $have"
    rm -rf "$VENV"
  fi
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[setup] Python $PYVER via uv"
  python3 -m pip install -q uv
  python3 -m uv python install "$PYVER"
  python3 -m uv venv --seed -q -p "$PYVER" "$VENV"
fi
PIP="$VENV/bin/pip"
"$PIP" install -q --upgrade pip setuptools wheel packaging ninja
echo "[setup] torch 2.5.1 (CUDA 12.4)"
"$PIP" install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
echo "[setup] transformers 4.49.0"
"$PIP" install -q transformers==4.49.0
echo "[setup] mamba-ssm 2.2.4 + causal-conv1d 1.5.0.post8 (if this takes >5 min it is compiling; let it finish)"
"$PIP" install -q --no-build-isolation causal-conv1d==1.5.0.post8 mamba-ssm==2.2.4
echo "[setup] plant-read-placement"
"$PIP" install -q -e .

# Run prp with a clean PYTHONPATH even when the notebook calls it directly.
if [[ ! -e "$VENV/bin/prp-real" ]]; then
  mv "$VENV/bin/prp" "$VENV/bin/prp-real"
  printf '#!/usr/bin/env bash\nunset PYTHONPATH\nexec "%s/bin/prp-real" "$@"\n' "$VENV" > "$VENV/bin/prp"
  chmod +x "$VENV/bin/prp"
fi

"$VENV/bin/python" - <<'PY'
import sys
import torch
import triton
assert torch.cuda.is_available(), "No GPU visible. Kaggle: Settings > Accelerator > GPU T4 x2"
from mamba_ssm import Mamba
m = Mamba(d_model=16).cuda()
m(torch.randn(1, 64, 16, device="cuda"))
print(f"[setup] OK: Python {sys.version.split()[0]}, torch {torch.__version__}, triton {triton.__version__}, "
      f"{torch.cuda.get_device_name(0)} (compute capability {torch.cuda.get_device_capability(0)}), Mamba forward pass ran")
PY
