#!/usr/bin/env bash
# Build an isolated environment on a Kaggle (or Colab) GPU notebook using PlantCaduceus's own pinned
# versions (torch 2.5.1 / CUDA 12.4, transformers 4.49.0, mamba-ssm 2.2.4, causal-conv1d 1.5.0.post8),
# without touching the notebook's preinstalled packages. Run from the repository root.
set -euo pipefail
VENV=${VENV:-/tmp/prp-venv}
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV" 2>/dev/null || { pip install -q virtualenv && python3 -m virtualenv -q "$VENV"; }
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
"$VENV/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "No GPU visible. Kaggle: Settings > Accelerator > GPU T4 x2"
from mamba_ssm import Mamba
m = Mamba(d_model=16).cuda()
y = m(torch.randn(1, 64, 16, device="cuda"))
print(f"[setup] OK: torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
      f"compute capability {torch.cuda.get_device_capability(0)}, Mamba forward pass ran")
PY
