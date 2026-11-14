#!/usr/bin/env bash
# Create the `prp` conda environment. CPU tools by default; `--gpu` adds PlantCaduceus dependencies.
set -euo pipefail
GPU=0
[[ "${1:-}" == "--gpu" ]] && GPU=1
ENV=prp
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
  conda create -y -n "$ENV" python=3.11
fi
conda activate "$ENV"
conda install -y -c conda-forge ncbi-datasets-cli

if [[ $GPU == 1 ]]; then
  # PlantCAD's own pinned requirements first, so torch matches the mamba-ssm build they support.
  rm -rf /tmp/PlantCaduceus && git clone -q --depth 1 https://github.com/kuleshov-group/PlantCaduceus.git /tmp/PlantCaduceus
  pip install -r /tmp/PlantCaduceus/requirements.txt
  pip install -e ".[plantcad,dev]"
  pip install mamba-ssm==2.2.4 causal-conv1d==1.5.0.post8 --no-build-isolation
else
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -e ".[dev]"
fi
pip install git+https://github.com/rrwick/Badread.git

if [[ $GPU == 1 ]]; then
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not visible to PyTorch: check nvidia-smi and the torch build"
from mamba_ssm import Mamba  # noqa: F401
print("GPU ready:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
PY
fi
prp --version
badread --version
datasets --version
echo "Environment '$ENV' ready. Run: conda activate $ENV && pytest"
