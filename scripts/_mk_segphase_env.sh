#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
echo START $(date -Is)
if conda env list | awk '{print $1}' | grep -qx segphase_eval; then
  echo exists
else
  conda create -y -n segphase_eval --clone PS
fi
conda run -n segphase_eval pip install -q einops torchinfo
conda run -n segphase_eval python -c "import torch,einops,torchinfo; print('OK', torch.__version__, torch.cuda.is_available())"
echo ENV_DONE
