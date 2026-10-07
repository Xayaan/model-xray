#!/usr/bin/env bash
# Prepare a fresh Lium pod: install model-xray and, when they build, the fast Mamba kernels.
#   bash lium/remote_setup.sh [python]
set -euo pipefail
PY=${1:-python3}
cd /workspace/model-xray

"$PY" -m pip install -q --upgrade pip
# Lium's PyTorch images ship a Debian-managed `cryptography` that pip cannot upgrade in place.
"$PY" -m pip install -q --ignore-installed cryptography
"$PY" -m pip install -q -e .

# Optional: fused Mamba CUDA kernels. Their source builds compile for six GPU generations and take
# well over 25 minutes, so they are off by default; the PyTorch path ran Nemotron 3 Nano 4B at
# ~12.8k tokens/s on an H100. Set BUILD_MAMBA_KERNELS=1 to build them anyway.
if [ "${BUILD_MAMBA_KERNELS:-0}" = 1 ] && ! "$PY" -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then
  echo "building Mamba kernels (causal-conv1d, mamba-ssm)..."
  MAX_JOBS=${MAX_JOBS:-16} "$PY" -m pip install -q --no-build-isolation causal-conv1d mamba-ssm \
    || echo "WARNING: Mamba kernels unavailable; using the PyTorch path"
fi

"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0))
try:
    import mamba_ssm, causal_conv1d
    print("mamba kernels: yes")
except ImportError:
    print("mamba kernels: no")
EOF
