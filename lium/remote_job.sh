#!/usr/bin/env bash
# Runs on the pod, detached: setup, a two-minute preflight on a tiny Nemotron, then the full
# pipeline on GPUS GPUs and the findings for CONFIG. Leaves /workspace/XRAY_DONE or XRAY_FAILED.
#   bash lium/remote_job.sh CONFIG [GPUS]
set -uo pipefail
CONFIG=${1:?usage: remote_job.sh CONFIG [GPUS]}
GPUS=${2:-1}
export PYTHONUNBUFFERED=1
cd /workspace/model-xray
rm -f /workspace/XRAY_DONE /workspace/XRAY_FAILED

PY=python3
for candidate in python python3; do
  if command -v "$candidate" >/dev/null && "$candidate" -c "import torch" 2>/dev/null; then
    PY=$(command -v "$candidate")
    break
  fi
done

stage() { echo; echo "===== $* ($(date -u +%H:%M:%S) UTC)"; }

run() {
  stage "setup" && bash lium/remote_setup.sh "$PY" &&
  stage "preflight" &&
  "$PY" -m xray.cli run configs/preflight-nemotron-tiny.yaml &&
  "$PY" -m xray.cli findings configs/preflight-nemotron-tiny.yaml --n 8 --steer-n 4 --scales 0 1 &&
  stage "train + eval + features + browser: $CONFIG on $GPUS GPU(s)" &&
  "$PY" -m xray.cli run "$CONFIG" --gpus "$GPUS" &&
  stage "findings" &&
  "$PY" -m xray.cli findings "$CONFIG" &&
  stage "preview" &&
  "$PY" -m xray.cli browse "$CONFIG" --sample 250 &&
  stage "all done"
}

if run; then
  touch /workspace/XRAY_DONE
else
  stage "FAILED"
  touch /workspace/XRAY_FAILED
fi
