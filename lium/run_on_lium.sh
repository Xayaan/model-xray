#!/usr/bin/env bash
# Run an xray config on a rented Lium GPU and bring the results home.
#
#   lium/run_on_lium.sh configs/nemotron3-nano-4b-pilot.yaml
#
# Environment:
#   GPU       GPU type to rent (default RTX5090)       COUNT     GPUs per pod: 1, 2, 4 or 8 (default 1)
#   BUDGET    stop spending after this many dollars     TTL       ...or after this long (default 4h)
#             (default $3)
#   TEMPLATE  Lium template id (default: PyTorch 2.11 + CUDA 12.8 devel)
#   NAME      pod name                                  KEEP_POD=1  leave the pod running at the end
#
# Lium removes the pod by itself once BUDGET or TTL is reached, whichever comes first, so a
# forgotten run cannot keep billing. The job runs detached on the pod (lium/remote_job.sh), so a
# dropped connection does not stop it; this script polls it, downloads the results, then removes the pod.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:?usage: lium/run_on_lium.sh CONFIG.yaml}
GPU=${GPU:-RTX5090}
COUNT=${COUNT:-1}
BUDGET=${BUDGET:-3}
TTL=${TTL:-4h}
TEMPLATE=${TEMPLATE:-cbc0ffdc-aa35-441a-abaa-170a87c51a64}
NAME=${NAME:-xray-$(date +%m%d-%H%M%S)}
RUN=$(awk '/^name:/ {print $2; exit}' "$CONFIG")
REMOTE=/workspace/model-xray

cleanup() {
  if [ "${KEEP_POD:-0}" != 1 ]; then
    echo "Removing pod $NAME"
    lium rm "$NAME" --yes --name-only || true
  fi
}
trap cleanup EXIT
trap 'echo; echo "Stopped watching. $NAME keeps running until its budget or TTL; remove it with: lium rm $NAME --yes"; KEEP_POD=1; exit 130' INT

echo "Renting ${COUNT}x $GPU as $NAME (budget \$$BUDGET, TTL $TTL)"
lium up --gpu "$GPU" -c "$COUNT" -t "$TEMPLATE" --name "$NAME" --budget "$BUDGET" --ttl "$TTL" -y --json > ".lium-$NAME.json"

lium rsync "$NAME" ./ "$REMOTE" --exclude .venv --exclude runs --exclude .git --exclude __pycache__ \
  --exclude .pytest_cache --exclude '*.tgz' --exclude '.lium-*.json'
lium exec "$NAME" -d --log /workspace/xray.log "bash $REMOTE/lium/remote_job.sh $CONFIG $COUNT"

echo "Job started on $NAME. Progress every 2 minutes; Ctrl-C stops watching without removing the pod."
fails=0
while true; do
  sleep 120
  if ! status=$(lium exec "$NAME" "if [ -e /workspace/XRAY_DONE ]; then echo XRAY_DONE; elif [ -e /workspace/XRAY_FAILED ]; then echo XRAY_FAILED; else tr '\r' '\n' < /workspace/xray.log | grep -v -i warn | tail -n 1; fi" 2>/dev/null); then
    fails=$((fails + 1))
    echo "  (could not reach $NAME, attempt $fails)"
    if [ "$fails" -ge 5 ]; then echo "Lost contact with $NAME; it may have hit its budget or TTL."; exit 1; fi
    continue
  fi
  fails=0
  case "$status" in
    *XRAY_DONE*|*XRAY_FAILED*) break ;;
    *) echo "  $(date +%H:%M) $status" ;;
  esac
done

lium exec "$NAME" "mkdir -p $REMOTE/runs/$RUN && cp /workspace/xray.log $REMOTE/runs/$RUN/remote.log; tar czf /workspace/xray-$RUN.tgz -C $REMOTE runs"
lium scp "$NAME" "/workspace/xray-$RUN.tgz" ./ -d
tar xzf "xray-$RUN.tgz" && rm "xray-$RUN.tgz"

if [[ "$status" == *XRAY_FAILED* ]]; then
  echo "The job failed; see runs/$RUN/remote.log"
  exit 1
fi
echo "Done: runs/$RUN"
echo "Browse it with: python -m http.server -d runs/$RUN/features 8000  (then open http://localhost:8000)"
