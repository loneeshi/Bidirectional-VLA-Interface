#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pshuai/bvi-research/runs/standardized-teleport16-20260921
SOURCE="$ROOT/source-v2"
PANEL="$ROOT/panel"
export PYTHONPATH="$SOURCE/src"

for chunk in 1 2 3 4 5 6 7 8; do
  /home/pshuai/bvi-research/envs/acdit/bin/python \
    "$SOURCE/scripts/run_standardized_teleport16.py" \
    --source-manifest /home/pshuai/bvi-research/runs/sac-interface-baseline-20260920/manifest.json \
    --paired-panel /home/pshuai/bvi-research/runs/goal-tools-paired16-20260920 \
    --output "$PANEL" \
    --checkpoint-root /home/pshuai/bvi-research/checkpoints/mshab \
    --execute --max-new 2 >> "$ROOT/chunk-runs.log" 2>&1
  status=$(python3 -c "import json; print(json.load(open('$PANEL/panel-status.json'))['status'])")
  case "$status" in
    finished) exit 0 ;;
    chunk_complete) ;;
    *) echo "unexpected panel status: $status" >&2; exit 2 ;;
  esac
done

echo "supervisor exhausted chunks before panel completion" >&2
exit 3
