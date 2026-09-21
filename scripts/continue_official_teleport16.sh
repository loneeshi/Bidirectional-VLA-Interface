#!/usr/bin/env bash
set -u

OUT=/home/pshuai/bvi-research/runs/official-teleport16-20260921
PY=/home/pshuai/bvi-research/envs/acdit/bin/python
RUNNER=/home/pshuai/bvi-research/runs/run_official_teleport16.py
export CUDA_VISIBLE_DEVICES=GPU-b7ebba23-7824-7601-df32-be55628936c3
export MS_ASSET_DIR=/home/pshuai/bvi-research/assets
export PYTHONHASHSEED=0

run_chunk() {
  local number="$1"
  echo "CHUNK_START $number $(date -u +%FT%TZ)" >> "$OUT/continuation.log"
  "$PY" "$RUNNER" \
    --source-manifest /home/pshuai/bvi-research/runs/sac-interface-baseline-20260920/manifest.json \
    --task-plans /home/pshuai/bvi-research/assets/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json \
    --paper-source /home/pshuai/bvi-research/src/mshab-paper-4729821 \
    --checkpoint-root /home/pshuai/bvi-research/checkpoints/mshab \
    --output "$OUT" --max-new 2 --episode-timeout 1200 --execute \
    > "$OUT/chunk-$number.log" 2>&1
  local rc=$?
  local status
  status=$($PY -c 'import json;print(json.load(open("/home/pshuai/bvi-research/runs/official-teleport16-20260921/panel-status.json"))["status"])')
  echo "CHUNK_END $number rc=$rc status=$status $(date -u +%FT%TZ)" >> "$OUT/continuation.log"
  [[ $rc -eq 0 && ( "$status" == chunk_complete || "$status" == finished ) ]]
}

test ! -e "$OUT/runner.lock" || exit 20
for number in 01 02 03 04 05 06 07 08; do
  status=$($PY -c 'import json;print(json.load(open("/home/pshuai/bvi-research/runs/official-teleport16-20260921/panel-status.json"))["status"])')
  [[ "$status" == finished ]] && exit 0
  run_chunk "$number" || exit 21
done
status=$($PY -c 'import json;print(json.load(open("/home/pshuai/bvi-research/runs/official-teleport16-20260921/panel-status.json"))["status"])')
[[ "$status" == finished ]] || exit 22
