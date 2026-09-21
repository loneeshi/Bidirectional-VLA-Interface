#!/usr/bin/env bash
set -u
cd /home/pshuai/bvi-research/runs/goal-tools-paired16-20260920
export PYTHONHASHSEED=0
export MS_ASSET_DIR=/home/pshuai/bvi-research/assets
export CUDA_VISIBLE_DEVICES=GPU-b7ebba23-7824-7601-df32-be55628936c3
export PYTHONPATH=/home/pshuai/bvi-research/src/official-mshab-runtime/mshab:/home/pshuai/bvi-research/runs/goal-tools-paired16-20260920/source-v5/src
PY=/home/pshuai/bvi-research/envs/acdit/bin/python
BASE=(--source-manifest /home/pshuai/bvi-research/runs/sac-interface-baseline-20260920/manifest.json
      --output /home/pshuai/bvi-research/runs/goal-tools-paired16-20260920
      --checkpoint-root /home/pshuai/bvi-research/checkpoints/mshab
      --bridge-dir /home/pshuai/bvi-research/runs/sac-interface-baseline-20260920/bridge
      --authorization-id SAC-INTERFACE-BASELINE-20260920-BATCH01 --goal-tools --execute
      --max-new-per-arm 2)
run_chunk() {
  local name="$1"; shift
  echo "CHUNK_START $name $(date -u +%FT%TZ)" >> continuation-v5.log
  "$PY" source-v5/scripts/run_ppo_sac_paired16.py "${BASE[@]}" "$@" > "continuation-v5-${name}.log" 2>&1
  local rc=$?
  local state
  state=$($PY -c 'import json;print(json.load(open("panel-status.json"))["status"])')
  echo "CHUNK_END $name rc=$rc status=$state $(date -u +%FT%TZ)" >> continuation-v5.log
  [[ $rc -eq 0 && ( "$state" == chunk_complete || "$state" == finished ) ]]
}
test ! -e runner.lock || { echo "BLOCKED existing runner.lock" >> continuation-v5.log; exit 20; }
run_chunk duplicate-repair-02-03 --repair-duplicate-seeds 2 3 || exit 21
for n in 01 02 03 04 05 06 07 08; do
  status=$($PY -c 'import json;print(json.load(open("panel-status.json"))["status"])')
  [[ "$status" == finished ]] && exit 0
  run_chunk "panel-$n" || exit 22
done
status=$($PY -c 'import json;print(json.load(open("panel-status.json"))["status"])')
[[ "$status" == finished ]] || { echo "INCOMPLETE final_status=$status" >> continuation-v5.log; exit 23; }
