#!/bin/bash
set -euo pipefail
source /workspace/bvi/activate.sh
cd /workspace/bvi/code
python scripts/run_coordinator.py \
  --organizer --tool-family-interface --organizer-slice-steps 40 \
  --navigation-policy lightnav --navigation-camera fetch_nav \
  --manipulation-policy fetch-pi05 --manipulation-chunk-steps 1 \
  --workspace-camera --seed 1 --policy-type rl_per_obj \
  --expected-plan-uid tidy_house-sequential-val-90-0 --stop-after-subtasks 4 \
  --max-navigation-predictions 80 --max-manipulation-predictions 200 \
  --max-calls 16 --max-env-steps 650 --max-wall-seconds 900 --skill-wall-seconds 90 \
  --checkpoint-root /workspace/bvi/mshab_checkpoints --output /workspace/bvi/runs/mshab011 \
  --transport bridge --bridge-timeout-seconds 120 --provider openai --model gpt-5.6-luna \
  --authorization-id MSHAB011 --max-api-cost-usd .32 --request-cost-ceiling-usd .02 \
  --max-output-tokens 700 --max-input-bytes 200000 --image-detail low --reasoning-effort none
