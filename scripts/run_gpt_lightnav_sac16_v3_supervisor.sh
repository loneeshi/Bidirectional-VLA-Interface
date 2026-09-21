#!/usr/bin/env bash
set -euo pipefail

out=/home/pshuai/bvi-research/runs/gpt-lightnav-sac16-20260921-rerun-v3
src="$out/source-v3"
panel="$out/panel"
python=/home/pshuai/bvi-research/envs/acdit/bin/python
export PYTHONPATH="$src/src:$src/scripts"

for chunk in $(seq 1 8); do
    if [ -f "$panel/panel-status.json" ]; then
        status=$($python -c "import json; print(json.load(open('$panel/panel-status.json'))['status'])")
        case "$status" in
            finished)
                printf 'finished\n' > "$out/supervisor-status.txt"
                exit 0
                ;;
            stopped_infrastructure|interrupted|blocked_*)
                printf '%s\n' "$status" > "$out/supervisor-status.txt"
                exit 2
                ;;
        esac
    fi
    if ! ss -ltn '( sport = :8050 )' | grep -q LISTEN; then
        printf 'lightnav_not_listening\n' > "$out/supervisor-status.txt"
        exit 3
    fi
    "$python" "$src/scripts/run_gpt_lightnav_sac16.py" \
        --source-manifest /home/pshuai/bvi-research/runs/sac-interface-baseline-20260920/manifest.json \
        --reference-panel /home/pshuai/bvi-research/runs/goal-tools-paired16-20260920/panel-status.json \
        --output "$panel" \
        --checkpoint-root /home/pshuai/bvi-research/checkpoints/mshab \
        --bridge-dir "$out/bridge" \
        --authorization-id GPT-LIGHTNAV-SAC16-20260921 \
        --lightnav-url ws://127.0.0.1:8050 \
        --max-new 2 --execute > "$out/chunk-$(printf '%03d' "$chunk").log" 2>&1
done

status=$($python -c "import json; print(json.load(open('$panel/panel-status.json'))['status'])")
printf '%s\n' "$status" > "$out/supervisor-status.txt"
test "$status" = finished
