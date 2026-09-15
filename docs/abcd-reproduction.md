# Reproduce the first-object A/B/C/D diagnostic

These commands run real GPU simulation with an oracle dispatcher and zero paid
VLM API requests. They are not the five-object benchmark evaluation. First follow
[environment setup](reproduction.md). Use separate simulator, LightNav, and
openpi Python environments: their NumPy, PyTorch, and Transformers versions differ.

The verified simulator uses Python 3.11, torch 2.4.1+cu124, NumPy 1.26.4,
OpenCV 4.11.0.86 and SAPIEN 3.0.0b1 on an RTX A6000. Source pins are MS-HAB
`e9ff3d23496d38e4431c8d913e147ffa007f7f72`, ManiSkill
`17121e3f96e3ee3ed0c03610b17f8bc2864617af`, LightNav
`c6f40e3220edbf7011e4f17eaf2c865416737d4d`, and openpi
`215abfb217dbac7d5f1273282331b9b1866c0479`.

## Simulator commands

Run from this repository in the simulator environment. Set `MS_ASSET_DIR` and
`MSHAB_CHECKPOINT_DIR` to the downloaded assets and official checkpoints.
Always select a fresh output directory; retain failed attempts.

```bash
COMMON=(--dry-run --seed 1 --policy-type rl_per_obj
  --navigation-camera fetch_nav --max-calls 4 --max-env-steps 1200
  --max-wall-seconds 900 --skill-wall-seconds 240
  --expected-plan-uid tidy_house-sequential-val-90-0
  --checkpoint-root "$MSHAB_CHECKPOINT_DIR")
NAV=(--navigation-policy lightnav
  --navigation-instructions configs/lightnav-seed1-disambiguated.json
  --navigation-recovery-instructions configs/lightnav-seed1-orientation.json
  --max-navigation-predictions 250)
MANIP=(--manipulation-policy fetch-pi05 --manipulation-chunk-steps 1
  --max-manipulation-predictions 200 --fetch-pi-url ws://127.0.0.1:8051)

# A: official PPO + SAC. B: LightNav + SAC.
python scripts/run_coordinator.py "${COMMON[@]}" --output runs/A-new
python scripts/run_coordinator.py "${COMMON[@]}" "${NAV[@]}" --output runs/B-new

# C/D are experimental until their actual results pass the task checks.
python scripts/run_coordinator.py "${COMMON[@]}" "${MANIP[@]}" --output runs/C-new
python scripts/run_coordinator.py "${COMMON[@]}" "${NAV[@]}" "${MANIP[@]}" --output runs/D-new
```

A3 and B11 are the published successful recordings; the common command above
uses a slightly larger runtime ceiling than A3. The explicit per-run metadata
and event files are the authoritative executed configuration. B11 used zero
orientation recoveries. Neither hidden goal coordinates nor a navigation-policy
fallback are used by LightNav. PPO/SAC retain their official privileged inputs.

## Start the actual model services

In the isolated LightNav environment, with the pinned VLN checkpoint downloaded:

```bash
lightnav-serve --task vln --model_path "$LIGHTNAV_MODEL_DIR" --backend hf \
  --host 127.0.0.1 --max_batch_size 1 --ready_file "$NEW_READY_PATH" \
  --record_dir "$NEW_RECORD_DIR"
```

Use a new ready path each launch. Wait for the actual READY log and successful
connection before running B/D. The ready marker is **empty**, so use `test -f`,
not `test -s`. Cold startup on this host took about a minute.

In the openpi environment, after the Fetch conversion/training steps in
[the adaptation notes](abcd-debugging.md):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.40 WANDB_MODE=disabled \
  python /path/to/this/repo/scripts/fetch_openpi.py serve \
  --repo-id bvi/fetch-seed1-state-v3 --work "$FETCH_WORK_DIR" \
  --openpi-root "$OPENPI_REPO_DIR" \
  --checkpoint "$FETCH_WORK_DIR/checkpoints/pi05_fetch_lora_state/seed1-diagnostic/1999"
```

Use checkpoint `1999` only after the 2,000-step training finishes successfully.
The server metadata must declare active state conditioning, the Fetch 15-state /
13-action contract, and the actual checkpoint path. The earlier 200-step pilot
has a different, vision-only configuration and failed. A DROID checkpoint is not
compatible with this adapter. First JAX inference also incurs compilation time.

The inference memory fraction permits the two model servers and one simulator
to share the 48GB A6000; recheck real memory before launching. Run training by
itself, then release its process before starting model servers. Service process
timeouts and simulator timeouts are not resource billing stops.

## Resource deadline and artifacts

The current Runpod path uses the Official plugin for resource control and SSH
for execution. A separate Pod-scoped `runpodctl 1.14.15` watcher was live-tested
on a disposable Pod: its resource became EXITED and was then deleted through
the plugin. The reusable `scripts/resource_deadline.py` defaults to this CLI
contract. Its ready file proves arming, not authentication or final shutdown.
Verify real self-stop on a disposable resource before relying on a new template.

Allocate each run's duration, GPU/storage cost, and prediction limits in the
private ledger before execution. Run the watcher independently of the workload
and SSH session, with an explicit UTC deadline. Preserve checkpoint `params/`,
`assets/`, dataset manifest, package/source pins, event logs and videos before
that deadline. Verify archive hashes after transfer. Use the control plane to
check final Pod state; delete a temporary Pod only after preserving needed data.
Stopped persistent storage can still incur charges.

Success requires four consecutive successful skills in one continuous episode,
grasp maintained through carrying, and a visibly correct final placement.
Inspect `first_object_chain_success` separately from full `task_success`; the
diagnostic's `benchmark_result` remains false. See the full
[failure and success record](abcd-debugging.md).
