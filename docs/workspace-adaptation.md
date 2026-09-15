# Fetch workspace-camera adaptation

This is a training-scene engineering experiment. The model uses the same
validation initialization used to collect the demonstrations. It does not
establish held-out performance, full five-object task success, or VLM planning
benefit. The dispatcher in these comparisons is the labelled oracle.

## Model contract

The student starts from OpenPi pi05 weights and is adapted to Fetch using LoRA.
This is not an off-the-shelf Fetch policy. V8 warms from the V7 checkpoint;
the exact archived V7 inference SHA256 is
`7892c5f1bfadbafc1feeee7ee4b3f5c510e68972a60a20d19a9abda9c1718550`.

| Component | Contract |
|---|---|
| Workspace RGB | `fetch_workspace`, 224 × 224, declared robot-mounted camera |
| Wrist RGB | Native `fetch_hand`, 128 × 128 |
| Proprioception | Named Fetch qpos followed by qvel, 30 values |
| Root translation | Subtract measured skill-start root-joint x/y; not an SE(2) transform |
| Output | 10-step chunks of 13 normalized Fetch controller channels |
| Applied control | Clamp to [-1, 1], then the existing stationary-head mask |
| Completion | Native environment feedback, independent of model output |

Object pose, goal pose and grasp truth are not policy inputs to the pi student.
The official SAC teacher does receive native extra state. The teacher/student
observation gap must be retained in any comparison.

The 10-step prediction horizon, number of flow denoising steps, and number of
actions executed before requesting another chunk are different settings. The
initial V8 evaluation uses 10 denoising steps and executes one action per query.

## Collection and data selection

V7 used 693 frames: 13 successful Pick segments and three Place segments.
V8 adds 535 expert recovery frames: 11 Pick segments and three Place segments.
There are 1,228 frames across 30 segments from 24 selected run directories.
All use `tidy_house-sequential-val-90-0` / `val/tidy_house/episode_18.json`.

The recovery collection executes a bounded pi prefix, then records SAC actions.
It never changes the simulator pose or success predicate. Only expert tails
whose corresponding native subtask completed enter training. Failed tails and
the pi prefix are excluded from the labels. For a recovery tail, the relative
root origin remains the **learner's skill-start origin**, not the takeover pose.

The initial 12 attempts used prefixes 5, 10, 20, 30, 20, 30 after PPO navigation
and the same six prefixes after LightNav navigation. Two further prefix-5
attempts used the corrected oracle remaining-time scheduling. All 14 raw
attempts are retained. Three mixed teacher chains succeeded; none is a C/D
evaluation result. [Per-run collection audit](results/workspace-recovery-training-data.json).

The verified local recovery archive SHA256 is
`3b3d644819c1b011ae1c34cd095e6a28ba826086de11ef7eb3052d41aca83754`.
The separate verified V7 teacher archive SHA256 is
`7d135fc4b3ce41dbcef69e797cfd32da9f388230ca21c5cd383c919cee187591`.
These archives and checkpoints are preserved locally; the repository does not
bundle the large training data or model weights. A fresh collection may differ
because the simulator and navigation outputs are not bitwise deterministic.

## Commands

Use the pinned simulation environment from the [deployment guide](reproduction.md). Run the
Fetch-trained V7 OpenPi server separately before collecting recovery data. The
following is one labelled training run, not an evaluation command:

```bash
python scripts/run_coordinator.py \
  --dry-run --training-collection --record-demonstrations \
  --collect-recovery-after 5 \
  --manipulation-policy fetch-pi05 --manipulation-chunk-steps 1 \
  --max-manipulation-predictions 100 \
  --workspace-camera --navigation-camera fetch_nav \
  --seed 1 --policy-type rl_per_obj \
  --max-calls 4 --max-env-steps 500 \
  --max-wall-seconds 180 --skill-wall-seconds 90 \
  --checkpoint-root /workspace/bvi/mshab_checkpoints \
  --output runs/recovery-prefix5
```

For a LightNav teacher run, also supply:

```bash
--navigation-policy lightnav \
--navigation-instructions configs/lightnav-seed1-disambiguated.json \
--navigation-recovery-instructions configs/lightnav-seed1-orientation.json \
--max-navigation-predictions 80 \
--expected-plan-uid tidy_house-sequential-val-90-0
```

Stop model-serving processes before training on the single A6000. Inside the
pinned OpenPi environment, use the exact successful source directories from the
archived selection manifest. `fetch_openpi.py` independently excludes failed
expert segments during conversion:

```bash
python scripts/fetch_openpi.py convert \
  --include-velocity --relative-base --base-camera fetch_workspace \
  --repo-id bvi/fetch-seed1-workspace-recovery-v8 \
  --runs /path/to/selected/run1 /path/to/selected/run2

python scripts/fetch_openpi.py norm \
  --include-velocity --relative-base --base-camera fetch_workspace \
  --repo-id bvi/fetch-seed1-workspace-recovery-v8 \
  --work /workspace/fetch-pi-workspace-recovery

XLA_PYTHON_CLIENT_MEM_FRACTION=.75 WANDB_MODE=disabled \
python scripts/fetch_openpi.py train \
  --include-velocity --relative-base --base-camera fetch_workspace \
  --repo-id bvi/fetch-seed1-workspace-recovery-v8 \
  --work /workspace/fetch-pi-workspace-recovery --steps 2000 --batch 4 \
  --init-checkpoint /workspace/fetch-pi-workspace/checkpoints/pi05_fetch_lora_workspace/seed1-diagnostic/1999/params
```

The executed adaptation source SHA256 is
`ab99fcc1728c487ea2dcde2656291071abeaf5606a16f3aae553d06169e077a3`.
The actual experiment used an immutable copy of that file. Norm statistics and
the state/camera contract are saved with the final checkpoint.

## Pure C/D evaluation

Serve the final V8 checkpoint with the matching repo ID, state convention and
camera flags. Omit **both** `--collect-recovery-after` and
`--training-collection` from evaluation. Keep the task UID fixed. C uses PPO
navigation; D uses LightNav. Both must use the same pi checkpoint.

Once the final checkpoint has been saved, start this in the OpenPi environment:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.40 WANDB_MODE=disabled \
python scripts/fetch_openpi.py serve \
  --include-velocity --relative-base --base-camera fetch_workspace \
  --repo-id bvi/fetch-seed1-workspace-recovery-v8 \
  --work /workspace/fetch-pi-workspace-recovery \
  --checkpoint /workspace/fetch-pi-workspace-recovery/checkpoints/pi05_fetch_lora_workspace/seed1-diagnostic/1999 \
  --denoising-steps 10
```

Then run C in the simulator environment:

```bash
python scripts/run_coordinator.py \
  --dry-run --seed 1 --policy-type rl_per_obj \
  --manipulation-policy fetch-pi05 --manipulation-chunk-steps 1 \
  --workspace-camera --navigation-camera fetch_nav \
  --max-manipulation-predictions 200 \
  --expected-plan-uid tidy_house-sequential-val-90-0 \
  --max-calls 4 --max-env-steps 700 \
  --max-wall-seconds 600 --skill-wall-seconds 240 \
  --checkpoint-root /workspace/bvi/mshab_checkpoints --output runs/C
```

For D, start the LightNav server as described in [A/B reproduction](abcd-reproduction.md),
add the LightNav flags above, set `--max-env-steps 1000`, and change the output
directory to `runs/D`. Server, simulator and checkpoint provenance are recorded
in the resulting event logs and run metadata.

Validate each run with:

```bash
python scripts/audit_chain.py runs/C --output runs/C-audit.json
python scripts/audit_chain.py runs/D --output runs/D-audit.json
```

The audit rejects mixed and training runs. It also checks that actual
manipulation controls match pi output after the declared clamp/head mask,
that grasp persists through carry, and that the target is released at the goal.
Inspect the video in addition to the automated checks. Training loss and
teacher-forced action error are not substitutes for online task success.
