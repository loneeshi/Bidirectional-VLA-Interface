# LightNav control diagnostic: actual motion, no task success

[Rollout video](media/lightnav-control-003-trial2.mp4) · [Structured evidence](results/lightnav-control-003.json) · [Official-policy grasp/hold calibration](media/fetch-hold-calibration.mp4)

The 512×512, 20 FPS video contains actual MS-HAB frames without the verbose
parameter overlay. It shows LightNav controlling Fetch for 270 steps (13.5 seconds
of simulated action; 13.55-second recording including the initial frame). It does
**not** show successful manipulation or a completed skill chain.

## Observed results

| Run | Observation | Outcome |
|---|---|---|
| Physical calibration | +0.15 m/s for 20 steps moved +0.131 m; +0.3 rad/s for 20 steps changed yaw +0.298 rad | Direction/unit checks passed |
| Official grasp and hold | PPO Navigate 29 steps, SAC Pick 53 steps; then 40 navigation-hold steps with all grasp checks true | Local hold handoff passed; no LightNav in this calibration |
| Trial 1 | One LightNav prediction, five control steps | Deferred-image interface bug; fixed before trial 2 |
| Trial 2 | 55 predictions, 270 steps | Model returned `stop=true`, but `navigated_close=false` and `oriented_correctly=false`; SAC was never reached |
| Trial 3 | Intended official Navigate/Pick prefix followed by LightNav | Service connection refused before decisions, predictions or control; no handoff result |

All trials use the seed-1 first subtask `tidy_house-sequential-val-90-0` and oracle
high-level dispatch. No GPT/Opus requests, pi0.5 control, training, or aggregate
benchmark evaluation occurred. The 56 model predictions are included in the
rented GPU usage, not billed as a separate model API.

A resource-control failure exceeded the experiment's approved wall-clock and
cost limits. The temporary Pod was deleted after evidence preservation. Process
`timeout` stopped the model service but did not release the rented GPU. Future
paid runs require a verified resource-level deadline that remains effective when
the interactive agent pauses; process timeouts and conversational polling alone
are insufficient. The private ledger preserves the overrun and billing evidence.

## What failed and what to test next

The first integration bug came from deferred PNG encoding: transition snapshots
had no images after the first step. The navigation skill now obtains the current
capture with `observe()` and rejects a mismatched frame ID. This does not perform
another simulator step or call its state-mutating evaluator. A regression test
covers replanning with empty transition images and stale-camera rejection.

The second trial proves model prediction → physical action → fresh observation,
but not navigation competence. The head camera is only 128×128, and the held
initial arm posture substantially occludes its view. The manually written counter
goal is an unvalidated instance-grounding assumption. Camera domain mismatch,
visible target identification, stopping pose and trajectory tracking therefore
remain separate hypotheses, rather than a demonstrated model defect.

The `ignore_arm_checkers` diagnostic option also removes grasp from navigation
success checks. Explicit grasp checks are essential when interpreting carrying
trials; the 40-step calibration establishes only that bounded hold.

Before another paid trial: verify the target in unobstructed images, select a
navigation-only approach task and a separate SAC-pick → carrying-navigation task,
confirm camera-to-base geometry, and install a resource-level spending guard.
Do not fine-tune or claim a full VLA chain on the strength of this video.

## Interface and reproduction

At skill entry the adapter verifies controller classes, flattened order, original
physical action ranges, normalization, base/body joint names and20Hz frequency.
It captures arm/body posture and the existing absolute gripper drive target.
Each step compensates arm/body drift while holding that gripper target. An all-zero
action would request an unintended gripper position, so it is not used as a hold.

LightNav head RGB + manually authored visual language returns cumulative local
SE(2) waypoints. Every row is transformed from the image-capture pose into the
same planar root-joint frame; rows are not summed. A conservative forward-only
tracker commands at most0.25m/s and0.6rad/s, replanning every5 simulation steps.
These are conservative engineering defaults, not benchmark-tuned values.
There is no independent collision avoidance or learned manipulation-ready pose
selector. Force failures/timeouts and official subtask advancement remain the
environment's feedback source. Model stop alone does not count as success.

The official camera is128x128 with a wide FOV; compatibility of camera geometry,
semantic destination, stopping pose and held-object clearance remains unverified.
The native camera image is forwarded without parameter overlays. Existing video
recording defaults to no debug text; raw metric/action logs remain JSONL.

## Bounded execution commands

1. Verify GPU/Vulkan and run physical forward/turn/hold checks first. Compare
   measured base motion with controller units/signs; abort on disagreement.
2. Reproduce official PPO+SAC seed1 with the current simulator/asset revisions.
3. Run one LightNav+SAC first-object diagnostic, recording all failures. Permit
   at most3 trials and240 total LightNav predictions in the proposed experiment.
   Do not describe these development trials as the fixed10-episode evaluation.
4. Download videos/logs; stop compute and delete any authorized temporary resource.

Example inside the pinned MS-HAB environment, with a separately running LightNav
HF server accessible on localhost8050:

```bash
python scripts/run_coordinator.py --dry-run --seed 1 --policy-type rl_per_obj \
  --navigation-policy lightnav --lightnav-url ws://127.0.0.1:8050 \
  --navigation-instructions configs/lightnav-seed1-diagnostic.json \
  --expected-plan-uid tidy_house-sequential-val-90-0 \
  --max-navigation-predictions 80 --max-calls 4 --max-env-steps 500 \
  --max-wall-seconds 1200 --skill-wall-seconds 240 \
  --checkpoint-root /workspace/bvi/mshab_checkpoints \
  --output /workspace/bvi/runs/lightnav-chain-seed1
```

`--dry-run` here means oracle high-level dispatch and no GPT/Opus API calls. It
still invokes LightNav and real GPU simulation. Missing instructions or a sampled
plan UID mismatch reject execution. Instructions are development annotations
based on the earlier scene; destination identity remains to be verified.
They are not generated by a VLM or a general solution to instance grounding.

The first-object chain is a diagnostic slice. Neither finishing four skills nor
recording a new video establishes the full five-object benchmark score, free
planning gains, or pi0.5 Fetch compatibility.

Pinned source contracts:
[Fetch](https://github.com/haosulab/ManiSkill/blob/17121e3f96e3ee3ed0c03610b17f8bc2864617af/mani_skill/agents/robots/fetch/fetch.py),
[base velocity](https://github.com/haosulab/ManiSkill/blob/17121e3f96e3ee3ed0c03610b17f8bc2864617af/mani_skill/agents/controllers/pd_base_vel.py),
[joint position](https://github.com/haosulab/ManiSkill/blob/17121e3f96e3ee3ed0c03610b17f8bc2864617af/mani_skill/agents/controllers/pd_joint_pos.py).

The optional `--lightnav-subtasks 2` explicitly selects LightNav only for carrying
navigation and routes other navigation indices to official PPO. This is declared
in run metadata and route events; it never silently falls back after a failure.
Its physical carrying rollout remains unvalidated because trial 3 did not start.

Physical calibration (real GPU usage, no model API):

```bash
python scripts/calibrate_fetch_navigation.py \
  --checkpoint-root /workspace/bvi/mshab_checkpoints \
  --output /workspace/bvi/runs/calibration
```

The calibration resets between sign tests and the official pick/hold episode.
Those two episodes must not be presented as one continuous task trajectory.
