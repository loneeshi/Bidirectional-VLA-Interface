# VLA replacement experiments

Status (2026-09-14): model-service clients and saved-frame probe entry point added.
Real candidate inference and candidate-controlled Fetch rollouts have NOT run.
The original GPT + PPO/SAC diagnostic remains the only live closed-loop result.

## Model boundaries

LightNav-0: RGB history + language -> cumulative local SE(2) waypoints, using the
[official JSON WebSocket protocol](https://github.com/lightorigins/LightNav-0/blob/main/docs/PROTOCOL.md).
The client preserves a connection/session, resets history, checks response sequence
and finite waypoint values. A predicted stop is not benchmark success.

Pi0.5: images + robot state + prompt -> action chunk, using the
[official openpi codec](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/websocket_client_policy.py).
The [DROID example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py)
uses seven joint velocities plus gripper position at 15 Hz. This is NOT the
Fetch 13-dimensional normalized action contract. Our DROID probe uses explicitly
synthetic zero proprioception and may test service availability only. It must
never be used as evidence that Fetch manipulation works.

Clients have finite receive deadlines and no automatic retry. Store model revision,
checkpoint hash, server environment and experiment authorization beside each probe.
Contract tests use fixtures, not model weights; they do not verify model inference.

## Reproduce the next inference probes

Use a separate Linux environment per model. Keep the old MS-HAB environment pinned.
Pin upstream commits before installation and record them in the experiment log.
Preflight HEAD references: LightNav `c6f40e3220edbf7011e4f17eaf2c865416737d4d`;
openpi `215abfb217dbac7d5f1273282331b9b1866c0479`. These are source references,
not downloaded/verified checkpoint hashes or evidence of a tested installation.
The probe client needs `websockets>=14`; the openpi branch additionally needs
`numpy`, `Pillow`, and the `packages/openpi-client` package from the pinned official
openpi checkout. Do not install the complete model training stack into MS-HAB.

Start LightNav using the official `lightnav-serve --task vln` workflow; start
openpi using its documented pi05_droid server configuration. Tunnel their ports
over SSH, e.g. `ssh -L 8050:127.0.0.1:8050 -L 8000:127.0.0.1:8000 <pod>`.
Only start paid servers within a recorded spending scope.

From this repository, with a running authorized server:

```bash
python scripts/probe_vla.py --backend lightnav --url ws://127.0.0.1:8050 \
  --head /path/to/head.png --instruction "Approach the kitchen counter" \
  --output runs/lightnav-probe-001.json
python scripts/probe_vla.py --backend pi05-droid --url ws://127.0.0.1:8000 \
  --head /path/to/head.png --hand /path/to/hand.png \
  --instruction "Pick up the can" --output runs/pi05-probe-001.json
```

Each command issues at most one inference and writes image hashes, output and
elapsed time. Failure/timeout is retained as uncertain, not retried. These commands
do not instantiate or step a simulator. The main coordinator runner still selects
official RL skills; candidate skill registration is pending the gates below.

## Before Fetch control

1. LightNav: verify native RGB camera orientation/FOV and goal language; measure
   latency/VRAM/history behavior. Track local cumulative waypoints using the robot
   pose at capture, not by summing consecutive rows. Verify actual Fetch controller
   DOF layout and physical limits from the installed environment; test forward and
   turn signs. Preserve arm/torso/gripper posture while driving. Do not invent an
   all-zero hold action. Terminate on benchmark feedback, not model stop alone.
2. Pi0.5: inspect and export named Fetch proprioception, controller units, joint
   order, gripper convention, camera images and control frequency. Create a Fetch
   dataset transform and normalization statistics; establish a Fetch checkpoint
   through suitable data/fine-tuning. Test individual grasp rollouts before skill
   registration. Padding/truncation of DROID or LIBERO actions is forbidden.
3. Register an implementation of the existing `Skill.start/act/feedback` protocol
   only after the controller mapping passes. Keep the serial full-vector owner;
   invalidate action chunks at skill changes. Evaluate handoff, not only inference.

## Task selection and evaluation

Primary task: a first-object TidyHouse chain (Navigate -> Pick -> Navigate -> Place).
It covers both handoff directions and carrying an object without articulation
skills. This is a diagnostic slice, not a complete official TidyHouse score.
Keep full scene/plan IDs and initial poses fixed across variants; do not select
only seeds that succeeded with one method. The existing seed-1 can chain is a
development example, not a held-out evaluation episode.

| Stage | Navigation | Manipulation | What it isolates |
|---|---|---|---|
| A | Official PPO | Official SAC | Reproduce baseline and clean video |
| B | LightNav + verified Fetch tracker | Same SAC | Navigation and manipulation-ready arrival |
| C | Same PPO | Fetch-adapted pi0.5 | Manipulation transfer and exit state |
| D | LightNav | Fetch-adapted pi0.5 | Combined handoff errors |

Use the same deterministic dispatcher for initial low-level A/B/C/D comparisons.
Then hold low-level policies fixed and compare a deterministic dispatcher with
VLM decisions. Current SequentialTask imposes a subtask order: exposing fictional
alternative skills does not establish free planning. An open-order coordinator
experiment requires a separately declared task/adapter and success predicates.

Pre-register 10 validation scene/plan/seed tuples before scored rollouts; save
explicit IDs and initial-state fingerprint rather than relying on seed alone.
Report all attempts, per-skill success, full-chain success, manipulation success
conditioned on candidate navigation arrival, collisions/force failures, dropped
objects, steps, inference latency and cost. Start with one development tuple;
do not claim the 10-episode evaluation until it is actually run.

Record clean video (`info_on_video=false`) with skill/model subtitles outside the
image and an additional gripper view. Retain raw footage and debug JSON separately.

Later: full five-object TidyHouse, then SetTable for Open/Close and constrained
access. SetTable needs additional articulation skills; PrepareGroceries adds
harder manipulation distributions. Neither belongs in the first replacement
smoke test. Task definitions: [MS-HAB official site](https://arth-shukla.github.io/mshab/).
