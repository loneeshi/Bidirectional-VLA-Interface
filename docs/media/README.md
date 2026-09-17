# Demo video index

Every video is linked by its exact filename and grouped by test task. Folder names identify the task and, when verified, its date or experiment batch. Historical dates are not inferred from file modification times. Previous recordings remain available, including failures.

## Naming and publication rules

- New batches use `<test-task>-YYYY-MM-DD[-run-id]/` (UTC recording date). For example: `organizer-2026-09-15/`; a later organizer experiment gets a new dated/run folder.
- Keep each recording distinct. Never overwrite a previous demo. Include `failed` or `injected-fault` in names where applicable.
- Every video mention must include the exact filename and a link/path to its task folder.
- Update this index, the SHA256 manifest, and referring documents when publishing or moving a video. Raw experiment archives retain their original paths.
- A successful first-object diagnostic is not full five-object benchmark success. A synthetic fault is not evidence of natural-failure recovery rate.

## Recordings

| Test task / batch | Video filename | What it establishes |
|---|---|---|
| `fetch-tapt-timing-2026-09-17-run01` | [legacy-pre-action-fetch-tapt-pick-episode000-failed.mp4](fetch-tapt-timing-2026-09-17-run01/legacy-pre-action-fetch-tapt-pick-episode000-failed.mp4) | Old pre-action timing; failed at26, no grasp. |
| `fetch-tapt-timing-2026-09-17-run01` | [post-action-fetch-tapt-pick-episode000-failed.mp4](fetch-tapt-timing-2026-09-17-run01/post-action-fetch-tapt-pick-episode000-failed.mp4) | Corrected timing; failed at27, no grasp; not paper reproduction. |
| `fetch-tapt-online-pick-2026-09-17-run01` | [fetch-tapt-pick-episode000-failed.mp4](fetch-tapt-online-pick-2026-09-17-run01/fetch-tapt-pick-episode000-failed.mp4) | Trained TAPT tool harness failed: premature reach/grasp progress thresholds; stopped on learned drop at26. No GPT/navigation, never grasped. |
| `sac-native-reference-2026-09-16-run01` | [sac-reference-seed2025.mp4](sac-native-reference-2026-09-16-run01/sac-reference-seed2025.mp4) | Official SAC reference: seed 2025 native Pick success at step 61. Observation-matched reset; not VLA or full-state paired evidence. |
| `acdit-native-pick-2026-09-16-run02` | [acdit-native-pick-seed2025-failed.mp4](acdit-native-pick-2026-09-16-run02/acdit-native-pick-seed2025-failed.mp4) | Seed 2025: native cumulative-force failure before grasp; no GPT or TAPT. |
| `acdit-native-pick-2026-09-16-run02` | [acdit-native-pick-seed2026-failed.mp4](acdit-native-pick-2026-09-16-run02/acdit-native-pick-seed2026-failed.mp4) | Seed 2026: native cumulative-force failure before grasp; no GPT or TAPT. |
| `acdit-native-pick-2026-09-16-run02` | [acdit-native-pick-seed2027-failed.mp4](acdit-native-pick-2026-09-16-run02/acdit-native-pick-seed2027-failed.mp4) | Seed 2027: native cumulative-force failure before grasp; no GPT or TAPT. |
| `acdit-native-pick-2026-09-16-run02` | [acdit-native-pick-seed2028-failed.mp4](acdit-native-pick-2026-09-16-run02/acdit-native-pick-seed2028-failed.mp4) | Seed 2028: native cumulative-force failure before grasp; no GPT or TAPT. |
| `acdit-native-pick-2026-09-16-run01` | [acdit-native-pick-episode000-failed.mp4](acdit-native-pick-2026-09-16-run01/acdit-native-pick-episode000-failed.mp4) | AC-DiT community checkpoint, native apple Pick, seed 2024: grasped, but failed to settle at the required rest pose; native termination at step 199. No GPT, LightNav or TAPT training. |
| `vlm-baseline-2026-09-14` | [vlm-seed1.mp4](vlm-baseline-2026-09-14/vlm-seed1.mp4) | Constrained VLM baseline; first object placed at step 233; full task incomplete. |
| `lightnav-control-003` | [lightnav-control-003-trial2.mp4](lightnav-control-003/lightnav-control-003-trial2.mp4) | Historical failed navigation rollout. |
| `fetch-grasp-hold-calibration` | [fetch-hold-calibration.mp4](fetch-grasp-hold-calibration/fetch-hold-calibration.mp4) | Official-policy grasp/hold calibration; not a complete task. |
| `abcd-baseline-reference` | [A-ppo-sac-clean.mp4](abcd-baseline-reference/A-ppo-sac-clean.mp4) | A: PPO/SAC, 248 steps. B: LightNav-0/SAC, 278 steps. First-object success with oracle dispatch; not full benchmark success. |
| `abcd-baseline-reference` | [B-lightnav-sac.mp4](abcd-baseline-reference/B-lightnav-sac.mp4) | A: PPO/SAC, 248 steps. B: LightNav-0/SAC, 278 steps. First-object success with oracle dispatch; not full benchmark success. |
| `pi05-state-adaptation` | [C-pi05-state-failed.mp4](pi05-state-adaptation/C-pi05-state-failed.mp4) | C/D failed manipulation adaptation trials; not task success. |
| `pi05-state-adaptation` | [D-lightnav-pi05-state-failed.mp4](pi05-state-adaptation/D-lightnav-pi05-state-failed.mp4) | C/D failed manipulation adaptation trials; not task success. |
| `pi05-recovery-adaptation` | [C-pi05-recovery-failed.mp4](pi05-recovery-adaptation/C-pi05-recovery-failed.mp4) | C/D failed recovery adaptation trials; not task success. |
| `pi05-recovery-adaptation` | [D-lightnav-pi05-recovery-failed.mp4](pi05-recovery-adaptation/D-lightnav-pi05-recovery-failed.mp4) | C/D failed recovery adaptation trials; not task success. |
| `pi05-workspace-adaptation` | [C-pi05-workspace-failed.mp4](pi05-workspace-adaptation/C-pi05-workspace-failed.mp4) | C10/D7 workspace-camera adaptation; Pick failed. |
| `pi05-workspace-adaptation` | [D-lightnav-pi05-workspace-failed.mp4](pi05-workspace-adaptation/D-lightnav-pi05-workspace-failed.mp4) | C10/D7 workspace-camera adaptation; Pick failed. |
| `pi05-v8-recovery` | [C-v8-recovery-failed.mp4](pi05-v8-recovery/C-v8-recovery-failed.mp4) | C12/D9 V8 evaluations; no successful grasp; native failure. |
| `pi05-v8-recovery` | [D-v8-recovery-failed.mp4](pi05-v8-recovery/D-v8-recovery-failed.mp4) | C12/D9 V8 evaluations; no successful grasp; native failure. |
| `organizer-2026-09-15` | [organizer-normal.mp4](organizer-2026-09-15/organizer-normal.mp4) | GPT-5.6 Luna execution organizer with PPO/SAC. Normal: 229 steps, 8 calls. Injected fault: 317 steps, 8 calls, retry after synthetic missed grasp. Both complete the first object only; no pi0.5, free task-order planning, or natural-failure recovery-rate claim. |
| `organizer-2026-09-15` | [organizer-injected-fault-recovery.mp4](organizer-2026-09-15/organizer-injected-fault-recovery.mp4) | GPT-5.6 Luna execution organizer with PPO/SAC. Normal: 229 steps, 8 calls. Injected fault: 317 steps, 8 calls, retry after synthetic missed grasp. Both complete the first object only; no pi0.5, free task-order planning, or natural-failure recovery-rate claim. |
| `mshab-vla-tools-2026-09-16-run01` | [gpt-lightnav0-pi05-episode000-failed.mp4](mshab-vla-tools-2026-09-16-run01/gpt-lightnav0-pi05-episode000-failed.mp4) | First MS-HAB GPT + LightNav-0 + Fetch π₀.₅ closed loop: navigation subtask advanced after 330 steps, then repeated missed grasps; 406 steps and 16 VLM calls, task failed. |
| `mshab-pick-matrix-2026-09-16-run01` | [template-stop-episode000-failed.mp4](mshab-pick-matrix-2026-09-16-run01/template-stop-episode000-failed.mp4) | π₀.₅ with training-template instruction and interrupting monitor: missed grasp after 34 Pick steps. |
| `mshab-pick-matrix-2026-09-16-run01` | [gpt-stop-episode000-failed.mp4](mshab-pick-matrix-2026-09-16-run01/gpt-stop-episode000-failed.mp4) | π₀.₅ with the recorded GPT instruction and interrupting monitor: missed grasp after 33 Pick steps. |
| `mshab-pick-matrix-2026-09-16-run01` | [template-observe-episode000-failed.mp4](mshab-pick-matrix-2026-09-16-run01/template-observe-episode000-failed.mp4) | π₀.₅ with training-template instruction and non-interrupting diagnostic monitor: native benchmark failure after 95 Pick steps. |
| `mshab-pick-matrix-2026-09-16-run01` | [gpt-observe-episode000-failed.mp4](mshab-pick-matrix-2026-09-16-run01/gpt-observe-episode000-failed.mp4) | π₀.₅ with the recorded GPT instruction and non-interrupting diagnostic monitor: no grasp within 120 Pick steps. |
| `mshab-pick-matrix-2026-09-16-run01` | [sac-reference-episode000.mp4](mshab-pick-matrix-2026-09-16-run01/sac-reference-episode000.mp4) | Official object-specific SAC diagnostic reference: Pick succeeds from the identical replayed state in 51 steps; not a VLA result. |


## LIBERO tool-family / TAPT run01

All 15 evaluation videos: [batch index](tapt-libero-2026-09-15-run01/README.md).

| Filename | Configuration / native outcome |
|---|---|
| [baseline-episode000.mp4](tapt-libero-2026-09-15-run01/baseline-episode000.mp4) | baseline: success, 241 steps |
| [baseline-episode001.mp4](tapt-libero-2026-09-15-run01/baseline-episode001.mp4) | baseline: success, 264 steps |
| [baseline-episode002.mp4](tapt-libero-2026-09-15-run01/baseline-episode002.mp4) | baseline: success, 242 steps |
| [baseline-episode003.mp4](tapt-libero-2026-09-15-run01/baseline-episode003.mp4) | baseline: success, 253 steps |
| [baseline-episode004.mp4](tapt-libero-2026-09-15-run01/baseline-episode004.mp4) | baseline: success, 284 steps |
| [vlm-standard-episode000.mp4](tapt-libero-2026-09-15-run01/vlm-standard-episode000.mp4) | vlm-standard: success, 286 steps |
| [vlm-standard-episode001.mp4](tapt-libero-2026-09-15-run01/vlm-standard-episode001.mp4) | vlm-standard: success, 280 steps |
| [vlm-standard-episode002-failed.mp4](tapt-libero-2026-09-15-run01/vlm-standard-episode002-failed.mp4) | vlm-standard: failure, 174 steps |
| [vlm-standard-episode003-failed.mp4](tapt-libero-2026-09-15-run01/vlm-standard-episode003-failed.mp4) | vlm-standard: failure, 40 steps |
| [vlm-standard-episode004.mp4](tapt-libero-2026-09-15-run01/vlm-standard-episode004.mp4) | vlm-standard: success, 255 steps |
| [vlm-tapt-episode000-failed.mp4](tapt-libero-2026-09-15-run01/vlm-tapt-episode000-failed.mp4) | vlm-tapt: failure, 190 steps |
| [vlm-tapt-episode001-failed.mp4](tapt-libero-2026-09-15-run01/vlm-tapt-episode001-failed.mp4) | vlm-tapt: failure, 255 steps |
| [vlm-tapt-episode002-failed.mp4](tapt-libero-2026-09-15-run01/vlm-tapt-episode002-failed.mp4) | vlm-tapt: failure, 425 steps |
| [vlm-tapt-episode003-failed.mp4](tapt-libero-2026-09-15-run01/vlm-tapt-episode003-failed.mp4) | vlm-tapt: failure, 240 steps |
| [vlm-tapt-episode004-failed.mp4](tapt-libero-2026-09-15-run01/vlm-tapt-episode004-failed.mp4) | vlm-tapt: failure, 120 steps |

[Integrity manifest](manifest.json) records all video sizes and SHA256 hashes. The reorganization changes paths only, not video bytes.

Rendered using [MS-HAB](https://github.com/arth-shukla/mshab), [ManiSkill](https://github.com/haosulab/ManiSkill), and upstream ReplicaCAD/Fetch assets. Those projects and assets retain their respective licenses. No model weights or source scene assets are redistributed here.

The LIBERO batch uses LIBERO, robosuite and MuJoCo; their assets and licenses are distinct from the MS-HAB recordings above.


## LIBERO repaired evaluation run03

[All20episodes and exact filenames](tapt-libero-2026-09-15-run03/README.md). Native outcomes:5/5,5/5,4/5,5/5; one SSH infrastructure failure retained.
| `mshab-pick-paired-2026-09-16-run01` | [gpt-observe-seed0-failed.mp4](mshab-pick-paired-2026-09-16-run01/gpt-observe-seed0-failed.mp4) | Paired-noise Fetch pi05 Pick diagnostic; no target grasp; not a benchmark success. |
| `mshab-pick-paired-2026-09-16-run01` | [gpt-observe-seed1-failed.mp4](mshab-pick-paired-2026-09-16-run01/gpt-observe-seed1-failed.mp4) | Paired-noise Fetch pi05 Pick diagnostic; no target grasp; not a benchmark success. |
| `mshab-pick-paired-2026-09-16-run01` | [gpt-stop-seed0-failed.mp4](mshab-pick-paired-2026-09-16-run01/gpt-stop-seed0-failed.mp4) | Paired-noise Fetch pi05 Pick diagnostic; no target grasp; not a benchmark success. |
| `mshab-pick-paired-2026-09-16-run01` | [gpt-stop-seed1-failed.mp4](mshab-pick-paired-2026-09-16-run01/gpt-stop-seed1-failed.mp4) | Paired-noise Fetch pi05 Pick diagnostic; no target grasp; not a benchmark success. |
| `mshab-pick-paired-2026-09-16-run01` | [sac-reference.mp4](mshab-pick-paired-2026-09-16-run01/sac-reference.mp4) | Official object-specific SAC positive control: native Pick succeeds in 49 Pick steps; excluded from VLA results. |
| `mshab-pick-paired-2026-09-16-run01` | [template-observe-seed0-failed.mp4](mshab-pick-paired-2026-09-16-run01/template-observe-seed0-failed.mp4) | Paired-noise Fetch pi05 Pick diagnostic; no target grasp; not a benchmark success. |
| `mshab-pick-paired-2026-09-16-run01` | [template-observe-seed1-failed.mp4](mshab-pick-paired-2026-09-16-run01/template-observe-seed1-failed.mp4) | Paired-noise Fetch pi05 Pick diagnostic; no target grasp; not a benchmark success. |
| `mshab-pick-paired-2026-09-16-run01` | [template-stop-seed0-failed.mp4](mshab-pick-paired-2026-09-16-run01/template-stop-seed0-failed.mp4) | Paired-noise Fetch pi05 Pick diagnostic; no target grasp; not a benchmark success. |
| `mshab-pick-paired-2026-09-16-run01` | [template-stop-seed1-failed.mp4](mshab-pick-paired-2026-09-16-run01/template-stop-seed1-failed.mp4) | Paired-noise Fetch pi05 Pick diagnostic; no target grasp; not a benchmark success. |
| `mshab-pick-paired-2026-09-16-run01` | [template-teacher-arm-torso-seed0-failed-teacher-assisted.mp4](mshab-pick-paired-2026-09-16-run01/template-teacher-arm-torso-seed0-failed-teacher-assisted.mp4) | Labelled teacher-assisted channel intervention; Pick still fails; excluded from pure VLA results. |
| `mshab-pick-paired-2026-09-16-run01` | [template-teacher-base-seed0-failed-teacher-assisted.mp4](mshab-pick-paired-2026-09-16-run01/template-teacher-base-seed0-failed-teacher-assisted.mp4) | Labelled teacher-assisted channel intervention; Pick still fails; excluded from pure VLA results. |
| `mshab-pick-paired-2026-09-16-run01` | [template-teacher-gripper-seed0-failed-teacher-assisted.mp4](mshab-pick-paired-2026-09-16-run01/template-teacher-gripper-seed0-failed-teacher-assisted.mp4) | Labelled teacher-assisted channel intervention; Pick still fails; excluded from pure VLA results. |


## Fetch current-progress handoff collection

Six retained recordings and exact filenames: [batch index](fetch-current-handoffs-2026-09-17-run01/README.md). Training/validation data collection: 2 native Pick successes, 4 failures, all retained. No navigation or GPT.


## Fetch current-frame calibration evaluation

[seed2025-fetch-tapt-pick-episode000-failed.mp4](fetch-current-progress-eval-2026-09-17-run01/seed2025-fetch-tapt-pick-episode000-failed.mp4) · [seed2030-fetch-tapt-pick-episode000-failed.mp4](fetch-current-progress-eval-2026-09-17-run01/seed2030-fetch-tapt-pick-episode000-failed.mp4). Fixed selected-checkpoint diagnostics:0/2 Pick completion, neither grasped. No GPT/LightNav.
