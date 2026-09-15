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

[Integrity manifest](manifest.json) records all video sizes and SHA256 hashes. The reorganization changes paths only, not video bytes.

Rendered using [MS-HAB](https://github.com/arth-shukla/mshab), [ManiSkill](https://github.com/haosulab/ManiSkill), and upstream ReplicaCAD/Fetch assets. Those projects and assets retain their respective licenses. No model weights or source scene assets are redistributed here.
