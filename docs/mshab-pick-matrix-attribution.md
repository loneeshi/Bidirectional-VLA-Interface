# MS-HAB Pick: pre-contact spatial failure, not grip loss

Local audit of MSHAB012, 2026-09-16 UTC. No new GPU workload, model API request, training, or simulation was run for this analysis. The evaluated model remains Fetch v8 π₀.₅ LoRA, not a completed Fetch TAPT/tool-family model.

## What the contact logs establish

All four π₀.₅ rollouts miss the target before contact. The two recorded finger–target force vectors are zero at every sampled control boundary, `is_grasped` remains false, and target-origin movement is below 0.0000004 m. The data do not support a slipping or weak-grip explanation for these runs.

| Configuration | Closest TCP–target origin distance | First command < −0.5 (Pick step; distance after action) | Observed result |
|---|---:|---:|---|
| Template, interrupt | 21.4 cm, step 28 | 24; 22.8 cm | Interrupt at 34 |
| GPT text, interrupt | 52.3 cm, step 24 | 28; 64.9 cm | Interrupt at 33 |
| Template, observe | 13.9 cm, step 24 | 26; 18.8 cm | Force-limit failure at 95 |
| GPT text, observe | 39.7 cm, step 24 | 26; 45.9 cm | Diagnostic cap at 120 |
| SAC reference | 2.88 cm, step 28 | 25; 2.88 cm | Grasp at 25; native Pick advances at 51 |

These are **TCP-origin to object-origin** distances, not surface clearances or finger alignment errors. Pick step 1 is environment step 331. The gripper command threshold is the monitor's heuristic; it is not a measured-contact event. Contact samples cannot exclude brief contacts between control boundaries, although the stationary target and absence of any grasp support a pre-contact failure.

SAC first obtains bilateral contact at Pick step 25: force norms are approximately 45.0 and 44.8 in the simulator force units. It maintains a grasp through step 51 and advances the native subtask. This is a positive control for the contact instrumentation and physical reachability, not a VLA success.

In the template/observe run, at step 95 `robot_cumulative_force=5125.876953125` and `cumulative_force_within_limit=false`; there are still 105 native subtask steps left. Thus the native failure is associated with the cumulative collision-force limit, not timeout. This score sums per-step force magnitudes; it should not be called a measured impulse. The logs do not identify which non-target collider caused the force. See the pinned local `runs/abcd-preflight/sequential_task.py` `evaluate()`/Pick checker for the native failure condition.

## Why the hand closes without the target

The behavioral sequence is observable: approach, close while spatially offset, then move away or continue an unsuccessful motion. It is not yet evidence of the model's internal reason for selecting that sequence.

At Pick step 24, SAC is 4.1 cm from the target; the π₀.₅ cases are already 13.9–52.3 cm away. The base pose differs as well as the arm pose. For example, the SAC base yaw is −2.002 rad, compared with −1.927 for template/observe and −1.763 for GPT/observe. This motivates separating base approach, arm alignment and gripper timing; changing gripper force alone does not address the observed failure.

The current monitor counts six consecutive closing commands without native grasp. It does not require measured aperture closure, approach to the target or target contact. Its `missed_grasp` label therefore means “unsuccessful closure heuristic,” not “the hand touched and dropped the can.” Reissuing the same skill does not itself implement realignment or recovery. Native collision limits must remain active even in diagnostic observe-only mode.

## Interface audit and remaining hypotheses

The offline audit compares every π₀.₅ request against the **pre-action** diagnostic state. All 282 request states match `[qpos with skill-start XY subtracted, qvel]` exactly. Executed controller actions match the logged bounded actions with the declared stationary-head mask; the largest discrepancy is below 3×10⁻⁸ (float precision). No stale-state or unexplained client-to-controller channel change was found by these checks.

The training converter stores 13 controller-normalized action channels, disables the additional LIBERO delta transform, and overrides the default 7-channel LIBERO output slice with a 13-channel Fetch slice. Its use of `LiberoInputs` copies the supplied state/images; it does not silently reinterpret the 30 states as a LIBERO end-effector pose. This source audit makes a simple seven-versus-thirteen-channel error unlikely. It does **not** replace a numerical test of the actual loaded model's normalization, tokenizer and action output.

Clipping is frequent: 24/34, 22/33, 35/95 and 63/120 action rows respectively contain at least one raw component outside [−1,1]. Saturation is a symptom worth measuring; removing the controller bounds would not be a justified fix, and this observation does not establish clipping as the cause.

The v8 selection contains 1,228 frames from 24 source runs. For first-object Pick, it includes 13 full-start segments (10 PPO-arrival, 3 LightNav-arrival) and 11 recovery tails; these are not 24 independent task distributions. Several full-start poses are near-duplicates. The closest full-start XY position is 6.55 cm from the current one but its yaw differs by 0.406 rad (23.3°). Two LightNav starts are 13.76 cm away with a yaw difference of 0.094 rad. These are comparisons to the archived segment starts, not a nearest-neighbor analysis over every image/state in training.

The current initial yaw velocity is −1.200 rad/s, but two training LightNav starts also have approximately −1.200 rad/s; continued rotation at handoff alone is not a newly isolated out-of-distribution condition. Only head-tilt state index 6 lies outside its marginal training 1–99% quantile range at the first request: 0.56220788 versus [0.56189724, 0.56210840]. The excess is only about 0.00010 rad. Do not infer a major calibration fault from this marginal check. Conversely, being within the other marginal ranges does not establish joint state/image coverage.

**Leading hypothesis, still unproven:** the adapted policy is insufficiently robust to the coupled base/arm/image state at navigation handoff and executes a learned closure/withdrawal pattern without correcting its target-relative error. Prompt sensitivity, stochastic action sampling, actual model normalization and action semantics remain possible contributors.

## Why this is not yet a clean four-cell causal experiment

The four start checks (qpos, qvel, TCP pose, target pose) were equal, but complete scene state, controller internal targets, visual identity and policy random state were not certified equal. The OpenPI service keeps an RNG across requests. The diagnostic client opens a new connection for each case, but that does not reset the server policy RNG.

For the two identical-template cases, the first bounded action already differs by up to 0.620; for the identical-GPT-text cases it differs by up to 1.225. These differences occur before the monitor can interrupt. Consequently, different terminal outcomes cannot be assigned solely to the monitor or prompt. The better template distances are suggestive, not proof of prompt superiority.

## Next verification, in order

1. **Repair diagnostic pairing before more attribution claims.** Give each paired case the same inference noise sequence or restart/reseed the policy with a recorded seed. Verify identical same-prompt inputs/actions up to the first intervention; compare rendered input hashes and controller state. Repeat several paired seeds. Retain native safety termination.
2. **Audit numerical transforms on recorded data.** Confirm actual checkpoint statistics, transformed state tokens, gripper sign, 13-channel output and pre-state/action pairing. Compare predictions with successful SAC demonstration frames and these failure frames. Evaluate both action error and resulting physical behavior; a lower imitation loss is not success.
3. **Separate approach from closure at the same start.** Use explicit diagnostic interventions to distinguish base approach, arm alignment and gripper timing. A live SAC teacher queried at the current learner state is preferable to blindly splicing teacher actions from a different state. Any substituted-action trajectory is labelled teacher-assisted and excluded from VLA results. Simply holding the gripper open cannot repair a 14–52 cm miss.
4. **Only then choose the fix.** If the numerical interface is wrong, repair it first. If it is correct but spatial robustness is inadequate, collect diverse successful handoffs/recovery demonstrations, split by whole trajectory/start state, then retrain. Do not clone this one successful reference trajectory to fabricate data diversity.
5. **Return to the requested baseline.** First require an unassisted GPT + LightNav-0 + π₀.₅ single-object navigation→Pick→carrying navigation→Place success. Fetch tool-family adapters and learned progress remain separate unfinished migration gates. This audit does not pass them.

## Reproducible evidence

- [Contact summary](analysis/mshab012/contact-summary.json): source JSONL SHA256, closure/contact events, bounds and state/action audit.
- [All 333 Pick-step samples](analysis/mshab012/contact-trace.csv): four π₀.₅ rollouts (282) and SAC reference (51).
- [Training support](analysis/mshab012/training-support.json): source hashes, full-start/recovery distinction and normalization quantiles.
- Raw archive: `runs/mshab012/mshab012-final.tar.gz`, SHA256 `9dd57423bfbf81219ad2dd7f367f4d51f59dabd03ba9265ee64782e14f3215ab`. Derived data do not replace this archive.

From the repository root, using the preserved workspace archives:

```powershell
.venv/Scripts/python.exe scripts/analyze_pick_matrix.py --input ../../runs/mshab012/final-extracted/runs/mshab012 --output docs/analysis/mshab012
.venv/Scripts/python.exe scripts/audit_pick_training_support.py --backup ../../runs/abcd004-batch1/data-backup --reference ../../runs/mshab012/final-extracted/runs/mshab012/reference-start.json --norm ../../runs/mshab012/v8-norm-stats.json --output docs/analysis/mshab012/training-support.json
```

`v8-norm-stats.json` preserves the parsed JSON content of `./assets/bvi/fetch-seed1-workspace-recovery-v8/norm_stats.json` inside the preserved `fetch-v8-inference.tar.gz` checkpoint archive; JSON whitespace may differ. The training-support output hashes the supplied copy. No model weights or secrets are added to the public repository.
