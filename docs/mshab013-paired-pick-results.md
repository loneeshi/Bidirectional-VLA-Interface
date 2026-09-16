# MSHAB013c paired Pick diagnosis

Run date: 2026-09-16 UTC. Task: first `002_master_chef_can-0` Pick after replaying the audited 330-step navigation prefix. External API calls: 0. Training updates: 0. These are single-start causal diagnostics, not benchmark success rates.

## Pairing repair

The earlier MSHAB013/MSHAB013b attempts failed their complete-scene equality gate because scene construction consumed Python `set` iteration without a fixed process hash seed. The fixed run launches every case in a fresh process with `PYTHONHASHSEED=0`.

All 12 MSHAB013c starts pass the full serialized simulator and captured controller-state comparison with zero differences. For each seed/prompt pair, the non-interrupting case reproduces the interrupting case exactly through the point where the monitor would stop it:

| Pair | Verified identical prefix | Maximum action error |
|---|---:|---:|
| Template, seed 0 | 31 steps | 0 |
| Recorded GPT instruction, seed 0 | 31 steps | 0 |
| Template, seed 1 | 32 steps | 0 |
| Recorded GPT instruction, seed 1 | 32 steps | 0 |

Each model request also carries acknowledged request-scoped noise. The first request in every case is repeated without a simulation step and must return the same action. A gate failure stops the experiment.

## Results

| Configuration | Pick steps | Closest TCP–target origin | Distance at first close | Contact / grasp | Outcome |
|---|---:|---:|---:|---|---|
| Template, seed 0, interrupt | 31 | 9.68 cm | 14.66 cm | none | `missed_grasp` |
| Template, seed 0, observe | 90 | 9.68 cm | 14.66 cm | none | force-limit `benchmark_fail` |
| GPT text, seed 0, interrupt | 31 | 10.68 cm | 16.45 cm | none | `missed_grasp` |
| GPT text, seed 0, observe | 120 | 10.68 cm | 16.45 cm | none | diagnostic cap |
| Template, seed 1, interrupt | 32 | 19.25 cm | 26.51 cm | none | `missed_grasp` |
| Template, seed 1, observe | 120 | 19.25 cm | 26.51 cm | none | diagnostic cap |
| GPT text, seed 1, interrupt | 32 | 18.62 cm | 26.26 cm | none | `missed_grasp` |
| GPT text, seed 1, observe | 120 | 18.62 cm | 26.26 cm | none | diagnostic cap |
| Official SAC reference | 49 | 2.87 cm | 2.87 cm | bilateral contact and grasp at step 25 | native Pick succeeds |
| π₀.₅ + SAC base, seed 0 | 35 | 9.33 cm | 9.33 cm | none | force-limit `benchmark_fail` |
| π₀.₅ + SAC arm/torso, seed 0 | 120 | 7.75 cm | 7.75 cm | no sampled finger contact or grasp | diagnostic cap |
| π₀.₅ + SAC gripper, seed 0 | 120 | 10.46 cm | 74.33 cm at step 41 | none | diagnostic cap |

Distances are TCP-origin to object-origin, not surface clearances. Contact forces are sampled at control boundaries and do not exclude a brief substep contact. The teacher-assisted rows query the official SAC policy at the current learner state and replace only the named action channels for a bounded window. They are interventions, not VLA results.

## Attribution

The interrupting monitor is not the cause of these four failures. With identical starts, inputs, noise and actions, allowing the paired rollouts to continue does not produce a grasp. Prompt wording changes the trajectory, but neither tested prompt succeeds under either noise seed.

The failure occurs before stable target contact. Across all eight pure π₀.₅ cases, `is_grasped` remains false, sampled finger–target force remains zero, and target displacement is below 0.4 micrometres. π₀.₅ closes while the TCP origin is still 14.7–26.5 cm from the target origin.

No single tested control channel is sufficient to repair the trajectory. Replacing base actions still misses and triggers the collision-force limit. Replacing arm and torso actions gets closer, 7.75 cm, but does not establish contact or grasp. Replacing gripper actions delays closure but the base/arm motion has already carried the TCP away. This supports a coupled base–arm–gripper coordination or learned spatial-policy problem rather than an isolated gripper-force setting. It does not distinguish model perception, normalization, training coverage and joint action prediction.

The official SAC reference reaches 2.87 cm, forms bilateral contact and advances the native Pick subtask. It demonstrates physical reachability and validates the contact signal from this start. It does not make the teacher-assisted mixed policies successful and is excluded from pure VLA claims.

Frequent raw action saturation remains relevant: the eight pure π₀.₅ cases clip 23–84 action rows each. Saturation is correlated with the failed behavior here; this run does not establish it as the causal mechanism.

## Next implementation decision

The next paid stage should compare the actual v8 checkpoint's transformed state/action values and predictions on successful teacher frames versus these paired failure frames, then collect diverse successful joint recovery demonstrations only if the numerical interface passes. Training one channel independently is not supported by these results. A full unassisted GPT + LightNav-0 + π₀.₅ navigation→Pick→carry→Place chain remains unachieved.

Raw final backup: `runs/mshab013/evidence-retry-c/current-evidence.tar.gz`, SHA256 `4a6f9caa55436d8203ed3815f52f96df8d6253aef2002d2a46725d7562089943`. Earlier failed-gate runs are retained separately.

All 12 videos are indexed under [`docs/media/mshab-pick-paired-2026-09-16-run01/`](media/mshab-pick-paired-2026-09-16-run01/). The only successful video is the SAC positive control `sac-reference.mp4`; teacher-assisted files are explicitly labelled and all π₀.₅ files retain `failed` in their names.
