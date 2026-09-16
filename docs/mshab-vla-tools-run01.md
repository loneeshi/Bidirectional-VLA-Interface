# MS-HAB VLA-tools run01

This run is the first real MS-HAB execution of the versioned `mshab-tool-family/1` interface with GPT-5.6 Luna as coordinator, LightNav-0 for navigation, and the Fetch-adapted π₀.₅ V8 checkpoint for manipulation.

## Result

- The coordinator made 16 paid, image-conditioned decisions and consumed bidirectional runtime feedback.
- LightNav-0 advanced the first benchmark subtask after nine bounded navigation calls and 330 simulator steps.
- π₀.₅ then attempted Pick. The first attempt ran 40 steps; six subsequent retries were interrupted after six steps each by the benchmark grasp monitor.
- The run ended at the configured 16-call limit after 406 simulator steps. The first object chain and benchmark task did not succeed.
- This establishes the initial GPT → tool-family → LightNav-0/π₀.₅ → MS-HAB → feedback loop. It does not establish a successful mobile-manipulation baseline.

The exact public recording is [gpt-lightnav0-pi05-episode000-failed.mp4](media/mshab-vla-tools-2026-09-16-run01/gpt-lightnav0-pi05-episode000-failed.mp4). SHA256: `9d36671ac0e75e7c842351886e3525fc7e445b1826219b1aaf8400f96d932700`.

## Failure attribution

The failure is localized to Pick after navigation, but policy failure and monitor-induced interruption are not yet causally separated. Across 76 manipulation steps, `is_grasped` remained false and the logged `robot_target_pairwise_force` was zero. This aggregate field does not replace a per-finger contact audit. The six retries each issued six consecutive close commands and were interrupted after 0.3 seconds at 20 Hz. The monitor checks commanded closure, not actual finger aperture or completion of an approach. It may therefore prevent recovery; disabling it in a bounded diagnostic is needed before attributing failure solely to the policy.

37 of 76 manipulation actions were clipped. Overshoot affected arm, gripper, torso and base-forward channels; raw torso values reached about -1.36 and gripper values about -1.13. This is an observable symptom, not proof of a normalization bug. GPT changed the text on each retry, but the exposed choices were Pick or abort; no explicit reopen/reapproach controller or family-specific residual selection was demonstrated. The relative base origin remained constant across retries, ruling out an origin reset on these calls.

Earlier infrastructure failures are retained in the raw MSHAB011 archive: missing `openpi_client`, output-directory reuse, an overly restrictive 200 KB request gate, and a local bridge idle timeout. Each failed before simulator actions or before a paid call, except the bridge timeout which created a remote request but made no provider call.

## Limits

Navigation completion used the benchmark subtask transition as completion evidence; LightNav-0 did not provide learned progress. Manipulation progress was likewise unavailable, so feedback came from explicit runtime and benchmark predicates. The run used serial control ownership and one seed.
