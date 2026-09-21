# Closed-loop first-divergence diagnostic

`src/bvi/closed_loop_divergence.py` is an observation-only Level 3 diagnostic.
It reports the first post-action `qpos` channel whose absolute error exceeds an
explicit caller-supplied threshold, plus policy/expert TCP-to-target distance and
cumulative-force curves.  It neither changes actions nor defines success.

The comparison deliberately fails closed.  A run is comparable only when both
result files have the same task, seed, and exact `reference_state_sha256`, and
both report state-restoration error no larger than the configured tolerance.
Missing expert `qpos`, gaps in event steps, or a channel-count mismatch also make
the result `not_comparable`; no first-divergence value or paired curve is then
emitted.  Merely sharing a seed is not pairing evidence.

Thresholds are not guessed by the implementation.  Supply a JSON file with all
15 Fetch `qpos` channels, for example:

```json
{
  "qpos_abs_error": {
    "root_x_axis_joint": 0.01,
    "root_y_axis_joint": 0.01,
    "root_z_rotation_joint": 0.05,
    "torso_lift_joint": 0.01,
    "head_pan_joint": 0.05,
    "shoulder_pan_joint": 0.05,
    "head_tilt_joint": 0.05,
    "shoulder_lift_joint": 0.05,
    "upperarm_roll_joint": 0.05,
    "elbow_flex_joint": 0.05,
    "forearm_roll_joint": 0.05,
    "wrist_flex_joint": 0.05,
    "wrist_roll_joint": 0.05,
    "r_gripper_finger_joint": 0.005,
    "l_gripper_finger_joint": 0.005
  }
}
```

Those numbers are an invocation example, not preregistered scientific
thresholds.  The chosen file is hashed into each result.  Run the analyzer as:

```powershell
python scripts/analyze_closed_loop_divergence.py `
  --policy-events <policy-run>/events.jsonl `
  --expert-events <exact-start-expert-run>/events.jsonl `
  --policy-result <policy-run>/result.json `
  --expert-result <exact-start-expert-run>/result.json `
  --thresholds <frozen-thresholds.json> `
  --output <new-result.json>
```

Exit code 2 means the artifact was written but strict comparability failed.  The
same core can be inserted into a runner with `ClosedLoopDivergenceRecorder`:
construct it from a frozen expert event sequence and two run identities, call
`record(row)` after each environment step, then save `summary()`.  Existing
`eval_ia_s1_call.py` is intentionally not modified by this diagnostic patch.

Current S1 caution: the original official H5 expert replay does not preserve a
full simulator/controller start snapshot and therefore cannot satisfy this
pairing gate.  The later D1 SAC replay artifacts do carry the same saved
`reference_state_sha256` as the IA reach calls and are eligible only if their
restoration evidence and event channels also pass the checks above.

The seed-2025 threshold was subsequently frozen with three fresh-process,
same-snapshot, same-SAC-action controls.  See
[s1-level3-first-divergence-2026-09-18.md](s1-level3-first-divergence-2026-09-18.md):
all three controls were identical, and the exact-start S1-IA/SAC comparison
first diverged after action 1.  That result is a paired diagnosis, not a success
rate estimate or a universal threshold for other configurations.
