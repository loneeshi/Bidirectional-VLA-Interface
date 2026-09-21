# Level 3 null-drift calibration runner

This runner calibrates the numerical/physics drift floor before any IA-versus-
SAC first-divergence analysis.  It does not accept an IA event path and never
runs a learned policy.

The frozen v1 design is:

- seed 2025, one exact `reach-state.pt` SHA256 and one recorded SAC `events.jsonl` SHA256;
- the same physical GPU, GPU backend, minimal shader and native Pick config;
- three serial child processes, each constructing a fresh environment, restoring
  the same snapshot and replaying the same first 35 SAC actions;
- 105 actions total, zero training, zero model/API calls, outer wall cap 600 s;
- every repeat must restore all exposed simulator/controller leaves within
  `1e-5`, preserve every action value, remain nonterminal, and emit exactly 35
  post-action rows containing 15 qpos values, TCP-target distance and cumulative
  force;
- per-channel threshold:
  `max(1e-4 unit floor, 2 * maximum pairwise difference across 3 repeats and 35 steps)`.

The `1e-4` floor is metres for translational qpos channels and radians for
rotational channels.  `launch.json` records this rule before the first child.
`thresholds.json` is written only after all three repeats pass.  Raw repeat
events remain under `repeat-00/` through `repeat-02/`.  The result is scoped to
this GPU/backend/configuration and is not a universal physics tolerance.

Bounded lab-server command used by the completed run:

```bash
cd ~/bvi-research/src/Bidirectional-VLA-Interface
PYTHONPATH=src:scripts timeout --signal=TERM --kill-after=20s 600s \
  ~/bvi-research/envs/acdit/bin/python scripts/run_s1_null_drift_calibration.py \
  --seed 2025 \
  --reference-state ~/bvi-research/runs/s1-ia-calls-2026-09-18-run01/starts/seed2025/reach-state.pt \
  --expected-reference-sha256 332928111d25f1cf03060cd760bb17d99c4204d5f0890d807ce24f3e00b20c5b \
  --actions-jsonl ~/bvi-research/runs/s1-ia-calls-2026-09-18-run01/starts/seed2025/events.jsonl \
  --expected-actions-sha256 f68ef63fd460a382625339c29a0cc6854071543d10fb2aa7bbcce006c1d11d01 \
  --expected-task-plan-sha256 135915353c8a56230fcbc68ada7f6a7d58d86119d8c8d1acbb15790d6da08c52 \
  --expected-spawn-sha256 26e0a345b396bf9aee609eee07cf462424d1cfb71650fea88e118b7fe36c362c \
  --sim-python ~/bvi-research/envs/acdit/bin/python \
  --output ~/bvi-research/runs/s1-level3-null-drift-2026-09-18-run01 \
  --repeats 3 --max-actions 35 --per-repeat-seconds 170 --max-seconds 570 \
  --execute
```

Omit `--execute` for a read-only preflight manifest printed to stdout.  The
parent launches children serially with a process timeout, refuses a busy GPU,
and fails if GPU memory does not return below 1024 MiB at completion.

Execution result: run01 failed closed before action 1 on a controller metadata
compatibility field; run02 completed all 105 actions.  The three raw event files
were identical and all calibrated qpos thresholds therefore used the `1e-4`
unit floor.  The subsequent exact-start comparison is reported in
[s1-level3-first-divergence-2026-09-18.md](s1-level3-first-divergence-2026-09-18.md).
