# SAC native-spawn pilot v1

Authorized by the user's 2026-09-21 request to implement, publish and run the experiments.
First executable gate only: 24 isolated official train-spawn cases, 200 actions each,
at most 3600 cumulative runner seconds, GPU1 only, zero API calls, no training.
Lab monetary cost remains unknown; zero API use does not mean free laboratory compute.

Two object categories (bowl/gelatin box), Pick/Place, seeds 100–102, spawn indices 0/1.
Actual parent UID and scene are saved after reset. Seeds are not presumed distinct
scenes: group results by actual parent binding, disclose duplicate parents and do
not claim 24 independent samples. Native spawn support is not a success guarantee.

Official checkpoint eval configuration and depth preprocessing are retained with
one environment and a 200-action observation horizon. This is not the RGB-D
long-horizon comparison. No force/fail-based early exit; numerical or infrastructure
failures stop the batch. Native success, force violations and final info are saved.
The full 24-row denominator is retained; failures and unrun rows are never dropped.

Each worker has a 300-second technical timeout, charged to the shared one-hour cap.
Resume skips all previously attempted cases; unclean running rows require manual
budget reconciliation. Full initial state, RNG and observation are archived, but
exact controller/frame-stack replay is NOT yet validated. Videos remain private
run artifacts until publication with hashes under the media policy.

This runner does not implement C0/C1/C2 continuous long-horizon scheduling, generate
a success-region prior, or demonstrate a navigation-to-manipulation causal effect.
Those gates remain pending; no learned prior is invented from checkpoint metadata.

## Verified launch

Executable commit: `9879e59`, branch `research/sac-spawn-probe-20260921`.
Pinned official runtime: `e9ff3d23496d38e4431c8d913e147ffa007f7f72`.
Remote root: `/home/pshuai/bvi-research/runs/sac-spawn-probe-20260921`.
Active attempt: `attempt-004`, launcher PID at verification: `3226646`.
The first case completed 200 actions and reached native Pick success at step 35;
the next case was running. This is not a panel SR or a long-horizon result.
Prior attempts failed before rollout (single-env scene allocation, unsupported
numeric scene ID, and missing official float conversion). All were preserved;
their total elapsed time was 90.838 seconds, leaving a conservative 3300-second
cap on the current attempt. No API requests or new rented resources.

Runner SHA256: `7e46ff4639f854be9a5cc4ae62f2a8bd204e1bacbc9fc9d3d2cfc5a0f0d1beea`.
Source archive and immutable per-attempt logs remain under the remote root.
