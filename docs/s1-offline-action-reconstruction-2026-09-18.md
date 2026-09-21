# S1-IA Level-1 offline action reconstruction (2026-09-18)

## Result

The frozen held-out-selected S1-IA checkpoint `best/855` completed 512
teacher-forced predictions on 128 recorded validation observations: 32 distinct
parent/invocation calls, four temporally spread observations per call, and four
independent flow samples per observation.  The roster contains 47 reach, 39
grasp and 42 move observations.  No simulator was created; simulator steps,
rollouts, training updates and external API calls are all zero.

This result rules against a wholesale 13-channel permutation or normalization
failure.  The distribution-mean prediction beats the zero-controller reference
on every one of the eleven active non-head channels when pooled across the
fixed sample.  It does **not** clear the full offline gate: yaw is barely better
than zero when pooled, and yaw/torso contain family-conditioned slices that are
worse than zero.  New VLA training therefore remains paused.

The table is the strict first action at recorded observation `t`.  RMSE uses the
physical command passed to the pinned sub-controller after quantile
unnormalization, clipping and the stationary-head mask.  These are command
units, not measured post-physics displacement.  `Zero RMSE` is the error from a
zero normalized controller command.  Correlation is same-channel Pearson
correlation across the 128 observation targets.

| Ch. | Command | Unit | Policy RMSE | Zero RMSE | Ratio | Pearson r | Raw OOB |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | shoulder pan | rad delta | 0.0505 | 0.0655 | 0.772 | 0.568 | 5.5% |
| 1 | shoulder lift | rad delta | 0.0426 | 0.0887 | 0.480 | 0.852 | 20.1% |
| 2 | upper-arm roll | rad delta | 0.0555 | 0.0832 | 0.667 | 0.651 | 20.3% |
| 3 | elbow flex | rad delta | 0.0314 | 0.0880 | 0.357 | 0.934 | 27.7% |
| 4 | forearm roll | rad delta | 0.0318 | 0.0856 | 0.372 | 0.923 | 24.8% |
| 5 | wrist flex | rad delta | 0.0331 | 0.0934 | 0.354 | 0.935 | 31.3% |
| 6 | wrist roll | rad delta | 0.0449 | 0.0585 | 0.769 | 0.602 | 0.6% |
| 7 | gripper absolute target | m | 0.0116 | 0.0273 | 0.424 | 0.905 | 19.9% |
| 8 | head pan, masked | rad delta | 0 | 0 | — | — | 0% |
| 9 | head tilt, masked | rad delta | 0 | 0 | — | — | 0% |
| 10 | torso lift | m delta | 0.0557 | 0.0666 | 0.836 | 0.577 | 11.7% |
| 11 | base forward | m/s | 0.3855 | 0.6863 | 0.562 | 0.797 | 13.3% |
| 12 | base yaw | rad/s | 0.8976 | 0.9099 | 0.986 | 0.283 | 0% |

The raw head predictions are approximately zero (`1.8e-8` and `1.6e-8`
controller RMSE), but the wrapper still records that it explicitly overwrote
them.  Raw out-of-bounds frequency is reported as a symptom; clipping is the
pinned controller contract and is not itself a causal diagnosis.

## Family-conditioned weaknesses

The pooled result hides the decisive failures:

- reach shoulder-pan RMSE is 1.029 times the zero reference; reach yaw is 0.990;
- grasp torso RMSE is 1.060 times zero;
- move torso is 1.156 times zero;
- move yaw is 1.246 times zero, with same-channel correlation `-0.116`.

Conversely, elbow, forearm, wrist-flex and gripper reconstruction is strong in
the pooled result.  Some off-diagonal correlations exceed the diagonal because
expert joints co-move; that is not evidence of a permutation when the same
channel itself remains strongly correlated.  Yaw is the one clear weak channel,
not a global channel-order failure.

## Scope and evidence

The primary comparison uses the mean of four stochastic samples.  The raw
evidence also reports expected single-sample RMSE and per-observation stochastic
spread.  Later chunk positions are retained as an explicitly open-loop
secondary analysis and are not mixed with the first-action claim.

This is a fixed diagnostic sample from the held-out conversion split of the
official train trajectories, not a benchmark success estimate and not the
MS-HAB validation protocol.  It cannot establish closed-loop stability,
RGB-domain sufficiency or task success.

- [machine-readable summary](results/s1-ia-offline-reconstruction-2026-09-18-run02/summary.json)
- [per-channel table](results/s1-ia-offline-reconstruction-2026-09-18-run02/per-channel.csv)
- [cross-channel correlations](results/s1-ia-offline-reconstruction-2026-09-18-run02/cross-channel-correlation.csv)
- [full timeline](results/s1-ia-offline-reconstruction-2026-09-18-run02/timeline.csv)
- [time curve data](results/s1-ia-offline-reconstruction-2026-09-18-run02/time-curve.csv)
- [time curve SVG](results/s1-ia-offline-reconstruction-2026-09-18-run02/time-curves.svg)
- [sample arrays](results/s1-ia-offline-reconstruction-2026-09-18-run02/reconstruction-arrays.npz)

Run02 wall time was 531.065 seconds, mean inference time 0.916 seconds and p95
0.923 seconds.  The completed process released GPU1 to 15 MiB / 0%.
Run01 is preserved as a zero-inference infrastructure failure caused by missing
remote helper files.  The combined raw archive is
`D:\AI\embodied intelligence\runs\s1-offline-action-reconstruction-2026-09-18-run01-run02.tar.gz`,
1,194,807 bytes, SHA-256
`aa185b3c1bff8e5140d34601808bb03077ac51c9ea5509da90dc9e4286f9daf0`.
