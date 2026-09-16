# Stage 1: native capability and interface fidelity, 2026-09-16

Still at mainline stage 1. Fetch TAPT, DROID and GRPO have not started on AC-DiT.

## Fixed native AC-DiT screen

Predeclared seeds 2024–2028, apple Pick validation, 200 action budget; seed 2024 reuses the archived native episode. No retries or success filtering. Community checkpoint, FP32 on lab GPU1; privileged context retained. This is a small capability screen, not reproduction of a paper score.

| Seed | Steps | Ever grasped | Native result |
|---|---:|---|---|
| 2024 | 199 | yes | Failed: rest/static predicates did not coincide |
| 2025 | 24 | no | Cumulative force 5250.28 > 5000 |
| 2026 | 53 | no | Cumulative force 5027.69 > 5000 |
| 2027 | 58 | no | Cumulative force 5434.25 > 5000 |
| 2028 | 43 | no | Cumulative force 5018.41 > 5000 |

**0/5 native success.** These logs establish failure stages, not the causal contribution of checkpoint capability versus precision, observations or start conditions. Do not relax force thresholds, add hidden controllers or count grasp as native success. Next: separately labeled official SAC reference with verified matching starting state; then isolate remaining train/eval differences.

Evidence: [batch](results/acdit-fixed-seeds-2026-09-16/batch.json), per-seed result/events in the same directory; prior seed 2024 in `results/acdit-native-2026-09-16/`. Downloaded raw archive SHA256 `dac8ec74b4cd248eb4d38f800102770892d13db3e576e44915a09d67b6104ce6`.

- [acdit-native-pick-seed2025-failed.mp4](media/acdit-native-pick-2026-09-16-run02/acdit-native-pick-seed2025-failed.mp4)
- [acdit-native-pick-seed2026-failed.mp4](media/acdit-native-pick-2026-09-16-run02/acdit-native-pick-seed2026-failed.mp4)
- [acdit-native-pick-seed2027-failed.mp4](media/acdit-native-pick-2026-09-16-run02/acdit-native-pick-seed2027-failed.mp4)
- [acdit-native-pick-seed2028-failed.mp4](media/acdit-native-pick-2026-09-16-run02/acdit-native-pick-seed2028-failed.mp4)

## LightNav installation and inference

Pinned author source `c6f40e3220edbf7011e4f17eaf2c865416737d4d` and checkpoint `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64`. Isolated torch 2.10/CUDA 12.8 environment. Explicit opt-in FP32 patch accommodates Turing GPU; default author BF16 remains unchanged. Author CPU tests: 478 passed, 9 skipped.

One archived real MS-HAB navigation frame passed HF inference with `task_key=vlnce`, `task_type=vlnce_traj`: 1.43 seconds, peak allocated 18.02 GB (decimal), finite decoded waypoints. [Result](results/lightnav-native-2026-09-16/result.json). Its first waypoint has near-zero translation and -0.3141 rad yaw. This is **not** an online navigation evaluation or proof of pathological spinning.

Diagnostic-only history scope and first-waypoint mapping are in `navigation_fidelity.py`; existing baseline unchanged. Project tests: 150 passed, 1 skipped. Online history/instruction pairing is pending. Sparse archived frames cannot reconstruct the original full observation history.

## Resource status

All jobs in this batch exited; lab GPU1 returned to 15 MiB. Model API calls 0, training updates 0, new rented compute 0. Lab service price unknown. Two historical Runpod pods remain EXITED; their total 40 GB persistent storage remains chargeable (~USD 0.266667/day at previously verified rates).

## SAC reference input audit (subsequent batch)

Official apple SAC checkpoint pinned to MS-HAB weights revision `91e96be85128df43728a7511355c3fa999bd2c94`, SHA256 `1d77da6845a190768792d08a63c2f3e6b19f1022865c350b9ea34f0c9b3aa5cb`. The AC-DiT environment fork adds 9 base position/velocity dimensions; official SAC expects 42 state dimensions, so the standalone reference explicitly selects the four official extra fields in source order. Official `evaluate.py` casts depth tensors to float before inference; the reference now does the same without additional normalization. Three integration failures (state schema, dtype, noncontiguous tensor layout) stopped before any action and are preserved separately.

Seed 2025 reset qpos and both camera RGB/depth arrays exactly match the archived AC-DiT reset. This is still observation-level matching only: the old AC-DiT run has no complete initial simulator/controller snapshot. The new reference saves simulator state, public controller state and RNG data for subsequent audit; contact solver internals and all task-private counters are not claimed to be captured. Do not describe it as exact full-state causal pairing.


The corrected reference completed **native Pick successfully at step 61** (seed 2025), cumulative force 3073.72 < 5000; grasp, rest and static predicates all true. AC-DiT previously failed this seed at step 24. This supports feasibility of the observed starting configuration; it does not isolate AC-DiT precision/interface/capability causes. SAC uses its own official policy contract and is not counted as a VLA success.

Video: [sac-reference-seed2025.mp4](media/sac-native-reference-2026-09-16-run01/sac-reference-seed2025.mp4). [Result](results/sac-native-reference-2026-09-16/seed2025-reference-v1/result.json). Raw archive SHA256 `baff08a7d156a0829bfd45209e450f3bc3ddfb54dfe87728b8841380071918eb`; failed integration attempts retained. GPU1 returned to 15 MiB, API/training 0. Next: remaining AC-DiT input/history/precision audit and navigation online paired checks before TAPT.
