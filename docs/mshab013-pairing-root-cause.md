# MSHAB013 pairing failure: unpinned Python hash seed

2026-09-16 UTC. This diagnoses the experiment's pairing failure, not a solved Fetch grasp or a benchmark result.

## Evidence

MSHAB013b used separate simulation processes, but did not set `PYTHONHASHSEED`. The failed second case had 114 differing serialized leaves, of which 96 exceeded the numerical tolerance or differed as discrete values. Robot qpos/qvel, TCP, target pose and the newly captured controller targets matched. The same task/build IDs `[2]`/`[69]` had different internal initialization indices `[31]`/`[33]`.

Thirty-six hidden actors had different locations, with a largest displacement of 540 m in their far-away hiding area. This is bookkeeping/construction-order variation, not a 540 m displacement of the grasp target. Visible non-target actors also differed: the largest measured positional difference was 1.69 mm for `010_potted_meat_can-1`. Its angular velocity differed by approximately 0.0687 rad/s.

The pinned MS-HAB source constructs `_init_config_names` as a `set`. The pinned ManiSkill ReplicaCAD rearrangement builder iterates that collection to construct initialization indices, and iterates another `obj_ids` set to create actors. An environment reset seed does not fix Python's process-start string hash seed. A fresh process therefore did not guarantee the same construction order.

Two fresh simulation processes were then started with **`PYTHONHASHSEED=0`** and each replayed the same 330 navigation actions from reset. Their complete serialized simulator/public-controller states compared byte-for-byte equal (`cmp`, exit 0). No saved state was restored and no success flag was changed. Source evidence is preserved in the MSHAB013 archive under `runs/hash0-a/0/state.json`, `runs/hash0-b/0/state.json`, and `evidence/hash0-a.log` / `hash0-b.log`.

This confirms that fixing the hash seed is sufficient for these two measured start-state replays. It does not establish global PhysX determinism over arbitrary scenes, seeds or long trajectories. The earlier interpretation that the failure was simply irreducible physics nondeterminism was premature.

## Fix and verification boundary

The diagnostic entry point now fails before environment creation unless `PYTHONHASHSEED=0` was set **before launching Python**. Setting it inside the running interpreter is insufficient. The value is recorded in each diagnostic configuration. Independent processes, complete-state comparisons, explicit controller targets, request-scoped policy noise, repeated-inference checks and matched observation/action prefixes remain enabled.

The next full matrix uses a new output directory `mshab013c`, preserving the failed `mshab013` and `mshab013b` runs. Its results are pending at the time of this note. A failed pairing gate is not counted as a completed failed task episode.

## Separate manipulation failure

The completed template/stop run in MSHAB013b still missed the grasp in 31 Pick steps. Its closest TCP-origin to target-origin distance was 0.102805 m; its first closing command below −0.5 occurred at step 26, with distance 0.149517 m after that action. No sampled finger–target contact occurred. All 31 model requests acknowledged their deterministic noise; the first identical-input/noise repeat had maximum action error 0.

Thus the sampling repair is working for the measured requests, while spatial approach/alignment remains unresolved. These observations do not distinguish insufficient visual feedback, state/action adaptation, training coverage, or base–arm coordination. The controlled matrix and labelled teacher channel interventions are intended to make that distinction. Teacher-assisted trajectories cannot be counted as pure VLA successes.
