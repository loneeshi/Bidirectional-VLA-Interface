# Fetch TAPT preparation — not yet trained

The native AC-DiT backbone is frozen. Native success rate is a baseline measurement, not a required threshold before TAPT. Interface errors remain gates.

Official SetTable Pick/Place sources are pinned and inspected: [source audit](results/fetch-tapt-preparation-2026-09-16/source-audit.json). Each contains1000 trajectories but lacks world base velocities required by the AC-DiT loader. Do not silently zero-fill. A separate fixed SAC teacher collector records these fields, pointclouds and images under the pinned AC-DiT environment. These are newly generated teacher demonstrations, not original official records or proven LightNav handoff coverage.

Fixed seeds3000–3024 per task:20 training,5 validation before segmentation. Retain all failures. Pilot Pick seed3000 succeeded in29 steps with required fields. Collection uses GPU1,3600-second outer timeout and240-second episode limits. No parameter training/API calls yet.

`src/bvi/acdit_tapt.py` implements explicit four-bank residuals plus a shared learned progress head. `ACDiTFamilyTool` preserves native action compute_loss and captures manipulation DiT action-token features before final projection. Rank8/alpha8 and projection insertion are an AC-DiT port, not claimed identical to OpenPI. Final sites must be explicitly configured and recorded. Two small CPU tests passed; full-model gradients, memory, input preprocessing and training/inference progress distribution remain unverified. No simulated progress substitutes for learning.

Next: segment real demonstrations with observable completion predicates; verify feature extraction, frozen weights and bank isolation on the full model; run20 training updates before the bounded2000-update/two-hour run. DROID/GRPO are still unimplemented.


Observable pilot segmentation is implemented in `fetch_segments.py` (reach8cm, place15cm, stable grasp3 observations; explicit port annotation choices). Completion uses predicate evidence, never simply end-of-recording. Three boundary tests passed. The20-update full-model gate is implemented and queued after collection: all four families in both splits, idle GPU, native loss/preprocessing, bank-gradient isolation, frozen-native full hash, held-out examples and recoverable checkpoint. It uses one fixed example per family/split for integration only, not full SFT. Runtime results remain pending. Gate timeout900s, outer wait4600s; do not launch concurrent copies.


## Actual gate result and current bounded SFT

[Twenty-update result](results/fetch-tapt-gate-2026-09-16/result.json): passed, all frozen native tensors unchanged, each of4family banks updated168tensors and learned progress parameters changed. Peak allocated12,920,481,792bytes. Local checkpoint SHA256 `b541531b57cc388f7560e55ee3d4080ecbc89da901cc89e4a53f318760118390`. This is a real training integration gate, not online skill success.

All50teacher episodes collected. Place reset initially has no physical contact, so segmentation now begins at an evidenced held interval rather than rejecting the entire episode. Segments train19/19/19/2 and validation3/3/4/1 for reach/grasp/move/release; low release coverage remains a limitation. [Collection and labels](results/fetch-tapt-gate-2026-09-16/collection.json).

Bounded native-start SFT pilot is launched with `gate_fetch_tapt_training.py --full`: per-call start/middle/end features across all labeled trajectories, uniform family batches, microbatch1/accumulation8, at most2000updates or6900training seconds. Outer9000s includes preprocessing/cleanup. Save every50updates; validate and checkpoint every200; select by held-out joint loss. This is an AC-DiT TAPT port with no DROID/GRPO or verified navigation-handoff coverage. Full SFT outcome and online evaluation remain pending.
