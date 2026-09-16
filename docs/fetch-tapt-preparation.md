# Fetch TAPT preparation — not yet trained

The native AC-DiT backbone is frozen. Native success rate is a baseline measurement, not a required threshold before TAPT. Interface errors remain gates.

Official SetTable Pick/Place sources are pinned and inspected: [source audit](results/fetch-tapt-preparation-2026-09-16/source-audit.json). Each contains1000 trajectories but lacks world base velocities required by the AC-DiT loader. Do not silently zero-fill. A separate fixed SAC teacher collector records these fields, pointclouds and images under the pinned AC-DiT environment. These are newly generated teacher demonstrations, not original official records or proven LightNav handoff coverage.

Fixed seeds3000–3024 per task:20 training,5 validation before segmentation. Retain all failures. Pilot Pick seed3000 succeeded in29 steps with required fields. Collection uses GPU1,3600-second outer timeout and240-second episode limits. No parameter training/API calls yet.

`src/bvi/acdit_tapt.py` implements explicit four-bank residuals plus a shared learned progress head. `ACDiTFamilyTool` preserves native action compute_loss and captures manipulation DiT action-token features before final projection. Rank8/alpha8 and projection insertion are an AC-DiT port, not claimed identical to OpenPI. Final sites must be explicitly configured and recorded. Two small CPU tests passed; full-model gradients, memory, input preprocessing and training/inference progress distribution remain unverified. No simulated progress substitutes for learning.

Next: segment real demonstrations with observable completion predicates; verify feature extraction, frozen weights and bank isolation on the full model; run20 training updates before the bounded2000-update/two-hour run. DROID/GRPO are still unimplemented.
