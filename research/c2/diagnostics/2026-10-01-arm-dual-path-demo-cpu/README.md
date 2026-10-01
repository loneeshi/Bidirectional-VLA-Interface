# Astra issued paths and actual execution: CPU media export

Stage 2 media delivery, 2026-10-01. The five completed V-mobile attempts now have 512×512, 20 fps scene demos matching the user's example format. [Video index](../../../../docs/media/c2-pick-arm-dual-path-2026-10-01-mobile-r1/README.md).

Pink dashed lines retain Astra's issued TCP goal connectors; cyan retains the measured TCP trajectory; orange shows issued base commands. Astra supplied goal poses, not a per-step physical forecast. Historical points are reprojected through the current camera. Commands appear only after issue, never before. The X-ray overlay has no depth-buffer occlusion and is labelled evaluation-only.

Sources are original online recordings and archived evaluator trajectories. No action replay, new simulator steps, API calls, or GPU evaluation were performed. Original recordings and experiment outcomes remain preserved: one strict success and four failures. Source and output SHA-256 hashes, frame counts, resolution, recording times and outcomes are in the export receipts and verification.json.

Validation: 18 CPU tests passed, covering issued-history retention, no future commands, simultaneous planned/actual overlays, existing trajectory projection, and arm-runtime regression. A frame from arm-dev-000 was visually inspected. All five output hashes were verified.

The current finalizer now adds this scene demo as the preferred export for future attempts. The future mobile freeze includes the updated finalizer and renderer. Historical frozen execution packages were not regenerated. Next gate: user review of the presentation; no new experiment authorization is implied.

`register.py` registers this batch and stages only its additions to shared media indexes, preserving unrelated working edits. It is a local recordkeeping helper.
