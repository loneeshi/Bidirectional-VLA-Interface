# Visual interface r3: attempted fix, retest still blocked

2026-09-30 UTC. User authorized repair and retest. Scope GPU1 serial cases5/55/58, 3x300=900 process seconds, API0, no transferred balance. Bundle SHA256 7b411510dc2ab050a196d21dfd14596164a3096d3a7d04df8d99c6d5771b4859. CPU452 passed.

Pinned ManiSkill render_rgb_array shows hidden visual actors and returns early for a single camera; GPU hide_visual translates collision-free visual actors. We added a visibility-restoring recording wrapper while retaining exact full-state equality. This reproduced mechanism in CPU mocks but did NOT resolve the real GPU state mismatch.

Case5 stopped during initial recording: evaluation recording changed simulator state after visibility restoration. Simulator0/API0. Cases55/58 not started. This is an interface/infrastructure censoring, not a Pick failure. No motion trajectory/video is available; initial frame is preserved. [Result](remote-results/results/plan-005-visual-interface/result.json), [delivery](remote-results/results/plan-005-visual-interface/delivery/README.md).

Actual GPU1 34.48411449044943 process seconds; project charge0 per user, no invoice. Unused scope closed; no automatic retry. [Final resources](resource-final-check.json). No Astra requests and no new success. Finance ledger-eef-visual-interface-20260930r3.json reconciled.

After this batch, CPU-only field-diff instrumentation was added to recording; it is NOT in the r3 frozen deployment. It preserves before/after values and exact numeric differences in recording-state-diff.eval-only.json and retains the strict guard. Ten focused tests pass. Cause remains unresolved pending field evidence; do not claim a tolerance problem or loosen checks.

Next gate: a separately bounded zero-action render diagnostic on case5 before another full interface batch. No model pilot until interface passes.
