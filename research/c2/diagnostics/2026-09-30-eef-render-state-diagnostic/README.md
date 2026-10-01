# Case5 zero-action render diagnostic: cause isolated, recording path fixed

2026-09-30 UTC. User approved the annotated zero-action diagnostic. One GPU1 process, cap120 seconds, API0 and environment actions0. Actual35.074310678988695 process seconds. GPU1 exited; final resource check retained; unused scope closed.

## Findings
The old visibility-restoring wrapper changed51 state entries: ee_rest_goal pose and50 object state records (quaternion components and some y-position components), maximum non-marker component difference2.86102294921875e-6. This is not a robot action or a model decision. Simply restoring hidden flags is insufficient. The exact-state guard correctly detected the changes; it has NOT been relaxed.

The diagnostic then directly updated/read the single frozen human-render camera, without calling environment.render() or toggling marker visibility. Full-state differences:0. Subsequent robot-sensor read differences:0. Evidence: [summary](summary.json), [full result](remote-results/result/result.json), [original diff](remote-results/result/recording-state-diff.eval-only.json).

## Fix and validation
AttemptRecording.capture_environment now uses scene.update_render(update_sensors=False, update_human_render_cameras=True) and the single camera image. It retains exact full-state equality and evaluation-only field-diff reporting. Robot sensor inputs and official physics/scoring are unchanged. No force/progress interruption added.

[452 CPU tests passed after fix](cpu-tests-after-fix.json); [fixed-source hashes](fix-source-sha256.json). Candidate was checked after the old wrapper within this zero-action diagnostic, not in a fresh full movement trial. A full interface replay from a clean snapshot remains necessary before claiming the interface/G2 gate passed. No strict Pick success or Astra capability claim.

## Artifacts and accounting
[Usage](remote-results/usage.json), [resources](resource-final-check.json). Raw initial captures reside in remote-results/result/recording-frames/. They show rendering variants at zero simulation steps; there is no motion trajectory/video and we do not manufacture one.

Lab charge0 per user confirmation, no invoice; other unknown historical costs unchanged. Finance ledger-eef-render-diagnostic-20260930-r1.json reconciled. No old budget transferred, no automatic follow-on GPU/API run.

Next gate: freeze corrected source/deployment and run the full API0 interface checks under their own bounded approval. Old r2/r3 failures remain intact.
