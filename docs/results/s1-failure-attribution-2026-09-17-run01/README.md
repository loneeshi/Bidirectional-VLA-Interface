# S1 failure attribution evidence

The native panel remains 0/10. No training updates or new online episodes were performed in this diagnostic batch.

- `rollout-metrics.json`: CPU summary of all ten archived native episodes; exact seed roster and complete event counts rechecked.
- `s1-teacher-action-audit-2026-09-17-run01/`: retained initial audit. Its frame100 samples are outside the success-trimmed training exports and **must not be used as evidence of training-observation fit**.
- `s1-teacher-action-audit-2026-09-17-run02/`: corrected eight-frame audit, using first/middle positions inside each parent's exported interval. Actual model predictions and expert actions are in the NPZ files; this is a small diagnostic, not benchmark evaluation or checkpoint selection.
- `provenance-and-resource-check.json`: retrospective CPU verification of H5 → exact dataset manifest → train-only normalizer binding. It also records GPU1 at 15 MiB / 0% and exited outer processes after both audits.
- `manifest.json`: SHA256 and byte lengths of published evidence files; manifest itself excluded.

The source archive `s1-action-audit-evidence-2026-09-17.tar.gz` is backed up locally (653,488 bytes, SHA256 `26d409a4f97bd80154b7a9bf6c55bd416723ca0c70b764a95dec7016fdf10b54`). Authentication material was excluded before archiving. Original logs are preserved; metadata's initial `loading` state is superseded by each server's `result.json` and audit `summary.json`.

Analysis and limitations: [S1 native failure analysis](../../s1-native-failure-analysis.md).
