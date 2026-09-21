# Current-frame calibration fixed online diagnostics

Both episodes failed to complete native Pick, neither ever grasped. Selected checkpoint: additional step20 from1199-update warm start. Seed2025 hit native cumulative-force failure; seed2030 requested coordinator replanning on learned-progress drop. This diagnostic has no GPT, so it stops at the request. No navigation.

| Exact filename | Seed | Actions | Native success | Stop |
|---|---|---|---|---|
| [seed2025-fetch-tapt-pick-episode000-failed.mp4](seed2025-fetch-tapt-pick-episode000-failed.mp4) | 2025 | 31 | False | native_terminated |
| [seed2030-fetch-tapt-pick-episode000-failed.mp4](seed2030-fetch-tapt-pick-episode000-failed.mp4) | 2030 | 26 | False | learned_drop_requires_coordinator |
