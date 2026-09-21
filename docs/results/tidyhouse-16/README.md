# TidyHouse 16-plan public result

| Setting | Completed episodes | Full-task SR | Completed objects | Mean / episode |
|---|---:|---:|---:|---:|
| Fixed PPO + SAC | 16/16 | 0/16 | 14/80 (17.5%) | 0.875 |
| GPT + PPO + SAC | 16/16 | 0/16 | 13/80 (16.25%) | 0.813 |
| Teleport + SAC | 16/16 | 0/16 | 21/80 (26.25%) | 1.313 |

[`summary.json`](summary.json) is the minimal public machine record. It contains
the 16 per-episode rows for each setting, including plan UID, seed, completed
objects, termination reason, environment steps, API requests, initial-state
hash and source-panel hash.

The human-readable configuration, observations and conclusion are in the
[experiment log](../../log/2026-09-21-tidyhouse-three-settings.md).

This is a 16-plan diagnostic, not the 1,000-rollout benchmark. All three
settings have 0/16 complete-task success. Teleport's 21/80 records partial
progress under a different navigation handoff distribution and is not evidence
of a superior learned navigation policy.
