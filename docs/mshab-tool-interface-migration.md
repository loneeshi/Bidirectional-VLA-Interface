# MS-HAB tool interface migration

2026-09-16: LIBERO recovery ablation is deferred. Mainline is GPT + LightNav-0 + Fetch-adapted pi0.5 in MS-HAB.

`--tool-family-interface --organizer --navigation-policy lightnav --manipulation-policy fetch-pi05` enables `mshab-tool-family/1`. Every GPT response includes `tool_family`, exact scene-grounded `instruction`, and version; family must match the admissible skill. Both backends receive that instruction. Legacy fixed prompts remain available only outside this opt-in mode. Navigation cannot silently substitute configured recovery text; it returns control to GPT. Action/history caches reset at each invocation, while Fetch's training-relative base origin persists across bounded calls inside the same native subtask.

This first migration uses coarse `navigate/pick/place` skills and distinct navigation/manipulation backbones. It does **not** implement four Fetch reach/grasp/move/release residual banks, a learned Fetch progress head, or full TAPT training. Progress is explicitly null/unavailable; completion and grasp monitoring disclose benchmark/rule provenance. The benchmark still restricts admissible skills, so this is constrained execution coordination, not free-order task planning.

Initial diagnostic: seed 1, plan `tidy_house-sequential-val-90-0`, first four subtasks (navigate, pick, carried navigation, place). Passing this chain is partial-task engineering success, not full five-object benchmark success. No SAC takeover in the new interface mode. The existing V8 Fetch checkpoint previously failed grasping; changing the interface must not relabel it as validated.

Offline checks cover strict schema/version/family validation, GPT text preservation, cache invalidation, navigation history reset and Fetch relative-origin persistence. 124 tests pass; new online outcome pending. Next gates are real model input/output compatibility, actual 13-channel actions, native grasp retention, and first-object chain completion. Keep all failed runs and per-request cost records.
