# Official-state r1: Astra goals and real execution in the same 3D scene

Run `arm-official-state-20261001-r1`, recorded 2026-10-01 UTC. Same five development cases as WP2. Astra additionally receives the official MS-HAB state fields `obj_pose_wrt_base`, `tcp_pose_wrt_base`, `goal_pos_wrt_base` and `is_grasped`; prompt `guidance-official-state-v1`, same executor as WP2. Strict Pick 2/5 within the official 200 steps and 5/5 within the non-official 600-step horizon. All five attempts are official successes; none is a failure or a censored run.

CPU postprocessing of the original online recordings; no simulator or API rerun, no action replay. Pink dashed: issued Astra TCP waypoint connectors (not controller interpolation or a physical forecast). Cyan: measured TCP history. Orange: base metric commands. X-ray overlay, without depth occlusion; evaluation-only.

| Case | Outcome | Steps | Video | SHA-256 |
|---|---|---:|---|---|
| arm-dev-000 | success (after 200) | 213 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `fefba069b82f5ee6…` |
| arm-dev-001 | success (after 200) | 239 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `8e27c07fae2c2faf…` |
| arm-dev-002 | success | 123 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `dcd31b87f8e1cc2e…` |
| arm-dev-003 | success (after 200) | 257 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `37184a3346835b88…` |
| arm-dev-004 | success | 145 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `84b93e01a2c50e84…` |

Linked analysis pages per case (evaluation-only):

| Case | Trajectory | Step history | Contacts |
|---|---|---|---|
| arm-dev-000 | [trajectory.html](../c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/trajectory.html) | [history.html](../c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/history.html) | [contacts.html](../c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/contacts.html) |
| arm-dev-001 | [trajectory.html](../c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/trajectory.html) | [history.html](../c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/history.html) | [contacts.html](../c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/contacts.html) |
| arm-dev-002 | [trajectory.html](../c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/trajectory.html) | [history.html](../c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/history.html) | [contacts.html](../c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/contacts.html) |
| arm-dev-003 | [trajectory.html](../c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/trajectory.html) | [history.html](../c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/history.html) | [contacts.html](../c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/contacts.html) |
| arm-dev-004 | [trajectory.html](../c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/trajectory.html) | [history.html](../c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/history.html) | [contacts.html](../c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/contacts.html) |

Contacts are sampled from the last physics substep once per control step; contact positions are unavailable, and these samples do not replace the official cumulative force. No contact information entered the model input.

Full video hashes, byte sizes and recording times: [export-receipts.json](export-receipts.json); analysis page hashes: [attachments.json](attachments.json). Experiment log: [2026-10-01 official-state input](../../log/2026-10-01-c2-arm-official-state.md).
