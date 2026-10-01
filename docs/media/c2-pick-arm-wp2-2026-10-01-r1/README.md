# WP2 gripper geometry r1: Astra goals and real execution in the same 3D scene

Run `arm-wp2-geometry-20261001-r1`, recorded 2026-10-01 UTC. Same five development cases as the earlier mobile r1; the only model-input change is a robot gripper geometry paragraph. Strict Pick 2/5 within 200 steps and 2/5 within 600 steps.

CPU postprocessing of the original online recordings; no simulator or API rerun, no action replay. Pink dashed: issued Astra TCP waypoint connectors (not controller interpolation or a physical forecast). Cyan: measured TCP history. Orange: base metric commands. X-ray overlay, without depth occlusion; evaluation-only.

| Case | Outcome | Steps | Video | SHA-256 |
|---|---|---:|---|---|
| arm-dev-000 | success | 137 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-8a8902b7/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `13f30022a792bc24…` |
| arm-dev-001 | failure — model gave up | 521 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-90194498/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `44ca30857398146d…` |
| arm-dev-002 | failure — cumulative force limit | 199 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-2c00e1ec/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `ae351249be790d6e…` |
| arm-dev-003 | failure — wrong object, model reported done | 313 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-83a042d7/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `de6b400ca6ec22f9…` |
| arm-dev-004 | success | 130 | [astra-commanded-vs-executed-3d-demo.mp4](../c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-f17e6eca/delivery/astra-commanded-vs-executed-3d-demo.mp4) | `7d2ec94fb1483e01…` |

Full hashes, byte sizes and recording times: [export-receipts.json](export-receipts.json). Experiment log: [2026-10-01 WP2 gripper geometry](../../log/2026-10-01-c2-arm-wp2-gripper-geometry.md).
