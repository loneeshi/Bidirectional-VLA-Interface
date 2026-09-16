# MS-HAB Pick diagnostic matrix run01

All five cases reconstruct the same audited 330-step navigation prefix from reset. The equality gate measured zero maximum difference for qpos, qvel, TCP pose and target-object pose in every case. No model API was called and no training occurred.

| Case | Pick steps | Grasped | Outcome |
|---|---:|---:|---|
| template + interrupt | 34 | no | monitor `missed_grasp` |
| recorded GPT instruction + interrupt | 33 | no | monitor `missed_grasp` |
| template + observe only | 95 | no | native `benchmark_fail` |
| recorded GPT instruction + observe only | 120 | no | diagnostic step limit |
| official object-specific SAC reference | 51 | yes | Pick subtask advanced |

## Attribution

**Updated attribution (2026-09-16):** the observed π₀.₅ failures occur before target contact. They are not evidence of a grasp followed by slipping. The observe-only cases show that permitting longer execution did not rescue these two rollouts. They do not establish that the monitor or instruction has no causal effect: the policy service was not reseeded between cases, and the same-prompt cases already differ on their first action. SAC establishes that this measured start permits a successful Pick with that reference policy; it does not establish that the start is supported by the π₀.₅ training distribution.

See [the contact and training-support audit](mshab-pick-matrix-attribution.md) for distances, closure timing, the force-limit failure, interface checks, and the next controlled experiment. This correction supersedes the earlier wording that excluded the monitor, prompt or navigation pose as causes.

The SAC case is an independent diagnostic reference and is excluded from pure-VLA success claims. These are single-start exploratory diagnostics, not a randomized causal estimate or benchmark success rates. Start equality covered four logged arrays, not the complete simulator/controller state or policy RNG.

Raw evidence archive: `runs/mshab012/mshab012-final.tar.gz`, SHA256 `9dd57423bfbf81219ad2dd7f367f4d51f59dabd03ba9265ee64782e14f3215ab`.
