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

The monitor shortens failed rollouts, but it is not the root cause: π₀.₅ still fails with both instructions when allowed to continue. Prompt wording is also not the leading cause because neither prompt condition grasps. The identical-state SAC success shows that the post-navigation pose is reachable and the benchmark Pick can succeed from it. The remaining leading fault domain is the Fetch π₀.₅ policy/data/interface adaptation: action/state normalization, camera/state distribution, action semantics, or insufficient successful manipulation data.

The SAC case is an independent diagnostic reference and is excluded from pure-VLA success claims. These are single-state causal diagnostics, not benchmark success rates.

Raw evidence archive: `runs/mshab012/mshab012-final.tar.gz`, SHA256 `9dd57423bfbf81219ad2dd7f367f4d51f59dabd03ba9265ee64782e14f3215ab`.
