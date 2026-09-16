# AC-DiT native Fetch validation — 2026-09-16

This batch validates deployment and the community AC-DiT checkpoint on its native
MS-HAB task before training or integration with GPT and LightNav. It is not TAPT,
a VLM collaboration result, or a reproduction of the paper's reported score.

## Fixed inputs

- AC-DiT source: `90ad00a926f34da04816ed9c3312aaf3bc845b7f`, with the documented
  [device precision patch](lab-server-deployment.md). FP32 on one Quadro RTX 8000;
  the other GPU belongs to an existing workload.
- Community checkpoint repository: `JJho1314/AC-DiT-MSHab-Reproduction`, revision
  `f57e782c6a152c5ada83a33d5c29273c857003fd`; mobility step 30,000 and main step 25,000.
  Hashes and provenance remain in [the source audit](acdit-mshab-mainline.md).
- SigLIP: `google/siglip-so400m-patch14-384`, revision
  `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`; both vision and LIFT3D constructors use
  the same pinned offline cache. [Encoder file hashes](results/acdit-native-2026-09-16/encoder-lock.json).
- Native task: `set_table/pick/013_apple`, validation split, seed **2024**, at most
  **200 actions**, `PickSubtaskTrain-v0`, CPU physics, GPU rendering, 20 Hz.
- Explicit instruction: **“Go to the apple and grab it.”** It is the first
  matching upstream training instruction. Tokenization retains that task's
  original batch padding; the selected text is encoded directly and hashed.
  FP32 text encoding is another consequence of the documented precision choice.
- Upstream pointcloud sampling, image history and whole-body control are retained.
  The native policy uses simulator-privileged 18D context. No task-family residual
  or learned tool progress is claimed.

## Gates

1. **Strict model loading passed.** Both model load calls use strict state-dict
   matching. Main policy state has 2,264 tensors; unique policy parameters total
   1,825,218,818. Peak tensor allocation during loading: 12,727,791,104 bytes.
   [Machine-readable model check](results/acdit-native-2026-09-16/full-model-check.json).
2. **Headless RGB/depth passed.** Asset-free SAPIEN rendering produced nonuniform
   RGB and finite depth with 12,100 visible pixels. PCI address verification binds
   rendering to the intended physical GPU.
   [Render check](results/acdit-native-2026-09-16/headless-check.json).
3. **Real Fetch reset and interface passed.** Full qpos has 15 joints, observation
   qpos has 12; name-based state mapping equals the upstream index mapping.
   Cameras are 128×128 RGB/depth/segmentation, pointcloud is 1×1024×6, and normalized
   13D actions split as arm 0:7, gripper 7:8, body 8:11, base 11:13.
   [Reset record](results/acdit-native-2026-09-16/reset-result.json) and
   [both camera observations](results/acdit-native-2026-09-16/reset-cameras.png).
4. **Native episode completed; task success failed (0/1).** The subtask's internal
   horizon terminated at action **199**, within the external 200-action budget.
   It first reported grasping at step 16, briefly reported no grasp at step 21,
   then continuously reported grasping from steps 22–199. This verifies physical
   grasp capability for this initial state, not native Pick completion.

## Episode result and failed phase

The robot carried the apple into the required rest region at steps **31–38**:
`is_grasped`, `ee_rest`, `robot_rest`, and the force condition were all true.
However, **`is_static` was false**, so the conjunction required for success never
became true. It then moved away from the rest region. At termination it held the
apple and was sufficiently static, but TCP-to-rest distance was **0.10838 m**,
above the unmodified **0.05 m** threshold.

The final shoulder-lift position was −1.220999 rad, at its −1.221 lower limit;
wrist flex was 2.160002 rad, at its +2.16 upper limit. During the last 20 actions
the shoulder-lift command remained between −0.982 and −0.860, pressing toward
the saturated direction. These records locate a **return-and-settle failure**.
They do not yet isolate checkpoint quality, FP32 versus training precision, or
remaining training/evaluation differences as its cause. Do not change the success
threshold, force a hold, or add an unreported controller to turn this into success.

There were 100 two-action predictions, 199 executed actions, and 75 executed
channels clipped to the existing [-1, 1] bounds; the unused last queued action was
discarded on termination. Median chunk inference latency was about **1.89 s**;
this is offline evaluation, not real-time 20 Hz inference.

- [Episode result](results/acdit-native-2026-09-16/result.json)
- [All predictions, actions and native feedback](results/acdit-native-2026-09-16/events.jsonl)
- [Derived phase/limit evidence](results/acdit-native-2026-09-16/analysis.json)
- [Exact executed runner](results/acdit-native-2026-09-16/executed-runner.py)
- Failure recording: [acdit-native-pick-episode000-failed.mp4](media/acdit-native-pick-2026-09-16-run01/acdit-native-pick-episode000-failed.mp4).
  This contains the unmodified head and hand policy cameras, with no text overlay.
  The native 128×128 camera resolution is retained.

Raw archive `acdit-native-2026-09-16.tar.gz` was downloaded and every manifest hash
verified; archive SHA256 is
`77cca13053702b2021a9f5faebb879eea08eb4aec47672dc609b4ab412fd7106`.
All our validation jobs exited, and GPU 1 returned to 15 MiB occupied memory.
No training, model API calls, or new Runpod compute occurred. Laboratory pricing
is unknown. Historical stopped Runpod storage remains 40 GB, estimated
USD 0.266667/day; the private finance ledger records the resource verification.

The next capability gate is controlled diagnosis of the return-and-settle phase,
with the same initial state and native criteria, before claiming native task
success or starting handoff TAPT data collection. GPT and LightNav integration,
TAPT SFT, DROID and GRPO remain unstarted in this batch.

## Harness corrections and limitations

The first reset was launched just before archive extraction completed and failed
on a missing scene directory. The launcher now checks the completed asset manifest.
The next reset reached the scene but failed in diagnostic logging because
`CombinedController` exposes child controller configs, not a top-level `config`.
That logging access was fixed. The third reset passed. Neither failed setup
attempt executed a policy action; both raw errors are retained in the run archive.

Unlike the upstream evaluator, this harness stops on **any** native termination
or truncation, selects an explicit task instruction, honors its supplied seed,
and records raw versus clipped actions. It preserves failures and does not repeat
episodes until success. Clipping at the normalized environment boundary is logged;
there is no gripper threshold, base freezing, SAC recovery, or fake progress head.

## Reproduce on the prepared laboratory environment

```bash
cd ~/bvi-research
envs/acdit/bin/python prepare_lab_acdit_encoder.py
timeout 600 envs/acdit/bin/python check_lab_acdit_model.py --gpu-index 1
timeout 90 envs/acdit/bin/python check_lab_headless.py --gpu-index 1
timeout 1500 env PYTHONHASHSEED=2024 envs/acdit/bin/python -u run_lab_acdit_native.py \
  --gpu-index 1 --seed 2024 --output "$HOME/bvi-research/runs/acdit-native-new-batch"
```

Use a new output directory per attempt. The runner refuses an occupied GPU and
never starts training or an external model API. The wall-clock timeout bounds the
entire process independently of its 200-step simulator budget.

Local verification: **146 passed, 1 skipped**. The existing websocket deprecation
warning remains; it is unrelated to this native runner. These tests complement,
not replace, the real simulator gates above.
