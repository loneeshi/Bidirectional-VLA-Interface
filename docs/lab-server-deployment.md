# Laboratory server deployment — 2026-09-16

SSH authenticated after the user connected the university VPN. Connection
credentials and the host-key record remain in ignored local files. Subsequent
compute uses this server; do not create or resume paid Runpod GPUs.

## Verified hardware and environment

- Ubuntu 20.04.6, 112 logical CPUs; home filesystem initially had about 942 GiB
  available. No user disk quota was reported by `quota -s`.
- Two **Quadro RTX 8000, 48 GiB each**, compute capability **7.5**. This is not
  RTX A6000. GPU 0 was occupied; the bounded arithmetic probe used GPU 1 only.
- No Slurm/PBS command was found in the login PATH. This does not prove that the
  laboratory has no scheduling or reservation rules; those remain to be clarified.
- System Python is 3.8.10; default `nvcc` is CUDA 10.1. CUDA **12.8** also exists
  at `/usr/local/cuda-12.8`, and the installed NVIDIA driver is 570.124.06.

Deployment creates `~/bvi-research` without sudo or changes to system Python,
drivers or CUDA. Python **3.11.13**, uv **0.8.22**, and the two source checkouts are
isolated there. All 13 AC-DiT files in the source lock match on the server.

Reproducible stages, run on the server:

```bash
python3 bootstrap_lab_acdit.py
python3 install_lab_acdit_base.py
python3 install_lab_acdit_extensions.py
~/bvi-research/envs/acdit/bin/python check_lab_cuda.py --gpu-index 1 --output ~/bvi-research/cuda-smoke.json
```

The first three scripts hide GPUs during installation. PyTorch3D compilation
targets `sm_75` and uses `MAX_JOBS=2`. Its pinned source revision is
`33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` (v0.7.9), recorded as a resolved
deployment dependency, not an author-provided environment lock.

The CUDA check selects the requested physical GPU by UUID and refuses to run
when more than 1 GiB is already occupied. This is a short arithmetic guard, not
a substitute for any laboratory GPU reservation policy.

## Observed checks

- PyTorch **2.7.0+cu128** imported and ran finite FP32 and FP16 matrix products
  on GPU 1. Peak tensor allocation was 9,502,720 bytes; the process exited.
- Native BF16 support returned **false**. [NVIDIA's capability table](https://developer.nvidia.com/cuda/gpus)
  identifies RTX 8000 as 7.5; [NVIDIA prerequisites](https://docs.nvidia.com/bionemo-recipes/latest/main/getting-started/pre-reqs/)
  place native BF16 at compute capability 8.0 or later.
- SAPIEN, ManiSkill environments and MS-HAB environments imported successfully.
- The first AC-DiT model import failed because PyTorch3D was missing; its bounded
  two-worker CPU build was then started. Completion is recorded separately below.

**Completed at 19:18 UTC:** PyTorch3D built successfully in about 13 minutes.
Subsequent imports of `sapien`, `mani_skill.envs`, `mshab.envs`, `pytorch3d.ops`,
`models.acdit_runner`, and `model_wrappers.mshab_model` all passed. Both community
checkpoints were downloaded on the server and verified against their pinned
hashes. The setup, build and download jobs have finished. Machine-readable evidence
and the precision diff are in [the deployment record](results/lab-server-2026-09-16/deployment.json).

## Remaining deployment gates

AC-DiT had hard-coded BF16 conversions in `models/weighting.py` and the pointcloud
path in `model_wrappers/mshab_model.py`; changing only the top-level dtype was
insufficient. The explicit compatibility patch in `scripts/patch_acdit_precision.py`
has been applied to the dedicated server checkout, with input/output hashes and
the exact diff saved. It retains BF16 as the upstream default and adds the
evaluator option `--precision fp32` (or `fp16`). No observation, action, reward,
completion or recovery mechanism was changed. Start with FP32 correctness before
testing FP16.

`scripts/check_acdit_precision.py` compared the perception-weighting component
against the pinned upstream code using identical weights: the default BF16
outputs were bitwise identical, and FP32 outputs were finite. This is a component
check; it does not certify the full model or task success. The source lock now
contains 14 files, including the weighting component.

**Next batch:** encoder pinning, full-model strict loading, asset-free headless
RGB/depth rendering, and real Fetch reset/interface checks passed. See the
[native validation report](acdit-native-validation.md) for the episode outcome and
remaining task-level gate. Handoff data and TAPT remain subsequent work.
No training or model API calls have been started by these checks. No new
Runpod compute was rented. Historical stopped storage remains separately billed.

Private local evidence is under `.runtime/lab-server/`; server logs and package
freeze are under `~/bvi-research/`. Hostnames, credentials, and private paths are
not included in the public experiment report.
