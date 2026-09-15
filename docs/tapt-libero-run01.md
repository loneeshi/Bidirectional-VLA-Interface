# LIBERO tool-family/TAPT component reproduction — run01

Status: executing, 2026-09-15. This is an invocation-aligned adaptation of an **already LIBERO-trained** pi0.5 model. It omits DROID intermediate post-training and does not reproduce the paper's score.

## Fixed experiment

See [machine-readable configuration](../configs/tapt-libero-run01.json), [source audit](vlas-as-tools-code-audit.md), [evaluation evidence](results/tapt-libero-2026-09-15-run01/), and [video batch](media/tapt-libero-2026-09-15-run01/).

Official OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479` supplies the unmodified baseline. Author fork `f4eb160ba52b22c1e85fe432de59c24bbbac6187` supplies the progress head and LoRA-capable modules. LIBERO is pinned to `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`. Added scripts live in this repository; no hidden author-tree patch is required.

Task: LIBERO-10 task 1, both cream cheese box and butter into basket. Initial states 0–4, simulation seed 7, sampling seed 0, 20 wait steps, 520 action steps. Policy predicts 10 and executes 5 actions at a time. Baseline completed 5/5 with 241, 264, 242, 253, 284 action steps. The two GPT conditions are pending; this baseline result is not evidence of a tool-family benefit.

## What is actually trained

One frozen pi05_libero backbone; four independent LoRA banks at the author's existing insertion sites; one shared author progress head. VLM residual rank/alpha 16/16 and action-expert rank/alpha 32/32. These two experts are architectural components, **not** two tool families. Every family has residuals in both components. Selection is explicit, not inferred from the prompt. Parameters are explicit JIT arguments so switching cannot silently reuse captured previous weights.

Action flow-matching loss is retained, plus 0.1 times progress MSE. Microbatch 1, accumulation 8, AdamW 5e-5, no weight decay, global-norm clipping 1. Frozen backbone parameters are excluded from the optimizer. Actual selected/unselected hashes are recorded in `update-audit-*.json`.

## Data and limits

Source: `physical-intelligence/libero`, the official OpenPI dataset, pinned to `a4336d589d589045d1c56423ffdf3b88a0e19b1f`. All25 downloaded file hashes match the Hub LFS metadata at this revision. Original main-branch URLs remain in the historical manifest; the source lock and downloader use the fixed revision. Sort matching demonstration IDs, first 20 train, next 5 validation, then segment. The manifest stores per-file hashes and source URLs. There are 200 windows, 160 train and 40 validation.

The parquet export has RGB, robot state and actions but no simulator contacts. Boundaries use gripper changes, motion and manually inspected images. Empty/unsustained attempt e166:201–214 was rejected despite passing a numerical lift proxy; other failed attempts are excluded. Labels are invocation-local normalized time, **not measured task completion**. They can be wrong when the accepted boundary is imperfect. No failed clip is marked complete merely at video end. The final reported model must include held-out progress diagnostics and actual closed-loop failures.

Initial validation accidentally sampled one trajectory. Those records are retained separately. Training resumed at step200 with optimizer state restored and the data sampler restarted at seed7 (recorded deviation); exact executed scripts are archived. It uses complete five-trajectory validation, and only the corrected validation set is used for model selection. Periodic metrics use midpoint samples. Final selection re-evaluates every checkpoint at fixed 10%, 50% and 90% positions of all40 held-out windows (120samples) before any GPT test. Constant-50% progress error is also reported. Test success is never used for selection.

## Communication and evaluation

The GPT request yields versioned family + grounded instruction. Both original and trained VLA receive the validated instruction verbatim. Only the trained condition selects learned family residuals. During a call, family/instruction are fixed. Interrupt/switch drops pending actions.

The trained condition returns the author's chunk-prefix progress predictions. The checked author server emits an array while its evaluator casts progress to a scalar; this implementation uses index0 (current-observation progress) for events and logs the complete sequence. This readout choice is explicit. Thresholds: reach/move 0.9, grasp/release 0.6; two consecutive above-threshold predictions. Replan on >0.03 regression or less than 0.03 growth across at least 10 predictions, with initial 3-prediction and replan 15-prediction cooldowns. First-call rollback is disabled as in the author evaluator. Native environment success is logged separately. The standard condition uses explicitly disclosed simulator rules. Its already-open fallback is corrected from0.39 to0.039m: a calibration replay measured0.039235m at open, so the original value did not detect it. The author1mm opening-delta rule remains; opening is not proof of a stable placement. Learned thresholds are unchanged. The comparison is a **combined method comparison**, not a causal isolation of TAPT.

GPT is accessed through the local SSH file bridge with durable claims, no SDK retries, max20 calls/episode and shared max200/$1. Credentials stay local. Input cap16000 bytes, two128x128 JPEG images, max600 output tokens, conservative reservation$0.005/request. Per-call usage and reconciled estimates will be published separately from provider billing facts.

## Reproduction commands

Commands assume `/workspace/tapt` inside the temporary Linux container. Install with `bash scripts/setup_tapt_libero.sh`, then copy this repository's scripts there and `src/bvi` to `/workspace/tapt/bvi`. Never copy `.env.local` to the GPU host.

```bash
cd /workspace/tapt
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/workspace/tapt:/workspace/tapt/author/third_party/libero
author/.venv/bin/python download_libero_checkpoint.py
sim-env/bin/python download_libero_pilot.py
sim-env/bin/python review_libero_segments.py
# Inspect segmentation-contact.png and preserve reviewed exclusions.
sim-env/bin/python prepare_libero_invocations.py --data data
author/.venv/bin/python serve_libero_baseline.py
# In another shell:
sim-env/bin/python eval_libero_baseline.py --output evidence/baseline
# Stop that model server before training on the same GPU.
XLA_PYTHON_CLIENT_MEM_FRACTION=.85 author/.venv/bin/python train_libero_family.py \
 --data data --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
 --output evidence/training --steps 2000 --seconds 7200
```

The commands above describe a clean run. This recorded run stopped the first trainer after step 200, restored its optimizer checkpoint in `training-v2`, and restarted the sampler at seed 7. The exact executed versions are retained in `results/tapt-libero-2026-09-15-run01/provenance/train-v1.py` and `train-v2.py`; this interruption is part of the run provenance.

After the trainer exits and releases GPU memory, select on validation data before launching either GPT condition:

```bash
author/.venv/bin/python validate_libero_family.py \
 --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
 --training evidence/training-v2 --data data
# Record selected path and metadata in evidence/evaluation-lock.json.
author/.venv/bin/python serve_libero_baseline.py
# Separate terminal; stop this server after all five episodes:
sim-env/bin/python eval_libero_family.py --mode standard --output evidence/vlm-standard
# SELECTED is the path from validation-selection.json, never test success.
author/.venv/bin/python serve_libero_family.py \
 --checkpoint /root/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
 --adapters "$SELECTED" --audit-data data
# Separate terminal:
sim-env/bin/python eval_libero_family.py --mode tapt --output evidence/vlm-tapt
```

Before either GPT evaluation, run the bridge **on the local machine** with a working SSH alias `tapt`. Use absolute local paths for `SSH_CONFIG`, `LOCAL_ENV`, and `BRIDGE_LOGS`. Reuse the same log directory after interruption so durable claims preserve the shared limit.

```bash
python scripts/serve_vlm_bridge.py --ssh-config "$SSH_CONFIG" --ssh-alias tapt \
 --remote-bridge-dir /workspace/tapt/bridge --provider openai --model gpt-5.6-luna \
 --credentials-file "$LOCAL_ENV" --authorization-id TAPT007 \
 --max-calls 200 --max-api-cost-usd 1 --request-cost-ceiling-usd .005 \
 --max-output-tokens 600 --max-input-bytes 16000 --image-detail low \
 --reasoning-effort none --idle-timeout-seconds 5400 --max-wall-seconds 7200 \
 --output "$BRIDGE_LOGS"
```

Inspect all cost limits before starting a new paid run; the historical authorization ID does not authorize another paid experiment. Checkpoint pickle files are trusted local training artifacts; never deserialize arbitrary downloaded pickle files.

## Pending acceptance

Training completion, held-out progress quality, GPT-standard 5 episodes, GPT-TAPT 5 episodes, learned progress participating in a real closed loop, final videos/hash inventory, per-call costs, backup verification and resource teardown remain pending. MS-HAB migration begins only after the LIBERO gates pass.
