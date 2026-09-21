# S1 exact-start first-action probe

This is the shortest zero-training gate after the Level-3 first-step split. It
does not estimate task success and does not treat SAC as an expert action label.

## Frozen design

1. P0 deterministic parity, before any inference: on the saved `step000.npz`
   for seeds 2024--2028, compare the actual training transform path
   (`cfg.data.create`: repack, data transforms, quantile normalization, model
   transforms) with the frozen server path (`LiberoInputs`, checkpoint
   quantile normalization, model transforms). The saved online uint8 HWC images
   are converted to LeRobot's raw float32 CHW `[0,1]` representation on the
   training side, so the real training image-parse branch is covered. The action label is a dummy and
   is removed before comparing every shared `image`, `image_mask`, `state`,
   `tokenized_prompt`, and `tokenized_prompt_mask` leaf by shape, dtype,
   `array_equal`, and SHA-256. Any mismatch blocks inference.
2. For each of the five exact starts, reset the frozen model to the recorded
   episode seed and make exactly 16 sequential calls on the unchanged saved
   request. Sample zero must reproduce both the recorded RNG key and raw first
   action within `1e-6`; the parameter, normalizer, state-contract, task, seed,
   reference-state SHA, and restoration-error identities must also match.
3. Compute each sample's RMSE to that start's official SAC first action over
   the 11 applied non-head channels. A start is an RNG-tail candidate only if
   the deployed draw is one of the two most SAC-distant of 16 (inclusive rank
   at least 15/16) **and** the per-channel policy median is at least 25% closer
   to SAC than the deployed draw.
4. All five starts must pass the exact-reproduction identity gate. The
   SAC-distance vote then excludes seed 2024 because its SAC replay failed the
   native task, leaving four documented-success references (2025--2028). The
   stochastic-tail hypothesis is supported only if at least 3/4 of those cases
   meet the per-start tail rule. Otherwise it is not supported and the next
   attribution target is input/conditioning or adaptation. This does not prove
   that any SAC action is uniquely correct. Seed 2024 metrics remain visible as
   diagnostic/non-success evidence.

Under independent exchangeable ranks, the reference probability of at least
three top-two ranks in four cases is about 0.00708. Independence is not claimed;
the value is recorded only to make the strictness of the frozen rule legible.

## Lab command

Run in the pinned `envs/openpi` environment from the repository root:

```bash
python scripts/probe_s1_exact_start_first_action.py \
  --policy-root /home/pshuai/bvi-research/runs/s1-ia-calls-2026-09-18-run01 \
  --sac-root /home/pshuai/bvi-research/runs/s1-ia-diagnostics-2026-09-18-run02 \
  --checkpoint /home/pshuai/bvi-research/runs/s1-ia-epoch-2026-09-18-run01/best/855 \
  --normalizer /home/pshuai/bvi-research/runs/s1-official-medium-norm-2026-09-17-run01 \
  --output /home/pshuai/bvi-research/runs/s1-exact-start-first-action-2026-09-18-run01 \
  --max-seconds 900
```

Fixed cost envelope: 80 frozen-model calls, zero simulator steps, zero rollout
episodes, zero training updates, and zero model/API calls outside the local lab
checkpoint. The server is bounded to those 80 calls and is shut down on success
or terminated in `finally` on failure.

Outputs are `summary.json`, `transform-parity.json`, `per-seed.json`,
`per-channel.csv`, `samples.npz`, server provenance, and an artifact manifest.
