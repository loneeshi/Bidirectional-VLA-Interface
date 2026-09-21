# BC/IA pairing and normalization audit — 2026-09-18

CPU-only audit, no rollout/model inference or parameter updates.
Raw result: [saved-state and statistics checks](results/bc-ia-static-2026-09-18.json).
Runner: `scripts/audit_bc_ia_static.py`.

## Pairing

BC and S1-IA use the same seed identifiers2024–2028. Saved robot/controller and
non-actor simulator fields match exactly. Actor-state numerical differences
remain: maximum across leaves versus the IA panel is1.79e-7–2.38e-7; versus the
IA reach snapshots it is3.58e-7–4.77e-7. All are below the earlier1e-5 restore
tolerance, but this is not bitwise-identical initial state. BC uses CPU physics;
IA/D1 uses GPU physics and restored snapshots. Hidden contact state and backend
dynamics are not certified by serialized-state comparison.

Thus this is the same seeded scene/start design, numerically near matched, but
BC4/5 versus IA0/5 is not a strictly controlled paired-policy experiment. Such
an experiment requires the same physics backend and restoration procedure.

The previous BC three-path equality is an exact comparison on its observed
trajectories, not a statistical success-rate test. It does not prove equality
for unseen inputs or untested RGB/point-cloud/VLA paths. BC success rules out a
protocol that makes every policy fail, not a differential effect of stopping
criteria on other policies. Four of five is not an established80% population rate.

## S1-IA normalization

Training identity, selected best855 checkpoint's actual norm_stats.json and
recorded inference metadata all have SHA256
`0f6264129ebbe89f8adee0c0caa2f204314674e5a95c882e1710413ee734e8d2`.

Training: H5 invocation sample → LiberoInputs → quantile Normalize → model
transforms. Inference: model sample → quantile Unnormalize → select first13
channels → client selects first action → clip[-1,1] → zero head8:10 → env.step.
The native24 config explicitly disables extra delta conversion.

Pinned formulas: z=2(x-q01)/(q99-q01+1e-6)-1;
x=(z+1)(q99-q01+1e-6)/2+q01. A CPU formula roundtrip on1000 synthetic13D
vectors gives maximum error3.33e-16. This checks direction/algebra, not a
fresh execution of the full trained model. No pre-unnormalization clipping was
found. Channels8/9 have q01=q99=0; their behavior is epsilon-scaled before the
explicit head mask. Other quantiles are approximately within controller[-1,1].

These checks do not support stale statistics, reversed transformation, or
clipping before unnormalization as the S1-IA explanation. They do not prove
the original stats were semantically correct for every action channel or that
the checkpoint learned the desired distribution. Olderπ₀.₅/V8 and other
checkpoints must be audited independently.

## AC-DiT static path and outer execution

Visible source `data/hdf5_mshab_dataset.py` loads H5 actions and embeds the13
channels directly in128D. `train/dataset.py` preserves those actions and
`train/train.py` casts their dtype. Per-trajectory state statistics exist but
are not an action unnormalization stage in the inspected path.
`model_wrappers/mshab_model.py` extracts the same13 output channels directly.
`configs/config.yaml` sets prediction_type=sample and clip_sample=False.

`src/bvi/acdit_contract.py` checks(2,13), clips to environment bounds, preserves
raw outputs, and enqueues both actions. `scripts/run_lab_acdit_native.py`
consumes the queue before predicting again and clears it at termination.
IA instead predicts10 and executes1. These different temporal contracts remain
separate audit targets; out-of-range raw actions do not distinguish a
normalization bug from model prediction error or temporal rollout distribution
shift. A correct unbounded regressor/diffusion output can overshoot[-1,1].

The visible AC-DiT training code is not proof of the community checkpoint's
actual historical training recipe. Full-harness runtime tracing, controller
channel semantics, RGB preprocessing, and point-cloud processing remain open.

## Next gates

1. Compare recorded teacher actions, packed training targets and inference
   transforms per channel, including physical/controller normalization.
2. Inspect real execution traces at model output, postprocessing, queue pop and
   env.step; keep full-harness results separate from the BC diagnostic adapter.
3. Audit RGB resize, scale, channel order and camera identity independently.

No new GPU, API, training or rental usage. Historical cloud storage billing
was not refreshed; no change to resource state is asserted by this CPU audit.
