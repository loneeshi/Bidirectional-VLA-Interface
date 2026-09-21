# Author π₀.₅ components on Fetch — 2026-09-17

> Superseded launch order: the user inserted a frozen-V8 native capability gate,
> verified wrong-handoff inputs and a ≤1/5 weekly π₀.₅ training stop rule.
> Run03 automatic training controller was stopped before any updates. Existing
> collection has completed50 trajectories. The native gate returned0/5, so this
> week’s π₀.₅ TAPT updates are stopped. Eight handoff inputs passed replay
> verification, not learned-head acceptance. See the [actual results](fetch-v8-native-capability.md) and current
> the then-current priority protocol (that planning document has since been
> removed as superseded). Historical execution below
> is retained, not an instruction to resume the old pipeline.


Current mainline: stage 2, Fetch TAPT migration. The author OpenPI environment is
installed; checkpoint transfer, model interface verification and a new compatible
teacher dataset are prerequisites to training. This run does not retrain LIBERO.

## Fixed model contract

- Author source: `f4eb160ba52b22c1e85fe432de59c24bbbac6187`.
- Fetch V8 archive: 6,337,444,039 bytes, SHA256
  `c78c75642e2b23b07d4ae56dcdf3682ac00590916ba718133917f33d31f9300d`.
- Preserve V8 pretrained values and LoRA, use its quantile normalization.
- State: robot qpos15 + qvel15; XY relative to the original Pick/Place start,
  never reset independently at a family boundary. Discrete state conditioning.
- Images: workspace224 and hand128, with the original V8 workspace extrinsics.
- Actions: internal32/horizon10, external Fetch13, no second delta transform.
- Add only the author's observation-prefix progress head. It starts untrained.

The model gate compares actions with/without progress readout, progress under
changed action noise, and the training-forward/inference progress outputs on
the same real historical observation. Same-author-graph parity does not prove
parity with the older upstream OpenPI runtime or native task success.

## Data gate and bounded collection

The frozen official per-object SAC teacher retains its original depth/state
inputs. Only the V8 workspace sensor and recording are added. Smoke outcomes:

| Task / seed | Steps | Native success | Data contract |
|---|---:|---|---|
| Pick / 3000 | 29 | Yes | Passed |
| Place / 3000 | 199 | No | Passed; failure retained |

Reports: [Pick](results/fetch-pi05-author-gate-2026-09-17-run01/teacher-pick-smoke.json),
[Place](results/fetch-pi05-author-gate-2026-09-17-run01/teacher-place-smoke.json).
There is no new video in this phase; raw H5, events and camera PNGs are retained.

Fixed collection: both tasks, seeds3000–3019 train and3020–3024 validation,
50 original trajectories total. Each episode ≤200 actions and240 seconds;
aggregate collection ≤3600 seconds across resumes. Infrastructure failure stops
the batch. Native failure is retained, not replaced by a newly sampled seed.
Model staging pauses collection at an episode boundary to avoid GPU overlap.

Only independently verified subintervals receive completion labels. The final
verified observation has progress supervision but no action supervision; future
padding has neither. All descendants keep their original parent split.

This dataset covers native subtask starts, **not actual LightNav handoffs**.
Release coverage may remain small. Neither changing thresholds nor relabelling
failed ends is an acceptable remedy for missing family coverage.

## Training gate

The new entry point is `scripts/train_fetch_pi05_family.py`. Four explicit LoRA
banks initialize from V8, one shared author prefix head is learned, and the
backbone remains frozen. Native action loss and author progress loss are used
with separate validity masks. The first bounded run is20 optimizer updates,
microbatch1, accumulation8; training requires full fixed collection and all
families in both splits. A longer run is a separate recorded phase after this
gate. Selected-family updates, unchanged other banks/backbone, optimizer/RNG
resume and validation-based checkpoint selection must be verified.

Not yet claimed: trained TAPT, online manipulation success, GPT/LightNav chain,
complete DROID intermediate stage, GRPO, or paper-level reproduction.

## Resources and evidence

Lab physical GPU1 only, UUID `GPU-b7ebba23-7824-7601-df32-be55628936c3`.
CPU environment bootstrap took57.25 seconds. API requests0; new rentalUSD0;
laboratory charges unknown. Previous stopped Runpod storage continues accruing.
See [experiment config](results/fetch-pi05-author-gate-2026-09-17-run01/config.json)
and the project's separate finance ledger for bounded execution and receipts.

## Execution update

Attempt run01 stopped during full-model import because the author's runtime imports
`pytest`, while the original bootstrap excluded its locked dev dependency group.
The bootstrap now uses the complete frozen author lock and imports the actual
model module as its CPU check. This passed; the author source remains unchanged.
Failure evidence is retained in [run01 model result](results/fetch-pi05-author-gate-2026-09-17-run01/model-gate-failed.json).

Attempt run02 is running. It strictly loaded 71 pretrained tensors from V8, with
only the new author progress head initialized. Numerical inference checks are
pending; model loading does not imply trained TAPT or task success. Related
CPU tests: 72 passed. The pipeline automatically proceeds only when each gate
passes, and stops after the first bounded 20-update training gate.

## Numerical attribution and continuation

The BF16 gate passed exact paired-action equality and both inference/training
progress independence from action noise. Its strict training/inference progress
comparison failed: maximum absolute difference0.0042010546. We retained that
[result](results/fetch-pi05-author-gate-2026-09-17-run02/result.json).

A separate float32 computation retained identical input, checkpoint and new-head
hashes; the same difference fell to0.0000010133, with exact action/noise checks
still passing. See the [precision result](results/fetch-pi05-precision-2026-09-17-run01/result.json)
and [bounded review](results/fetch-pi05-precision-2026-09-17-run01/review.json).
The author uses a joint prefix+suffix attention shape for training and a
prefix-only shape for cached inference; the experiment supports reduced-precision
numerical attribution. It does not prove every learned head will have the same
error magnitude or that online threshold decisions are unaffected.

We keep author BF16 training and the original completion thresholds. The reviewed
evidence permits only the first20-update gradient/memory check; after training,
measure the trained head's inference gap again before online acceptance. This
is an explicit engineering gate decision, not a claim that the original strict
BF16 comparison passed.

Pipeline run03 has resumed fixed collection from the completed14-episode boundary;
15/50 had finished at the last startup check. Training remains conditional on all
data gates. No new video or model API request was generated.
