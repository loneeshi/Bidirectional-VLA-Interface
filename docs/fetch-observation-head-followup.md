# Future option: observation-conditioned Fetch progress head

Status: source audit and design option only. **Not implemented or evaluated.**
The running current-observation calibration and its action-token head are
unchanged. No model configuration, training, or GPU operation was performed for
this audit. Consider this option only after the current fixed online outcomes.

## Source evidence

Author checkout: `repo/openpi-vlas-tools-audit`, commit
`f4eb160ba52b22c1e85fe432de59c24bbbac6187`.

- `src/openpi/models/pi0.py:155–186`: prefix tokens contain image and language
  embeddings, with explicit validity masks. This is not the action suffix.
- `:323–330`: inference encodes the prefix independently of the denoising
  action suffix. Sampling noise is drawn earlier, but is not an input to that
  prefix encoder.
- `:238–244`: masked mean pooling returns `[B, prefix_width]`.
- `:13–15,39–57,246–256,373–384`: project to 1024 dimensions, add a sinusoidal
  chunk-offset embedding, and apply a shared 1024→512→128→1 ReLU/sigmoid head.
- `src/openpi/training/data_loader.py:675–679`: progress target is
  `clip((frame_index + offset) / max(episode_len - 1, 1), 0, 1)`.

AC-DiT source was read remotely at
`/home/pshuai/bvi-research/src/AC-DiT`, commit
`90ad00a926f34da04816ed9c3312aaf3bc845b7f`; references below are relative to
that directory. Dimensions are inferred from active source/configuration, not
from a new model execution. Some upstream docstring dimensions are stale;
implementation/configuration take precedence.

| Clean input before diffusion | Inferred shape | Source |
| --- | --- | --- |
| SigLIP instruction tokens | `[B, L, 1152]` | `model_wrappers/mshab_model.py:243–245,265–269`; `configs/config.yaml:48` |
| Image/history tokens | `[B, 6*729, 1152]` | wrapper `:232–234`; `models/weighting.py:58,85` |
| Point-cloud/history tokens | `[B, 2*128, 1152]` | wrapper `:259–263`; `models/acdit_runner.py:298–309`; weighting `:59,86` |
| Current unified proprioception | `[B, 1, 128]` | wrapper `:237–240`; config `:53` |
| Valid state/action channel mask | `[B, 1, 128]` | wrapper `:273`; runner `:642` |
| Privileged context | wrapper supplies `[B, 18]` | wrapper `:247–257`: goal 3, grasp 1, object pose 7, TCP pose 7 |

These tensors already exist in the recorded `predict_action(**batch)` input.
They depend on observation/history/instruction/state, not sampled actions or a
diffusion timestep. Their visual encoders/preprocessing may have their own
randomness; this is a separate issue from action diffusion and should be tested
using fixed recorded input tensors.

`models/acdit_runner.py:652` applies the perception-aware multimodal adaptor
before calling the mobility diffusion sampler at `:664`. Its weighted image/PC
features depend only on the supplied language/image/PC tensors:
`models/weighting.py:54–86`. Main manipulation adaptors at runner `:100–130`
and `:339–346` map language, weighted image, weighted PC, masked state, and
privileged context to width **2048** (`configs/config.yaml:57`). These individual
adaptor calls could be reused without invoking either diffusion sampler.

## Inputs to exclude

- **Mobility latent tokens are not observation-only.** Runner `:455–485`
  creates random actions and runs diffusion; `:664–676` obtains and flattens its
  hidden states. `latent_mobility_cond`/`lm_c` therefore contains sampled-action
  and timestep dependencies, even though it is passed as a “condition.”
- Do not pool all DiTCC `conditions`: `models/rdt/model.py:498–508` concatenates
  language, image, PC **and those mobility latents**.
- Do not substitute final state/context tokens from the manipulation DiT.
  `models/rdt/model.py:480–493` inserts the diffusion timestep alongside
  state/action tokens; `:511–515` runs shared transformer blocks. The blocks
  include self-attention (`models/rdt/blocks.py:171–179`), so those outputs are
  no longer guaranteed independent of sampled actions.
- Training's `state_action_traj` at runner `:573–585` includes noisy teacher
  actions. A future observation branch must instead adapt the original single
  current state concatenated with its channel mask, as inference does at `:655`.

## Minimal possible future port

Create a separately trained observation-progress branch from the already
recorded clean inputs. Reuse frozen main condition adaptors individually, omit
mobility latents, and pool clean condition tokens with an explicit mask. A new
1024-dimensional projection, chunk-offset embedding, and shared progress MLP
could follow the author's readout dimensions. Use exactly the same extraction
function for training and inference, before either diffusion sampler; keep the
existing action path unchanged.

This would be an **AC-DiT observation-conditioned port**, not an identical
author prefix architecture. AC-DiT's independent modality adaptors do not supply
the author's jointly contextualized image-language transformer prefix. Adding
a fusion encoder would be a further architectural change, not a free equivalent.
The present family LoRA banks reside inside the manipulation transformer; this
new branch would bypass them unless a separate, explicitly designed family
conditioning mechanism were added. Instruction tokens still identify the call.
Old action-token progress weights cannot be reinterpreted as this new head.

## Masks and research risks

- The wrapper replaces missing cameras/history with background images
  (`model_wrappers/mshab_model.py:185–204`) and duplicates the current point
  cloud when the previous one is missing (`:259–260`). Existing input batches
  do **not** include image/PC availability masks. Derive and record these from
  raw history for a new branch, or explicitly declare that background/duplicate
  tokens are treated as valid. Do not infer missingness from learned embeddings.
- The native language mask is all true for the individually encoded instruction
  (`:267–269`); any future padded batching must preserve actual validity masks.
- Naively averaging 4374 image tokens, 256 PC tokens, a short language sequence,
  and one state/context token strongly weights image token count. Per-modality
  pooling/concatenation or balanced pooling may be preferable, but constitutes
  another design choice to validate rather than an author guarantee.
- Including the 18-dimensional context preserves the existing privileged
  simulator setting; it does not establish vision-only or deployable sensing.
- An observation head removes action-diffusion dependence by construction, but
  does not establish calibration, recovery competence, physical completion, or
  native task success. Invocation-start zero anchors remain reconstructed local
  time labels, not author-provided failure labels.

Before any future training: verify exact train/inference feature equality on
recorded inputs, unchanged progress under changes to action noise/timesteps,
mask behavior, frozen-parameter/buffer invariance, and action-output identity
under restored RNG. Preserve trajectory-level splits and evaluate on the same
predeclared online cases. This document authorizes no new experiment.
