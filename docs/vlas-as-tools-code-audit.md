# Author-code availability and LIBERO-first preparation

Checked 2026-09-15. **Correction: it is not accurate to say that no relevant author code is public.** No standalone, paper-labeled release was found, but the co-first author's public OpenPI fork contains substantial relevant implementation. Completeness and correspondence to the paper remain to be established.

## Evidence

- [arXiv](https://arxiv.org/abs/2605.13119) still says code will be released.
- [Changxing Liu's homepage](https://cxliu0314.github.io/) identifies this paper and links his GitHub account. Its paper entry has a Paper link, without a Code link.
- [Author OpenPI fork](https://github.com/cxliu0314/openpi), pinned at `f4eb160ba52b22c1e85fe432de59c24bbbac6187`, is 5 commits ahead and 7 behind upstream main at this check. Repository search alone had missed these changes. His OpenVLA-OFT fork was identical to upstream main at this check.
- [Implementation notes](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/progress.md) describe an opt-in LIBERO evaluation port with subtask instructions, predicted-progress switching, stagnation/rollback and optional multimodal replanning.
- [Progress-head training](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/scripts/train_chunk_progress_head_only.py) is actual source code, not just a roadmap. [Subtask training orchestration](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/exp_scripts/run_openpi_subtask_pipeline.sh) contains local dataset paths and multi-GPU defaults; it must not be launched unchanged on one A6000.

GitHub repository searches for `2605.13119`, `VLAs-as-Tools`, and `TAPT VLA` did not return a dedicated author release. This is bounded search evidence, not proof of global absence. The first author's OpenReview page was blocked by a browser challenge.

## Preparation completed

The author fork was cloned into the local sibling checkout `repo/openpi-vlas-tools-audit`; only its official LIBERO submodule was initialized, at `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`. No training scripts were executed, dependencies installed, model weights downloaded, external model calls made or GPU rented by this audit. No demo was generated.

## Immediate LIBERO work order

1. Audit the author implementation for family-specific residual selection, server progress output, exact training configuration, dataset/checkpoint availability and evaluator privileges. Generic OpenPI VLM/action experts must not be mistaken for tool-family experts.
2. Establish the original OpenPI `pi05_libero` baseline on `libero_10` using the [official example](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero). Pin an unmodified upstream revision separately from the author fork. Validate reset, RGB, actions, native success and recording before enabling a planner.
3. Use a bounded single-task diagnostic first. The full-suite evaluator must not be mistaken for a single-episode smoke command. Preserve initial-state IDs and native horizons for later matched comparison.
4. Run the author's subtask evaluator with explicit progress-source labeling. An environment-predicate fallback does not validate a learned progress head. Route any external replanner through our per-call budget and ledger before enabling it.
5. Recover available TAPT components from the author code before implementing missing pieces. Compare ordinary VLA, VLM + ordinary VLA and VLM + tool-aligned VLA under matched conditions. Missing weights/data prevent claiming paper-score reproduction even if the interface executes.

MS-HAB/Fetch adaptation is deferred until this LIBERO path is validated. Before paid execution, reconcile the research ledger and derive a bounded experiment cost from the actual setup; this audit has not reserved new paid resources.
