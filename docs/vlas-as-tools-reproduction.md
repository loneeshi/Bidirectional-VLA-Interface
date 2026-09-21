# VLAs-as-Tools reproduction: priority and acceptance gates

**Historical priority, 2026-09-17 UTC:** MS-HAB was that week's main delivery.
Limit LIBERO to a2–4hour author-component/reference audit; full DROID/LIBERO
reproduction is not a prerequisite for the MS-HAB method port. The historical
results below remain component evidence, not paper-level reproduction.

Status updated 2026-09-15: [TAPT009](tapt-libero-run03.md) completed corrected online evaluation: standard 5/5, old TAPT 5/5, old TAPT with memory 4/5 (one infrastructure failure), new TAPT with memory 5/5. The minimal single-task method loop now has native-success evidence. This does not establish recovery gains or reproduce paper scores. Next: [implementation alignment and controlled recovery design](tapt-method-alignment-and-recovery.md). No further training is justified by these ceiling-level results alone.

**Availability correction:** [relevant code was found in the co-first author's public OpenPI fork](vlas-as-tools-code-audit.md). The sections below preserve the **pre-execution** audit and implementation proposal, not the current LIBERO implementation inventory. Their missing-interface findings still must be checked separately for the older MS-HAB path.

## Verified source

[Lei et al., arXiv:2605.13119v1](https://arxiv.org/pdf/2605.13119), Sections 3–4, 5.1 and Appendices A–D:

The invocation pairs a family selector with a grounded instruction. Family-specific low-rank residuals share a backbone. An auxiliary head predicts invocation-local progress for event-triggered intervention. TAPT combines invocation-aligned action supervision and progress regression; intermediate DROID-split post-training precedes benchmark adaptation. The experiments use LIBERO, RoboTwin and CALVIN, not MS-HAB. The abstract says code will be released. Initial searches missed relevant code in an author fork. The follow-up audit linked above corrects that finding; a complete paper release and split/checkpoint package remain unverified.

## Local-code findings

| Component | Existing implementation | Gap to resolve |
|---|---|---|
| Invocation | `src/bvi/protocol.py`: `SkillRequest` has skill/target IDs, requirements and limits | No grounded instruction field emitted by the VLM |
| VLM contract | `src/bvi/coordinator.py`: strict schema/parser | Must version schema and carry the instruction end to end |
| Manipulation language | `src/bvi/fetch_pi_skill.py`: prompt from scene configuration or task-plan template | The actual VLA prompt is not a VLM-issued instruction |
| Navigation language | `src/bvi/lightnav_skill.py`: instruction indexed by subtask | Same end-to-end language gap |
| Feedback | `src/bvi/organizer.py`: bounded slices and privileged grasp-state heuristics | No learned VLA progress head; heuristics must retain separate provenance |
| Family execution | Named skill wrappers and existing Fetch checkpoint client | No demonstrated shared-backbone, family-selected residual bank |
| Evidence | Real GPT + PPO/SAC organizer episodes | Infrastructure evidence; neither TAPT nor VLM + TAPT-trained VLA validation |

These findings concern the checked-in implementation. Old Fetch adaptation pilots must not be relabeled TAPT after the fact.

## Proposed implementation sequence

1. **Invocation contract.** Introduce an explicit versioned family/instruction request. Log the VLM instruction, validated invocation, selected adapter identifier/hash and exact model input. Test that changing the invocation changes the transmitted instruction, and that legacy runs retain their old semantics. Invalid or unavailable families must be rejected before motion, with no silent SAC fallback.
2. **Family routing and feedback.** Add an inference-service adapter registry and a progress output contract. Lock the selected residual for a call; clear cached action chunks when the call changes. Distinguish learned progress, simulator predicates and heuristic warnings. A proposed local monitor uses completion, stagnation, regression and timeout events; its thresholds must be calibrated on held-out data, not represented as recovered author defaults.
3. **Training-data audit.** Specify invocation boundaries, grounded text, family label, aligned actions and progress supervision. Keep trajectory-level train/validation/test splits before windowing. Check camera timestamps, action conventions and incomplete/failure segments. In particular, do not assign successful terminal progress to a failed clip merely because it ends.
4. **TAPT implementation pilot.** Verify family-specific trainable parameters and progress-head gradients on a tiny offline batch, then measure memory and throughput before pricing any A6000 run. For pi0.5, retain its appropriate action-training objective; the exact feature tap, residual insertion layers and progress weighting require explicit implementation choices or author code. A tiny local-data pilot is a component experiment, not the full intermediate post-training recipe.
5. **Paper-task reproduction first.** Use one LIBERO-Long task for an initial end-to-end diagnostic once dependencies/assets are checked. Pin task, initial states, checkpoint and horizon before comparing ordinary VLA, VLM + ordinary VLA, and VLM + tool-aligned VLA. Expand evaluation only after the diagnostic passes. Missing author data/configuration must be documented as a deviation, never filled in as fact.
6. **MS-HAB transfer second.** Map validated manipulation families to Fetch, then add navigation as a separately identified extension. LightNav and pi0.5 are heterogeneous backends; calling them both tools does not make them a shared-backbone residual family. Preserve the original benchmark evaluator and label any free-order diagnostic environment separately.

## Required evidence before claiming success

- Protocol tests: instruction fidelity, adapter routing, invalid-family rejection, progress provenance, event handling and action-cache invalidation.
- Training tests: selected adapter receives updates; unselected adapters do not; held-out progress quality and failure behavior are measured.
- Paired closed-loop runs: same initial states and evaluator, real VLM requests, actual VLA actions, family/hash trace and per-call usage ledger.
- Distinct reports for task completion, following the requested object/relation, over-execution, recovery and VLM cost. A single successful trajectory cannot establish a comparative improvement.
- Reproduce author metrics only when their task definitions and evaluation details are available; otherwise name ours as local diagnostic metrics.

The current budget does not establish that full-scale reproduction is affordable. Read and reconcile the finance ledger before any billable experiment, measure a bounded pilot, and price remaining work from observed throughput. No new top-up is requested at this audit stage.

## Open reproducibility questions

Confirm official repository/checkpoint availability; exact DROID-split and downstream split manifests; adapter ranks and insertion layers; progress target construction and head feature location; monitor thresholds; training hyperparameters and random seeds; and exact counterfactual task assets. Unresolved details are implementation assumptions, not paper facts.

## Future recording convention

No new video was generated by this audit. Future recording batches use `docs/media/tapt-interface-YYYY-MM-DD-run-id/` or `docs/media/tapt-libero-YYYY-MM-DD-run-id/`, with the actual recording date. Every delivered recording must have its exact filename, outcome, backend and training status in the central media index. Existing organizer recordings retain their historical labels.
