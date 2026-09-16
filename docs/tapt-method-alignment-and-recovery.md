# TAPT method alignment and next recovery experiment

Audit date: 2026-09-15. Offline source inspection only; no GPU deployment, training or coordinator API requests. The experiment below is a design, not an implemented runner or executed result.

**2026-09-16 implementation update:** `scripts/run_recovery_experiment.py`, `scripts/recovery_runtime.py` and the family evaluator now implement the proposed loop. Local tests pass; GPU/simulator gates and online evaluation remain unexecuted because three A6000 provisioning attempts returned no capacity. TAPT010 is authorized within the existing cumulative budget, with internal caps GPU/storage $2 and API $1. No resource was created by those attempts.

Concrete protocol refinement before execution: use all five TAPT009 new-memory traces, cut at the first cream-cheese grasp request, and replay the preceding recorded actions in a fresh seeded environment. This reconstructs controller dynamics rather than restoring only MuJoCo qpos/qvel. Hash controller numeric state, simulation state, warm-start acceleration, actuator controls, observations, call counters and history; restore the policy RNG by the exact prior prediction count, including trigger-only predictions. Both arms start at a fresh invocation with empty queues. This is a paired diagnostic conditional on a historical successful-run prefix, not an independent full-task robustness sample. Historical prefix GPT requests consume the logical per-episode budget but incur no new API charge. Do not describe it as a newly sampled common prefix.

All ten state/intervention pairs must pass two independently reconstructed branch audits before any new coordinator call. Online branches must reproduce their audited hash again. Invalid support/contact geometry or state mismatch stops the paid stage; no state substitutions or favorable displacement search. The current implementation aborts the batch on such a gate failure rather than silently skipping it. Prefix frames are retained, while branch action logs begin after the intervention; use the recorded prefix source for full trajectory accounting.

## What is established

[TAPT009](tapt-libero-run03.md) resolves the old character/byte interruption in the tested online runs and establishes native task success with learned feedback. It does not establish recovery robustness or a success-rate benefit of TAPT008 training. Three configurations reached 5/5 on one familiar task; the remaining 4/5 includes a transport failure that stays in the denominator.

## Source-grounded alignment

Primary paper: [VLAs-as-Tools v1](https://arxiv.org/pdf/2605.13119), sections 4.1–4.2, 5.1 and appendices A–B. The local PDF text was inspected for this audit. Author implementation is pinned to [f4eb160](https://github.com/cxliu0314/openpi/tree/f4eb160ba52b22c1e85fe432de59c24bbbac6187); a public author fork is evidence of reusable code, not proof that every paper experiment has been released.

| Item | Confirmed implementation | Interpretation / limitation |
|---|---|---|
| Invocation | `src/bvi/tool_family.py`, `scripts/eval_libero_family.py`, `scripts/serve_libero_family.py`: family plus grounded instruction, call-locked adapter, action queue reset | Matches section 4.1 interface structure. Real GPT text reaches the executor. |
| Residual families | `scripts/train_libero_family.py`: four independently selected LoRA banks with shared frozen backbone and progress head | Matches deterministic family routing in section 4.2.2. Our bank-management code is a local implementation. |
| LoRA layout | Author `src/openpi/models/gemma.py:96,107`: attention/FFN ranks and alphas 16/16 for VLM, 32/32 for action expert | Source-code defaults reused; these are not four tool families. Four banks each contain the selected expert residuals. Do not call these ranks paper-specified hyperparameters. |
| Progress head | Author `src/openpi/models/pi0.py:373` and `scripts/train_chunk_progress_head_only.py:117`: pooled image/prompt prefix plus chunk-step embedding | Reuses author head. Current scheduler consumes the first predicted chunk value; the full vector is logged. It is not measured future physical success. |
| Labels | Local `scripts/train_libero_family.py:127`: `(index-start)/(end-start-1)`, clipped/masked to a window | Invocation-local temporal proxy, compatible with normalized segment progress in appendix A. Not a calibrated success probability; progress on failed/off-distribution attempts remains unvalidated. |
| Loss | Local `scripts/train_libero_family.py:243`: action flow-matching loss + 0.1 masked progress MSE | Real joint training of selected residual and head. Author progress training default weight is 0.1; our joint training script is not identical to the author's head-only script. |
| Events | Author `examples/libero/openvla_eval_port/run_libero_eval_openpi.py:84,374–438`; local `src/bvi/progress_monitor.py` | Reuses reach/move .9, grasp/release .6, two threshold observations, drop .03, stagnation ten observations/.03, cooldown 15. Local monitor is a port, not byte-for-byte evaluator replication; ordering, call lifecycle and planner behavior differ. |
| Data pipeline | 20 complete demonstrations train, five validation, then 200 invocation windows | Small local adaptation. Appendix A describes global video context and multimodal labeling; our gripper/state segmentation and review do not reproduce that full pipeline. |
| Training scope | Already LIBERO-trained `pi05_libero`, TAPT007 adaptation then TAPT008 language variants | No DROID intermediate stage, broad-task evaluation or RL stage. This is minimal method/component reproduction, not reproduction of paper scores. |
| Memory/recovery | `src/bvi/task_memory.py`, manually reviewed language variants | Local engineering extensions. Their effects cannot be attributed solely to TAPT. |

Important hidden coupling: `task_memory.transition_rejection` reads progress and progress-derived stop reasons. Merely hiding numeric progress in the GPT prompt leaves a feedback channel through rejected grasp transitions. A feedback ablation must address both paths.

## Proposed next experiment: fixed-checkpoint paired disturbance

Do not retrain. Freeze TAPT008 step200, checkpoint SHA256 `ee48e60dd979985cfd17ee126b1c703bd7c29f82907821cb2e801e66d90b67f2`. Keep the existing GPT model, images, instruction schema, object-memory format, action limits and native evaluator. Disable the progress-dependent transition guard in **both** arms; retain object execution memory with progress fields redacted in the no-feedback arm.

- **Learned feedback:** current learned event monitor and progress messages.
- **Bounded-call control:** no learned progress values or learned events reach the scheduler/planner/memory. The head may be computed for diagnostic logging only. Replan at the declared invocation step cap; retain the same total action/request caps.

This compares the **combined feedback and event-triggering mechanism**, not the isolated causal effect of numeric progress. Actual request counts need not match; report them and cost. A separate matched-frequency visual-monitoring control would be needed to isolate event timing from information content; it is not included in the first pilot.

Use the same task and five official initial states as TAPT009. Generate one common prefix per initialization up to the first valid pre-grasp boundary for the cream-cheese box. Save simulator state, observation hashes, model RNG, action queue, family/call state, monitor state, planner history and budget counters. Fork both conditions from that snapshot. Same initial state alone is insufficient to claim paired recovery states.

For each initialization, use both a sham intervention and one fixed +3 cm world-X displacement of the target before grasp: 5 states × 2 interventions × 2 arms = 20 branch episodes. This is a synthetic perturbation diagnostic, not an official benchmark score. Preserve orientation; validate table support, nonpenetration and absence of existing gripper contact. If invalid, record the exclusion and stop that pair; do not search for a favorable displacement or replace the initialization. Both arms receive identical post-intervention camera observations, but no intervention flag, target pose, native subgoal predicate or diagnostic contact data.

The 520-action/20-request limit includes the common prefix; clone remaining counters. Prefix API expenses are recorded once, then branch expenses individually. GPT randomness remains a limitation even with paired simulator state; log all requests/responses rather than claim deterministic causal isolation.

## Gates before spending

1. Implement state capture/restore and demonstrate identical initial branch observations, state and RNG hashes. Verify complete model/environment state restoration before calling this paired evaluation.
2. Test information isolation: changing diagnostic progress must not alter the control scheduler, prompt, memory, transition acceptance or queue behavior.
3. Validate injection, sham and invalid-state rejection without GPT. No task-policy tuning against the five evaluation outcomes.
4. Freeze source commit, checkpoint, configuration and perturbation manifest. Reconcile funds/resources and price a bounded smoke run before deployment; this design authorizes no new spend.

Report every branch: native success, post-intervention target grasp and completion latency (diagnostic state predicates only), steps, event sequence, object-choice errors, request count, usage cost, exclusions and infrastructure failures. Define recovery latency from intervention to first valid target grasp; unrecovered attempts are censored, never recorded as zero. Report disturbed and sham success separately and paired outcome counts; five pairs do not support broad superiority claims.

## Migration gate

If the learned-feedback arm completes at least one disturbed case with a documented learned event followed by valid recovery, it supplies a recovery example; aggregate paired results still determine whether benefit is supported. Even a negative comparison can close this diagnostic if fully reported. Then start MS-HAB observation/action/data-boundary adaptation and a Fetch manipulation pilot. Do not transfer LIBERO normalization or arm weights as if robot embodiments matched. LightNav remains a separate navigation backbone and follows manipulation validation.

No new recording was produced by this audit. Future executed batches must receive actual UTC recording dates, unique task-named directories, exact per-episode filenames, preserved failures and updated media hashes.
