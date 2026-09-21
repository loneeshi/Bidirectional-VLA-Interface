# SAC interface audit amendment — 2026-09-20

Source: [Lei et al., arXiv:2605.13119v1](https://arxiv.org/pdf/2605.13119v1), Table 3/4 (p.8), §5.5 (p.9), Appendix D (p.17). This amendment follows observation of the first two current-run outcomes. It is not a preregistration for those observations or the already-running batch.

## Interpretation corrections

The present MS-HAB system is a heterogeneous LightNav/SAC integration baseline, not a TAPT reproduction. No training is requested. Equality of `tool_family` and `skill` enforces consistent routing; the actual restriction of choice comes from the benchmark-derived admissible-call set. Navigation consumes language; SAC manipulation does not. Both may nevertheless be affected by invocation boundaries, policy restarts, retries, and early termination. Navigation degradation and manipulation neutrality are hypotheses, not established mechanisms. A negative MS-HAB result alone would not reproduce Table 3's LIBERO ablation.

Appendix D specifies 50 trials per evaluated task. Set the staged coverage target to seeds 0–49 while retaining all 1000 manifest rows. The currently frozen batch remains seeds 0–19, with batches of at most 20. Fifty MS-HAB seeds have only comparable sample count, not matched task/initial-state semantics. This target does not authorize exceeding 800 total API requests, USD 2, GPU1-only constraints, or the existing deadline. At 40 calls each, 50 episodes require provision for 2000 calls; reconcile the current batch before scheduling remaining seeds. Unrun rows remain visible.

## Metrics amendment

`interface_metrics` derives explicitly labelled local proxies from serialized `skill_results`. An intervention is an interrupted call caused by `missed_grasp` or `grasp_lost`; ordinary step-limit continuation is excluded. A recovery attempt begins only when the next call repeats the interrupted skill and target. Recovery succeeds when native success is recorded within at most two consecutive matching calls. Another interruption ends that recovery window. This is a bounded grasp-recovery measure, not semantic plan-change detection or the paper's exact Table 4 metric.

Report intervention-episode frequency over episodes with available histories, successful recovery windows over attempted windows, and all API calls per completed episode. Missing histories produce null, not zero. Preserve observed denominators. Historical summaries already contain call IDs, skills, targets, feedback and steps, so these proxies can be reconstructed after the batch without altering execution. Missing images, instructions or richer semantic replan decisions cannot be inferred from these fields. Future evaluation needs explicit event IDs, trigger, parent call, native subtask identity and outcome before claiming semantic Replan SR or navigation recovery coverage.

Recovery success refers to the matching invocation's native success, not whole-episode success: a successful Pick recovery remains successful if a later Place fails. The two-call, same-skill/target window misses delayed or cross-skill resolutions, tending to undercount such recoveries. Its success-rate bias has no guaranteed direction because the restricted attempt denominator also excludes some strategies, and native success alone does not establish causal resolution of the triggering anomaly.

## Batch-close accounting priority

First reconcile actual global request consumption across all episodes, failed attempts, smoke calls and retries. The current batch alone can use 20 × 40 = 800 requests, the entire authorization. Its episode cost caps sum to USD 1.00 (20 × USD 0.05), below the global USD 2 cap; cost headroom cannot substitute for request headroom. Schedule no continuation, extra panel or rerun on assumed unused requests. Record actual remaining authorization before choosing any next batch.

Report the number and seeds of episodes terminated by the 40-call cap or API cost cap separately from natural termination, other limits and infrastructure failures, using terminal reasons and budget-error evidence. If the reason only records a generic exception, mark the cause unresolved until logs establish it. Report the all-calls mean as observed resource consumption under caps. For budget-terminated episodes, unconstrained completion demand is right-censored (at least the observed calls, potentially never completing); this is not missing data and must not become null or be silently excluded. Also report uncapped-terminal and budget-terminal counts and conditional means, with their selection caveat. Do not compare the capped overall mean directly with Table 4's means as equivalent completion demand. Other horizon limits also restrict interpretation and must remain visible.

## Five-step control

Do not treat `--organizer-slice-steps 5` alone as a faithful direct-monitoring arm. Under the unchanged 40-call cap it permits at most 200 physical actions, versus 1600 under 40-step slices (both also obey 7000 episode action cap). Every call starts an executor invocation; observing an ongoing invocation every five steps is a different operation. A proper comparison needs persistent execution across monitoring-only queries, equal action/time horizons, declared intervention semantics and a separately feasible API allocation. This arm is recorded as not_run_budget_and_protocol_gate. The existing coordinator rejects five-step slices in frozen benchmark mode; retain that guard.

## Source and environment identity

Progress monitor constants are now labelled local settings. The pinned author fork is a related reference, not proof of exact numeric provenance. Exact upstream file/line verification remains open.

The binder now deep-copies EnvConfig for each seed and checks the serialized pre-factory identity before passing it to the mutating environment factory. Evidence states the snapshot stage. This prevents cross-seed mutation for future bindings; it does not certify earlier bindings, identical physical states, or full RNG-state capture. Existing bindings remain preserved and require a separate reproducibility check before being reused as matched-condition evidence.

## Author enquiry draft — not sent

Subject: Implementation details for VLAs-as-Tools / arXiv:2605.13119

Dear authors,

I am implementing a no-training integration baseline inspired by your tool-family interface, using LightNav navigation and object-specific SAC manipulation on MS-HAB. I understand this differs from your evaluated TAPT systems. Could you share the intended public implementation or clarify the progress-head feature tap and thresholds, DROID split manifest, residual insertion layers and ranks, and the definitions/denominators of Table 4's intervention, replanning success and average calls? Does direct monitoring preserve the executor state between five-step queries? This would help us distinguish faithful implementation from adaptations. Thank you.

The PDF lists eight authors; three names in the supplied note are an abbreviated selection. Institutional affiliation does not establish a personal connection. No email has been sent.

## Execution boundary

These are local code/document changes only. Do not deploy them over the running frozen batch. Wait for the current batch to finish as requested, then reconcile results, API usage and GPU state, preserve raw attempts, and produce a versioned derived-metrics report with its source hashes. Keep continuation instructions with the manifest. Current completion and cost are not asserted by this amendment.
