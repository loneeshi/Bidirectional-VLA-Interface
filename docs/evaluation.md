# Evaluation and reporting

The project currently reports a working constrained VLM interface and a successful one-object skill chain, not aggregate benchmark performance. A reproducible empty-scene rollout, a successful skill, a complete task demonstration, and a validation score are different results.

G1 has passed on one real ReplicaCAD TidyHouse validation scene: seed 0, build index 69, task-plan index 23, 200 zero-action steps. It verifies loading, observations/actions, and rendered video. It does not test a learned policy or establish task success. The [recorded G1 result](reproduction.md#recorded-g1-result) lists the measured interface and timing. After two unsuccessful seed-0 policy diagnostics, a seed-1 live VLM run supplied G2–G4 engineering acceptance evidence. All three policy results are retained below.

## First official-policy diagnostic

The first run used the pinned upstream evaluator through `scripts/run_official.py`, with the official oracle dispatcher and `rl_all_obj` checkpoints. It attempted one TidyHouse validation trajectory at seed 0, build configuration index 69 and task-plan index 23.

| Measure | Result |
|---|---|
| Coverage | One validation scene, one sampled plan, one trajectory |
| Executed horizon | 7,000 control steps |
| `success_once` | 0 |
| `success_at_end` | 0 |
| Completed tasks | 0 / 1 diagnostic attempts |
| `subtask_fail_counts` | `{"1": 1}` |
| Wall time | 686.43 s, including initialization |
| Observed GPU memory use | Approximately 4.8 GB; a runtime observation, not an established peak-memory requirement |

The trace advances from Navigate to Pick at **zero-based step index 128**. At index **143** (`elapsed_steps=144`), the first Pick failure appears with `cumulative_force_within_limit=false`; both `robot_force` and `robot_cumulative_force` are approximately `11109.0264` in the environment's reported force signals. The remaining official horizon preserves the failed outcome. Pick does not succeed and Place is not reached, so neither G2 nor G3 passes.

The force-check failure is observed evidence. The trace alone does not identify the exact collision geometry or prove a causal explanation such as a poor approach pose. The cumulative signal follows MS-HAB's per-step aggregation and must not be presented as a physical impulse. Likewise, the end-of-episode subtask count identifies where the trajectory ended, not a complete cause classification.

These results are a reproducible **single-case diagnostic failure**, not a population success rate or a published benchmark comparison. The run demonstrates that the official evaluation path can finish and produce metrics; useful benchmark performance remains to be established.

## Per-object oracle protocol diagnostic

A subsequent seed-0 run used `scripts/run_coordinator.py --dry-run --policy-type rl_per_obj` to execute the policies through the `bvi` runtime. It used an oracle coordinator and made no model API calls.

| Invocation | Control steps | Outcome | Total steps |
|---|---:|---|---:|
| Navigate | 128 | Succeeded | 128 |
| Pick | 64 | Succeeded | 192 |
| Navigate while holding | 163 | Succeeded | 355 |
| Place | 115 | Benchmark failure | 470 |

This run establishes successful Navigate/Pick invocations and continuous progression into Place through the protocol. Place did not succeed, so G2's all-skills criterion and G3's full-chain criterion remain unmet. The failure status does not by itself identify the physical cause. This is an early-stop diagnostic with different policies and dispatch code from the full-horizon all-object run above; their outcomes must not be pooled as a matched benchmark comparison.

## Live VLM protocol diagnostic

A seed-1 run used OpenAI `gpt-5.6-luna`, the SSH file bridge, and official `rl_per_obj` checkpoints. The recorded environment was TidyHouse `val`, build configuration index 69, task-plan index 2 (`tidy_house-sequential-val-90`). The scene was `v3_sc3_staging_00.scene_instance.json`. Six API requests produced six successful skill invocations:

| Invocation | Control steps | Outcome | Total steps |
|---|---:|---|---:|
| Navigate | 29 | Succeeded | 29 |
| Pick | 48 | Succeeded | 77 |
| Navigate while holding | 140 | Succeeded | 217 |
| Place | 16 | Succeeded | 233 |
| Navigate to the next object | 70 | Succeeded | 303 |
| Pick the next object | 33 | Succeeded | 336 |

The first object's four-skill chain completed at step 233. Execution stopped at the configured **six-request experiment limit** after step 336, with `task_success=false` and `benchmark_result=false`. This is a partial five-object TidyHouse episode, not a completed full task. The recorded runner wall time was 183.67 s; the exported 512 × 512 video duration was 16.85 s. Playback duration is simulation time, not API or end-to-end latency.

The coordinator is **constrained by oracle task-plan targets and supported calls**: all six decisions had exactly **one allowed skill/target pair**. Skill completion uses the environment's oracle checks. This setup tests real image transport, request validation, skill invocation, and feedback delivery. It does not establish autonomous task decomposition, observation-only verification, failure recovery, or a causal benefit from VLM reasoning. The successful seed-1 run and unsuccessful seed-0 run use different sampled plans and cannot establish a coordinator improvement.

The evidence audit verified one environment reset, no automatic reset between skills, and no teleportation path in the invoked adapter. All 336 executed actions were finite 13-dimensional vectors within bounds; 203 steps had raw policy outputs outside the normalized range that were correctly clipped before execution. Four loaded checkpoint hashes matched the saved official metadata.

The first two requests used different head/hand camera frames. Their decoded image bytes matched the saved PNGs by SHA-256; the second request included feedback matching the first skill's actual completion event. Six provider responses, accepted requests, and skill results matched by attempt/call identifiers. These checks support **G2–G4 as single-case engineering gates**, with the constrained coordinator limitations above.

The local API client round trip averaged 2.88 s per request; the remote coordinator's file-bridge wait averaged 12.02 s, including transfer, polling, and API waiting. These are observed application latencies, not isolated model-compute measurements. The gap identifies bridge overhead to investigate before making runtime comparisons.

## Three evaluation levels

| Level | Purpose | Required report |
|---|---|---|
| Infrastructure smoke test | Verify imports, GPU/rendering, and reset/step | Version manifest, observation/action contract, steps executed, logs and frames |
| Diagnostic skill or chain | Locate policy and composition failures | Explicit scene/plan/seed, skill outcomes, resets or teleportation, full transition trace |
| Benchmark evaluation | Measure performance under a declared official protocol | Coverage, matched settings, official metrics, all included episodes, uncertainty and failures |

The first demonstration targets Navigate → Pick → Navigate while holding → Place. A shortened one-object task is useful for diagnosis and must be labeled as such. It is not the full TidyHouse benchmark, which includes multiple objects and a longer fixed horizon.

## Protocol requirements

Use `SequentialTask-v0` for long-horizon skill composition. The subtask training environments support isolated policy checks and have different initialization/reward behavior. Keep the official task semantics and success/failure thresholds for benchmark reporting.

MS-HAB defines 63 training scenes and 21 validation scenes, with 10,000 training plans and 1,000 validation plans for each long-horizon task. A single environment does not automatically cover all validation scenes across resets. Use explicit scene/plan sharding or a verified parallel allocation, and publish the actual coverage.

For the pinned baseline, the official full TidyHouse horizon is 7,000 control steps with `continuous_task=True`. Match the navigation configuration to the selected upstream evaluation, including the recorded `ignore_arm_checkers` setting. Confirm these values in the pinned code and save the resolved configuration rather than treating a shell script's defaults as the full validation protocol.

The upstream evaluator's policy dispatcher uses the environment task pointer and target information. Reproducing that dispatcher is an oracle baseline. To evaluate VLM coordination, record the VLM's actual decisions and how invalid or incompatible requests are handled. The coordinator may not modify the environment pointer or declare success on its own.

## Required reporting fields

| Category | Fields |
|---|---|
| Environment | Source commits, dependency versions, task, split, scene IDs, plan IDs, seeds, number of environments |
| Episode semantics | Horizon, `continuous_task`, success/failure thresholds, navigation mode, reset/teleport behavior |
| Policies | Checkpoint identity/revision, observation wrappers, controller/action shape, deterministic or sampled actions, raw out-of-range/clipping counts |
| Coordinator | Scripted/oracle/VLM, provider and model, prompt/config revision, images supplied, allowed state, request limits |
| Outcomes | Number attempted/completed, `success_once`, `success_at_end`, trajectory length, timeouts and failure events |
| Resources | Runtime, simulator throughput, peak GPU memory, VLM latency and request/token usage |

Report `success_once` and `success_at_end` separately. A task can temporarily satisfy a condition and fail to preserve it. Do not omit unsuccessful episodes or label a final subtask index as a proven failure cause.

## Demonstration acceptance

A first end-to-end demonstration must preserve one physical episode across all skill transitions, use real image inputs to the coordinator, save structured requests and feedback, and export a viewable video. Record an initial plan and at least one subsequent decision based on execution feedback. A scripted coordinator remains useful for debugging but does not meet this criterion.

For failure analysis, separate observable events from explanations. Examples of reportable events include approach pose outside the policy's successful region, a lost grasp, an official completion check remaining false, policy timeout, an invalid VLM request, or a provider error. Causal explanations require supporting evidence.

## Comparison design

Once the infrastructure is stable, compare a scripted/oracle dispatcher and a VLM coordinator using the same scenes, plans, seeds, low-level checkpoints, observation permissions, and completion checks. If a condition uses simulator ground truth for feedback, identify it as oracle completion. A learned or observation-only verifier is a separate condition.

A small subset result must state the subset explicitly. Aggregate benchmark claims require the declared official coverage and must not be extrapolated from a selected success video. The full-horizon all-object result and early-stop per-object trace above have different purposes and settings; no aggregate benchmark score is available.

## Official references

- [MS-HAB benchmark and task definitions](https://github.com/arth-shukla/mshab)
- [Evaluation implementation](https://github.com/arth-shukla/mshab/blob/main/mshab/evaluate.py)
- [Evaluation launch script](https://github.com/arth-shukla/mshab/blob/main/scripts/evaluate_sequential_task.sh)
- [Task and success/failure configuration](https://github.com/arth-shukla/mshab/blob/main/mshab/envs/planner.py)
