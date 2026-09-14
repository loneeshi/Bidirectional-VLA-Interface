# Evaluation and reporting

The project currently reports infrastructure progress, not benchmark performance. A reproducible empty-scene rollout, a successful skill, a complete task demonstration, and a validation score are different results.

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

A small subset result must state the subset explicitly. Aggregate benchmark claims require the declared official coverage and must not be extrapolated from a selected success video. No benchmark numbers are currently available.

## Official references

- [MS-HAB benchmark and task definitions](https://github.com/arth-shukla/mshab)
- [Evaluation implementation](https://github.com/arth-shukla/mshab/blob/main/mshab/evaluate.py)
- [Evaluation launch script](https://github.com/arth-shukla/mshab/blob/main/scripts/evaluate_sequential_task.sh)
- [Task and success/failure configuration](https://github.com/arth-shukla/mshab/blob/main/mshab/envs/planner.py)
