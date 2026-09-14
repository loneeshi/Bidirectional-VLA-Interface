# Interface and runtime design

This document describes the intended MS-HAB baseline contract. A design requirement is not a claim that the simulator adapter or VLM path has passed an end-to-end test. See the [current status](../README.md#current-status) for verified gates.

## Module boundaries

| Module | Input | Output and responsibility |
|---|---|---|
| Environment adapter | Control action and reset configuration | Timestamped observations, official environment events, and reproducible episode identity |
| Observation adapter | RGBD, proprioception, allowed task context | VLM observation packet; separate checkpoint-compatible policy observations |
| Coordinator | Task, image observations, supported skills, prior results | Structured request for one supported skill |
| Runtime | Request and adapter capabilities | Validated execution, bounded lifetime, explicit state transitions |
| Skill adapter | Checkpoint-compatible observation and supported target binding | One action at the environment control rate |
| Feedback adapter | Execution history and permitted evidence | Completion/failure/timeout plus requirement states and evidence references |
| Recorder | Observations, requests, actions, events, provider usage | Replayable events, configuration manifest, evaluation summaries, and video |

## Invocation and feedback

An invocation needs a stable call identifier, skill family, target reference, desired effect, supported requirements, and execution limits. Target references must be checked against the current episode and observation. Request identifiers enable duplicate-call detection. Unsupported constraints must be rejected or returned as unsupported; silently dropping them would change the request's meaning.

Feedback needs the same call identifier, a terminal or running status, elapsed control steps, a reason code, and evidence references. Requirement verification uses three distinct values:

| Requirement state | Meaning |
|---|---|
| `satisfied` | Available evidence supports the requested condition |
| `unsatisfied` | Available evidence supports that the condition is not met |
| `unknown` | The permitted evidence does not resolve the condition |

Progress is not proof of completion. A timeout is not task success. Failure descriptions should distinguish observed events from hypotheses about their causes. Any future learned verifier must preserve this distinction and be evaluated separately from the skill's physical success.

The first RL adapters support the invocation fields exposed by their training and environment interfaces. They do not gain arbitrary language-conditioned constraints by receiving JSON. More detailed instructions, such as a required grasp location, need an appropriate policy and evidence source before they can be advertised as supported.

## Control ownership

The first runtime executes a single skill at a time. The official Fetch policy action has 13 dimensions under `pd_joint_delta_pos`:

| Indices | Controller component |
|---|---|
| 0–6 | Seven arm joint commands |
| 7 | Gripper command |
| 8–9 | Head pan and tilt |
| 10 | Torso lift |
| 11–12 | Base forward and yaw velocity |

The tested external action space is `Box(-1, 1, (13,), float32)`. These normalized inputs must not be confused with physical joint or base limits. A stationary-head wrapper masks indices 8–9 but preserves the vector length. Fetch does not expose a separate lateral base velocity in this controller.

The active skill retains control of the full action vector. Navigation/manipulation overlap will need actuator ownership declarations, an arbiter, synchronized observations, and interruption semantics. Adding a second concurrent action producer without those mechanisms is not a supported extension.

The strict runtime checks shape, finite values, and controller bounds before stepping. Policy raw outputs and controller-ready actions are different stages: PPO mean actions can exceed the `Box` range, and the upstream controller clips them. An adapter must preserve the same clipping behavior before submitting actions to the strict runtime, recording raw values and out-of-range/clipping counts. Nonfinite values or the wrong dimension are errors, not values to repair by clipping. Stopping a skill must execute an explicitly defined safe hold for the relevant controller, rather than assuming that an all-zero normalized vector preserves the gripper state.

## Observation and evaluation boundaries

The official RL pipeline uses stacked head/hand depth and robot/target state. Its privileged target information is an explicit property of this baseline. VLM RGB observations are a separate path; a depth-only checkpoint wrapper does not automatically retain suitable VLM images.

The environment advances its own task pointer when its configured completion checks are met. The coordinator cannot set that pointer, teleport the robot, reset a failed skill into a favorable state, or relax success thresholds inside a reported episode. Reinitialization is allowed for isolated diagnostics only and must be marked as such.

The first feedback adapter may use simulator checks, labeled **oracle completion**. RGB/proprioception-only verification is a subsequent research condition. Simulator ground truth used for evaluation must be recorded separately from information made available to the coordinator.

## Limits and failure handling

An invocation is bounded by simulation control steps and wall-clock time. The coordinator has a separate request/token budget. Invalid output, missing images, stale observations, policy loading errors, NaN actions, an unsupported target, timeout, and environment failure must yield distinct recorded events.

The simulator need not advance while an API request is pending. The log should record both simulation time and wall-clock time so that model latency remains visible. A fallback coordinator must be labeled in the run manifest; a scripted fallback cannot satisfy the image-based VLM gate.

## References

- [MS-HAB environment and wrappers](https://github.com/arth-shukla/mshab/tree/main/mshab/envs)
- [Fetch controller implementation](https://github.com/haosulab/ManiSkill/blob/mshab/mani_skill/agents/robots/fetch/fetch.py)
- [MS-HAB evaluator](https://github.com/arth-shukla/mshab/blob/main/mshab/evaluate.py)
