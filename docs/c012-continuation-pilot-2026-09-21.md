# C0/C1/C2 continuation pilot

This is a six-episode development smoke: two previously analyzed validation
plans times C0/C1/C2. It is not the final 16-plan panel and is not unseen test.

- C0: continuous execution, no retry after an unsuccessful full-horizon
  tool/target attempt.
- C1: same interface and budgets, at most three revisits per abandoned target.
- C2: C1 plus frozen `spawn-prior-v3.json` derived only from 24 train-spawn
  SAC probes. The prior exposes policy-input relative XY distance ranges,
  sample/success/strict-success counts and uncertainty. It is not an optimum.

All conditions use GPT-5.6 Luna, official PPO navigation and per-object SAC.
C0 permits at most 20 valid physical tool decisions, one opportunity for each
of the twenty unique skill/target subtasks. Its API request cap is 24 so up to
four schema/incomplete responses can be retried without granting any subtask a
second physical attempt. C1/C2 retain up to 40 to permit bounded recovery.
All use 7000 episode actions and 900 seconds. The executor owns
the official per-invocation horizon: Navigate 500 actions and Pick/Place 200.
Native completion returns early. GPT does not output `max_steps` and is called
again only after completion, full-horizon timeout, or another terminal result.
The simulator continues after native force/subtask failure; those events latch
`official_episode_valid=false`. Infrastructure/numerical failure still stops.
Final completed objects are rescored from the five Place predicates rather than
the native sequential pointer. A repeat after full-horizon timeout is a retry.

The pilot is launched serially and stops on the first infrastructure failure.
Every attempt and the fixed 6-row denominator remain in `panel-status.json`.
API ceiling: 240 requests, 2048 output tokens/request, conservative reservation
USD 0.30. Actual provider and laboratory charges remain pending reconciliation.
No training, GPU0 or RunPod.

Launch source: commit `1068263`; remote `panel-v2`. The first panel attempt is
preserved as a zero-action/zero-API infrastructure failure caused by both runner
layers creating the same output directory. After the atomic-ownership fix, C0
seed 0 passed 120 physical actions and two accounted provider responses. This
establishes stable startup only, not an outcome or SR.

`panel-v4` is retained but invalid for the subtask-boundary study: it used
40-action slices and called GPT again after each slice. Its results must not be
mixed with the official-horizon rerun.

## Code reading path

1. `scripts/run_continuation_c012.py::command/main` builds the C0/C1/C2 panel.
2. `scripts/run_coordinator.py::main` resets the environment and owns the agent-tool loop.
3. `src/bvi/goal_tools.py::GoalToolAdapter.view` creates the five-object task,
   grounded target catalog, admissible tool calls and camera observation.
4. `src/bvi/feedback/digest.py::summarize` compacts invocation history; C2 adds
   `spawn_prior` through `src/bvi/feedback/spawn_prior.py::load_prior`.
5. `src/bvi/coordinator.py::VLMCoordinator.decide` constructs the exact system
   prompt, JSON context, images and strict response schema. `parse_request`
   validates returned JSON and injects the executor-owned horizon.
6. `src/bvi/goal_tools.py::GoalRLSkill` binds the selected target to the
   official PPO/SAC policy input. `src/bvi/runtime.py::SerialRuntime.execute`
   executes until success, official horizon or another terminal result.
7. `src/bvi/continuation.py::ContinuationGoalRLSkill.feedback` produces native
   evaluator feedback; `RetryLedger` implements the C0/C1/C2 retry difference.
8. `scripts/run_coordinator.py` appends feedback and trajectory summaries to
   history before the next GPT decision.
