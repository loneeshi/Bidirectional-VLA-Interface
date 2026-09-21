# GPT + PPO/SAC progress-feedback protocol v1

Status: implemented locally; CPU contract tests passed. No GPU rollout has yet
validated this behavior. Existing oracle-feedback experiment results remain
historical and are not relabeled.

## Feedback boundary

New GPT + official PPO/SAC runs launched by `run_ppo_sac_paired16.py --goal-tools`
also pass `--progress-feedback`. The selected tool returns invocation-local
continuous progress in `[0,1]` with source
`heuristic_policy_observation_relative_geometry_v1`.

The proxy uses only quantities already represented in the official low-level
policy observation: base/target relative geometry for Navigate, TCP/object
relative geometry and grasp state for Pick, and object/goal relative geometry
plus grasp state for Place. It does not call `_navigate_check_success`,
`_pick_check_success`, `_place_check_success`, or stateful `evaluate()`.

This is a rule-based progress baseline, not learned progress and not a
reproduction of the paper's progress head. A later learned provider must use a
separately trained and accepted head and declare its checkpoint and data.

## Control behavior

- Native evaluator state remains available only for background scoring and
  irreversible benchmark failure or whole-episode termination.
- A tool invocation does not return `succeeded` from a native subtask predicate.
- Each invocation normally runs to its requested bounded slice. `step_limit`
  returns `timed_out` while preserving the latest progress value and source.
- Requirement completion remains `unknown` when only progress is available.
- GPT receives the new image, bounded-call result, progress value, provenance,
  and prior history. Progress is explicitly not proof of completion.

## Historical compatibility

`GoalRLSkill` retains the old native-oracle completion path for exact historical
reproduction. `ProgressGoalRLSkill` is selected only by `--progress-feedback`.
The paired runner now enables it for new GPT goal-tool attempts. Old attempts,
videos, summaries, and scores must remain labeled native-oracle feedback.
