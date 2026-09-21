# Current evaluation architecture

The repository is evaluation-only. `bvi-eval` is the only public entry point;
there is no training, fine-tuning, data-collection, or calibration command.

## Data flow

```text
frozen profile + 16-plan manifest
              |
              v
       panel coordinator
              |
       one episode at a time
              |
              v
 MS-HAB adapter -> serial runtime -> PPO / SAC / teleport
              ^             |
              |             v
       GPT bridge <- goal tools and bounded feedback   (GPT only)
              |
              v
 panel-status.json -> bvi-eval summarize -> public summary
```

## Active modules

| Module | Responsibility |
|---|---|
| `evaluation.py` | Frozen profiles, 16-plan binding, resume state, locking and aggregation |
| `mshab_runner.py` | One MS-HAB episode with native success and horizon checks |
| `mshab_adapter.py` | Official environment observations, events and PPO/SAC policy state |
| `runtime.py` / `protocol.py` | One skill owner at a time and validated request/feedback schemas |
| `coordinator.py` / `goal_tools.py` | GPT selection from the allowed skill/target set |
| `bridge.py` / `bridge_server.py` | Local credentials, bounded API calls and cached responses |
| `teleport_skill.py` | Standardized navigation replacement used only by the teleport profile |

## Invariants

- The plan roster contains exactly 16 unique five-object TidyHouse validation plans.
- GPT and teleport rows must bind to the fixed row's plan and initial-state hash.
- Native horizons remain Navigate 500, Pick 200 and Place 200 control steps.
- Episode limits are 7,000 environment steps and 900 wall-clock seconds.
- Fixed and teleport make zero model API requests.
- GPT has at most 40 decisions; each requested skill slice is at most 40 actions and 180 seconds.
- All profiles have `training_updates = 0`.
- Execution is serial and resumable; attempts are appended rather than overwritten.
