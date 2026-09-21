# Evaluation contract

## Frozen settings

| Setting | Navigation | Manipulation | Dispatcher | API calls |
|---|---|---|---|---:|
| `fixed` | Official PPO | Official per-object SAC | Official fixed order | 0 |
| `gpt` | Official PPO | Official per-object SAC | VLA-as-Tools goal protocol | Bounded |
| `teleport` | Standardized teleport | Official per-object SAC | Official fixed order | 0 |

The three settings use the same 16 plan UIDs. GPT and teleport must also carry
the fixed row's recorded initial-state hash. Teleport nevertheless changes the
state delivered to manipulation after navigation; its partial-progress result
must not be interpreted as a stronger learned navigation policy.

## Domain parameters

| Parameter | Frozen value |
|---|---|
| Task / split | MS-HAB TidyHouse sequential validation |
| Plans | 16 unique five-object plans; 80 objects total |
| Seeds | 0–11, 13, 14, 16, 19 |
| Observation | RGB-D, three-frame stack, official 42D policy state |
| Controller | `pd_joint_delta_pos`, serial full-action-vector ownership |
| Native horizons | Navigate 500; Pick 200; Place 200 |
| Episode limits | 7,000 environment steps; 900 seconds |
| GPT model | `gpt-5.6-luna` |
| GPT limits | 40 decisions; 40 actions / 180 seconds per request; 2,048 output tokens |
| Training updates | 0 |

## Reported metrics

- completed episodes out of the fixed 16-plan denominator;
- complete five-object task success rate;
- completed objects out of 80 and per-episode mean;
- infrastructure failures, termination reason, environment steps and API requests;
- plan UID, seed, initial-state hash and source hash for provenance.

Infrastructure failures do not become benchmark failures and do not reduce the
denominator. Budget termination is reported separately from a native task
failure. A completed object is counted only when the native sequential scorer
advances through Place.

## Interpretation boundary

The current result is a matched 16-plan diagnostic. It is not the published
1,000-rollout benchmark and cannot support a broad ranking of navigation
methods. All three settings have 0/16 complete-task success; differences in
14/80, 13/80 and 21/80 describe partial progress only.
