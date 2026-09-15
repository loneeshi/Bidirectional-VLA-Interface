# A/B/C/D demonstration completion plan

Four actual continuous first-object TidyHouse rollouts are required, using the
same explicit scene/plan/initial-state fingerprint and an oracle dispatcher to
isolate the low-level policy change. A later VLM comparison is separate.

| Variant | Navigation | Manipulation | Current evidence | Remaining gate |
|---|---|---|---|---|
| A | Official PPO | Official SAC | Earlier successful first-object chain | Re-record with clear, marker-free video |
| B | LightNav-0 + Fetch tracker | Official SAC | 270 navigation steps; no arrival | Ground the correct destination, check camera/base geometry, achieve arrival and both handoffs |
| C | Official PPO | Fetch-adapted pi0.5 | DROID inference with synthetic state only | Fetch data/action transforms, normalization, suitable checkpoint and individual Pick/Place validation |
| D | LightNav-0 + Fetch tracker | Same Fetch-adapted pi0.5 | No combined control evidence | Pass B/C component gates, then test combined handoffs |

The deliverables are A.mp4, B.mp4, C.mp4 and D.mp4 with matching manifests,
per-skill events, object/grasp evidence and final success/failure. Four successful
chains cannot be guaranteed before adaptation experiments. A failed rollout must
be labelled failed; a prediction probe, fallback policy or concatenation of
different episodes cannot substitute for the requested method.

## Green object diagnosis

The green sphere at the gripper in the current B video is the benchmark's
`ee_rest_goal` debug visualization. The pinned upstream source constructs a
radius-0.05 green sphere with no collision and positions it at the desired EE rest
pose when rendering. Trial B never reached Pick, and its step records show
`is_grasped=false`; that sphere is not evidence of an object being carried.

The runners now default to `invisible_goals_in_human_render=True` as well as no
parameter overlay. `--show-goal-markers` enables a labelled debug view. This uses
the upstream render option; it does not replace object meshes, remove collision
geometry, or edit pixels in recorded footage. A new GPU render is still required
to verify appearance and actual object visibility, including a gripper view.

Source: [pinned MS-HAB environment](https://github.com/arth-shukla/mshab/blob/e9ff3d23496d38e4431c8d913e147ffa007f7f72/mshab/envs/sequential_task.py).

## pi0.5 adaptation gate

Do not apply the eight DROID outputs directly to Fetch's 13 normalized controller
channels. The seven arm joints belong to different robots; numeric padding is
not kinematic transfer. The concrete route is successful MS-HAB manipulation
demonstrations with named Fetch state, synchronized head/wrist RGB, language,
and the actual action convention, then a pi0.5 Fetch data transform, train-only
normalization statistics, and a bounded fine-tuning pilot. Hold out episodes
before training. Record failed expert rollouts separately from training data.

Define explicitly whether the learned manipulation policy owns arm/gripper only
or all 13 channels, and preserve required torso/base behavior consistently between
the demonstrations and evaluation. Reusing official SAC actions during a pi0.5
segment would invalidate C/D attribution.

## Paid execution gate

The preceding experiment's authorization is closed. New GPU time, storage,
request limits and any training require a new bounded scope in the private ledger.
Start with resource-level automatic stopping plus early artifact synchronization;
test the stopping path before substantive work. A model-process timeout does not
stop Pod billing. Recheck UTC and authorization before every new run after a pause.

Do not describe a shell command or an untested watchdog as a verified provider
spending cap. The available Pod tool schema has no native deadline field; an
independent authenticated resource-stop mechanism must be verified before use.
