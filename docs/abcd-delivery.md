# A/B/C/D demonstration completion plan

Four actual continuous first-object TidyHouse rollouts are required, using the
same explicit scene/plan/initial-state fingerprint and an oracle dispatcher to
isolate the low-level policy change. A later VLM comparison is separate.

[Environment, model-service and A/B/C/D commands](abcd-reproduction.md).

| Variant | Navigation | Manipulation | Current evidence | Remaining gate |
|---|---|---|---|---|
| A | Official PPO | Official SAC | [Clean successful chain, 248 steps](media/A-ppo-sac-clean.mp4) | Broader evaluation |
| B | LightNav-0 + Fetch velocity adapter | Official SAC | [Successful chain, 278 steps](media/B-lightnav-sac.mp4); grasp maintained on all 115 carrying steps | Broader evaluation |
| C | Official PPO | Fetch-adapted pi0.5 | [Actual state-conditioned control, Pick failed](media/C-pi05-state-failed.mp4); velocity-input pilot underway | Individual Pick/Place and composed success |
| D | LightNav-0 + Fetch tracker | Same Fetch-adapted pi0.5 | [LightNav succeeded, pi05 Pick failed](media/D-lightnav-pi05-state-failed.mp4) | Adapt manipulation policy, then verify full chain |

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
geometry, or edit pixels in recorded footage. A3 and B11 were inspected with
these settings: the green marker and overlay are absent and the blue can's
texture is visible during grasping and transport.

Source: [pinned MS-HAB environment](https://github.com/arth-shukla/mshab/blob/e9ff3d23496d38e4431c8d913e147ffa007f7f72/mshab/envs/sequential_task.py).

## pi0.5 adaptation gate

Do not apply the eight DROID outputs directly to Fetch's 13 normalized controller
channels. The seven arm joints belong to different robots; numeric padding is
not kinematic transfer. The concrete route is successful MS-HAB manipulation
demonstrations with named Fetch state, synchronized head/wrist RGB, language,
and the actual action convention, then a pi0.5 Fetch data transform, train-only
normalization statistics, and a bounded fine-tuning pilot. The current training-
scene demo deliberately overlaps seed 1; held-out evaluation must use a separately
declared split. Record failed expert rollouts separately from training data.

Define explicitly whether the learned manipulation policy owns arm/gripper only
or all 13 channels, and preserve required torso/base behavior consistently between
the demonstrations and evaluation. Reusing official SAC actions during a pi0.5
segment would invalidate C/D attribution.

## Paid execution gate

GPU time, storage, request limits and training require a bounded allocation
within the user's authorization, recorded in the private ledger.
Start with resource-level automatic stopping plus early artifact synchronization;
test the stopping path before substantive work. A model-process timeout does not
stop Pod billing. Recheck UTC and authorization before every new run after a pause.

Do not describe a shell command or an untested watchdog as a verified provider
spending cap. The available Pod tool schema has no native deadline field; an
independent authenticated resource-stop mechanism must be verified before use.
