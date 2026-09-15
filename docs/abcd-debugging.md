# A/B/C/D debugging and Fetch adaptation

This is a **training-scene, first-object diagnostic**, not the full five-object
TidyHouse benchmark. The task in seed 1 (`tidy_house-sequential-val-90-0`) is to
carry the blue `002_master_chef_can-0` from the kitchen counter to the seat of the
gray armchair beside the blue sofa. The earlier description of a wooden-table
destination was incorrect. The gray beanbag near the stairs is a different object.

All new runs use an oracle dispatcher to isolate the skill interfaces. They do
not use GPT/Opus or establish VLM planning gains. No external model API is called.

## Verified changes

- Human-render videos have no parameter overlay and hide the green `ee_rest_goal`
  debug sphere. Real cans retain their scene textures. A3 completed four skills
  in 248 steps; a second official recording failed at Place and is retained.
- Native head RGB was obstructed by the resting arm. An explicitly added
  `fetch_nav` camera at base-relative `[0.35,0,1.9]`, pitched down 0.15 radians,
  supplies 448x256 RGB. It is an observation augmentation, not an official-sensor
  benchmark result. The original head/wrist inputs to PPO/SAC remain intact.
- The LightNav adapter can use the upstream unicycle mapping: predicted forward
  displacement and yaw divided by the 0.25-second execution window, bounded to
  0.6 m/s and 1.2 rad/s. The original position tracker is retained as an explicit
  option. Its slow progression was a material mismatch with model cadence.
- A model STOP now commands a full second of braking and settling before checking
  completion. STOP by itself never means task success.
- Optional orientation recovery takes **boolean benchmark feedback** (close but
  not oriented) and a declared language instruction, resets LightNav history, and
  permits only model-predicted rotation. It is capped at two attempts. It does not
  read goal coordinates or relax benchmark checks; the privileged feedback must
  nevertheless be disclosed.

Source for velocity semantics: the pinned LightNav repository's
`src/lightnav/velocity.py` and `src/lightnav/habitat/policy.py`. Source for
completion semantics: [pinned MS-HAB environment](https://github.com/arth-shukla/mshab/blob/e9ff3d23496d38e4431c8d913e147ffa007f7f72/mshab/envs/sequential_task.py).

## Observed B failures and progress

| Run | Observation |
|---|---|
| B1/B2 | Clearer camera; STOP while distance check remained false |
| B3 | Route instruction with old tracker; 499-step navigation failure |
| B4 | New velocity mapping: Navigate and Pick succeeded; shared prediction cap interrupted carrying navigation |
| B5 | Navigate and Pick succeeded; carried object to target distance, kept grasp, stopped with wrong orientation |
| B6 | Repeat with route instruction did not reach the first goal within its official horizon |
| B7 | Direct goal instruction: first Navigate succeeded in 77 steps and Pick in 46; second navigation confused the beanbag with the armchair |
| B11 | Disambiguated target: all four skills succeeded in 77/47/115/39 steps; held-object check true on all 115 carrying steps; zero orientation-recovery calls |

Service-start/connection failures are logged separately from executed model
failures. Success is not established by any individual successful component.
The [complete B chain](media/B-lightnav-sac.mp4) is now verified for one object.
[A's clean reference video](media/A-ppo-sac-clean.mp4) uses the same camera augmentation.

## Fetch pi0.5 adaptation

`scripts/fetch_openpi.py` runs inside the isolated, pinned openpi environment
(`215abfb217dbac7d5f1273282331b9b1866c0479`). It converts successful continuous
Pick/Place segments to LeRobot, computes training-only normalization, fine-tunes
`pi05_base`, and serves a Fetch-specific checkpoint. No DROID action padding is
used. The policy owns all 13 normalized controller channels during manipulation.

Data contains native 128x128 head/wrist RGB, named 15-joint qpos, language, and
the 13 actions after official wrapper masking. Each action is paired with its
pre-action frame and its actual next-step event. Failed teacher segments are
excluded; their original runs remain available. Navigation actions are not used
as manipulation training examples.

The initial **vision-only** pilot used 118 frames and 200 LoRA steps. Training
loss decreased to about 0.11, but C1 failed Pick after 173 manipulation steps.
Audit found that copying LIBERO's `discrete_state_input=False` omitted joint
conditioning from pi0.5 despite carrying the state through the wire interface.
It also retained pre-mask head-action labels. This pilot is a failed experiment,
not an accepted Fetch policy.

The corrected configuration enables discrete state input and matches the actual
post-wrapper action events. A live tokenizer check confirmed that changing the
15 joint values changes the input tokens. The runtime rejects checkpoints that
do not declare the Fetch action contract **and** active state conditioning.
The corrected 2,000-step run uses 785 frames across 20 successful manipulation
segments, including B11's actual handoff states. C/D evaluation is pending.

```bash
# In the openpi environment, after collecting successful teacher runs:
python scripts/fetch_openpi.py convert --repo-id bvi/fetch-seed1-state-v3 --runs RUN_DIRS
python scripts/fetch_openpi.py norm --repo-id bvi/fetch-seed1-state-v3 --work WORK_DIR
python scripts/fetch_openpi.py train --repo-id bvi/fetch-seed1-state-v3 --work WORK_DIR --steps 2000 --batch 4
python scripts/fetch_openpi.py serve --repo-id bvi/fetch-seed1-state-v3 --work WORK_DIR --checkpoint CHECKPOINT_DIR
```

The current data intentionally overlaps the seed-1 diagnostic. A successful
overfit demo would demonstrate closed-loop control, not held-out generalization.
Benchmark evaluation requires a separately declared training/evaluation split.

## Result fields

`first_object_chain_success` requires four consecutive successful
Navigate/Pick/Navigate/Place invocations. `task_success` retains the environment's
full-task result. `benchmark_result` stays false for this diagnostic runner.
Video, checkpoint identity, all model requests, actual actions and feedback must
agree before a C/D success claim is made.
