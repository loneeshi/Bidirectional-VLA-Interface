# C0/C1/C2 continuation pilot

This is a six-episode development smoke: two previously analyzed validation
plans times C0/C1/C2. It is not the final 16-plan panel and is not unseen test.

- C0: continuous execution, no revisit after switching away from an unfinished
  tool/target attempt.
- C1: same interface and budgets, at most three revisits per abandoned target.
- C2: C1 plus frozen `spawn-prior-v3.json` derived only from 24 train-spawn
  SAC probes. The prior exposes policy-input relative XY distance ranges,
  sample/success/strict-success counts and uncertainty. It is not an optimum.

All conditions use GPT-5.6 Luna, official PPO navigation and per-object SAC,
40 model calls, 40 actions per call, 7000 episode actions and 900 seconds.
The simulator continues after native force/subtask failure; those events latch
`official_episode_valid=false`. Infrastructure/numerical failure still stops.
Final completed objects are rescored from the five Place predicates rather than
the native sequential pointer. Adjacent slices are continuations, not retries.

The pilot is launched serially and stops on the first infrastructure failure.
Every attempt and the fixed 6-row denominator remain in `panel-status.json`.
API ceiling: 240 requests, 2048 output tokens/request, conservative reservation
USD 0.30. Actual provider and laboratory charges remain pending reconciliation.
No training, GPU0 or RunPod.
