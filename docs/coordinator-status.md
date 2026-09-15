# What the six external API requests establish

## Execution organizer implementation (September15)

`--organizer` now enables a VLM execution organizer using the existing paid-call budget and request trace. It adds an explicit `abort_task` choice, caps each physical skill invocation at40steps by default, and yields on missed-grasp/grasp-loss heuristics. New images and the result return to the next VLM decision. Native failure/termination still takes precedence, and no task pointer is modified. A retry invokes the feasible skill again; it does not imply a new repositioning controller.

Two real API decisions on saved C12 images were made after the initial six requests: at the Pick start and after a recorded six-step failed closure. Both selected a40-step Pick request. These were static-image probes with no executed actions, not evidence of successful recovery. The CPU suite has98passing tests, including monitor yield, native outcome precedence and an offline VLM abort path. Live simulation validation is recorded separately below when complete.

The remaining text describes the earlier baseline and why this extension was necessary. Free task-order planning and learned repositioning are still outside the implemented execution organizer.

## Live results

| Episode | Real VLM calls | Physics steps | Outcome |
|---|---:|---:|---|
| Normal PPO/SAC |8|229|Navigate→Pick→carry Navigate→Place completed|
| Synthetic closure fault + PPO/SAC |8|317|Six-step injected closure caused `missed_grasp`; VLM selected retry, Pick completed51steps later, then carry and Place completed|

[Normal video — organizer-normal.mp4](media/organizer-2026-09-15/organizer-normal.mp4) · [Injected-fault recovery video — organizer-injected-fault-recovery.mp4](media/organizer-2026-09-15/organizer-injected-fault-recovery.mp4) · [Normal audit](results/organizer006-normal.json) · [Fault audit](results/organizer006-fault.json) · [Per-call API usage](results/organizer006-api-usage.json).

Both use GPT-5.6 Luna through the local-key SSH bridge, actual current workspace/wrist images, official PPO navigation and SAC manipulation. Normal invocations are capped at40steps; the fault test uses60steps. The fault is explicitly injected once by `InjectedClosureFault`, not a natural model failure. Both passed native first-object progress and grasp-on-every-carry-step checks; video contact sheets were visually inspected. The fault test reached the fourth subtask at the last allowed call, so its original summary says `experiment_call_limit` despite completing that boundary. The raw record is preserved; the runner now prioritizes an achieved boundary before loop exhaustion.

The total known external request count is24: original6 + saved-image2 + live16. The16live requests cost an estimated$0.00696985 from returned usage at [official model pricing](https://developers.openai.com/api/docs/models/gpt-5.6-luna); this is not a reconciled provider bill. The temporary GPU was deleted after verified artifact download. No pi05 training or inference ran in these organizer episodes.

This establishes a first execution-organizer loop and recovery from one controlled fault. It does **not** establish natural-failure recovery rate, a benefit over a deterministic retry rule, free choice of object order, or the complete five-object benchmark. The executed choices were feasible skill invocation/continuation/retry; `abort_task` is implemented and CPU-tested but was not selected in these live episodes. A genuine reposition skill and a task-order planner remain separate extensions.

Reproduction on the pinned Linux simulation environment (local bridge must be running with matching authorization and limits):

```bash
python scripts/run_coordinator.py --organizer --organizer-slice-steps 40 \
  --stop-after-subtasks 4 --seed 1 --policy-type rl_per_obj --workspace-camera \
  --expected-plan-uid tidy_house-sequential-val-90-0 --max-calls 16 \
  --max-env-steps 650 --max-wall-seconds 900 --skill-wall-seconds 60 \
  --checkpoint-root /workspace/bvi/mshab_checkpoints --output /workspace/bvi/runs/organizer006 \
  --transport bridge --bridge-timeout-seconds 120 --provider openai --model gpt-5.6-luna \
  --authorization-id YOUR_APPROVED_SCOPE --max-api-cost-usd .32 \
  --request-cost-ceiling-usd .02 --max-output-tokens 600 --max-input-bytes 200000 \
  --image-detail low --reasoning-effort none
```

Use a new output directory. For the controlled fault test add `--inject-closure-fault` and set slice60, calls8, steps421, API cap.16. These are per-run technical limits, not authorization to incur additional charges. The CPU suite now has99passing tests.

The high-level image-conditioned `VLMCoordinator` exists in `src/bvi/coordinator.py`. The normal runner constructs it and invokes `decide(observation, history)`; `--dry-run` bypasses it and chooses `observation.allowed_calls[0]` using benchmark task-plan information. Dry-run still executes real simulation and the selected low-level policies.

The initial six external API requests exercised image input, structured skill requests, execution feedback and subsequent decisions. Later A/B/C/D skill comparisons and contact-replay diagnostics used the oracle dispatcher and made zero external API requests. Self-hosted LightNav/π inference consumes rented GPU time, not OpenAI API requests. The six-request count is therefore expected from the experiment history.

Crucially, `MSHABAdapter._snapshot` currently exposes one target and one admissible skill from the current native subtask pointer. Even turning the VLM back on would not demonstrate unconstrained decomposition or meaningful choice between several skills. The generic protocol supports multiple allowed calls; the current benchmark adapter does not expose that decision space.

Current claim: constrained VLM integration smoke test plus separately evaluated low-level skills. Not established: planning benefit, VLM-driven failure recovery, or successful VLM+LightNav+π task completion.

The current runner asks the coordinator between skill invocations, not every simulation step. A missed grasp at step23 does not immediately trigger another VLM call: the active Pick skill continues until success, native failure, or timeout. A recovery-capable organizer needs an explicit interruption/failure event and a callable recovery skill, not merely an API budget. The current `benchmark_feedback` reports the native skill outcome; it does not implement an early missed-grasp detector.

Next organizer evaluation should explicitly expose feasible alternatives such as inspect, retry grasp, and reposition with measured preconditions; deliver compact execution/failure feedback; compare VLM decisions with a fixed dispatcher under the same skills and episode budgets. Additional recovery skills and benchmark-compatible execution semantics must be implemented and tested first. Do not simply relax the allowed-call validator or change the native task pointer to manufacture a choice or success.
