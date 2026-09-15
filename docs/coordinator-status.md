# What the six external API requests establish

The high-level image-conditioned `VLMCoordinator` exists in `src/bvi/coordinator.py`. The normal runner constructs it and invokes `decide(observation, history)`; `--dry-run` bypasses it and chooses `observation.allowed_calls[0]` using benchmark task-plan information. Dry-run still executes real simulation and the selected low-level policies.

The initial six external API requests exercised image input, structured skill requests, execution feedback and subsequent decisions. Later A/B/C/D skill comparisons and contact-replay diagnostics used the oracle dispatcher and made zero external API requests. Self-hosted LightNav/π inference consumes rented GPU time, not OpenAI API requests. The six-request count is therefore expected from the experiment history.

Crucially, `MSHABAdapter._snapshot` currently exposes one target and one admissible skill from the current native subtask pointer. Even turning the VLM back on would not demonstrate unconstrained decomposition or meaningful choice between several skills. The generic protocol supports multiple allowed calls; the current benchmark adapter does not expose that decision space.

Current claim: constrained VLM integration smoke test plus separately evaluated low-level skills. Not established: planning benefit, VLM-driven failure recovery, or successful VLM+LightNav+π task completion.

The current runner asks the coordinator between skill invocations, not every simulation step. A missed grasp at step23 does not immediately trigger another VLM call: the active Pick skill continues until success, native failure, or timeout. A recovery-capable organizer needs an explicit interruption/failure event and a callable recovery skill, not merely an API budget. The current `benchmark_feedback` reports the native skill outcome; it does not implement an early missed-grasp detector.

Next organizer evaluation should explicitly expose feasible alternatives such as inspect, retry grasp, and reposition with measured preconditions; deliver compact execution/failure feedback; compare VLM decisions with a fixed dispatcher under the same skills and episode budgets. Additional recovery skills and benchmark-compatible execution semantics must be implemented and tested first. Do not simply relax the allowed-call validator or change the native task pointer to manufacture a choice or success.
