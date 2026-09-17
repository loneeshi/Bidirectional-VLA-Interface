# Verified Fetch wrong-handoff inputs

Eight fixed cases from five archived parent histories passed two independent reconstructions each. This verifies input readiness only: `trained_progress_validated=false`, `training_ready=false`. The frozen V8 native0/5 weekly stop remains in force.

Initial simulator/controller, robot observations and original camera pixels match the archives exactly. Recorded action prefixes reproduce qpos/qvel, extra observations, native info and rewards within declared 1e-5 physical tolerance (booleans exact); repeated boundary state/observation hashes match exactly. Workspace224 and wrist128 RGB are actual rendered inputs, with raw state30 relative to the original episode XY.

Original runner horizon200 retained. The six-origin collection straddled a later time-limit code revision; historical process source was not saved. All selected boundaries precede both80/200limits, and exact replay evidence is retained; this does not prove historical full-horizon code identity.

| Role | Case | Boundary TCP-object distance | Held | Actual input NPZ |
|---|---|---:|---|---|
| validation | validation-seed3020-grasp-step17 | 0.01279m | False | [validation-seed3020-grasp-step17.npz](fetch-current-handoffs-2026-09-17-run01--validation-seed3020/validation-seed3020-grasp-step17.npz) |
| validation | validation-seed3020-move-step20 | 0.01816m | True | [validation-seed3020-move-step20.npz](fetch-current-handoffs-2026-09-17-run01--validation-seed3020/validation-seed3020-move-step20.npz) |
| validation | validation-seed3021-grasp-step31 | 0.42219m | False | [validation-seed3021-grasp-step31.npz](fetch-current-handoffs-2026-09-17-run01--validation-seed3021/validation-seed3021-grasp-step31.npz) |
| validation | validation-seed3021-move-step34 | 0.34312m | False | [validation-seed3021-move-step34.npz](fetch-current-handoffs-2026-09-17-run01--validation-seed3021/validation-seed3021-move-step34.npz) |
| locked_diagnostic | locked_diagnostic-seed2025-grasp-step26 | 0.15175m | False | [locked_diagnostic-seed2025-grasp-step26.npz](fetch-current-progress-eval-2026-09-17-run01--seed2025/locked_diagnostic-seed2025-grasp-step26.npz) |
| locked_diagnostic | locked_diagnostic-seed2030-grasp-step18 | 0.30565m | False | [locked_diagnostic-seed2030-grasp-step18.npz](fetch-current-progress-eval-2026-09-17-run01--seed2030/locked_diagnostic-seed2030-grasp-step18.npz) |
| locked_diagnostic | locked_diagnostic-seed2025-grasp-step16 | 0.24119m | False | [locked_diagnostic-seed2025-grasp-step16.npz](fetch-tapt-timing-2026-09-17-run01--legacy-pre-action/locked_diagnostic-seed2025-grasp-step16.npz) |
| locked_diagnostic | locked_diagnostic-seed2025-move-step18 | 0.23054m | False | [locked_diagnostic-seed2025-move-step18.npz](fetch-tapt-timing-2026-09-17-run01--legacy-pre-action/locked_diagnostic-seed2025-move-step18.npz) |

Parents3020/3021 are fixed validation parents; 2025/2030 and the original24.1cm history are locked diagnostics, excluded from training/checkpoint selection. Normal and wrong examples from different parents are not claimed to be identical-state counterfactuals; model-arm comparisons must use the identical saved input.

No failed endpoint is labeled complete. Invocation-local first-frame progress0 is a temporal label, not a physical-success reward. Future learned-feedback acceptance must measure false completion, stagnation/regression events and actual recovery separately.

[Machine-readable results](result.json) · [Artifact hashes](archive-manifest.json). No policy calls, API or training. GPU1 released after completion.
