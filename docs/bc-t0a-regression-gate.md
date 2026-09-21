# Official BC T0a regression gate

This is a lightweight, offline CI check for the three already measured paths:

- A: official wrappers on the official MS-HAB environment;
- B: project-style BC input/action assembly on the official environment;
- C: project-style BC assembly on the AC-DiT environment fork.

It checks the fixed seeds 2024--2028, pinned checkpoint/task/source provenance,
bitwise initial-state and policy-input equality, exact per-step input/action/qpos/
info comparison summaries, zero raw-action delta, equal outcomes and step counts,
and at least one native success from the known-good official BC.  The last item
is deliberately a **nonzero diagnostic**, not a success-rate threshold.  Five
fixed initial conditions do not estimate paper or benchmark performance.

## Offline CI command

From the repository root, this validates the checked-in evidence without a GPU,
simulator, model download, training, or API call:

```powershell
.\.venv\Scripts\python.exe scripts/check_bc_t0a_regression.py --output .runtime/bc-t0a-gate.json
.\.venv\Scripts\python.exe -m pytest tests/test_bc_t0a_regression.py -q
```

The check exits zero only when all invariants pass.  Its JSON explicitly marks
`diagnostic_only=true` and `benchmark_success_rate_estimate=false`.  CI should
archive `.runtime/bc-t0a-gate.json`; it should not publish its `4/5` fingerprint
as an empirical success-rate claim.

## Explicit fresh-rollout commands

Fresh evidence is a separate, GPU-backed diagnostic job.  On the configured lab
server, from a synchronized repository root with GPU1 free, use new output
directories (never overwrite the checked-in evidence):

```bash
python scripts/run_bc_t0a.py \
  --output "$HOME/bvi-research/runs/bc-t0a-ci-NEW-official" \
  --environment official --paths official project --seeds 2024 2025 2026 2027 2028
python scripts/summarize_bc_t0a.py \
  "$HOME/bvi-research/runs/bc-t0a-ci-NEW-official"

python scripts/run_bc_t0a.py \
  --output "$HOME/bvi-research/runs/bc-t0a-ci-NEW-acdit" \
  --environment acdit --paths project --seeds 2024 2025 2026 2027 2028
python scripts/summarize_bc_t0a.py \
  "$HOME/bvi-research/runs/bc-t0a-ci-NEW-acdit" \
  --reference "$HOME/bvi-research/runs/bc-t0a-ci-NEW-official"

python scripts/check_bc_t0a_regression.py \
  --official "$HOME/bvi-research/runs/bc-t0a-ci-NEW-official" \
  --acdit "$HOME/bvi-research/runs/bc-t0a-ci-NEW-acdit" \
  --output "$HOME/bvi-research/runs/bc-t0a-ci-NEW-gate.json"
```

The rollout scripts retain their occupancy, renderer-device, per-episode, and
outer time bounds.  A fresh run is still a laboratory-resource operation and
must follow the project's preflight, finance logging, and final GPU-state checks.
No result from this gate clears RGB, point clouds, VLA-specific normalization,
precision, temporal chunk execution, or the complete coordinator harness.
