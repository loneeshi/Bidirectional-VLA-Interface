# C2: feedback-guided manipulation recovery

This isolated research implementation does not replace the published three-setting
`bvi-eval` baseline at the repository root. Run it in a separate Python process:
both implementations intentionally use the `bvi` package name.

See [experiment design](docs/sac-capability-experiment-design.md) for the three
feedback variants, task selection, evidence boundaries and budgets. C2 is running
on three development plans per variant; no final C2 success-rate claim is made.

## Loop

`run_c2_chunks.py` → `run_c2_feedback.py` → `run_coordinator.py` →
`bvi.feedback.c2_context` → `bvi.coordinator` → `bvi.runtime` → PPO/SAC or
`bvi.pose_goal` / `bvi.pose_recovery`.

Each manipulation recovery round permits two GPT decisions: request a bounded
base pose, then inspect arrival evidence and decide whether to invoke SAC.
Failed arrival never grants a SAC retry. At most three recovery rounds are
allowed per manipulation target; ordinary navigation has no additional retries.
The basic/trace/experience variants differ in supplied feedback, not executors.

## Offline checks

From this directory, in a Python environment with pytest, numpy and Pillow:

```sh
python -m pytest -q
PYTHONPATH=src:scripts python scripts/run_c2_feedback.py --help
```

## Execution prerequisites

On the simulator machine set `PYTHONPATH=src:scripts`, `BVI_GPU` to the explicitly
approved GPU UUID, `BVI_MSHAB_ROOT` to the compatible official MS-HAB source root,
and `MS_ASSET_DIR` to the installed assets. Supply checkpoints, frozen three-plan
manifest, excluded-parent case bank, verified controller smoke evidence, shared
bridge spool and output directory using the required runner flags. The runner
checks source hashes against smoke evidence: do not bypass it after source edits.

The local `serve_vlm_bridge.py` keeps credentials local and requires explicit
request/cost/token caps. Claims persist across attempts; network retries consume
budget. Use `run_c2_chunks.py` with the same arguments as `run_c2_feedback.py` to
run at most five chunks of up to two episodes. Infrastructure faults halt the
chain; resume preserves completed rows and immutable attempt directories.

This export includes the transitive Python dependencies of the C2 entry points.
Some dependencies retain historical optional policy branches, which are not
enabled by this experiment. Private datasets, credentials, host identity, GPU
identity, authorization IDs, raw logs and model assets are intentionally absent.
Public deployment paths/GPU selection are environment variables instead of the
private launch defaults; these packaging changes were not deployed to the active
experiment. Existing strict benchmark results remain separate from this relaxed
failure-continuation development protocol.
