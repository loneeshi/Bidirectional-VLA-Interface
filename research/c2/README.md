# C2: feedback-guided manipulation recovery

This isolated research implementation does not replace the published three-setting
`bvi-eval` baseline at the repository root. Run it in a separate Python process:
both implementations intentionally use the `bvi` package name.

See the [current pose-capability design](docs/sac-capability-experiment-design.md)
and [A1–A3 Pick result log](../../docs/log/2026-09-23-c2-sac-pose-capability.md).
The [failure-feedback selection design](docs/failure-feedback-selection-experiment-design.md)
and [exploratory B0/B1 result log](../../docs/log/2026-09-23-c2-failure-feedback-selection.md)
cover the completed small, offline information ablation.
The [Pick start supervision design](docs/pick-start-supervision-recovery-experiment-design.md)
was frozen before the later start-time GPT batch. The exploratory batch is
[archived as diagnostics](diagnostics/2026-09-23-pick-start/README.md) because
the candidate geometry supplied to GPT was insufficient for a grounded pose
choice. Its raw trajectories and finance records are retained; the three batch
notes are not in the formal experiment-log index.
The [single-seed geometry-guided online Pick design](docs/seed9-geometry-online-pick-experiment-design.md)
and [seed8 post-SAC feedback design](docs/seed8-online-pick-feedback-experiment-design.md)
were run as separate cases. Their results are
[archived as exploratory diagnostics](diagnostics/2026-09-24-pick-feedback/README.md):
seed9 included a real movement but a deterministic movement control also
succeeded; seed8 GPT twice chose to continue and never moved. Neither case
establishes added benefit from GPT. The
[next input-interface proposal](docs/pick-gpt-input-contract-proposal.md)
audits the supplied geometry and images without adding an experiment.
The [frozen force-event design](docs/pick-force-event-online-recovery-experiment-design.md)
was then run as single-seed experiment #014. Its
[diagnostic result](diagnostics/2026-09-24-pick-feedback/2026-09-24-seed8-force-event-recovery.md)
shows real GPT calls and error feedback, but the selected turn failed dynamic
arrival validation; it does not establish same-episode Pick recovery.
The [#015 contact and rescue-feasibility design](docs/pick-contact-and-rescue-feasibility-experiment-design.md)
was run without GPT. Its [single-experiment diagnostic result](diagnostics/2026-09-24-pick-feedback/2026-09-24-pick-contact-rescue-feasibility.md)
identifies seed8 base-link contact with a sofa and tests 30 fixed base-adjustment
candidates across seed8 and seed4. None produced a legal, arrived, strict Pick
rescue; seed4's force-trigger boundary did not occur. The raw trajectories and
process receipts are retained, and no GPT selection benefit is claimed.
The [seed4 live Oracle feasibility probe](docs/seed4-oracle-feasibility-probe-design.md)
then tested assistant decisions at real Pick checkpoints while SAC controlled
the arm. Its [diagnostic record](diagnostics/2026-09-24-pick-feedback/seed4-oracle-live-diagnostic.md)
shows two failed strict Pick trajectories: base assistance collided with the
sofa, and the 39-action handoff left too little time to detour and grasp. It is
not a formal GPT comparison or a recovery witness.
The [seed4 fixed-base SAC design](docs/seed4-fixed-base-sac-pick-experiment-design.md)
then tested the user's simpler control: zero forward and turn commands throughout
one native Pick. The [single-case diagnostic](diagnostics/2026-09-25-fixed-base-pick/seed4-fixed-base-sac-pick.md)
shows a valid hold with no new collision, but no target contact or grasp before
the 39-action timeout. This isolates the obstacle effect from the SAC reach
failure at that handoff; it does not prove the object is kinematically unreachable.
The [same-snapshot extended-horizon design](docs/seed4-fixed-base-extended-horizon-experiment-design.md)
then changed only the remaining Pick counter from 39 to 200. Its
[single-case diagnostic](diagnostics/2026-09-25-fixed-base-extended/seed4-fixed-base-extended-horizon.md)
kept the first 39 SAC actions and physical feedback identical, then still
failed without bowl contact or grasp. The edited clock makes this a time-limit
counterfactual, not a native benchmark result.
The earlier feedback-recovery implementation below is retained as historical
development code; this experiment does not report a full-task success rate.

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
