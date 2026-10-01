# C2: feedback-guided manipulation recovery

This isolated research implementation does not replace the published three-setting
`bvi-eval` baseline at the repository root. Run it in a separate Python process:
both implementations intentionally use the `bvi` package name.

The real-handoff Pick rescue scan（附件未发布）
has completed its two-position preflight（附件未发布）
and 14-failure main scan（附件未发布）.
Five of fourteen reproducible failures were confirmed rescued by real base motion
and same-episode official strict Pick; the best fixed action covered four of five.
The 45/60 finite-library figure uses retrospective per-position action selection,
and three unrescued positions remain interface-unresolved.

See the current pose-capability design（附件未发布）
and [A1–A3 Pick result log](../../docs/log/2026-09-23-c2-sac-pose-capability.md).
The failure-feedback selection design（附件未发布）
and [exploratory B0/B1 result log](../../docs/log/2026-09-23-c2-failure-feedback-selection.md)
cover the completed small, offline information ablation.
The Pick start supervision design（附件未发布）
was frozen before the later start-time GPT batch. The exploratory batch is
archived as diagnostics（附件未发布） because
the candidate geometry supplied to GPT was insufficient for a grounded pose
choice. Its raw trajectories and finance records are retained; the three batch
notes are not in the formal experiment-log index.
The single-seed geometry-guided online Pick design（附件未发布）
and seed8 post-SAC feedback design（附件未发布）
were run as separate cases. Their results are
archived as exploratory diagnostics（附件未发布）:
seed9 included a real movement but a deterministic movement control also
succeeded; seed8 GPT twice chose to continue and never moved. Neither case
establishes added benefit from GPT. The
next input-interface proposal（附件未发布）
audits the supplied geometry and images without adding an experiment.
The frozen force-event design（附件未发布）
was then run as single-seed experiment #014. Its
diagnostic result（附件未发布）
shows real GPT calls and error feedback, but the selected turn failed dynamic
arrival validation; it does not establish same-episode Pick recovery.
The #015 contact and rescue-feasibility design（附件未发布）
was run without GPT. Its single-experiment diagnostic result（附件未发布）
identifies seed8 base-link contact with a sofa and tests 30 fixed base-adjustment
candidates across seed8 and seed4. None produced a legal, arrived, strict Pick
rescue; seed4's force-trigger boundary did not occur. The raw trajectories and
process receipts are retained, and no GPT selection benefit is claimed.
The seed4 live Oracle feasibility probe（附件未发布）
then tested assistant decisions at real Pick checkpoints while SAC controlled
the arm. Its diagnostic record（附件未发布）
shows two failed strict Pick trajectories: base assistance collided with the
sofa, and the 39-action handoff left too little time to detour and grasp. It is
not a formal GPT comparison or a recovery witness.
The seed4 fixed-base SAC design（附件未发布）
then tested the user's simpler control: zero forward and turn commands throughout
one native Pick. The single-case diagnostic（附件未发布）
shows a valid hold with no new collision, but no target contact or grasp before
the 39-action timeout. This isolates the obstacle effect from the SAC reach
failure at that handoff; it does not prove the object is kinematically unreachable.
The same-snapshot extended-horizon design（附件未发布）
then changed only the remaining Pick counter from 39 to 200. Its
single-case diagnostic（附件未发布）
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
