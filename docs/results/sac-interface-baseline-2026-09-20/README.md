# SAC-interface TidyHouse baseline

Current update (2026-09-20 01:14 EDT): batch ended, planned20, audited terminal16 / infrastructure4; API83, remaining717. New20 blocked by the request-allocation gate. GPU1 released. See [close, fixes and restart handoff](../../sac-followup-2026-09-20.md) and the separate versioned audit. The status and contract files in this directory remain historical pre-launch snapshots; they are not current live state. The original batch results are preserved.

Status at 2026-09-20 02:53 UTC: **implementation ready; rollout not started**.
The laboratory SSH endpoint remained unreachable, so the prior AC-DiT process
and GPU1 state are unknown and no simulator action or paid model request was
started. This directory must not be read as a score.

## Frozen scope

- Population: all 1000 MS-HAB TidyHouse validation rollouts.
- Monday-report batch: first 20 bound seeds, retained as a partial 20/1000 result.
- Stack: GPT-5.6 Luna organizer, LightNav-0 navigation, official per-object SAC
  Pick/Place.
- Training/fine-tuning: none.
- Per episode: 7000 physical actions, 40 organizer calls, 900 seconds; each skill
  invocation is bounded to 180 seconds and the organizer slice is 40 actions.
- Batch ceiling: 800 model requests and USD 2. GPU1 only; GPU0 remains untouched.

The 20-row batch is not an official full-benchmark score. Reports must show
coverage as 20/1000 and keep infrastructure failures, interrupted attempts, and
not-run rows.

## Resumption contract

`bind_sac_interface_manifest.py` creates the fixed 1000-row manifest and binds
the next 20 seeds through real environment resets. `run_sac_interface_baseline.py`
selects or resumes one persistent batch. It writes the manifest after every
attempt, uses a new `attempt-NNN` directory after interruption, skips terminal
rows, and shares one durable bridge spool. A restart therefore continues the
unfinished batch without deleting or replacing earlier evidence.

The local source/dirty snapshot taken before implementation is at
`D:/AI/Embodied Intelligence/VLA as Tools/runs/sac-interface-source-freeze-2026-09-20-preimplementation/`.

## Current blocker

Before execution, reconnect to the lab, safely stop and archive
`acdit-apple-bounded-20260919-run01` if it is still running, verify GPU1 is free,
then bind the first 20 real plans. Do not infer stop completion from an SSH
failure. The API model identifier `gpt-5.6-luna` was verified by a read-only
model retrieval; no generation request was made.
