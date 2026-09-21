# S1-IA D1/D2 diagnostic controls — 2026-09-18 run02

## Decision

- **D1: knife-edge threshold / policy precision insufficient.** All-zero and initial-aperture hold accumulated zero force in all 10 control trials. Official SAC succeeded on 4/5 starts, but seed 2024 itself crossed the 5000 cumulative-force limit (peak 6994.58). IA crossed the same limit on 5/5 starts. This excludes an idle-wrapper failure and shows the native threshold can also terminate the positive reference; the reach result is evidence of insufficient IA precision under this native gate, not evidence that every non-SAC policy is mechanically doomed.
- **D2: constant close is not distinguishable from IA grasp.** On the four evaluable starts, constant-close achieved the same three successes in exactly three steps (3/4). The prior IA grasp 3/4 therefore carries zero policy-specific information and must be removed from the capability table. The median start distance was 4.45 cm; seed 2024 was already terminal and excluded.
- **B3 native gate: 0/10 against 3/10.** Existing evidence was summarized, not rerun. S1-IA does not enter the mainline.

The requested B2 +3/+5 cm states were not executed because adding those offsets placed at least one target outside the preregistered 5–8 cm band for every evaluable start. They remain `not_evaluable`; no start was silently moved and no seed was replaced.

## Evidence and correction

`summary.json` is the immutable first machine summary. [adjudication.json](adjudication.json) supersedes three derived fields: D1's branch interpretation, IA raw-action maxima, and the scalar B0 median. Raw rollout files are unchanged.

The final archive hash is recorded in the external `archive-receipt.json` sidecar, avoiding a self-referential hash inside the archive. Earlier archive versions remain preserved and are not the final evidence package.

No training, API calls, threshold changes, new seeds, or rented compute were used. GPU1 ended at 15 MiB / 0%; laboratory charge remains unknown.
