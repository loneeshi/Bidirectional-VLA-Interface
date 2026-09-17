# Observation-progress pairing preparation

This batch contains CPU-only metadata preparation. No GPU, SSH, model API,
simulation, feature extraction, or training was performed. No new demo was recorded.

`pairing.json` binds 375 existing cache rows:336 training and39 development
validation. Its original report, split-manifest and reference-validation JSON match
the archived Linux LF source bytes. Cache and checkpoint hashes are checked against the previous
archive-verification evidence; tensor contents are not loaded by this preparation.

The fixed action policy includes selected step20's family LoRA banks, native
backbone/mobility expert and preprocessing. The independent head remains
unimplemented. Validation was previously used for checkpoint selection and does
not become unseen test evidence by renaming this experiment.

```powershell
.venv/Scripts/python.exe scripts/prepare_fetch_observation_progress.py --source docs/results/fetch-current-calibration-2026-09-17-run01 --output <new-output-path.json>
.venv/Scripts/python.exe -m pytest tests/test_observation_progress_preflight.py tests/test_fetch_current_labels.py -q
```

Output files refuse overwrite. The metadata output is deterministic; source JSON
byte hashes also record platform checkout line endings. The archived-source
comparison normalizes CRLF to LF explicitly, without changing JSON values.

Next numerical checks and architectural limits are documented in
[the follow-up design](../../fetch-observation-head-followup.md).
