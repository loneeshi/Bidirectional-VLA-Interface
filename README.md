# Bidirectional VLA Interface

Evaluation-only MS-HAB code for comparing three frozen TidyHouse settings:

| Setting | Episodes | Full-task SR | Completed objects | Mean / episode |
|---|---:|---:|---:|---:|
| Fixed PPO + SAC | 16/16 | 0/16 | 14/80 (17.5%) | 0.875 |
| GPT + PPO + SAC | 16/16 | 0/16 | 13/80 (16.25%) | 0.813 |
| Teleport + SAC | 16/16 | 0/16 | 21/80 (26.25%) | 1.313 |

The complete human-readable report is in
[`docs/log/2026-09-21-tidyhouse-three-settings.md`](docs/log/2026-09-21-tidyhouse-three-settings.md).
The machine-readable episode summary is
[`docs/results/tidyhouse-16/summary.json`](docs/results/tidyhouse-16/summary.json).

All three settings had zero complete five-object task successes. This is a
matched 16-plan diagnostic, not a reproduction of the published 1,000-rollout
benchmark. Teleport changes the navigation handoff distribution, so its larger
partial-object count is not evidence that a navigation policy is better.

Training and fine-tuning are outside the active scope. Every profile records
`training_updates = 0`, and the public CLI contains no training entry point.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval]"
```

MS-HAB, ManiSkill, assets, and official PPO/SAC checkpoints are external
dependencies. Provider and SSH dependencies are optional:

```bash
pip install -e ".[openai,bridge]"
```

## Run

`bvi-eval run` is a non-executing validation preview unless `--execute` is
supplied.

```bash
bvi-eval run --setting fixed \
  --source-manifest /path/to/manifest.json \
  --checkpoint-root /path/to/checkpoints \
  --output /path/to/fixed-panel

bvi-eval run --setting gpt \
  --source-manifest /path/to/manifest.json \
  --reference-panel /path/to/fixed-panel \
  --checkpoint-root /path/to/checkpoints \
  --bridge-dir /path/to/bridge \
  --output /path/to/gpt-panel

bvi-eval run --setting teleport \
  --source-manifest /path/to/manifest.json \
  --reference-panel /path/to/fixed-panel \
  --checkpoint-root /path/to/checkpoints \
  --output /path/to/teleport-panel
```

Real execution requires Linux, `--mshab-root`, and `--execute`. GPT execution
also requires a separately started `bvi-eval bridge` and an approved
`--authorization-id`. Credentials are accepted only from the local environment
or an ignored credential file, never from command-line values.

## Documentation

[`docs/README.md`](docs/README.md) is the complete documentation index. The
active tree intentionally excludes old plans, training notes, raw run dumps,
videos, and superseded diagnostics; they remain recoverable from Git history.
