# Bidirectional VLA Interface

An evaluation-only MS-HAB implementation for comparing fixed skill dispatch,
VLA-as-Tools dispatch, and standardized teleport navigation with the same
official per-object SAC manipulation policies.

## Current 16-plan diagnostic

The canonical [result table](docs/results/tidyhouse-16/README.md) and
[machine-readable evidence](docs/results/tidyhouse-16/summary.json) are
generated from preserved panel records by `bvi-eval summarize`; derived values
are not duplicated by hand in this README.

All three settings completed their 16 episodes and all three had zero complete
five-object task successes. This is a matched 16-plan diagnostic, not a
reproduction or estimate of the published 1,000-rollout benchmark. Teleport
changes the navigation handoff distribution, so its larger partial-object count
does not by itself show that a navigation policy is better.

Training and fine-tuning are outside the active scope. Next week's update count
is fixed at zero, every profile records `training_updates = 0`, and the public
command has no training entry point.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval]"
```

MS-HAB, ManiSkill, assets, and the official PPO/SAC checkpoints are external
research dependencies and must be installed separately. Provider SDKs and the
SSH bridge are optional:

```bash
pip install -e ".[openai,bridge]"
```

## Run

Every `run` command is a non-executing preview unless `--execute` is supplied.
The preview validates the 16-plan roster and prints the frozen profile without
creating the output directory.

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

Real execution additionally requires Linux, `--mshab-root`, and `--execute`.
The GPT setting also requires an approved `--authorization-id` and a separately
started `bvi-eval bridge`; credentials stay in the local environment or ignored
credential files and are never accepted as command-line values.

## Repository layout

- `src/bvi/`: protocol, serial runtime, coordinator, bridge, MS-HAB adapter,
  goal tools, teleport skill, frozen evaluation profiles, and the CLI.
- `tests/`: offline contract, accounting, security, and result-validation tests.
- `docs/results/tidyhouse-16/`: sanitized public evidence for the active table.
- `docs/media/` and older reports: historical evidence, including explicitly
  labelled failures and diagnostics; they are not active executable entrypoints.

See [the documentation index](docs/README.md) for evidence boundaries and
archived research reports.
