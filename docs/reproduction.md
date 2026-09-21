# Reproducing the public evaluation

## Requirements

- Linux with NVIDIA CUDA and Vulkan support for simulator execution;
- Python 3.10 or newer;
- official MS-HAB commit `e9ff3d23496d38e4431c8d913e147ffa007f7f72`;
- compatible ManiSkill/ReplicaCAD assets and official TidyHouse PPO/SAC checkpoints;
- this package installed with `pip install -e ".[eval]"`.

Provider and SSH dependencies are required only for the GPT setting:

```bash
pip install -e ".[openai,bridge]"
```

## Preview first

The default command performs configuration, plan-roster, path and pairing
validation. It does not start the simulator, GPU work or an API call.

```bash
bvi-eval run --setting fixed \
  --source-manifest /path/to/manifest.json \
  --checkpoint-root /path/to/checkpoints \
  --output /path/to/fixed-panel
```

Review the printed profile and 16 bindings. Real execution additionally needs
`--mshab-root` and `--execute`.

## Execute in order

Run fixed first because the other settings require its plan and initial-state
bindings. Use `--max-new` to bound how many new episode attempts a single
invocation may start.

```bash
bvi-eval run --setting fixed --execute \
  --source-manifest /path/to/manifest.json \
  --checkpoint-root /path/to/checkpoints \
  --mshab-root /path/to/mshab \
  --asset-dir /path/to/assets \
  --output /path/to/fixed-panel \
  --max-new 2

bvi-eval run --setting gpt --execute \
  --source-manifest /path/to/manifest.json \
  --reference-panel /path/to/fixed-panel \
  --checkpoint-root /path/to/checkpoints \
  --mshab-root /path/to/mshab \
  --asset-dir /path/to/assets \
  --bridge-dir /path/to/bridge \
  --authorization-id APPROVED_SCOPE \
  --output /path/to/gpt-panel \
  --max-new 2

bvi-eval run --setting teleport --execute \
  --source-manifest /path/to/manifest.json \
  --reference-panel /path/to/fixed-panel \
  --checkpoint-root /path/to/checkpoints \
  --mshab-root /path/to/mshab \
  --asset-dir /path/to/assets \
  --output /path/to/teleport-panel \
  --max-new 2
```

An existing panel resumes completed rows and appends a new attempt. Use
`--retry-infrastructure` only after confirming the previous failure was
infrastructure-related. The runner lock prevents duplicate writers.

## Generate the public table

```bash
bvi-eval summarize \
  --paired-panel /path/to/paired-fixed-gpt-panel \
  --teleport-panel /path/to/teleport-panel \
  --output-dir /path/to/public-summary
```

The command refuses incomplete panels, changed plan rosters, invalid initial
state pairing, nonzero API use in fixed/teleport, or a teleport total other than
21 completed objects for this frozen evidence set.
