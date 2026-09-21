# Documentation index

## Current evaluation

- [TidyHouse 16-plan result table](results/tidyhouse-16/README.md)
- [Machine-readable sanitized summary](results/tidyhouse-16/summary.json)

The active repository exposes only three frozen evaluation settings: Fixed
PPO+SAC, GPT+PPO+SAC through the VLA-as-Tools protocol, and standardized
Teleport+SAC. None performs training or fine-tuning.

## Evidence boundary

The current comparison contains 16 matched TidyHouse plans and zero full-task
successes in every setting. Partial completed-object counts are diagnostic. They
must not be reported as a 1,000-rollout benchmark reproduction, and teleport is
not an initial-state-equivalent navigation-policy ablation.

The public summary contains seeds, plan UIDs, outcomes, counts, and source-file
hashes. Machine paths, process IDs, GPU identifiers, SSH details, bridge paths,
and authorization references are intentionally excluded.

## Archive

The following documents and all older result directories describe historical
development experiments, failed capability gates, diagnostics, or superseded
work. They remain available as evidence but are not current plans, public
commands, or supported code paths:

- **Archive:** [architecture notes](architecture.md)
- **Archive:** [historical evaluation notes](evaluation.md)
- **Archive:** [historical environment setup](reproduction.md)
- **Archive:** [historical bridge procedure](bridge.md)
- **Archive:** [media index](media/README.md) and its
  [manifest](media/manifest.json)

Failure and synthetic-fault labels in `docs/media/` remain authoritative.
