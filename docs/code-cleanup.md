# Public evaluation-only cleanup

The active package has one public entry point, `bvi-eval`, and three immutable
evaluation profiles: `fixed`, `gpt`, and `teleport`. The `scripts/` directory no
longer contains experiment implementations.

Removed from the active tree:

- model adaptation and dataset-generation pipelines;
- superseded policy families and external benchmark integrations;
- stage-specific capability gates, collection utilities, and calibration jobs;
- one-off server probes, diagnosis scripts, and their dedicated tests;
- tracked Python caches, package metadata, and test/lint caches.

Retained in the active package:

- protocol and serial runtime;
- constrained coordinator, provider bridge, and goal tools;
- MS-HAB adapter and standardized teleport skill;
- frozen evaluation profiles, execution harness, and result aggregation.

Earlier research documents and recordings are archival evidence, not current
execution instructions. Their failure and diagnostic labels remain intact, and
media continue to be indexed by `docs/media/manifest.json`.
