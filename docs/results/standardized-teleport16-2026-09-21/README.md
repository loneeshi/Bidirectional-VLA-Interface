# Standardized Teleport + official per-object SAC (16 plans)

This directory is the local evidence copy of the completed standardized
Teleport panel. It is distinct from `official-teleport16-2026-09-21`, which
used the earlier paper-release evaluator and reported 12/80 objects.

## Result

- Panel status: `finished`
- Planned/completed episodes: 16/16
- Infrastructure failures: 0
- Full-task successes: 0/16
- Completed objects: 21/80 (26.25%)
- Mean completed objects: 1.3125 per episode
- Per-seed completed objects for seeds
  `0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,19`:
  `2,2,1,3,0,0,4,3,0,0,0,1,0,1,1,3`

The panel used the current paired evaluator/runtime (`e9ff3d23496d38e4431c8d913e147ffa007f7f72`),
the paper Teleport implementation from commit
`4729821db3fc94a2470cfd625e6f8ab439f01478`, and the same
`fetch_nav + fetch_workspace` RGB-D setting as the Fixed and GPT panels.

## Evidence

- Authoritative local panel: `panel/panel-status.json`
- Per-episode directories: `panel/seed-*/attempt-*`
- Videos: 16 MP4 files, one scored video per seed
- Episode summaries: 17 files (the retained seed-0 failed startup attempt plus
  all 16 scored attempts)
- Event logs: 17 files, retaining the same additive-attempt history
- Frozen sources: `source-v1.zip`, `source-v2.zip`
- Raw remote export: `raw-export.tar.gz`
- Remote source directory:
  `/home/pshuai/bvi-research/runs/standardized-teleport16-20260921`

## SHA256

- `raw-export.tar.gz`:
  `f9876f69eac6fa3c8a351e1d0b29e22b967e5e856fa6b7850f31cab2187b7a01`
- `panel/panel-status.json`:
  `76367f66a7cae6b25aa46ece7fb37038cf2412d037e005818b44dacdf6c90576`
- `source-v1.zip`:
  `2deea7ae9c94744d7cac0b186860d5c90c2261d6b8cf4de1700546cb71ff406c`
- `source-v2.zip`:
  `ca32951619c62b797dbf74cb858ec096d7bfd5713a2d48ba5b8708f8eab1acbb`
- `supervise-v2.sh`:
  `ec4f289c2a4cbdcf1f8507e60d1554b843b59b9b86ccae4a6b9078805ae50061`

