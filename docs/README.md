> **当前交付更新（2026-09-19 19:06 UTC）：** AC-DiT底盘187更新、全身尚未开始；本轮为训练中期交付。见[训练/评测核查](acdit-apple-evaluation-2026-09-20.md)、[统一VLM接入方案](vlm-integration-methodology-2026-09-20.md)、[证据与复核入口](results/acdit-apple-delivery-2026-09-20/README.md)。原生能力验证和完整权重/数据备份未完成。下方旧S1失败交付保留，不能代表当前无活动训练。

# Docs index

Experiment records for the Bidirectional-VLA-Interface project. Files are named by the batch that produced them and are never rewritten after the fact: a superseded document keeps its original findings and gains a pointer to what replaced it. Start here rather than reading them in directory order.

**Current scope:** [Delivery status and evidence](delivery-2026-09-20.md) · [Initial source-freeze receipt](results/two-day-delivery-2026-09-20/source-freeze.json). Superseded planning and authorization documents have been removed; the results and diagnostics they produced are kept in full. The comparison rules now in force are [baseline protocol revision 2](baseline-protocol-2026-09-20-v2.md).

**Final native gate:** Original S1-IA best/855 baseline **0/5**; frozen dev8-selected candidate best/250 **0/5**. All 10 oracle-assisted development episodes ended at the native cumulative-force limit without grasping. **G1 failed**, so no success rerun, LightNav/GPT integration, or Place evaluation followed. The candidate completed 500 updates and passed independent full CPU restore/frozen-parameter verification, but offline RMSE improvement did not establish native Pick capability. Both panels' full raw evidence, S1/S2 checkpoints, and the selected candidate checkpoint have verified complete second copies. The two earlier infrastructure-only baseline launches remain separate from the evaluated denominators. Historical fixed-sentence 0/10 belongs to ordinary S1 best6000, not IA best855; see the [identity correction](results/two-day-delivery-2026-09-20/historical-identity-audit/receipt.json).

[Current experiment evaluation](pi05-candidate-evaluation-2026-09-20.md) · [Fresh-process reproduction](two-day-reproduction-2026-09-20.md) · [Result manifest](results/two-day-delivery-2026-09-20/manifest.json)

## Experiment log

Human-readable summaries of each completed panel, alongside the machine-readable
evidence in `results/` and `runs/`. See [log/README.md](log/README.md).

| Log | Covers |
|---|---|
| [TidyHouse 16-plan, three settings](log/2026-09-21-tidyhouse-16plan-panel.md) | Fixed PPO+SAC (14/80 objects), GPT+PPO+SAC (13/80) and standardized Teleport+SAC (21/80) on the same 16 TidyHouse validation plans: 0/16 full-task SR in all three arms |

## Start here

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | Interface and runtime design: the MS-HAB baseline contract and its extension points |
| [evaluation.md](evaluation.md) | Evaluation protocol and the reporting checklist every result must satisfy |
| [reproduction.md](reproduction.md) | Setup, pinned version manifest, external assets, headless rendering, execution sequence |
| [bridge.md](bridge.md) | SSH bridge that keeps API credentials on the local machine during remote simulation |
| [lab-server-deployment.md](lab-server-deployment.md) | Laboratory server (two 48 GiB Quadro RTX 8000); replaces rented Runpod compute |

## Method source and reproduction scope

| Document | What it covers |
|---|---|
| [vlas-as-tools-reproduction.md](vlas-as-tools-reproduction.md) | Priority and acceptance gates against the paper; what counts as reproduced and what does not |
| [vlas-as-tools-code-audit.md](vlas-as-tools-code-audit.md) | Author code availability: the co-first author's OpenPI fork, pinned at `f4eb160` |
| [tapt-method-alignment-and-recovery.md](tapt-method-alignment-and-recovery.md) | Clause-by-clause alignment with the paper, plus the designed (not yet run) paired-disturbance recovery experiment |
| [vla-tools-mshab-fidelity-audit.md](vla-tools-mshab-fidelity-audit.md) | What was actually trained per backend, and candidate manipulation policies for MS-HAB |

## LIBERO: TAPT components (method validated, paper scores not reproduced)

| Document | Outcome |
|---|---|
| [tapt-libero-run01.md](tapt-libero-run01.md) | 2,000 updates, four LoRA banks + shared progress head. Baseline 5/5, GPT+original 3/5, GPT+TAPT 0/5 — six episodes truncated by a UTF-8 validation defect |
| [tapt-libero-run02.md](tapt-libero-run02.md) | 400 updates with language variants; checkpoint selected on validation loss only. No online evaluation |
| [tapt-libero-run03.md](tapt-libero-run03.md) | Corrected online comparison: 5/5, 5/5, 4/5 (one infrastructure failure), 5/5. Learned progress participated in the closed loop; task has no failure headroom |

## MS-HAB: interface and coordinator

| Document | Outcome |
|---|---|
| [coordinator-status.md](coordinator-status.md) | What the real GPT requests establish: organizer loop, one injected-fault recovery, and the constrained single-allowed-skill limitation |
| [mshab-tool-interface-migration.md](mshab-tool-interface-migration.md) | Migration of the versioned `mshab-tool-family/1` invocation interface |
| [mshab-vla-tools-run01.md](mshab-vla-tools-run01.md) | First GPT + LightNav-0 + π₀.₅ V8 chain: navigation advanced, Pick failed, ended at the call limit |
| [mshab-next-baseline-plan.md](mshab-next-baseline-plan.md) | MSHAB011 failure analysis and the baseline plan that followed |
| [lightnav-control.md](lightnav-control.md) | LightNav-0 driving Fetch through the adapter; control diagnostics and limits |
| [fallback-baseline-lock.md](fallback-baseline-lock.md) | Historical fallback configurations locked, with the comparison gaps disclosed |

## MS-HAB: why manipulation fails (diagnostics)

| Document | Finding |
|---|---|
| [abcd-delivery.md](abcd-delivery.md) · [abcd-debugging.md](abcd-debugging.md) · [abcd-reproduction.md](abcd-reproduction.md) | A/B (official PPO/SAC) complete the one-object chain; C/D (π₀.₅) never grasp across seven training variants |
| [workspace-adaptation.md](workspace-adaptation.md) | The V8 model contract: cameras, 30-value proprioception, 13 action channels, and the 1,228-frame single-scene training selection |
| [v8-final-status.md](v8-final-status.md) | V8 evaluation: both runs failed on cumulative force without grasping |
| [c-root-cause-analysis.md](c-root-cause-analysis.md) | Locating the first failed grasp; closure and base-motion attribution; replay pilot |
| [mshab-pick-matrix-run01.md](mshab-pick-matrix-run01.md) · [mshab-pick-matrix-attribution.md](mshab-pick-matrix-attribution.md) | Pre-contact spatial failure, not grip loss: π₀.₅ closes 14–52 cm from the target while SAC reaches 2.87 cm |
| [mshab013-paired-pick-results.md](mshab013-paired-pick-results.md) · [mshab013-pairing-root-cause.md](mshab013-pairing-root-cause.md) | Fully paired diagnosis (fixed `PYTHONHASHSEED`): no single control channel repairs the trajectory |
| [s1-offline-action-reconstruction-2026-09-18.md](s1-offline-action-reconstruction-2026-09-18.md) | Level-1 teacher-forced per-channel reconstruction: no global channel-order failure, but yaw and torso fail selected family-conditioned zero-action comparisons |
| [s1-expert-replay-level2-2026-09-18.md](s1-expert-replay-level2-2026-09-18.md) | Level-2 expert replay: wrapper mutation is exactly zero; GPU replay matches the recorded initial observation but diverges in object pose after one action and in robot state near first contact |
| [s1-level3-first-divergence-2026-09-18.md](s1-level3-first-divergence-2026-09-18.md) | Level-3 exact-start diagnosis: three same-SAC controls have zero repeat drift; S1-IA versus SAC diverges after the first action on 14/15 qpos channels |
| [s1-first-action-attribution-2026-09-18.md](s1-first-action-attribution-2026-09-18.md) | Training/inference preprocessing is bit-exact; fresh-server same-key behavior clusters 5/5, while the unchanged SAC-tail rule is only 1/4 and does not support the stochastic-tail explanation |
| [s2-native24-diagnostic-20-2026-09-18.md](s2-native24-diagnostic-20-2026-09-18.md) | Diagnostic-only 20-update S2 integration: all four banks route and freeze correctly; fixed validation loss falls 15.16%, with no native success test or capability admission |
| [bc-t0a-regression-gate.md](bc-t0a-regression-gate.md) | Offline fail-closed CI gate for the three already measured official-BC assembly paths; diagnostic, not a benchmark estimate |
| [closed-loop-divergence-diagnostic.md](closed-loop-divergence-diagnostic.md) | Strict exact-start Level-3 first-divergence analyzer; refuses to pair runs without matching saved-state evidence |
| [vla-integration.md](vla-integration.md) | VLA replacement experiment matrix |

## Fetch backbones: AC-DiT line (frozen) and author π₀.₅ line (current)

| Document | Outcome |
|---|---|
| [acdit-mshab-mainline.md](acdit-mshab-mainline.md) | AC-DiT source, community checkpoints and the migration gate |
| [acdit-native-validation.md](acdit-native-validation.md) | Native AC-DiT on the lab server: 0/1, return-and-settle failure |
| [stage1-native-fidelity-2026-09-16.md](stage1-native-fidelity-2026-09-16.md) | Fixed seeds 2024–2028: **0/5 native**; official SAC succeeds from the same start as a positive control |
| [fetch-tapt-preparation.md](fetch-tapt-preparation.md) | Tool-family segmentation and training entry point, before any training |
| [fetch-tapt-sft-and-online.md](fetch-tapt-sft-and-online.md) | 1,199-update bounded SFT completed; first online tool test failed |
| [fetch-tapt-progress-attribution.md](fetch-tapt-progress-attribution.md) | Two confirmed defects: progress supervised after the action but consumed before it, and wrong-handoff inputs outside training coverage |
| [fetch-progress-timing-fix.md](fetch-progress-timing-fix.md) | Timing fix with exact paired replay — the defect is real but does not explain the main misjudgement |
| [fetch-current-progress-calibration.md](fetch-current-progress-calibration.md) | Current-frame labels, 118 more updates: held-out progress MSE 0.0609 → 0.0284, online still 0/2 |
| [fetch-observation-head-followup.md](fetch-observation-head-followup.md) | Observation-only progress head: CPU pairing prepared, head and numerical gates **not implemented** |
| [fetch-pi05-author-migration.md](fetch-pi05-author-migration.md) | Author OpenPI π₀.₅ on Fetch: model contract, strict loading of 71 tensors, BF16/FP32 attribution |
| [fetch-v8-native-capability.md](fetch-v8-native-capability.md) | **Frozen V8 native gate: 0/5.** Stop-loss executed; π₀.₅ TAPT training stopped for the week |

## Proposals (not yet executed)

| Document | What it proposes |
|---|---|
| [codebase-proposal-2026-09-21.md](codebase-proposal-2026-09-21.md) | Preparing the codebase for the feedback-system week: module status labelling, a `bvi/feedback/` subpackage, the SAC spawn-prior scan, and the measurement changes the current 16-plan panel needs before it can detect an improvement |

## MS-HAB: the 16-plan paired panels (current)

| Document | What it covers |
|---|---|
| [baseline-protocol-2026-09-20-v2.md](baseline-protocol-2026-09-20-v2.md) | The comparison and repair rules now in force for every arm |
| [ppo-sac-paired16-2026-09-20.md](ppo-sac-paired16-2026-09-20.md) | Fixed official-order PPO+SAC arm: scope, authorization and accounting |
| [goal-tools-paired16-2026-09-20.md](goal-tools-paired16-2026-09-20.md) | Goal-grounded tool planning arm: what GPT does and does not see, and the resume history |
| [goal-tools-response-repair-launch-v5.md](goal-tools-response-repair-launch-v5.md) | Bounded repair retry of GPT seeds 0 and 1, with all original attempts retained |
| [gpt-ppo-sac-progress-feedback-v1.md](gpt-ppo-sac-progress-feedback-v1.md) | Progress-feedback variant of the tool-calling arm |
| [sac-interface-audit-amendment-2026-09-20.md](sac-interface-audit-amendment-2026-09-20.md) | Amendment against the paper's Table 3/4 and §5.5 after the first two outcomes |
| [sac-followup-2026-09-20.md](sac-followup-2026-09-20.md) | First SAC batch close and the request-budget gate blocking the restart |
| [vlas-as-tools-method-conformance.md](vlas-as-tools-method-conformance.md) | Living clause-by-clause alignment baseline; the reference for any "we reproduced X" claim |

## S1/S2 diagnostics: the 09-18 to 09-19 chain

| Document | Finding |
|---|---|
| [s1-diagnostic-checklist-v2.md](s1-diagnostic-checklist-v2.md) · [s1-native-failure-analysis.md](s1-native-failure-analysis.md) | The checklist the chain followed, and the original S1 native 0/10 attribution |
| [s1-medium-training.md](s1-medium-training.md) · [s1-ia-sft-revision.md](s1-ia-sft-revision.md) · [s1-ia-call-evaluation.md](s1-ia-call-evaluation.md) | Ordinary S1 SFT, the invocation-aligned revision, and the preregistered call-level evaluation |
| [s1-exact-start-first-action-probe.md](s1-exact-start-first-action-probe.md) | The zero-training gate that follows the Level-3 first-step split |
| [level3-null-drift-calibration.md](level3-null-drift-calibration.md) | Physics/numerical drift floor, calibrated before any IA-versus-SAC comparison |
| [bc-t0a-2026-09-18.md](bc-t0a-2026-09-18.md) · [bc-ia-normalization-audit-2026-09-18.md](bc-ia-normalization-audit-2026-09-18.md) | Official BC T0a diagnostic and the CPU-only pairing/normalization audit |
| [s1-s2-native-handoff-preparation.md](s1-s2-native-handoff-preparation.md) · [s2-native24-handoff-input-gate-2026-09-18.md](s2-native24-handoff-input-gate-2026-09-18.md) · [s2-native24-handoff-behavior-2026-09-18.md](s2-native24-handoff-behavior-2026-09-18.md) | S1→S2 handoff: preparation, the passed input gate, and the failed offline behavior sub-gate |
| [s2-offline-decomposition-2026-09-18.md](s2-offline-decomposition-2026-09-18.md) | step0/step20 head x bank offline decomposition over the fixed held-out rows |
| [vla-contract-audit-2026-09-19.md](vla-contract-audit-2026-09-19.md) | Static input/timing/action-consumption contract audit of frozen S1-IA best/855 |

## AC-DiT Apple line (2026-09-19 to 09-20)

| Document | What it covers |
|---|---|
| [acdit-reproduction-audit-2026-09-19.md](acdit-reproduction-audit-2026-09-19.md) | Reproduction check against the authors' two-stage seven-task recipe |
| [acdit-apple-bounded-2026-09-19.md](acdit-apple-bounded-2026-09-19.md) | Bounded Apple Pick training launch and its verification |
| [acdit-apple-evaluation-2026-09-20.md](acdit-apple-evaluation-2026-09-20.md) | Mid-training check and the native evaluation gate; capability validation not completed |

## Evidence

`media/` holds every published recording — see [media/README.md](media/README.md) and `media/manifest.json` for exact filenames, outcomes and hashes; failure and injected-fault labels are preserved. `results/` holds per-batch JSON/JSONL evidence, `analysis/` holds derived tables. Raw run archives stay outside this repository and are referenced by SHA-256.

## Conventions

Every result names its scene and task-plan coverage, seeds, checkpoint revisions, navigation mode, horizon, success criteria, coordinator type, and access to privileged observations. Failures and incomplete runs are reported alongside successes, and infrastructure failures are listed separately rather than removed from the denominator. A video or a single successful task is a demonstration, not an aggregate benchmark result.
