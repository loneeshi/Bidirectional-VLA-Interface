# Goal-grounded tool planning, version 1

Current comparison and repair rules: [baseline protocol revision 2](baseline-protocol-2026-09-20-v2.md).
The continuous PPO/SAC fixed arm remains the official-model baseline. Historical
duplicated-response GPT attempts require additive validity adjudication; their
raw records are retained. Runner default is now a maximum of 2 attempts per arm.

## Retry10 and 2+2 chunks

GPU0 remains occupied by unrelated PID672548 and only GPU1 is available. Concurrent simulator episodes are not used: they would share one GPU, risk memory/throughput interference, and invalidate wall-time comparison. The remaining panel is divided into invocations that start at most2 fixed rows and2 GPT rows; each invocation exits as chunk_complete and preserves the common panel/status/attempt history. Because fixed seed2 was already completed, chunk01 comprises GPT2 retry plus the next rows until exactly2 new fixed and2 new GPT attempts have started.

Source-v4 adds at most10 retries for APIConnectionError/APITimeoutError only. Each physical provider attempt creates its own durable claim and USD0.00125 reservation before execution; retry count, safe exception-type chain, and connection reset are logged. Model-output, authentication, rate-limit, budget, protocol, and policy failures are not retried. Global800-request/USD2 gates take precedence, so “10” is a per-logical-request ceiling, not permission to exceed the global budget. Ambiguous historical calls are never replayed.

CPU full suite637 passed,6 skipped. Archive `runs/goal-tools-source-v4-retry10-chunk2.zip`, SHA256 `25c41e02d0c742e73491ee9c7423077863eeab6e17de4433bbd6b1b4261af959`. Model availability read passed, no old runner/bridge/lock, and GPU1 had no compute process before launch. Chunk01 uses remote source-v4, runner output launch-v4-chunk01.log, local bridge parent33880 with --max-provider-retries10. Existing104 physical claims remain preserved.

## Connection-failure resume, September20 17:04 EDT

User requested immediate resume after minimal checks. Model availability read succeeded for gpt-5.6-luna; no old runner/bridge or lock remained, GPU1 had no compute process, GPU0 process672548 untouched. API ledger has97 claims (including the connection-failed attempt; charge unknown), leaving703 requests. Remaining14 GPT episodes cap560 requests: total657<=800. Original USD2/global resource cap retained; actual provider/lab bills remain unknown. No source, prompt, policy or threshold change.

Restarted original persistent bridge (parent3664; logs workspace runs/goal-tools-bridge-resume2.*.log). Runner uses the same frozen source-v3 and --retry-infrastructure, launch-resume2.log; skips completed3 fixed and2 GPT episodes, retries GPT seed2 into attempt002. Old attempts/results are preserved; no replay of the ambiguous failed API request. This is an episode retry with new request IDs, not a claim that the failed provider call was uncharged.

Verified runner1916921 / child1916922 active. Three new bridge calls returned ok; GPT seed2 executed40+40+18 actions and returned native requested-target success. This verifies restored API/action/success-feedback flow, not final episode or panel completion.

## Resume correction, September20 afternoon EDT

The initial launch stopped at02:09:48 EDT after only fixed seed0 and GPT seed0. GPT navigation reached native success at step129, but successful feedback omitted evidence references and the protocol validator rejected it. This is an interface defect, not model failure. Other30 rows remained not_run. Previous startup verification did not cover successful-return handoff and was insufficient.

User authorized resume. Revision source-v3 adds a call/frame-addressed reference to the emitted grounded_predicate event, without changing goals, models, thresholds, actions or scoring. Tests exercise successful navigate/pick/place feedback and failure precedence. Full CPU634 passed,6 skipped. Real train100 smoke now requires actual navigation success-return followed by a physical Pick action, rather than only one navigation action.

Native smoke-success-v3 passed:20 physical actions, API0, navigation success returned through SerialRuntime, then Pick executed a real action; state parity max difference0 and scoring state remained unchanged by observations/predicates. Detached bridge parent12804 uses the original persistent claims ledger. Formal resume starts from GPT seed0 attempt002; fixed seed0 remains untouched.

Resume verified about13:35 EDT: runner1892479 progressed beyond seed0 to GPT seed1. GPT seed0 attempt002 ended on ModelResponseError (non-JSON output,1 API request), retained as an invalid-model-output outcome rather than infrastructure; no prompt/model adjustment or retry to improve score. Fixed seed1 also terminated natively.88 cumulative claims at this snapshot (USD0.11 reserved, actual cost unknown). GPU1 continues authorized evaluation; GPU0 untouched. Original failed GPT attempt001 retained and counted in all-attempt API usage. No periodic monitor created.

Source archive SHA256 `5aa13199f567018ef714b45329544fab8b28793e835e0b4bb9779b86a3bf8d36` (`runs/goal-tools-resume-source-v3.zip`). Previous source-v2 remains unchanged. Resume uses source-v3 with --retry-infrastructure, skips completed fixed seed0, and creates GPT seed0 attempt002. Existing87 claims + maximum640 further requests =727<=800; reservation ceiling remainsUSD2, actual bills unknown. No repeat runner/bridge was active before smoke; only GPU0's unrelated process672548 was present. Resume validation/launch evidence follows below.

User requested stopping the constrained-supervisor panel and replacing it with whole-goal tool planning. This is a VLA-as-Tools architectural control using pretrained PPO/SAC executors, not a claim that PPO/SAC are language-conditioned VLAs or a reproduction of trained TAPT.

## Stopped predecessor

`/home/pshuai/bvi-research/runs/ppo-sac-paired16-20260920` is preserved. At stop verification no remote runner/child remained. Fixed seed0 completed with native benchmark failure, 373 actions, no completed object. GPT seed0 stopped before API/physical actions on initial-state hash mismatch; other 30 planned rows are not_run. The preceding missing-asset attempt is also retained. Local bridge parent47232/child48628 were stopped. GPU1 had no compute processes; GPU0 process672548 was untouched. No additional paid requests: persistent claims remain83.

The hash mismatch included hidden off-scene actor parking poses and init-config indices. Python hash randomization affects upstream set ordering; new subprocesses use PYTHONHASHSEED=0. Exact full-state equality is still a hard gate, not a tolerance or a removed check.

## New comparison

Same 16 unique historical plan UIDs: seeds0–11,13,14,16,19. Both arms have official PPO navigation and per-object SAC Pick/Place. Fixed chooses the next official subtask. GPT sees all five object-to-destination goals, all20 valid skill/target combinations, camera images and its own execution history, but no current native subtask index, type or next-call hint. Native object-order constraints are disclosed; unrestricted object reordering is NOT claimed.

Targets resolve independently of the scorer pointer. Requested object/goal poses rebuild official42D policy state using native flattening; real stacked depth images are retained. Requested object selects the SAC checkpoint. Pure upstream predicate helpers supply target-specific feedback. No reset, teleport, pointer write, evaluate call or substituted action is permitted. Original native sequential scoring, force checks and horizons remain authoritative. Completed objects are counted from native progression, never repeated successful place invocations. Both arms use navigation ignore_arm_checkers=True inherited from the existing sequential evaluation configuration; this is disclosed, not a claim of untouched upstream defaults.

GPT has no additional heuristic grasp interruption. Each invocation40 actions/180s; episode7000 actions/900s. Fixed can continue175 slices; GPT has40 API decisions, so decision-cap censoring is reported separately. This comparison includes GPT planning overhead and its finite decision budget, not an equal-compute ablation.

## Gates, accounting and continuation

CPU full suite631 passed,6 skipped. Real compatibility smoke is train seed100, one physical action, API0; verify native-state parity for matching goal, differing state for other goal, no scorer mutation, real requested-other-target navigation, and Pick/Place actor forward passes. Validation results never tune the interface or model.

New output `/home/pshuai/bvi-research/runs/goal-tools-paired16-20260920`. Source snapshots and all attempts are immutable; panel rows each have planned denominator16. Stop at infrastructure failure. Resume only with no live runner/child, same interface and preserved API ledger; failed attempts require explicit retry and remain archived. Do not mix the old supervisor results into this study.

API reuse authorization SAC-INTERFACE-BASELINE-20260920-BATCH01 and original local/remote bridge spool.83 used +640 worst-case new =723 within800. Reservations USD0.10375 +0.8 withinUSD2, actual bills unknown. GPU1 only; no training or RunPod. Original18h/deadline remain, panel outer cap33000s. Sequential episode caps total8h plus startup/teardown; typical ETA requires measured GPT speed.

## Verified launch (September20, about02:10 EDT)

Native train100 smoke passed: matching state max absolute difference0, requested alternative navigation index4 executed1 action while native pointer stayed0; pure observations/predicates left native state unchanged; Pick/Place produced finite13D actions. Evidence local `runs/goal-tools-smoke-summary.json` under the workspace.

New fixed/GPT seed0 reset hashes match exactly: `65dc99175b6cbbbdc82ba4c1ba05f32e6b969fd3957a862688e43f04934ef333`. GPT completed its first real model-selected navigate call (40 actions) and requested further feedback-conditioned decisions. No hidden next-call filtering is present. Native failures remain results, not grounds to tune policies.

Detached remote runner1871493; startup GPT child1876607, on GPU1 only (2579MiB at verification). Local detached bridge parent48780, original authorization/spool reused. Full source/tests snapshot `runs/goal-tools-paired16-source-v2-20260920.zip`, SHA256 `d2c7f2eac5a421875f785ba58387151d9998bcc81231352a81f20ffac5e8d668`, deployed at output/source-v2. Earlier source and stopped panel are retained. Startup status downloaded to `runs/goal-tools-startup-panel-status.json`; authoritative running status remains remote output/panel-status.json.

At startup verification87 persistent bridge claims (83 previous +4 new), reservation USD0.10875, actual charges unknown; remaining request capacity713 at that snapshot, decreasing as the detached panel runs. No new rental/storage purchases. LabGPU1 remains occupied by authorized evaluation; GPU0 unchanged. No additional monitoring automation created.

Resume command uses the recorded runner argv in the process evidence: `source-v2/scripts/run_ppo_sac_paired16.py --source-manifest /home/pshuai/bvi-research/runs/sac-interface-baseline-20260920/manifest.json --output /home/pshuai/bvi-research/runs/goal-tools-paired16-20260920 --checkpoint-root /home/pshuai/bvi-research/checkpoints/mshab --bridge-dir /home/pshuai/bvi-research/runs/sac-interface-baseline-20260920/bridge --authorization-id SAC-INTERFACE-BASELINE-20260920-BATCH01 --goal-tools --execute`. Only resume after checking all old PIDs and locks, with the original persistent bridge and exact source. Environment requires PYTHONHASHSEED=0, MS_ASSET_DIR=/home/pshuai/bvi-research/assets, pinned official MS-HAB PYTHONPATH plus source-v2/src, and GPU1 UUID. Completed rows are skipped; old attempts are never overwritten.
