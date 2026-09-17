# Fetch TAPT SFT and first online tool test

## Stage2 training completed, online success not achieved

The bounded native-start SFT pilot ended at1199 updates under its training time budget. Effective batch8, frozen AC-DiT backbone, four explicit residual banks and shared learned progress. All frozen native tensors remained unchanged; every bank changed168 parameter tensors and the progress head changed. Final held-out joint loss0.00619879; selected step1199. This loss does not establish progress calibration or task success. DROID, GRPO and actual LightNav handoff coverage remain absent; release supervision is only2train/1validation segments.

[Final training result](results/fetch-tapt-sft-2026-09-16-run01/final-result.json). Best and final checkpoints plus the full1,490,460,346-byte run archive are backed up locally and SHA256 verified. Archive SHA `29f8a23875b0fd954d334e22342d5e36cd34902323095507513b5d404266c69d`.

## Online tool harness (UTC2026-09-17)

A fixed reach→grasp→move diagnostic sequence, seed2025, used actual trained adapters, exact instruction embeddings, native whole-body actions and the existing learned-progress monitor. No GPT or navigation was used. Both thresholds and drop criteria were retained. This is a tool runtime test, not VLM collaboration or a controlled baseline comparison.

| Step | Learned event | Physical evidence |
|---|---|---|
|16|reach threshold; queue cleared, switched tograsp|TCP–object distance24.1cm, outside8cm training boundary|
|18|grasp threshold; queue cleared, switched tomove|Not grasped; TCP–object distance23.1cm|
|26|move progress drop; queue cleared, requested replanning|Not grasped; native successfalse|

**The learned signal actually controlled switching, but its completion predictions were premature.** No grasp occurred. Cumulative force at stop4981.37, below5000; the harness stopped for learned drop, not native safety failure. This separates interface execution from feedback correctness. The new risk is progress calibration/transfer to online states; training-versus-inference feature differences and limited data remain hypotheses to test, not established causes. Do not solve it by silently lowering/raising thresholds or substituting oracle completion.

Video: [fetch-tapt-pick-episode000-failed.mp4](media/fetch-tapt-online-pick-2026-09-17-run01/fetch-tapt-pick-episode000-failed.mp4). [Result and event evidence](results/fetch-tapt-online-pick-2026-09-17-run01/result.json). Raw folder retained its launch-plan date2026-09-16; publication uses verified recording UTCdate2026-09-17.

Next gate: compare trained progress from actual denoising inference with verified held-out call boundaries, separately log action/progress losses, and audit the AC-DiT port's feature path against the intended method before modifying training or thresholds. Then re-test tool feedback; GPT orchestration success remains untested.

## Resources

Training and online harness processes exited; labGPU1 returned to15MiB. No API calls or rented compute added. Lab price unknown. The two historical Runpod pods remainEXITED with40GB chargeable storage (~USD0.266667/day, previously verified rate).
