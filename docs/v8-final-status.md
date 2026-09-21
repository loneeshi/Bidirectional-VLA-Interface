# V8 evaluation: unsuccessful manipulation

The 2,000-update Fetch π₀.₅ LoRA checkpoint trained on 1,228 frames did not complete the single-object task. Both evaluations used the same checkpoint, seed and task plan, an oracle dispatcher, workspace/wrist RGB and proprioception. No SAC takeover occurred during these evaluations.

| Run | Navigation | Total steps | π predictions | Ever grasped | Result |
|---|---|---:|---:|---|---|
| C12 | Official PPO | 182 | 153 | No | Native failure; cumulative force 6196.565 |
| D9 | LightNav-0, 16 predictions | 236 | 159 | No | Native failure; cumulative force 9408.522 |

The [trace audit](results/v8-final-failures.json) confirms executed manipulation controls match the logged π outputs with the declared head mask. Tool connectivity and inference work; task success does not. A/B remain successful single-object demonstrations, not full benchmark results. The same-scene training data and privileged SAC teacher observations limit comparisons.

- [C12 failure video — C-v8-recovery-failed.mp4](media/pi05-v8-recovery/C-v8-recovery-failed.mp4)
- [D9 failure video — D-v8-recovery-failed.mp4](media/pi05-v8-recovery/D-v8-recovery-failed.mp4)

The descriptive-language C13/D10 ablation and offline language probe were prepared but not executed. More training is not yet supported by demonstrated closed-loop improvement. Before another paid run, investigate action/state conventions, handoff velocity and teacher/student observation mismatch with recorded trajectories, then select a bounded diagnostic.

The complete V8 checkpoint and raw evaluation archive were downloaded and SHA256-verified before the temporary GPU Pod was deleted. A scheduled stop command failed repeatedly; its cause is unresolved. The batch exceeded its duration/cost allocation. This is recorded as a resource-management failure, not an authorized extension. No further paid experiment was started.
