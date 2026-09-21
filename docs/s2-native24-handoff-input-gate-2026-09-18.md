# S2 native24 wrong-handoff 输入门（2026-09-18）

## 结论

`bvi.native24-handoff/1` 输入门已通过。固定 validation parents 3020/3021 各执行两次精确 source replay，在四个预声明边界重新采集 `head128 + wrist128 + state24`，覆盖：

- `near_grasp_candidate_not_completion`：seed3020 step17，TCP–apple `0.012786 m`，未持有、未完成；
- `held_move`：seed3020 step20，TCP–apple `0.018162 m`，已持有、未完成；
- `far_grasp_diagnostic`：seed3021 step31，TCP–apple `0.422189 m`，未持有、未完成；
- `unheld_move`：seed3021 step34，TCP–apple `0.343119 m`，未持有、未完成。

四例都使用 `current_observation_v2`，输入是 `128×128 uint8` 的 head/wrist RGB 与 `float32 state24`；`policy_input_privileged=false`，ground truth 来自重放物理谓词而非模型 progress。训练 parents 3000–3003 被 registry 明确排除，旧 workspace224/state30 与不能精确恢复的 H5 parent 没有进入。

## 执行与验证

两条 source 的最大重放前缀分别为 20 和 34，每条双重重放，实际共执行 `2 × (20 + 34) = 108` 个 simulator actions；外层约 `120.10` 秒。policy/model call、optimizer update、外部 API 和新租机均为 `0`。

机器可读输入 manifest 为 [native24-handoff-manifest.json](results/s2-native24-handoff-2026-09-18-run01/native24-handoff-manifest.json)，SHA-256 `8d164a2a0f0ab2fb03cc4b52dd42848d5ad5bc5368cb28c8ee72cba096a4d190`。独立 CPU validator 重新计算所有 NPZ、source trajectory、source replay、case 和训练 manifest 哈希后通过，输出见 [independent-validation.json](results/s2-native24-handoff-2026-09-18-run01/independent-validation.json)。父轨迹隔离证据见 [parent-registry.json](results/s2-native24-handoff-2026-09-18-run01/parent-registry.json)，最终运行状态见 [result.json](results/s2-native24-handoff-2026-09-18-run01/result.json)。

本地完整证据包为 `D:\AI\embodied intelligence\runs\s2-native24-handoff-2026-09-18-run01-evidence.tar.gz`，`149,017` bytes，SHA-256 `e86f7eb1b8d9fd3952f2f6e8b3ec24af89adfa4ed1290e4042baf21e7cc2da62`。

## 尚未通过的半门

这批证据只包含每类一个独立观测，所以 manifest 明确记录：

- `input_only_behavior_not_evaluated`；
- `thresholds_bound=false`；
- false-completion、stagnation、rollback 均不可评估；
- learned feedback、native success 与 capability admission 均未执行。

因此不能把“输入门通过”缩写成完整 wrong-handoff 行为门通过，也不能据此进入 matched closed loop 或恢复训练。下一步必须先把既有 monitor 合同（reach/move `0.9`，grasp/release `0.6`，连续两次；drop `0.03`；至少10次的 stagnation `0.03`）在看模型输出前冻结并绑定源码哈希，再从同一已验证 validation parents 精确重放采集连续 native24 序列。只有连续序列才能评价两击完成、停滞和回退；孤立帧不得通过重复复制伪造成时间证据。
