# S2 step0/step20 head×bank 离线分解（2026-09-18）

## 结论

固定的 69 个 held-out rows 已完成 `H0/B0`、`H20/B0`、`H0/B20`、`H20/B20` 四格分解，共 276 个参数组合行。step0 与 step20 的历史固定验证值均逐值复现，action/progress/joint loss 最大差异为 `0`；训练侧调用与进程内 deployment adapter 在 276 行上的 normalized action、反归一化外部 action 和 progress 均 bit-exact，最大差异为 `0`。

最直接的归因是：**grasp/release 的 action regression 来自对应的 step20 family bank，不来自共享 progress head。** 在 `B0` 固定时把 head 从 H0 换到 H20，四族 action loss 与首动作都完全不变；在 `H0` 固定时换入 B20，grasp action loss 上升 `4.87%`，release 上升 `8.25%`，move 下降 `40.34%`，reach 上升 `0.11%`。共享 H20 head 则改善四族 progress loss。

这支持继续冻结追加训练，并把下一次修复范围缩到 family bank 的数据、动作目标或更新，而不是共享 progress head。它仍只是固定离线 roster 上的参数块归因，不证明 bank 是 native failure 的唯一原因，也不构成 native capability admission。

## 身份与恢复说明

主计算在 `792.105` 秒内完成，optimizer update、simulator step、rollout 和外部 API 均为 `0`。原 runner 在写完 `rows.jsonl`、`predictions.npz` 与 `queue-audit.json` 后，因脚本部署在实验室根目录而把 core 源码相对路径解析成不存在的 `/home/pshuai/src/...`，只在最终身份报告阶段 fail-closed。原始失败 [result.json](results/s2-native24-offline-decomposition-2026-09-18-run02/result.json) 保留未覆盖。

随后使用独立的 CPU-only finalizer，在显式 SHA-256 绑定五份既有产物、三份实际 evaluation 源码和全部训练证据后重新验证：

- `276 = 69 × 4`，四格均覆盖相同 roster 与 RNG；
- 预测数组首动作与逐行记录一致；
- bank/head 路由、调用切换清队列和返回身份一致；
- step0/step20 历史 loss 逐值复现；
- normalizer、state contract、checkpoint 与数据 manifest 身份一致。

finalizer 没有导入或反序列化模型、没有查询 GPU、没有模型前向、optimizer 或 simulator。恢复终态见 [finalization-result.json](results/s2-native24-offline-decomposition-2026-09-18-run02/finalization-result.json)，状态为 `passed_in_process_deployment_adapter_parity_transport_not_evaluated_recovered_without_model`；完整身份见 [identity.json](results/s2-native24-offline-decomposition-2026-09-18-run02/identity.json)，汇总见 [summary.json](results/s2-native24-offline-decomposition-2026-09-18-run02/summary.json)。

## 四格结果

下表把 `H0/B0` 作为基线。head effect 是 `H20/B0 - H0/B0`，bank effect 是 `H0/B20 - H0/B0`；负值表示 loss 改善。

| family | H0/B0 action | H20/B0 action变化 | H0/B20 action变化 | 完整H20/B20 action变化 | H20/B0 progress变化 | 完整H20/B20 progress变化 |
|---|---:|---:|---:|---:|---:|---:|
| reach | 0.095914 | 0.00% | +0.11% | +0.11% | -41.56% | -43.24% |
| grasp | 0.085872 | 0.00% | **+4.87%** | **+4.87%** | -3.57% | -5.35% |
| move | 0.206517 | 0.00% | **-40.34%** | **-40.34%** | -12.82% | -14.82% |
| release | 0.093398 | 0.00% | **+8.25%** | **+8.25%** | -6.41% | -8.88% |

action 的 head×bank 交互项在四族均为 `0`。progress 存在小的交互项，因此不能把完整 joint loss 变化简单相加；但 action 归因在这套架构和固定 rows 上是清楚的。

首动作诊断没有给出同样单调的排序。例如 release 的逐块 action loss 变差，而固定 RNG 首动作 physical RMSE 从 `0.2256` 降到 `0.1802`，raw OOB 均值从 `12.18%` 降到 `6.41%`。所以 bank 选择不能依靠一个首动作样本或 OOB 单指标；per-family action loss non-regression、progress 和新的独立门应共同约束。

## 部署一致性的边界

本轮通过的是可复用的**进程内** deployment adapter：输入合同、family/bank 路由、调用切换时清空其内部队列、反归一化 chunk 与响应身份。它没有启动 socket/WebSocket transport，没有消费 action queue、调用 `env.step` 或验证客户端裁剪/head mask。因此不能称完整部署服务、外围执行链或 native success 已通过。

原始逐行证据为 [rows.jsonl](results/s2-native24-offline-decomposition-2026-09-18-run02/rows.jsonl)、[predictions.npz](results/s2-native24-offline-decomposition-2026-09-18-run02/predictions.npz) 与 [queue-audit.json](results/s2-native24-offline-decomposition-2026-09-18-run02/queue-audit.json)。本地完整证据包为 `D:\AI\embodied intelligence\runs\s2-native24-offline-decomposition-2026-09-18-run02-evidence.tar.gz`，`776,653` bytes，SHA-256 `e7d43db869b5019b9c213c2cbcc74d9b3e3081386a303cd18b60927f2c17ec76`。

## 下一步裁决

不追加 optimizer update。先冻结一个供后续独立诊断的选择假设：保留 H20 progress head，只在 move 使用 B20，reach/grasp/release 回到 B0。这个组合能在当前 rows 上做到四族 action non-regression，同时保留 H20 的 progress 改善；但它是看过本批结果后提出的，**不得在同一 69 rows 上自证通过**。它只能作为新 held-out wrong-handoff／shadow 门中的预注册候选，并且在新门通过前不能控制环境或替代现有能力结论。
