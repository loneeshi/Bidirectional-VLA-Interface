# S1 首动作随机尾部归因（2026-09-18）

## 结论

冻结的 S1-IA `best/855` 在五个 exact start 上，**首动作失败不能解释为偶然抽中了随机动作分布的 SAC-distance 尾部**。预注册判据要求四个原生成功 SAC 参照中至少 `3/4` 命中尾部候选，实际只有 seed 2026 命中，即 `1/4`；结论为 `stochastic_tail_not_supported`。

这项结论建立在两层分开的证据上：训练与推理预处理在五个请求上逐数组、逐哈希 bit-exact；跨进程的动作值虽不能满足原始 `1e-6` 复现门，但后续独立 fresh-server 试验以预先冻结的簇判据确认了 `5/5` 个 same-key 数值簇。原始 `5 × 16` 队列仍保留为 `invalid_cohort`，没有被事后改写成有效的 bitwise 复现实验。

## 证据链

### 1. 预处理逐级 bit-exact

五个 seed（2024–2028）的同一份 `step000.npz` 分别走训练侧和推理侧路径；四个阶段 `training_repack_vs_server_raw`、`libero_inputs`、`quantile_normalized`、`model_inputs` 全部 `array_equal=true`，最终图像、mask、state 和 token 叶子的 SHA-256 也一致。checkpoint normalizer 与显式提供的 normalizer 哈希同为 `0f6264…e8d2`，state contract 哈希为 `c4c444…eb40`。

静态量化范围检查发现轻微 state OOD：dim 3 在五个 seed 都越出训练 `q01–q99`，归一化绝对值为 `1.186–1.213`；dim 6 仅 seed 2026 越界，归一化值为 `1.071`。这是需要保留的弱异常，不等同于已确认根因。

证据：[独立预处理审计](results/s1-preprocessing-parity-2026-09-18-run01/report.json)；[模型输入级 transform 审计](results/s1-exact-start-first-action-2026-09-18-run01/transform-parity.json)。

### 2. 原始 5 × 16 队列严格保持 invalid

探索性运行对每个 exact start 连续采样 16 次，共 80 次冻结模型推理。原合同要求第一个样本与历史部署首动作在 raw action 上以 `atol=1e-6` 精确复现。虽然 checkpoint、normalizer、state contract、RNG key 和服务配置均匹配，五个 seed 的最大绝对差仍为 `0.001526–0.011555`，因此五例全部 `identity_failed`，整批判为 `invalid_cohort`。这项失败不会因后续结果被追溯性改名或删除。

证据：[原始摘要](results/s1-exact-start-first-action-2026-09-18-run01/summary.json)；[逐 seed 结果](results/s1-exact-start-first-action-2026-09-18-run01/per-seed.json)；[完整样本](results/s1-exact-start-first-action-2026-09-18-run01/samples.npz)；[文件清单](results/s1-exact-start-first-action-2026-09-18-run01/artifact-manifest.json)。

### 3. same-key 簇先作为探索性观察，再独立确认

对 invalid 队列做不改标签的探索性检查后发现：每个 seed 的历史部署动作与 exploratory sample 0，都是相对于 sample 1–15 的 mutual unique nearest neighbors。两者 RMSE 为 `0.000976–0.003902`，而最近的 different-key 样本 RMSE 为 `0.109574–0.295146`；same-key / nearest-different-key 比值仅 `0.00347–0.02546`，且没有 material sign flip。

随后在推理前冻结确认规则，启动独立 fresh server，每个 seed 只做一次 same-key 调用。要求 fresh action 同时比所有 different-key 样本更接近历史部署动作和 exploratory sample 0，使用严格不等式。结果 `5/5` 通过，RNG 与模型身份也全部匹配；fresh 的最差 same-key margin ratio 为 `0.00347–0.02771`。

因此可以确认**功能性的 same-key 数值簇**，但不能声称跨进程 GPU bitwise determinism；原 `1e-6` cohort 仍是 invalid。

证据：[fresh-server 确认摘要](results/s1-same-key-confirmation-2026-09-18-run01/summary.json)；[逐 seed 确认结果](results/s1-same-key-confirmation-2026-09-18-run01/per-seed.json)；[fresh actions](results/s1-same-key-confirmation-2026-09-18-run01/fresh-actions.npz)；[服务结果](results/s1-same-key-confirmation-2026-09-18-run01/server/result.json)。

### 4. SAC-tail 判据未改，结果为 1/4

确认 same-key 数值簇后，SAC-distance 仍使用运行前冻结的原规则：只计入原生成功的 seed 2025–2028；部署样本需位于 16 个样本中 SAC-distance 的 top two（percentile `>=0.9375`），同时样本中位 SAC RMSE 不高于部署 RMSE 的 `0.75`。结果如下：

| seed | SAC 原生成功 | deployed percentile | median / deployed RMSE | tail candidate |
|---:|:---:|---:|---:|:---:|
| 2024 | 否，仅诊断 | 0.3750 | 1.0307 | 不计票 |
| 2025 | 是 | 1.0000 | 0.7819 | 否 |
| 2026 | 是 | 0.9375 | 0.5624 | 是 |
| 2027 | 是 | 0.9375 | 0.7846 | 否 |
| 2028 | 是 | 0.3125 | 0.8643 | 否 |

有效票为 `1/4`，低于预注册的 `3/4`，所以 `stochastic_tail_not_supported`。这说明已观察到的五个部署首动作不能由“恰好都抽到坏尾样本”充分解释；它不证明某个特定动作才是唯一正确动作。

## 范围与限制

- 官方 SAC 首动作只是同起点参考，不是监督标签，也不是唯一可成功动作。
- 本实验没有 simulator step、rollout episode 或任务成功率测量；因此没有建立新 task success。
- 总计 85 次冻结模型推理（探索 80、确认 5），训练 update 为 0，simulator step 为 0，外部 API 调用为 0。
- 结论只用于首决策随机尾部归因；它不是 S1/S2 capability admission，也不能替代后续训练后闭环门。
- state quantile OOD 是相关诊断线索，不足以单独解释动作差异或失败。

## 本地证据与归档

- 预处理结果目录：[`docs/results/s1-preprocessing-parity-2026-09-18-run01/`](results/s1-preprocessing-parity-2026-09-18-run01/)
- 探索性 5 × 16 目录：[`docs/results/s1-exact-start-first-action-2026-09-18-run01/`](results/s1-exact-start-first-action-2026-09-18-run01/)
- fresh-server 确认目录：[`docs/results/s1-same-key-confirmation-2026-09-18-run01/`](results/s1-same-key-confirmation-2026-09-18-run01/)
- `D:\AI\embodied intelligence\runs\s1-preprocessing-parity-2026-09-18-run01.tar.gz` — SHA-256 `c417c9003bc25d744b23acb8bb4aa1d3ab76e104183d60760706fdb55ff94073`
- `D:\AI\embodied intelligence\runs\s1-exact-start-first-action-2026-09-18-run01.tar.gz` — SHA-256 `8124ccd614da379b0a8c0ec0bb58a826210d8109bfb149317600531279d889c6`
- `D:\AI\embodied intelligence\runs\s1-same-key-confirmation-2026-09-18-run01.tar.gz` — SHA-256 `782946c24ea40cdc8a2b84a09b57926124745f4385e85506f075e57151c4db61`
