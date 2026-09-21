# S2 native24 诊断性 20-update 门（2026-09-18）

## 结论

本次运行以状态 **`diagnostic_20_update_only_not_s2_capability_admission`** 完成：请求的 `20/20` 个 optimizer update 全部执行，训练与固定验证合计 `726.358` 秒；**没有执行 native task success 评测**，也没有 simulator rollout。因此，这一结果只证明 S2 四族训练、隔离更新、保存与固定验证链路能在真实 GPU 上闭环，不能解释为 S2 capability admission、完整 SFT 完成或在线能力提升。

固定 held-out joint loss 从 step 0 的 `0.132121` 降至 step 20 的 `0.112090`（`-15.16%`）。改善主要来自 move（`-39.00%`）；grasp（`+3.74%`）和 release（`+6.30%`）反而小幅变差。这个短门支持“梯度确实流到预期参数且验证器能测到变化”，不支持“所有族都已改善”。

## 准入范围与证据

这次运行使用单独的诊断准入，而没有改写正式 S2 准入条件。准入清单锁定了 S1-IA `best/855` 的参数、normalizer 和 state contract，并只授权最多 20 个 update；`capability_admission`、`bounded_sft_expansion`、`online_evaluation` 和 simulator rollout 均为 false。

准入依据为：

- 五个 exact start 的训练侧与推理侧预处理为 bit-exact；
- 原始 `5 × 16` 首动作队列因跨进程 `1e-6` 复现失败而继续保留为 `invalid_cohort`，没有被追溯性改名；
- 独立 fresh-server same-key 簇确认达到 `5/5`；
- 原生成功 SAC 参照中的随机尾部候选仅 `1/4`，结论为 `stochastic_tail_not_supported`；
- S1 参数 SHA-256 为 `7854919f17de40ea8c62ee966327904a61503cfe2c0eb1715e3f30ae6e72892e`，normalizer 为 `0f6264129ebbe89f8adee0c0caa2f204314674e5a95c882e1710413ee734e8d2`，state contract 为 `c4c4448d1db68590d47253c38ddff618bc8cb723c85fe5c3ecbc35f51860eb40`；
- native24 cache 在训练前通过 CPU dry-run，manifest SHA-256 为 `fad8f72d5ee3b288302292b60926669fc79cc7a0391954fa8814ca579528248d`，共 101 个窗口，覆盖 reach、grasp、move、release 四族。

证据：[诊断准入裁决](results/s2-native24-diagnostic-2026-09-18-run01/adjudication.json)、[运行配置与身份锁](results/s2-native24-diagnostic-2026-09-18-run01/config.json)、[最终 CPU 数据复核](results/s2-native24-diagnostic-2026-09-18-run01/final-validator-dry-run.json)。首动作准入的完整解释见 [S1 首动作随机尾部归因](s1-first-action-attribution-2026-09-18.md)。

## 训练与隔离门

训练按 `reach → grasp → move → release` 循环五轮，所以每族恰好获得 5 个 update，总计 20 个；每个 update 使用 microbatch `1`、gradient accumulation `8`。逐步 loss 与耗时保存在 [train.jsonl](results/s2-native24-diagnostic-2026-09-18-run01/train.jsonl)。

四个族各自在第一次更新时通过一次参数隔离审计：被选中的族 bank 发生变化，另外三个 bank 保持不变，共享 head 发生变化，冻结分区保持不变。四份证据分别是 [reach](results/s2-native24-diagnostic-2026-09-18-run01/update-audit-reach.json)、[grasp](results/s2-native24-diagnostic-2026-09-18-run01/update-audit-grasp.json)、[move](results/s2-native24-diagnostic-2026-09-18-run01/update-audit-move.json) 和 [release](results/s2-native24-diagnostic-2026-09-18-run01/update-audit-release.json)。冻结分区在运行配置与最终 checkpoint 中的 SHA-256 都是：

`27d751c8f58555c596928acf7ae698aa8952775aebbae2a8eb382e2daffa15af`

这证明冻结骨干没有被这 20 个 update 改写。最终共享 head 和四个 bank 的结构哈希记录在 [checkpoint.json](results/s2-native24-diagnostic-2026-09-18-run01/checkpoint.json)。

## 固定验证：step 0 → step 20

验证使用相同的 held-out 窗口和固定位置，selection 为 `heldout_joint_loss_only_fixed_first10_windows_per_family`。结果如下；负变化表示 loss 改善：

| 范围 | step 0 | step 20 | 相对变化 | 读法 |
|---|---:|---:|---:|---|
| overall | 0.132121 | 0.112090 | -15.16% | 改善 |
| reach | 0.108525 | 0.103173 | -4.93% | 小幅改善 |
| grasp | 0.096588 | 0.100197 | +3.74% | 小幅变差 |
| move | 0.217969 | 0.132952 | -39.00% | 明显改善 |
| release | 0.105402 | 0.112039 | +6.30% | 小幅变差 |

原始固定验证证据：[step 0](results/s2-native24-diagnostic-2026-09-18-run01/validation-0000.json) 与 [step 20](results/s2-native24-diagnostic-2026-09-18-run01/validation-0020.json)。这些数值是离线 joint loss，不是 benchmark success rate；特别是 grasp/release 的反向变化不允许用 overall 均值掩盖。

## Checkpoint、留存与可恢复性

- 最佳固定验证 checkpoint：remote `step-0020.pkl`，`2,431,704,949` bytes，SHA-256 `c7bfc42c6e177758188b9030dcd56fc751f15c1b021e11f4ca0af0548d3024df`；
- step 20 resume checkpoint：remote `resume.pkl`，`2,431,704,949` bytes，SHA-256 `0d5b8e35ad4e37c35165b8487f29f586dfb1f39fdfb4c2efe1d23019f9dd28e1`；
- 最佳 checkpoint 的验证选择记录见 [best.json](results/s2-native24-diagnostic-2026-09-18-run01/best.json)，resume 与参数分区哈希见 [checkpoint.json](results/s2-native24-diagnostic-2026-09-18-run01/checkpoint.json)。

完整权重仍只保留在实验室服务器的 `/home/pshuai/bvi-research/runs/s2-native24-diagnostic-2026-09-18-run01/`，远端目录共 `8,727,116,709` bytes；**没有完成本地全量权重备份**。本地只有 SHA-verified 元数据与验证证据归档 `D:\AI\embodied intelligence\runs\s2-native24-diagnostic-2026-09-18-run01-evidence.tar.gz`（`20,113` bytes，SHA-256 `35ab57bde9a832e608f026995810f6c829a49c415593b5b6fcbfec22ff0487da`）。留存状态与资源关闭证据见 [weight-receipt.json](results/s2-native24-diagnostic-2026-09-18-run01/weight-receipt.json)。

## 尚未通过的门

本次 20-update 运行没有解除后续门：

- **完整 S2 capability admission 仍未通过。** 本实验没有 native success episode；诊断准入不能替代现有正式路径中的 native capability 与 native24 wrong-handoff 验证。
- **完整或扩展 SFT 仍未获准。** 当前 adjudication 明确只允许恰好 20 个诊断 update，不能据此继续训练或扩大预算。
- **在线评测仍未获准。** 没有执行 rollout，也没有验证闭环任务成功、族间交接或 grasp/release 的在线影响。

因此下一道验收门应先补齐正式 S2 准入所需的 native capability / wrong-handoff 证据，再单独授权扩展 SFT 和训练后在线评测；不能从本次 offline loss 下降直接跳到 capability 结论。

机器可读的终态见 [result.json](results/s2-native24-diagnostic-2026-09-18-run01/result.json)，状态严格为 `diagnostic_20_update_only_not_s2_capability_admission`。
