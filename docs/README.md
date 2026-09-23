# 文档索引

这里仅保留公开仓库当前评测代码所需的文档。旧计划、训练路线、一次性诊断、原始
JSONL/NPY/视频和已停止的方法均不再作为当前文档维护；需要追溯时可从 Git 历史
中的 `193a17c` 恢复。

## 当前结果

| 设置 | 完成 episode | 完整任务 SR | 完成物体 | 每集平均 |
|---|---:|---:|---:|---:|
| Fixed PPO + SAC（官方固定任务顺序） | 16/16 | 0/16 | 14/80（17.5%） | 0.875 |
| GPT + PPO + SAC（VLA-as-Tools 通讯协议） | 16/16 | 0/16 | 13/80（16.25%） | 0.813 |
| Teleport + SAC（官方固定任务顺序） | 16/16 | 0/16 | 21/80（26.25%） | 1.313 |

- [三组实验记录](log/2026-09-21-tidyhouse-three-settings.md)
- [C2 Pick 站位能力与 GPT 选择](log/2026-09-23-c2-sac-pose-capability.md)：冻结交接的开发实验，不是完整任务 SR
- [机器可读摘要](results/tidyhouse-16/summary.json)
- [C2 最小机器摘要](results/c2-pose-capability-2026-09-22/summary.json)
- [负结果备忘](log/negative-results.md)

## 使用文档

当前开发实验：[C2 feedback-guided recovery](../research/c2/README.md)。它位于独立研究目录，不替换下列已发布 baseline，也尚无最终 SR。

| 文档 | 内容 |
|---|---|
| [architecture.md](architecture.md) | 当前评测代码的模块边界和数据流 |
| [evaluation.md](evaluation.md) | 三个固定 setting、指标和结论边界 |
| [reproduction.md](reproduction.md) | 安装、预览、执行、恢复和汇总 |
| [bridge.md](bridge.md) | GPT setting 的本地凭据桥接方式 |
| [log/README.md](log/README.md) | 实验日志规则与索引 |
| [log/TEMPLATE.md](log/TEMPLATE.md) | MS-HAB/VLA 实验日志模板 |

## 文档维护规则

1. `docs/log/` 只写人可以直接阅读的实验结论，不复制完整事件流。
2. `docs/results/` 只保留能重新计算公开表格的最小机器摘要。
3. 未执行的计划、已经放弃的方法和一次性排障过程不进入当前文档。
4. 重要负结果压缩到 `negative-results.md`，避免重复踩坑。
5. 任何涉及个人姓名、私有主机、用户名、PID、GPU UUID、凭据或授权 ID 的内容不得发布。
