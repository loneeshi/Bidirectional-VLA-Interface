# C2：视觉工具调用与抓取能力

当前处于主线第 2 阶段：接口执行与失败归因。开发组官方 Pick 出生站位上的固定底盘 r3 与可移动底盘 r1 结果，统一见 [5 例实验日志](../../docs/log/2026-10-01-c2-arm-capability-dev.md)。600 步是非官方时间预算，成功仍按官方严格判据评估，同时报告 200 步结果。

| 条件 | 严格成功 ≤600 步 | 严格成功 ≤200 步 |
|---|---:|---:|
| Astra 固定底盘 V | 1/5 | 0/5 |
| Astra 可移动底盘 V-mobile | 1/5 | 0/5 |
| 脚本参照 | 0/5 | 0/5 |

这些是同一组已暴露开发任务的诊断。执行条件存在差异，不能从此表单独推断底盘的因果收益。脚本参照未通过 ≥4/5 的测试组门槛；30 例测试组尚未运行。

## 当前方法与设计

- [抓取能力测评设计](../../docs/design/c2-eef-arm-capability-eval-design.md)：以末尾最新修订为准，删除 P，开发组缩为 5 例。
- [可移动底盘条件](../../docs/design/c2-eef-same-five-mobile-authorization-proposal.md)：已完成的有界开发组范围，不代表后续预算授权。
- [视觉输入与真值边界](docs/real-handoff-visual-stance-and-privileged-ablation-design.md)：主条件只用机器人传感器与自身状态，场景真值只在评估端。
- [工具循环设计](docs/real-handoff-eef-tool-loop-icl-design.md)：方法与后续修订。
- [执行器计划](docs/real-handoff-eef-executor-systematic-plan.md)：2×2 对照已暂停，协调 v2 保持冻结。
- [录像](../../docs/media/README.md)：粉色目标点连线、青色真实 TCP、橙色底盘指令；目标点连线不是逐步物理预测。

旧版本提案、一次性排障和逐请求证据统一通过 [历史索引](diagnostics/README.md)访问。历史 C2 实现的入口、测试和安装说明见 [CODE.md](CODE.md)。完整 Astra 执行器及冻结包在本地研究目录保留。
