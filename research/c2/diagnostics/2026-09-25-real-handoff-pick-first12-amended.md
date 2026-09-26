# 真实导航交接 SAC Pick：前 12 计划预检（修订复跑门）

**状态：前 12 计划完成；完整 40–60 交接批次未获授权、未启动。** 原始[实验设计](../../../docs/design/real-handoff-pick-baseline-census-experiment-design.md)和用户于 2026-09-25 批准的[复跑门修订](../../../docs/design/real-handoff-pick-census-replay-amendment-proposal.md)共同规定本次统计口径。固定 120 计划清单按首个 Navigate UID 的字面字典序取样；本次仅尝试位置 0–11，未替换任何计划。官方 PPO Navigate 后接官方 SAC Pick，不运行 Place、GPT、候选站位或训练。

## 首段结果

| 指标 | 结果 |
|---|---:|
| 固定计划尝试 | 12/12 |
| 原生 Navigate 未完成 | 1（位置 0，499 步） |
| reset 时已自动进入 Pick、Navigate 动作 0 步 | 1（位置 2，不计真实交接） |
| 有效 Navigate→Pick 交接 | 10/12，95% Wilson 区间 55.2%–95.3% |
| 首次自然 SAC 严格 Pick 成功 | 8/10，95% Wilson 区间 49.0%–94.3% |
| 首次自然 SAC 失败 | 2/10，均在官方 Pick 200 步时限用尽 |
| 可重复失败 | 1/2；另 1 例复跑变成严格成功 |
| 自动未归类 | 0/2 |
| 盲审人工标签 | 尚未取得；一致率不报告 |

位置 1 的真实交接在 Navigate 第 78 步取得：自然 SAC Pick 第 200 步失败，自动机制类为“抓后未完成”；同快照复跑第 167 步严格成功。因此它留在**自然基线失败**分母，不进入**可重复失败**分母。位置 9 的真实交接在 Navigate 第 151 步取得：自然 SAC Pick 和同快照复跑均在第 200 步失败，自动类均为“夹取失败”。其余 8 个有效交接严格 Pick 成功。

自然失败机制表为“抓后未完成”1/2、“夹取失败”1/2；稳定复跑子集仅“夹取失败”1/1。复跑的官方终止原因、终止步和自动机制类别各为 1/2 一致。不稳定失败占全部自然失败 1/2，超过修订规定的三分之一判读门；同时可重复失败仅 1 例，**不得据此判断主要失败机制或研究分流方向**。10 个有效交接中的严格成功率仅描述首次 Pick，不是完整 TidyHouse 任务成功率。

官方 episode 配置的冻结 120 行清单为每个目标提供源家具类型，清单 SHA-256 `ef4b6cae4ee43843fbc02952e140a1d924246c260eb8dd080d95c952d1bd2b1b`。位置 1 的源家具为沙发，位置 9 为桌；这只是协变量。对于位置 1，源家具由清单补全，原始旧版结果未被改写。

## 运行和证据

- 首段最终源码包 SHA-256 `8de2b88cbd20088a6c2eae14b101f4dba226b06ccd2be4a1521e5b38bd74da86`。它加入了 reset 已自动进入 Pick 的单列状态；第 2 个计划在更早的源码下曾被错误记为基础设施删失，原始失败回执仍保留。
- 12 个逐计划结果、回执、首次 Pick 轨迹 10 条、失败复跑轨迹 2 条的原始归档：`repo/Bidirectional-VLA-Interface/runs/real-handoff-pick-census-20260924/results-amendment-v3-evidence.tar.gz`，SHA-256 `fb683630a3aebdc02c0db64f70f8bad1eb229a4f34fa4f9930eca1d07765c4d7`。归档 294 个成员，路径检查通过。
- 前 12 计划摘要 SHA-256 `fae5c93728661f8bbfacd8de8ba6ca6600dba9956939578c4fcbea5cf60f7e4a`；评分 JSON SHA-256 `3663f003eaefa283443a47baa3dd72dced45a9eef5baef92bff164fe25ddc077`。
- 两张不含自动类别的人工复核图已生成：`blinded-review-amendment-v3-plan-001.png` 和 `blinded-review-amendment-v3-plan-009.png`，位于同一 ignored runs 目录。尚无独立人工复核意见，不把自动标签冒充人工一致性。
- 四项 CPU 检查通过：互斥分类优先级、导航失败分母、修订复跑统计、官方 episode 家具来源；完整批次脚本的 120 计划 dry run 通过。未修改官方 PPO/SAC checkpoint、42 维观测、500/200 步 horizon 或严格成功判据。

## 资源与停止门

首段实际累计 915.1961/3600 GPU1 进程秒，包含此前所有接口诊断和两次未产出新 Pick 结论的启动/重试；最终 12 计划批次的 517.558 秒包含复制的旧位置 0、1 回执，故没有重复加入实际累计。新最终批次位置 2–11 实耗 395.7445 秒。GPU1 收尾 15 MiB、0% 利用率；付费模型 API 0、训练 0、新 RunPod 资源 0。实验室单价、币种、账单与货币金额未知，未记为零。历史 RunPod 持久盘仍单列跟踪，详情见[独立账本](../../../../../BVI-research-plan-2026-09-14/finance/ledger-real-handoff-pick-census-20260924.json)。

首段预检已完成，但 2 例自然失败中 1 例复跑不稳定；完整批次即使继续到 40–60 交接，也须按修订口径并列报告自然结果与稳定子集。完整批次的另行授权和资源上限见[续跑申请](../../../docs/design/real-handoff-pick-census-full-batch-authorization-proposal.md)。
