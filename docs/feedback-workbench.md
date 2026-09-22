# Feedback 工作入口（2026-09-21）

最新范围：[SAC初态能力与重试实验设计](sac-capability-experiment-design.md)。
优先从官方预制spawn刻画工具能力，再测GPT调整/重试；信息摘要和image消融暂缓。
用户要求的新持续执行协议已写入设计，目前运行器仍是此前原生失败截停逻辑。
下方设置和历史表保留，不能视为新协议已运行。

本周围绕 evaluator feedback 改进 GPT 信息与执行反馈。默认 GPT-5.6 Luna +
官方 PPO + 逐物体 SAC；LightNav 导航替换路径保留。continuous progress 显式选择。
当前 progress 是规则 proxy，读取模拟器位姿和抓取状态；这些仍是特权策略观测，
不能称为非特权反馈或 learned progress。

## 入口与开关

[源码索引](../src/README.md) 由 registry 测试检查，不搬动旧模块。
批次入口 scripts/run_ppo_sac_paired16.py，单集入口 scripts/run_coordinator.py。

| 设置 | 默认 | 可选 |
|---|---|---|
| 批次 --feedback-mode | evaluator | continuous_progress（仅 PPO/SAC） |
| 单集连续模式 | 关闭 | --progress-feedback |
| --feedback-profile | raw_v0 | object_v1 / object_trajectory_v1 |
| --hide-feedback | 不屏蔽 | images / trajectory / progress / structured_goals，可组合 |
| 先验 | 无 | 单集 object_trajectory_prior_v1 + --spawn-prior + --evaluation-manifest |

raw_v0 保持本次修改前 JSON context/history 序列化字节。更早 source-v2/v5 的
system prompt 复现仍以对应 frozen source 为准。摘要模式保存 instruction，
按物体折叠，分别保留到物体与到目的地的 navigation。
轨迹模式记录距离首值/末值/最小值、最小值步数、曾抓住、末尾停滞步数
（epsilon=0.0001m/step）及结束原因；不调用额外成功谓词。

视图只改变呈现，不改 monitor/执行器。images 移除实际多模态 part；structured_goals
移除结构化描述和 task 中目标目录坐标，保留文字目标与合法路由 ID。
progress 移除数值和 progress provenance；evaluator 的成功反馈仍保留。
新模式/profile/view 必须使用新批次目录，resume 检查阻止混组。

## 实验表

| 设置 | 调度 | 工具 | GPT 反馈 | planned | 完成对象 | 完整 SR |
|---|---|---|---|---|---|---|
| Fixed 历史配对 | 官方顺序 | PPO + SAC | 无 GPT | 16 | 14/80 | 0/16 |
| GPT 历史配对 | GPT | 同一 PPO + SAC | native evaluator subtask completion + RGB + raw history | 16 | 13/80 | 0/16 |
| 标准化 Teleport | 官方顺序 | teleport + SAC | 无 GPT | 16 | 21/80 | 0/16 |
| continuous progress | GPT | PPO + SAC | 模拟器观测规则 proxy | 未预注册新批次 | 未运行 | 未运行 |
| 摘要/轨迹/先验 | GPT | 待冻结匹配配置 | 显式 profile/view | 未预注册新批次 | 未运行 | 未运行 |

三组历史实验匹配16 plan UID；PPO两臂保存初态哈希配对。Teleport改变后续交接位姿。
RGB-D + fetch_nav/workspace，evaluator e9ff3d2，Teleport算法源4729821。
每集7000动作/900秒，GPT40调用，每次最多40动作/180秒。预算截断、所有旧attempt
和失败请求均需纳入有效性及成本说明，详见[历史实验日志](log/2026-09-21-tidyhouse-16plan-panel.md)。

## CPU 日志重算

[48集逐行结果和SHA](results/feedback-structure-2026-09-21/native-subtasks.json)全部与
归档完成对象数一致。工具 scripts/summarize_native_subtasks.py 按原生指针重算，
缺失日志为null，不把重复工具成功计成多个物体。reached表示收到过动作的subtask。

| 组别 | Navigate完成/计划 | Pick完成/计划 | Place完成/计划 |
|---|---|---|---|
| Fixed | 49/160 | 20/80 | 14/80 |
| GPT | 47/160 | 21/80 | 13/80 |
| Teleport | 67/160（均为跳过导航） | 30/80 | 21/80 |

统计独立单位保持episode；20个subtask相关，不能把16集当成320个独立样本。

## 站位先验与后续验收

已实现冻结先验消费接口：必须记录训练plan UID、非空bins、样本数/成功数，
与完整evaluation manifest核验无重叠，加载时记录SHA并持有只读语义的副本。
GPU扫描器与4-plan配对pilot尚未实施/启动；属于后续单独冻结预算和协议的工作。
采样器须使用非评测plan，排除全部评测UID，保存相对位姿、全部失败原因及样本数、
Wilson区间；在线评测只读先验，不现场试姿态挑最好。先验是独立消融变量。

## 文档保留

日常只看根README、本页、源码索引、实验日志。docs/results、docs/media、历史诊断
和冻结协议保留；过期计划不作为当前指令。已落实的提案删除，剩余GPU事项记录于上节。
修改前源码/测试/提案保存在 runs/pre-feedback-structure-*.zip，可恢复删除的提案。
本次只实施CPU代码/文档，未启动新的GPU或API实验。

验收：完整CPU测试664 passed、6 skipped；保留一条已有WebSocket弃用警告。
备份文件：runs/pre-feedback-structure-20260921T195948.zip（含已删除提案）。
