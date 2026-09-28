# 实验日志

当前实验日志范围以 `BVI-research-plan-2026-09-14/Weekly/2026 9 28.md` 为准；未提到的实验日志已移除，周报中缺少对应日志的旧实验不补建。16-plan 与 three-settings 两篇对应同一项三设置实验。
实验日志必须让没有打开原始 JSON 的读者直接回答四个问题：为什么做、具体怎么配、
结果是什么、结论能说到哪里。不要写自我描述，也不要把
运行过程、PID、绝对路径或事件流复制进来。

## 当前日志

- [TidyHouse 三设置：Fixed / GPT / Teleport](2026-09-21-tidyhouse-three-settings.md)
- [C2 Pick 站位能力与 GPT 选择](2026-09-23-c2-sac-pose-capability.md)
- [C2 结构化失败反馈与 Pick 站位选择](2026-09-23-c2-failure-feedback-selection.md)
- [C2 真实导航交接 SAC Pick 基线普查](2026-09-25-c2-real-handoff-pick-census.md)
- [视觉空间理解与站位判断开发集校准](2026-09-28-c2-spatial-understanding-cap2000-pilot.md)
- [负结果备忘](negative-results.md)
- [新实验模板](TEMPLATE.md)

有日志的实验设计见 [docs/design/](../design/README.md)。

## 规则

1. 同一问题的多个 setting 在一篇实验日志中用表格对比；Q1–Q5 等题目用列表解释。
2. 设置表说明输入、提示、动作及真值使用；结果表每格一个指标，最多六列，注明分母和单位。
3. 均值必须说明离散程度与样本数；本仓库统一报告总体标准差（population SD）。
4. 机器记录只保留最小公开摘要，并从摘要生成派生数字。
5. 负结果必须保留结论和可能原因，但不保留成百上千个中间文件。
