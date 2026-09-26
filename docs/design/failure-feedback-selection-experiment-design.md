# C2 第二实验：结构化失败反馈是否改善候选站位选择

## 要回答的问题

同一个 Pick 的 C0 SAC 已失败时，向 GPT 增加一张**由真实 C0 轨迹确定性生成的失败卡片**，是否会改变它从固定新候选中选出的站位，并提高所选候选的官方严格 Pick 成功数？这是输入信息实验；候选站位、SAC 和评分表均固定。

本批只做探索性离线选择。候选的 SAC 结果来自各自独立重建的原交接状态；C0 力超限或时限耗尽后，不能在同一 episode 内把那些结果当作在线恢复成功。[第一实验记录](../log/2026-09-23-c2-sac-pose-capability.md)保留原样。

## 固定样本与执行条件

| 项 | 冻结规则 |
|---|---|
| 样本 | A1 中 C0 三次结果一致且失败的 seed8、seed4、seed9，三者都保留；seed19 的 C0 成功，无须触发失败后选择；seed10 未取得可重复 C0 |
| 候选与标签 | 沿用 A2 已执行的新候选及官方严格 Pick 结果。seed8、seed4 各 0/4，seed9 为 5/6；无效候选不进入选项，不增加或替换站位 |
| 共同输入 | `c0_outcome=failed`、当前 Pick 目标及持物状态、候选相对底盘位移和几何量、标明 C0/候选编号的头部与手部 SAC 前拼图；两组完全相同 |
| 模型与输出 | `gpt-5.6-luna`，`reasoning_effort=none`，两张图片 `high`，每次最多 2048 输出 token；只允许一个有效 `candidate_id` 和简短理由 |
| 隔离 | 输入只可来自 C0 轨迹和候选 SAC **前**的信息。A2 候选结果、Oracle、GPT 旧回答均不得进入输入或失败卡片。六份完整请求及图像哈希须在任何新模型调用前一起冻结，评分在六份回答落盘后进行 |

## 两个 setting

| setting | GPT 收到什么 | 相对上一组唯一变化 |
|---|---|---|
| B0 基础反馈 | 上表共同输入 | 起点 |
| B1 结构化失败反馈 | B0 全部输入，另加 `c0_failure_card` | 只增加一张 C0 失败卡片 |

失败卡片从官方 C0 逐步记录直接计算，固定字段为：`sac_steps`、`termination_reason`（力超限或官方剩余时限耗尽）、`final_robot_cumulative_force`、`force_limit`、`grasped_at_end`。字段缺失时显式填 `null`，不让模型或人工补写原因；不用候选结果筛选内容。本实验不叠加轨迹关键帧或旧 40 步经验片段。

## 完整指令与真实输入样例

两组共用以下 system prompt；每个状态的合法候选由 schema 的 `enum` 限定：

> 你要为一次 C0 已失败的 Pick，选择重新开始时最可能让同一 SAC 成功的新底盘站位。图片展示的是原冻结交接中 C0 和各候选在 SAC **执行前**的相机画面，不是 C0 失败后的现场。只能从 `eligible_candidates` 返回一个 `candidate_id`；不得选 C0、输出连续坐标、其他技能或第二次尝试。你看不到任何新候选的 SAC 结果。请根据输入给出简短可观察理由。只输出 JSON：`{"candidate_id":"C1","reason":"简短理由"}`。

下例是 seed9 的 B1 输入，数值来自真实候选与 C0 记录；实际请求中的浮点数统一保留三位小数。B0 输入只删除 `c0_failure_card`，其他字段与两张图像保持相同。两张 PNG 作为图像内容传入，文件名用于核对附件。

```json
{
  "case_id": "case-pick-03",
  "skill": "pick",
  "target_id": "object-7df706aa8cd2",
  "held_object": false,
  "c0_outcome": "failed",
  "pre_sac_official_pick_steps_left": 199,
  "images": {
    "fetch_head": "seed9-fetch_head.png",
    "fetch_hand": "seed9-fetch_hand.png"
  },
  "eligible_candidates": [
      {"candidate_id": "C1", "local_delta": {"forward_m": 0.2}, "target_distance_xy_m": 1.696, "target_relative_angle_rad": 0.087},
      {"candidate_id": "C2", "local_delta": {"forward_m": -0.2}, "target_distance_xy_m": 2.094, "target_relative_angle_rad": 0.070},
      {"candidate_id": "C3", "local_delta": {"left_m": 0.2}, "target_distance_xy_m": 1.921, "target_relative_angle_rad": 0.181},
      {"candidate_id": "C4", "local_delta": {"left_m": -0.2}, "target_distance_xy_m": 1.890, "target_relative_angle_rad": 0.028},
      {"candidate_id": "C5", "local_delta": {"yaw_rad": 0.35}, "target_distance_xy_m": 1.895, "target_relative_angle_rad": 0.427},
      {"candidate_id": "C6", "local_delta": {"yaw_rad": -0.35}, "target_distance_xy_m": 1.895, "target_relative_angle_rad": 0.273}
  ],
  "c0_failure_card": {
    "sac_steps": 26,
    "termination_reason": "cumulative_force_limit",
    "final_robot_cumulative_force": 10732.163,
    "force_limit": 5000,
    "grasped_at_end": false
  }
}
```

候选图像见[已冻结的 A3 图像](../media/README.md)；A3 的旧回答不作为本次 B0。

输出 schema 在每个状态使用相同结构，仅 `candidate_id.enum` 等于该状态的有效新候选 ID：

```json
{"type":"object","properties":{"candidate_id":{"type":"string","enum":["C1","C2","C3","C4","C5","C6"]},"reason":{"type":"string","maxLength":160}},"required":["candidate_id","reason"],"additionalProperties":false}
```

## 执行与判据

先完成失败卡片逐字段来源核对、六份输入冻结和无候选结果泄漏检查；再对每个状态、每个 setting 各发 **一次**请求，共最多 6 次，零重试。无效输出记 `model_protocol_failure`，不人工代选。请求顺序固定为 seed8 B0/B1、seed4 B0/B1、seed9 B0/B1；模型、图像、候选 ID、输出限制和评分口径不变。

主表逐状态列出 B0/B1 所选 ID 及其 A2 严格 Pick 结果，并列出候选成功比例与事后 Oracle。比较同一状态中 B1 相对 B0 是否选得更好；协议有效率单列。seed8、seed4 的候选全部失败，不能用它们判断 GPT 是否能找到成功站位；本批真正有成功/失败选择差异的只有 seed9，因此只作探索性证据，不做显著性或泛化结论。若 B1 改选成功候选，只说明这份卡片在该状态有用；若没有改进，也不能据此判定失败反馈总体无用。

## 资源与记录

复用已有图像与 A2 标签，**新仿真、GPU、训练和 RunPod 均为 0**。执行模型调用前，须另按财务规则确认最多 6 次付费请求、每次最多 2048 输出 token、USD 0.15 预留上限；预留额度不是实付金额。完整 prompt、请求 JSON、图片哈希、原始回答、provider 用量和结果写入新的 run 目录；阶段数值与异常写入结果记录，不回填本设计。
