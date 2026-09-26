# 第三步 Pick 前门控输入与输出契约 v1

> **执行后状态（2026-09-26）：** V1 开发输入和模型格式已完成 60/60 次请求；下文保留调用前契约原文。G3 全部选择干预，结果见[开发诊断](../diagnostics/2026-09-26-real-handoff-pick-gate-development.md)。

**状态：CPU 草案，未调用模型。** 主协议见[第三步设计](real-handoff-pick-gpt-handoff-check-experiment-design.md)。模型、分辨率、超时、token 上限及本文件哈希在首次付费开发请求前冻结；验收集开始后不得修改。

## 单次请求

请求发生在真实 PPO Navigate 完成、官方 SAC Pick 尚未执行任何动作的同一帧。输入只含以下字段：

```json
{
  "skill": "pick",
  "target_object_id": "<本次 Pick 对象实例 ID>",
  "frame_id": "<交接时当前帧 ID>",
  "head_rgb": "<同帧头相机图像>",
  "wrist_rgb": "<同帧腕相机图像>",
  "geometry_source": "privileged_simulator_at_handoff",
  "base_target_horizontal_distance_m": "<数值>",
  "target_bearing_left_rad": "<数值；以 base_link 为原点，左为正>",
  "target_height_m": "<数值>",
  "tcp_target_distance_m": "<数值>",
  "head_target_visible": "<若第 0 步原生输入可稳定取得则布尔值，否则 unavailable>",
  "allowed_intervention": "forward_10: 当前朝向前进 0.10 m；需通过路线及到位检查，耗用同一 Pick episode 的步数和累计力"
}
```

图片、几何、对象 ID 和帧号必须绑定到同一交接；不得附带普查结局、失败机制、可救回标签、后续 SAC 轨迹、其他候选的成功信息、Oracle 选择或验收集结果。`head_target_visible` 如来自仿真分割，必须另标 `source=privileged_simulator_segmentation`；不能把它当纯视觉标注。输入编译器对不存在、非有限或不同帧的字段应拒绝请求，不补造数值；若真实交接身份和官方 Pick 仍有效，该次模型门控按“不干预”计入全部交接分母，记录 `input_unavailable`。只有交接身份或官方评估本身无法核证才记接口删失。

系统指令的语义固定为：根据当前图像和标明来源的几何信息，估计**如果立即沿原站位继续官方 SAC Pick**严格失败的可能性，然后决定是否在 SAC 第 0 步先执行唯一允许的 `forward_10`。必须同时考虑移动可能破坏原本成功轨迹以及路线、步数和力预算。不能输出其他动作，也不能把预测终点当作已观察图像。理由只引用输入中的当前证据，不猜测未来接触或任务结果。

输出必须符合以下 JSON 对象；不允许额外键：

```json
{
  "p_fail": 0.0,
  "intervene": false,
  "reason": "<不超过 200 个字符的当前证据说明>"
}
```

`p_fail` 是原站位直接续跑 SAC 的严格 Pick 失败概率估计，有限数且在 `[0,1]`。`intervene` 是 GPT 原生门控决策；次级阈值组 G4 对同一次响应的 `p_fail` 应用开发集冻结阈值。格式错误、超时或 provider 错误均按“不干预”计入意向分析，保留原始错误和请求计数，不重试。`p_fail` 的 Brier 分数与原站位基线结果比较，需注明基线复跑随机性。

开发集可以调试请求格式，但须保留每次尝试、token 用量和费用回执。验收集只使用最后冻结的版本，不能依据验收结局改提示词、阈值或图像处理。
