# BVI / MS-HAB 高层 VLM 统一接入方案

状态：2026-09-19 编写，面向 9/20 交付；仅方法设计，未实现新高层模块、未调用机器人运行时 VLM API、未开展论文对比实验。

## 结论与具体例子

让高层模型从示范和真实执行记录中判断下一次调用，把连续控制交给已有低层策略。当前 AC-DiT 只应作为一个待验收的 `apple_pick_episode` 工具：高层请求“完成苹果 Pick”，工具在原生动作上限内执行，返回实际步数、停止原因和新观测。高层不能要求尚不存在的独立 `reach`、`grasp`、`move` 或 `place` 工具；控制器接受请求也不意味着苹果已经抓取并回收停稳。

例如，某次请求被拒绝且执行步数为零，下一轮上下文必须保留“拒绝 + 原因 + 当前观测”，不能写成“已尝试抓取”。若执行若干动作后累计力超限，应记录原生失败并结束本次环境，不能让高层绕过终止条件继续动作。整 episode 工具目前不能提供任意中途换指令或可安全恢复的局部调用能力。

## 原论文机制、项目事实与提案

| 依据 | 原文机制 | 已核实项目能力 | 本方案中的未验证适配 |
|---|---|---|---|
| VLAs-as-Tools §3、§4.1–4.2 | 调用 `(g,z)` 指定工具族及局部要求；有界执行返回反馈。TAPT 使监督窗口与调用边界一致，族残差适配器让 g 控制参数路径，辅助进度头预测当前调用进度，监视器据此继续或重规划 | 本轮是单任务普通 AC-DiT 两阶段训练，没有新增调用分段、族适配器或 learned-progress 训练 | 借用双向调用接口组织系统；不能把普通策略外包一层 planner 称为完整 TAPT |
| RoboICL §2、§4 | GPT 直接生成低层双臂 Cartesian 动作；参考和在线记录共用观测→请求→执行反馈→新观测语法。原记忆按时间目标保留完整交互块，并显式标记遗漏区间 | 尚无这种高层上下文编译器 | 把同一语法提升到工具调用层；按工具边界保留锚点，而非照搬原动作时间锚点。有效性未经实验验证 |
| GPT-Policy §3、附录 D | 区分异构任务参考与在线历史；编译带来源和视角的上下文，生成结构化请求，接收执行或拒绝反馈。原工具包括 Cartesian 路径与夹爪控制，不是 learned VLA 工具 | 有低层推理/原生环境代码，尚无本方案的上下文层 | 保留请求校验与反馈机制，把具体执行器换成已验收的任务策略；不能继承其低层控制成绩 |

RoboICL 对锚点策略的独立贡献仍提出消融需求；这里不能声称“锚点已证明提高 BVI 成功率”。GPT-Policy 附录 D 的历史示范不是当前动作目标；本项目也应明确区分历史参考和在线命令。[R2][R3]

## 模块与信息流

`任务 + 当前观测 + 固定示范 + 保留的执行历史 → 上下文编译 → 高层 VLM → 校验后的工具调用 → 受限低层执行 → 实际执行/拒绝反馈及新观测 → 下一次决策`

| 模块 | 输入与来源 | 输出及边界 | 当前状态 |
|---|---|---|---|
| 参考装载 | 预先选定的示范、观测时间戳、视角、真实执行记录、来源哈希 | 固定参考集，与本轮 LIVE 历史分开。没有真实调用边界的 SAC 连续轨迹不能伪装成高层调用示范 | 数据存在；调用级示范编译未实现，本轮不新采集 |
| 历史装载 | 已完成或已拒绝调用的日志、执行步数、终止事件、新观测 | 保留初始交互、最近交互和有上限的边界锚点；省略区间记录起止时间/调用 ID，不伪造连续轨迹 | 方案 |
| 上下文编译 | 任务、固定参考、当前 RGB/允许状态、带来源的历史 | 有序多模态输入，区分 REQUESTED / EXECUTED / OBSERVED；预算不足时删整块并留下 gap，不拆散请求与反馈 | 方案 |
| 高层 VLM | 上述编译结果、可用工具表、剩余调用预算 | 一个候选 `(g,z)` 及引用的证据 ID；拒绝输出未注册工具 | 方案；本轮无在线调用 |
| 校验层 | 请求、工具能力清单、活动调用 ID、环境是否已终止 | 接受或拒绝；检查对象/协议、重复调用、动作与时间上限，不把任意语言转成不支持的技能 | 需要独立实现及测试 |
| 低层执行 | 锁定权重/输入合同；见下一节 | 实际动作、执行前缀、未执行后缀、停止原因；不允许请求放宽原生判据 | 旧入口存在，本次候选入口未验收 |
| 反馈层 | 控制日志、真实后观测、可选且标明来源的 simulator 谓词 | 返回执行事实与证据；评分、oracle、learned progress、VLM 判断分别命名 | 原生日志存在；统一反馈格式为方案 |

建议记录格式（字段设计，不是已实现 API）：

```json
{
  "source": "reference_or_live",
  "call_id": "unique_id",
  "before_observation": {"artifact": "path", "timestamp": "UTC", "views": ["head", "hand"]},
  "requested": {"g": "apple_pick_episode", "z": "allowlisted_training_instruction"},
  "executed": {"accepted": false, "steps": 0, "action_log": null, "stop_reason": "unsupported_request"},
  "after_observation": {"artifact": "path", "timestamp": "UTC"},
  "feedback": {"source": "controller", "native_success": null, "learned_progress": null},
  "provenance": {"model_sha256": "required_when_available", "config_sha256": "required"}
}
```

如果后来有调用级示范，应以实际调用开始/结束切片，包含失败、拒绝和部分执行，不能从动作命令推断机器人完成了什么。人类示范只有图像时保留为视觉参考，不能补写未记录的机器人执行动作。

## 当前 AC-DiT 能接受什么

以下来自本轮下载的真实服务器代码快照，而非仅凭任务卡推断：

- `train_acdit_rdt_bounded.py` / `train/dataset.py` 使用预计算语言 embedding；Apple Pick 有 15 个 embedding 文件，其哈希在 `remote-source-provenance.json`。这证明存在语言条件输入，不证明理解任意局部指令或跨任务泛化。
- 训练窗口预测两步动作，输入两帧图像历史、三相机槽位（head、hand、空槽）、点云、13 维真实关节/底盘速度映射到 128 维 universal state、频率及 18 维 privileged context。两步 action chunk 不等于两步语义调用监督。
- 18 维 context 由 goal position（3）、is_grasped（1）、object pose（7）、TCP pose（7）组成。原生输入与动作合同不能在评测时默默删去这些条件。
- 阶段一学习底盘；阶段二全身模型接入冻结底盘 DiT 和对应五组输入投影，并转移点云编码器。点云编码器并未被这段转移代码显式设成全部冻结，应按实际 `requires_grad` 记录，不把“转移”混称“全部冻结”。
- 训练只绑定整任务 Apple Pick，没有本轮 TAPT 调用标签/族路由。因此当前注册表只能提出整个 episode 的 Pick 工具，且先标记 capability unvalidated。LightNav 的历史能力不证明本次交接，Place 不在本轮能力清单。

## 信息可见性

| 信息 | 高层提案可见 | 低层实际使用 | 教师/评分用途 |
|---|---|---|---|
| 当前 head/hand RGB、时间戳、允许的 proprioception | 是，视角和来源显式标记 | RGB 历史和 state | 可归档 |
| 点云、实际底盘速度、动作 mask、频率 | 默认不把完整张量塞入高层 | 是 | 输入审计 |
| goal/object/TCP 位姿与 grasp flag | 默认不暴露原始 privileged state | 当前 AC-DiT 使用 18 维 context | 仿真/教师提供；不能称纯视觉部署 |
| native success、累计力、终止谓词 | 若用于重规划，明确标记 `simulator_oracle` 条件 | 不作为额外新输入注入模型 | 保留原生评分；不能声称与 RoboICL 的信息屏蔽条件一致 |
| reward、评测答案、test 布局元数据 | 不进入拟议高层输入 | 无新增注入 | 仅评分/离线审计 |
| learned progress | 本轮无合格新头；null | 本轮未训练 | 未来须独立校准 |
| VLM 自评完成 | 仅判断，不能改写 success | 无 | 不能替代原生成功 |

## 失败条件与后续证据要求

没有可加载全身权重、输入缺字段、来源不明或已终止环境时，拒绝执行。未知进度保留 unknown，不以请求被接受代替成功。动作/时间预算耗尽返回 timeout 或 native truncation；基础设施异常单独分类，保留计划分母。若引入高层 oracle 反馈，必须单列实验条件。

优先证据是本次整 episode Pick 候选在固定开发回归面板的原生结果和成功复跑；此门未完成前，无证据说明上下文设计能弥补低层能力。未来讨论所需的最小证据包括：边界对齐日志正确性、请求/执行差异保真、gap 与锚点消融、同低层权重同预算的上下文比较、错误交接与信息泄漏审计。这里没有执行这些实验，也没有申请扩训或 API 预算。

## 原文版本与可复核位置

- [R1] 本地 `papers/vlas-as-tools-2605.13119.pdf`，arXiv:2605.13119v1；读取 §3、§4.1、§4.2 的文本，并查看已有 Fig.5 调用标签图。论文的监督分段不是当前 Apple Pick 训练已完成的事情。
- [R2] [RoboICL 作者报告](https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/)，页内证据日期 2026-09-18，读取 §2、§4。web 阅读器访问失败后通过 HTTPS 取得完整 HTML，存入本批 `papers/roboicl.html` 和文本快照；未从摘要补写机制。
- [R3] [GPT-Policy arXiv v1 HTML](https://arxiv.org/html/2609.19138v1)，读取 §3 和附录 D，并核对[作者项目页](https://cheng-haha.github.io/GPT-Policy/)。指定 PDF 只取得受限的前 25,000,000 字节，保留为私有 `.partial`，不作为已读全文；本轮未声称 PDF 与 HTML 逐字一致。完整 HTML/文本快照在本批 `papers/`。

代码、来源快照与哈希索引：[结果目录](results/acdit-apple-delivery-2026-09-20/README.md)。
