# 真实交接末端工具闭环抓取与上下文重试（设计草案，对齐 GPT-Policy）

**状态：草案，未冻结、未授权。** 本文没有执行任何 API、GPU 或仿真调用。来源是 2026-09-28 组会（笔记见 `BVI-research-plan-2026-09-14/Weekly/2026 9 28.md`）以及同日与用户的讨论。已决定事项汇总在文末，其中最重要的三点是：

- 本周跳过 SAC；
- 笔记中的 "nfactor control" 指末端执行器（end-effector）控制；
- 整体设计与 GPT-Policy 对齐。

## 研究问题

- **主问题**：在真实 PPO Navigate→Pick 交接处，Astra 不调用 SAC，只通过末端工具接口完成 Pick。它自己输出末端轨迹，拿到 IK 路径检查和执行反馈后再调整。这种方式能否达到官方严格 Pick 成功？
- **上下文学习问题**：同一交接失败后重试时，给 Astra 看此前各次尝试的轨迹、反馈、结局和教训，能否用更少的尝试次数做对？
- **不在本设计内**：prompt 以外的学习形式（Skill Pool、RPent 式记忆提炼）。这是组会定的长期方向，本周不做。候选方案留存在[跨尝试记忆系统候选方案](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/astra-memory-system-options.md)。

## 参考实现与对齐范围

主要参考 **GPT-Policy**（`cheng-haha/GPT-Policy-Eval`，arXiv 2609.19138，In-Context Robot Learning with VLM Agents）。该项目许可证尚未确定，也没有授予再分发权，所以**只对齐接口语义，自行实现，不复制代码**。

| GPT-Policy 做法 | 本设计 |
|---|---|
| 模型每轮只输出一个工具调用；程序执行完这一段后，返回新观测和上一步结果 | 相同 |
| 模型输出底座坐标系下的绝对末端位姿（位置＋单位四元数），不输出关节角 | 相同，坐标系换成 Fetch `base_link` |
| `move_eef_chunk`：保留每个 waypoint 及其顺序，只沿直线/SLERP 段加密采样、逐点解 IK、重排时序 | 相同；IK 结果再换算成官方 `pd_joint_delta_pos` 动作执行 |
| `check_path`：只读，只查 IK、关节限位和时序，不发指令，**不查碰撞**；要求不为通过 IK 而改变抓取方式 | 相同。导师 2026-09-28 裁定：IK 可以用，不算作弊（真实机器人也能在自身运动学模型上解 IK）；场景网格碰撞检测仍算作弊 |
| `locate_point`：像素→相机射线；无深度，需两视角三角化 | **改为深度反投影**：像素＋深度＋内参＋由 URDF 正运动学得到的相机外参 → `base_link` 下 3D 点 |
| `set_gripper` 单独调用，走轨迹时夹爪不变 | 相同 |
| `done` / `give_up` 必须带 hindsight | 相同；hindsight 写进下一次尝试的历史 |
| 历史示范按"关键帧图像＋末端目标/实测位姿＋夹爪事件"编排，并声明只作参考、不要盲目回放 | 相同；示范来源是其他交接的 SAC 成功轨迹 |
| 一次任务就是一个连续的会话 | 一次尝试从冻结交接快照开始；重试从同一快照重新开始，并附带历史 |

其他参考：

- **GPT-as-Policy**（银河通用）：Astra 只用末端控制（Direct），每次移动不超过 5 cm／0.35 rad，严格成功率 26%。Astra 审查并修正 π₀.₅ 动作的混合方式为 48%。这提醒我们纯 Astra 控制可能偏弱。小步闭环不作为本设计的主条件。
- **RPent / Harness VLA**：记忆提炼加失败重试，供长期的 Skill Pool 方向参考。

## 输入边界

以 `AGENTS.md` 中 "C2 GPT input policy (2026-09-27)" 为准，另加导师 2026-09-28 对 IK 的裁定：

- IK 可用：在机器人自身运动学模型上，对模型提出的末端位姿做检查、拒绝或执行，不算作弊；
- 仍然禁止：给 IK 输入场景真值（例如目标真值位姿），以及基于场景网格的碰撞检测，因为真实场景的网格坐标拿不到。

| 类别 | 内容 |
|---|---|
| **主条件允许** | 头部、手部相机 RGB-D；关节角与角速度；由关节角正运动学算出的 `base_link` 系末端位姿；夹爪状态；目标语义描述；机器人常量：休息位、末端（TCP）定义、相机内参、URDF 外参；`check_path` 的 IK／限位／残差反馈；执行反馈：目标与实测末端的差、到位与否、实际用掉的环境步数；重试时，此前各次尝试的官方结局与终止类别 |
| **主条件禁止** | 仿真物体位姿、目标相对真值、世界坐标、导航网格、碰撞网格、基于场景网格的碰撞检测、以场景真值为输入的 IK、第三人称相机、尝试内回滚、调用 SAC |
| **特权消融（E1）** | 额外提供 `base_link` 系下的目标真值位置，并标注为特权，不进主成绩 |
| **评估端** | 自由使用真值：严格成功判定、失败归类、G2 的反投影误差 |

相机外参沿用 `scripts/audit_real_handoff_spatial_camera_calibration_cpu.py` 的做法：由 URDF 和关节角计算，世界位姿只用于核对正运动学，不进请求。

## 工具接口（Fetch / MS-HAB 版）

| 工具 | 参数 | 程序行为与返回 |
|---|---|---|
| `locate_point` | 相机（`head`／`hand`）、像素 `[x,y]` | 深度反投影到 `base_link`；返回 3D 点、深度是否有效、邻域深度离散度。深度无效时如实说明，不补值 |
| `check_path` | 末端位姿列表 | 与执行相同的加密和 IK；逐点报告 IK 是否有解、关节限位、残差；不发指令、不查碰撞 |
| `move_to` / `move_eef_chunk` | 单个／多个绝对末端位姿 | 加密→逐点 IK（以实测关节角为初值）→按每步动作上限拆成 `pd_joint_delta_pos` 动作执行；返回规划点、实测末端、剩余误差、用掉的步数；遇官方终止立即返回终止类别 |
| `set_gripper` | 开度 0（闭）～1（开） | 执行后返回命令值与实测指距；抓住与否不直接告知 |
| `return_to_rest` | 无 | 在关节空间把躯干、头部、手臂移回 Fetch 休息关键帧的关节角，夹爪保持当前命令；返回实测关节角与休息位的最大偏差和用掉的步数。只用机器人常量，相当于真机上的"回原位"按钮 |
| `done` | 完成依据 `summary`、教训 `hindsight` | 结束本次尝试；由官方判定成败，模型的 `done` 不算成功标签 |
| `give_up` | 原因 `reason`、教训 `hindsight` | 结束本次尝试 |

- **执行方式（已定）**：程序自己解 IK，再换算成官方关节增量动作。这样控制器、累计力上限和成功判定都与 SAC 基线相同，`check_path` 和执行也用同一个 IK。
- **底盘**：主条件本周固定在交接站位，不提供底盘工具。瞬移换站位作为备选分支另行设计。
- **IK 自由度（已定）**：手臂 7 个关节加躯干升降，共 8 个自由度。
  - 休息关键帧中躯干处在最高位 0.386 m。普查失败组目标平均高 0.53 m，成功组 0.69 m，可重复失败中有 4 例够不到；降躯干是够到低处目标最直接的手段。
  - 官方 SAC 本身控制躯干，锁住躯干会让 Astra 比 SAC 少一个自由度。
  - Astra 只输出末端位姿，躯干怎么动由 IK 分配，不增加模型负担。
  - 头部两个关节不参与 IK，保持交接时的角度。
  - G0 同时跑"只用手臂 7 关节"的对照；两者差别不大时再简化。
- **严格成功条件**：MS-HAB Pick 要求以下五项同时成立：
  - `is_grasped`：抓住目标；
  - `ee_rest`：末端距休息位 ≤ 0.05 m；
  - `robot_rest`：躯干、头部、手臂**全部关节**（`qpos[3:-2]`）都在容差内回到休息关键帧；
  - `is_static`：机器人静止；
  - 累计力 < 5000。
  
  手臂有冗余，用 IK 把末端移回休息位不保证关节角也复原，所以提供关节空间的 `return_to_rest`。提示里写明：抓住并抬起后调用 `return_to_rest`。

## 尝试、重试与历史

- **一次尝试**：从冻结交接快照恢复，进入工具闭环，遇到 `done`、`give_up`、官方终止或工具调用上限即结束。官方 Pick 剩余时限照常计算；模型思考期间仿真暂停，不耗环境步。
- **重试（已定）**：从同一快照开始新的一次尝试，每个交接最多 **K=3** 次，成功后不再继续。
  - 每次尝试最多 **20 次工具调用**。一次正常抓取约需 8–10 次（定位 1–2、检查 1–2、接近与对准 2–3、闭合夹爪、抬起、回休息位、结束），20 次约留一倍余量用于尝试内纠错。
  - 定位和路径检查不耗环境步；环境步由官方 Pick 的 200 步时限约束（SAC 成功平均约 40 步），所以调用上限主要用来控制费用。
  - 3 次是能看出趋势的最少点数。若开发集上 H1 到第 3 次仍在上升，测试集冻结时改为 K=5，并在运行前写明。
  - 最坏情况每个交接 3 × 20 = 60 次调用。

历史条件：

| 条件 | 上下文里附带什么 | 对应组会 |
|---|---|---|
| **H0 无历史** | 每次尝试都从零开始，提示相同 | 对照：只靠随机性的重试 |
| **H1 自身历史** | 同一交接此前各次尝试：夹爪事件前后的关键帧、规划与实测末端轨迹、`check_path` 被拒记录、结局类别、hindsight | 第二步：输出轨迹、拿到反馈、再调整 |
| **H2 H1＋他例示范** | 再加其他开发集交接中 SAC 成功的轨迹，按 GPT-Policy 的 video+action 格式编排 | 第一步：参考之前做过的 trajectory |

- **泄题边界**：同一交接自己的 SAC 成功轨迹不得进入 H2，只可作上限消融。测试集的示范只能来自开发集。
- **结局反馈（用户 2026-09-28 确认合规）**：历史中告知每次尝试的官方结局（成功／失败）和终止类别：时限用尽、累计力超限、模型调用 `done`／`give_up`、工具调用达到上限。确认范围只到这里；`is_grasped`、接触物体身份和任何几何真值都不给，如要加入须另行决定。
- **随机性**：普查中 20 个失败原样复跑，有 4 个成功，所以 H1、H2 的收益必须和 H0 比，不能和"第 1 次尝试"比。

## 前置门（不花 API，必须先过）

> **2026-09-29 调整（用户决定）：**
> - 原 G0"回放 SAC 轨迹"已停止，改为工具冒烟测试，随后先做 E1 特权试点，见[冒烟测试与 E1 试点设计](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-tool-smoke-e1-pilot-design.md)。停止原因：回放考的是"模仿 SAC 的动作风格"，不是 Astra 使用工具的方式。
> - 前几轮记录保留在诊断目录。
> - 执行器保留三条经验：沿路径连续跟踪、关节空间回休息位、IK 选解偏好与防跳解。
> - G1 须用新执行器 v3 在 CPU 上重做，作为硬门。v3 的 IK 已加入选解偏好，旧版 dogbox 的 108/108 不能代替。
> - E1 不开放 `locate_point`；G2 改为在 E2 之前完成；G3 在 E1 之前完成。
> - 2026-09-29 v3 CPU 硬门回执：59 项测试、坐标/动作映射、第 11 号回归通过，G1 107/108 未过；尚不能申请冒烟 GPU。见[CPU 诊断](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-09-29-eef-tool-smoke-cpu/README.md)。
>
> 下表 G0 一行为原定义，仅作历史记录。

| 门 | 内容 | 通过标准（冻结前再定具体数值） |
|---|---|---|
| **G0 执行器上限（已停止）** | 从快照重跑开发集中 SAC 成功的交接，补记每步完整末端位姿（位置＋朝向）、关节角和夹爪（普查只记了末端位置）。把轨迹压成关键帧，用本设计的执行器回放，看能否拿到官方严格成功 | 回放成功率接近 SAC 自身的复跑成功率；否则先修执行器，不进入 API 阶段 |
| **G1 IK 检查可信度** | G0 中实测到达过的末端位姿全部送进 `check_path` | 应全部通过。现有有界 IK 检查在 Q5 中对 14 个起点都没找到完整抓取路径，还在 SAC 实际成功的 seed19 C4 上找不到可行解；要找出原因，不能沿用 |
| **G2 反投影精度** | 在评估端用真值分割取目标像素，经 `locate_point` 反投影，测到物体表面的距离，并统计深度无效率 | 误差分布要足以支撑抓取（厘米级）；否则主条件改用手部相机近距离定位 |
| **G3 请求合规** | 复用现有请求审计器，确认请求不含真值字段和第三人称图像 | 不合规的请求不发送 |

TCP 定义（MS-HAB `agent.tcp`，即 Fetch `gripper_link`）在 G0 中核对并冻结。G0 还要：

- 统计 SAC 成功轨迹中躯干的实际运动幅度；
- 比较"8 自由度"和"只用手臂"两种 IK 的回放成功率；
- 核对 `return_to_rest` 能否在官方容差内满足 `robot_rest`。

## 实验（每项单独申请授权）

| 编号 | 条件 | 样本 | 目的 |
|---|---|---|---|
| E1 | 特权：给目标真值位置（包围盒中心，只在开始给一次），H0，单次 | 开发集少量交接 | 已知目标位置时，Astra 能否通过工具完成严格 Pick。只排除目标定位误差，朝向、尺寸、抓取位置和障碍仍须 Astra 自己判断 |
| E2 | 主条件，H0，单次 | 开发集 | Astra 末端闭环本身能做到什么程度 |
| E3 | 主条件，H0／H1（／H2），K 次 | 先开发集，冻结后测试集只评一次 | 历史能否减少所需尝试次数 |
| E2-x | 同 E2，推理强度改为 `xhigh` | 开发集 3–4 个交接，各单次尝试 | 推理强度是不是瓶颈 |

- **推理强度（已定）**：主实验用 `medium`。
  - 与阶段 1 一致，已测到的 Astra 空间理解结果可以衔接。
  - 多轮闭环中每次调用都用 `xhigh`，费用和耗时成倍增加。
  - 推理强度未必是瓶颈：RPent 用 Astra low 模式在 LIBERO-PRO 上达到 92.63%，GPT-as-Policy 用 `xhigh` 仍只有 26%。
  - E2-x 与 E2 在相同交接上配对比较。

- **指标**：
  - 官方严格 success@1 与 success@K；
  - 首次成功所需尝试次数（K 次未成功按删失处理）；
  - 每次尝试的工具调用数与环境步数；
  - `check_path` 拒绝率与 IK 残差；
  - 失败归类（够不到、夹取失败、抓后未回休息位、力超限）；
  - token 与费用。
- **参照**：同一批交接上 SAC 首次 Pick 为 40/60（66.7%）。这只是参照，方法不同，不是配对对照。
- **分开报告**：接口或基础设施删失（传输失败、快照恢复失败）单列，不计入 Pick 失败。

## 预算与授权

- 本设计不构成任何付费或 GPU 授权。
- 此前 Astra 全量测评约 USD 0.05／请求，但那批每个请求只有一张图组、不带历史，不能直接外推。多轮会话的上下文会随调用次数增长，单次尝试的费用须在开发集小样本上实测后再定上限。
- 实验室 GPU1 用量按进程秒上限单独申请。费用登记遵守 `BVI-research-plan-2026-09-14/finance/README.md`。

## 已决定事项（用户 2026-09-28）

1. 本周跳过 SAC，整体对齐 GPT-Policy；"nfactor control" 指末端执行器控制。
2. IK 可用（导师裁定）；基于场景网格的碰撞检测仍算作弊。
3. 执行方式：程序自己解 IK，再换算成官方 `pd_joint_delta_pos` 动作。
4. 重试历史中告知此前各次尝试的官方结局与终止类别，合规。
5. IK 用手臂 7 关节加躯干升降，并提供 `return_to_rest`；G0 跑只用手臂的对照。
6. K=3，每次尝试最多 20 次工具调用，成功后停止。
7. 推理强度：主实验 `medium`，加测 E2-x（`xhigh`）。
8. [阶段 1 正例挑战集](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-spatial-stage1-astra-positive-challenge-amendment.md)暂停，GPU1 优先用于 G0。
9. （2026-09-29）停止回放 SAC 的 G0，改为工具冒烟测试，接着做 E1 特权试点（第 5、8、11 号开发交接，H0，单次），见[冒烟测试与 E1 试点设计](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-tool-smoke-e1-pilot-design.md)。躯干由程序按需分配；出现可归因于躯干的失败时锁定躯干。
10. （2026-09-29）底盘纳入 Astra 工具集，原"本周底盘固定"作废。依据：SAC 在 Pick 中移动底盘 0.32–1.06 m；E1 r2 第 11 号在固定底盘下够不到。新增 `move_base`（先转后直行，只用里程计闭环，不做路线预检），接着做 E1-B 特权试点，见[底盘工具与 E1-B 试点设计](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-base-tool-e1b-pilot-design.md)。
11. （2026-09-29，用户要求）**Astra 写推理说明，归因要看它的反馈**，适用于之后所有工具闭环实验。

    **推理说明**：
    - 每条工具调用必须带 `note` 字段，写 1–3 句：依据了哪些证据（哪张图、深度、里程计、上一条工具反馈），以及这条指令想达到什么效果；
    - `done`／`give_up` 继续写 hindsight；
    - 供应商接口若提供推理摘要，一并保存；
    - `note` 只要求说明依据与意图，不附加具体的方法提示（例如"先检查前方间距"）。方法指导是另一个实验因素，须单独决定；
    - 请求审计器放行该字段。

    **失败归因**：报告里要逐条引用或概括导致失败的关键指令对应的 `note` 与 hindsight，再归为以下之一：
    - 模型推理有误：说明里的判断本身错了；
    - 执行偏离意图：工具或控制器没有按说明执行；
    - 计划合理，但被环境挫败；
    - 无法判定。

    `note` 是模型的自我陈述，不一定真实反映其决策过程，要与实际动作和传感器数据交叉核对。

    E1-B v13 没有推理说明，其归因（例如三个场景第一条指令都是前进 0.85 m）只能标为"推测"，不事后让模型补写解释。
12. （2026-09-29，用户决定）**提示加入方法指导，并作为之后所有实验的标准提示。**
    - 内容：机器人常量（由 URDF 计算）与通用推理步骤，不含任何场景真值。推理步骤包括：先用深度估计前方间距再移动；以家具边缘而非目标为停车基准；分段靠近；先用 `check_path` 验证能否够到；预留回休息位的步数。
    - 版本记为 `guidance-v1`，先在 E1-B2 使用，之后 G2、E2、E3 沿用；修改须另记版本。
    - 见[E1-B2 设计](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-e1b2-method-guidance-design.md)。


## 2026-09-29 工具前置门与E1运行进度

用户授权保留IK8自主修复后，v11同版本工具冒烟3/3通过，官方严格Pick0/3单列。E1 r1因会话未接续而全部协议删失并关闭，原记录保留；r2通过previous_response_id保持同次尝试的首轮特权点上下文，各样本从新会话开始，模型/提示/工具及控制器冻结。当前r2第5号已达官方严格Pick（13工具槽、127环境动作、五项判据同时为真，Pick指针1→2），第8/11号仍在原方案下继续；不能将存在成功实例写成总体成功率。完整结果随后写独立diagnostics，E2尚未启动。


### E1最终回执与前置论据更正

上述运行已结束：E1 r2严格Pick1/3（5成功；8时限；11give_up），无删失；r1协议删失保留。工具门3/3不等同抓取率；此特权开发试点仅证实存在Astra成功实例。新增CPU证据说明部分移动失败后的边界缓存会错误拒绝零位移命令，须在后续条件前修复。SAC教师最大底盘移动0.324/0.496/1.059m，因此不能以其成功证明本轮固定底盘可行；不事后改分母。下一道门是接口修复、固定底盘可行性核对和G2，E2未运行。完整[最终结果](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-09-29-eef-e1-pilot-r2/README.md)含逐例、请求审计、源码哈希与财务收尾。

2026-09-29实施进展：底盘工具v12的CPU硬门通过（262项测试、108帧坐标、31份离线请求审计），物理冒烟仍待批准。冒烟20/23/5/8/11，5×180=900进程秒、API0；E1-B新样本预定1/14/26/42与回归5/8/11分别报告，未运行。详细冻结与授权见[底盘提案](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-base-smoke-authorization-proposal.md)。


2026-09-29物理冒烟回执：第20号转向/直行分别12/15步通过；监督器读取缺失strict_pick_success字段后停止，23/5/8/11未运行。实耗47.275746进程秒、API0，旧额度关闭，GPU1空闲。CPU已修复独立r2记录接口（264项通过），v12控制器不变；[剩余四例720秒提案](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-base-smoke-remaining-authorization-proposal.md)待批，E1-B未启动。


### 2026-09-29 底盘冒烟收口补记

保留第20号通过；剩余授权批次23、5通过，8因两调用合并航向误差0.034638 rad超过0.03 rad停止，11未启动。五例门未通过，未进入E1-B。现有证据为相对指令容差累积，不自动归因躯干。控制器、阈值与样本未改；修正方向待用户确认，复测需新有界授权。见[诊断](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-09-29-eef-base-smoke-v12-remaining/README.md)。


### 2026-09-29 净航向衔接修复（用户已确认方向）

冒烟SAC序列的末转改为wrap(冻结净航向−实测里程计航向)，第一调用、净目标及验收门不变，两调用仍共用80步。r3脚本独立保留旧版本；v12底层控制器不变。CPU 275项通过，尚无物理复测结果。5/8/11复测申请3×180＝540进程秒、API0，等待单独授权，旧余额不转入。见[复测提案](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-base-net-heading-r3-authorization-proposal.md)。


### 2026-09-29 r3物理复测结果

5/8通过，8的目标衔接bug在本次回放中验证修复；11第一调用61步，末转耗尽剩余19步，最终0.023691 m/0.051453 rad未过门。未调整门槛、未重跑、未进入E1-B，候选速度暂不冻结为最终参数。用户决定后续控制器修订方向；不自动归因躯干。见[诊断](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-09-29-eef-base-smoke-net-heading-r3/README.md)。


### 2026-09-29 CPU归因补记：第11号接触与漂移

末转K=2的理想速度响应需22步而只剩19步；69–80步评估端采样到forearm_roll_link与外部非目标标签body持续接触，末转20.4 mm漂移主要为横向。未力超限不等于无碰撞。首次零前进转向也有128.6 mm平移；不自动归因躯干。已提出独立v13停稳与转向收敛修订草案，尚未改控制器、启动仿真或申请GPU；原门不变。见[CPU归因](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-09-29-eef-base-turn-drift-cpu/README.md)和[v13草案](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-base-control-v13-revision-proposal.md)。


## 2026-09-30 UTC：E1-B2 CPU 准备与选样阻塞

已冻结 guidance-v1 机器人常量及方法文本，新增 note、终止反思与零步结束接口；原有 361 项加新增 32 项共 393 项 CPU 回归通过。没有 GPU/API/仿真调用，没有新增严格 Pick 成功。

沿用既有排除口径（含 CPU IK 调试暴露），仅剩 55、58、61 三个未用开发交接，均 >1.14 m；四新样本要求未满足，未修改分母、未借用测试集。待用户决定三新＋四配对或保留四新要求暂停。最终网络/部署整合与可启动部署包哈希尚未完成。

详情：[CPU 诊断](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-09-30-eef-e1b2-cpu/README.md)、[阻塞中的提案草稿](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-e1b2-authorization-proposal.md)。八例建议 USD1050 上限，七例备选 USD920，均未授权；旧余额不转入。


## 2026-09-30 UTC 用户选样修订（覆盖上述四新要求）

用户明确选择方案 1：新组 55、58、61 三例（均 >1.14 m），配对 1、14、26、42 四例；分别报告 k/3、k/4。三例为全部剩余未用开发交接；不放宽排除、不借测试集。配对组仍注明指导据此制定、结果偏乐观。此选择仅批准样本修订，不批准 GPU/API 启动。

其余条件与晋级门不变。资源申请改为最多 182 次 API、USD920、GPU1 串行 7×1800=12600 进程秒。CPU 回归 395/395，部署包与源码已冻结；下一道门为[七例启动提案](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-e1b2-authorization-proposal.md)的明确批准。此前“选样阻塞/待决定”段落保留为历史记录，以本修订为准。


## 已决定事项 13：2026-09-30 高清视觉主条件开发试点

用户已批准[visual-v1 / guidance-v2 实施设计](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-visual-pilot-design.md)：四路原生2048 RGB-D保留完整轮内历史，移除首轮特权目标点，新增主动转头与传感器像素定位，只读查询零环境步，v13运动与官方200步不变。先用已暴露开发5/55/58做H0单次诊断，再另提H0/H1重试比较；61及测试集保留。

本决定明确替代“E1-B2必须成功后才允许准备/申请视觉试点”的顺序：独立CPU、GPU接口和G2通过后可申请视觉试点。E1-B2未完成、未过门的原结果与分母保留，不追认通过。新接口与提示同步改变，整体试点不能单独证明分辨率收益。此处guidance-v2覆盖新条件，guidance-v1及旧版本保持原记录。

CPU436/436通过，3快照实验室CPU反序列化通过且未初始化CUDA；实际高清渲染、头部运动和G2仍待验证。下一道门为[API0接口启动提案](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-visual-interface-authorization-proposal.md)的独立批准：GPU1串行3×300=900进程秒。API阶段尚未申请启动，后续预算建议USD950只是完整历史保守估算。证据见[CPU诊断](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-09-30-eef-visual-cpu/README.md)。本轮无新Pick成功，无API/GPU实验/仿真调用，不发布GitHub。


## 2026-09-30：高清接口r5通过

453项CPU回归通过，5/55/58真实接口与双相机G2通过；API0，未运行Astra、未新增严格Pick成功。录像直接读相机避免状态副作用，主动头部与保持通道限幅分离；v13底盘与原40/200步门不变。guidance-v3完整轮内历史预算更新为USD1000.8816保守预留，独立提案申请1010（取代旧950估算，不构成授权）。见[试点授权提案](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-visual-pilot-authorization-proposal.md)和[接口诊断](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-09-30-r5-eef-visual-cpu/README.md)。


## 2026-09-30：用户改为640与近期图片窗口

用户明确选择640×640实际渲染，当前/上一轮图像+完整精简文字执行记录；不再接续previous_response_id，旧图显式回看，完整档案仍保存。guidance-v4-compact与新模块独立版本，旧条件原结果保留。详见[新设计](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/docs/real-handoff-eef-compact-640-design.md)。CPU474通过，物理接口与模型预算待批；r1/r2原额度关闭。

## 已决定事项 14：2026-09-30 执行器系统对照与基础工作（用户决定）

- **主线不变**：继续以直接控制末端为主线，不另开"Astra 选站位＋SAC 原语"的并行线。
- **先完成三件基础工作**：
  - API 调用搬到实验室服务器，验证是否能降低本机链路的传输删失；网络根因仍未确定；
  - 按普查的冻结流程生成新的开发集交接（开发集 2），测试集仍不动；
  - 冻结当前模型侧条件（视觉主条件、640、`guidance-v4-compact`、现有工具与历史方式），执行器达标前不再修改。
- **执行器**：停止按单个样本逐次修补，改为一次系统对照：
  1. 先补逐步可观测性，并用已归档状态做 CPU 重算；
  2. 用 15–20 条标准移动（至少 3 个交接，含未参与调试的交接）做 2×2 机制对照：关节目标来源 × 参考推进方式；
  3. 按预先冻结的规格选出候选，重复确认后冻结；
  4. 没有候选达标则停下，交用户做方向讨论。

详见[执行器系统对照与基础工作计划](real-handoff-eef-executor-systematic-plan.md)。


## 2026-10-01：基础工作实施回执

用户授权按验收顺序完成后直接启动。A 服务器归档请求探测 3/3 完成，响应不执行，估算 USD 0.4194625，实际账单待核；B 已冻结并启动独立原生 PPO/SAC 数据采集，最多 5400 GPU1 进程秒、API 0。544 项相关 CPU 回归通过、3 跳过。C 的 18 条草拟数值移动仍有规划/参考时长问题，物理入口与规定接触定义未验收，不能把 CPU 重建写成四条件物理失败，也不能跳过 C 启动 Astra。模型侧条件及旧分母保持不变。见[实施诊断](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-10-01-eef-systematic-foundations-cpu/README.md)。

## 已决定事项 15：2026-09-30 站位就绪条件下的抓取能力测评（用户决定，导师重视）

- **目的**：把"底盘停在哪里"这一步拿掉，单独测 Astra 规划手臂轨迹并完成抓取的能力。
- **站位**：使用 MS-HAB 官方 Pick 子任务的出生站位，优先用验证集。
- **条件**：关闭 `move_base`，其余沿用冻结的模型侧条件。两个条件配对：V（只给目标语义，主条件）与 P（另给目标真值位置，特权消融）。
- **参照**（只在评估端使用）：官方 SAC 从同一站位跑一次；另用脚本化抓取作为执行器可行性的下界参照。
- **样本**：开发组约 10 例；冻结测试组约 30 例，只评一次。
- **顺序**：排在执行器达标之后、带底盘的模型试点之前。

详见[抓取能力测评设计](../../../docs/design/c2-eef-arm-capability-eval-design.md)。


### 2026-10-01 B 收尾更新

开发集 2 已完成：24 次候选尝试取得 20 个有效交接，3 次导航失败、1 次初始化即满足导航；SAC 13/20，Astra 尝试 0。界外 18、界内 2；快照及 23 份有动作尝试媒体已校验。GPU1 1494.218746 进程秒，未用 3905.781254 秒关闭，API 0。C 尚未验收，不能启动 Astra；详情见开发集 2 诊断。

## 已决定事项 16：2026-10-01 抓取能力测评先行，执行器门槛改为任务级（用户决定）

- 抓取能力测评现在就开始，用逐实例的脚本化抓取参照区分模型失败与执行器失败，不再以执行器全面过门为前提。
- 时间预算放宽为每条移动 150 步、总共 600 步，标注为非官方条件；同时报告 200 步内的官方结果。
- 执行器只做一项改动：大幅移动改走关节空间插值，先在 CPU 上验证，并在开发组开跑前冻结。
- 测试组开跑门槛：开发组上脚本化抓取成功率 ≥ 70%。

详见[抓取能力测评设计](../../../docs/design/c2-eef-arm-capability-eval-design.md)末尾的修订。

### 已决定事项 16 的 CPU 落地回执（2026-10-01）

唯一关节空间候选未过门，按用户指定分支冻结协调v2；不再修执行器。验证集出生站位开发10/测试30及种子/哈希已冻结，guidance-v4-arm明确150/600步、禁move_base，P仅首次给特权中心。600步源码门通过，内部601/外层600补偿reset扣步，官方五项判据不变。详见[CPU诊断](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-10-01-arm-capability-cpu/README.md)及[开发组待批提案](../../../docs/design/c2-eef-arm-development-authorization-proposal.md)。脚本、V、P均未物理运行；开发集2保留给之后带底盘条件。本轮只本地提交，不发布。

## 已决定事项 17：2026-10-01 抓取能力测评只保留 V 条件，开发组 5 例（用户决定）

- 删除 P 条件（特权目标位置）；定位问题改由评估端比较 `locate_point` 返回点与目标表面来诊断。
- 开发组从 10 例减到 5 例；测试组开跑门槛改为脚本化抓取 ≥ 4/5。
- 开发组上限：API 125 次、USD 65；GPU1 13,500 进程秒。

详见[抓取能力测评设计](../../../docs/design/c2-eef-arm-capability-eval-design.md)末尾的"修订二"。

### 已决定事项17的实施回执

CPU修订已完成：原清单前5例无需高度层替换，仅V/SAC/script共15进程；定位表面距离只用于评估，P与配对报告已撤除。协调v2哈希不变；125次/USD65/13500秒硬门与4/5测试提案门已更新。579项相关CPU回归通过、3跳过，脚本及V未运行。见[重新提交提案](../../../docs/design/c2-eef-arm-development-authorization-proposal.md)，等待批准，不发布。
