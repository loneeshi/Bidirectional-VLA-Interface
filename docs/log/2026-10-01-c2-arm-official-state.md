# 2026-10-01 · 加入官方状态字段后的抓取复测：开发组 5 例（official-state r1）

日期按 UTC，本地为 10/1。
- **主线阶段**：第 2 步，接口执行与失败归因。
- **本次推进**：给 Astra 加上 MS-HAB 官方状态字段后，同样 5 例在 600 步内全部严格成功（WP2 为 2/5）；官方 200 步内为 2/5。
- **阻塞**：200 步内只有 2/5。远处目标的底盘分段移动约占 100 步，下探时推动目标、抬升被拒也会额外耗步。
- **下一验收门**：由用户决定下一步：先做 200 步效率分析，还是按既定顺序单独授权有界执行器阶段。测试组尚未授权。

**要回答的问题**：在官方出生站位上，Astra 拿到官方给出的目标位姿、末端位姿和抓持状态后，能不能把物体抓起来？能不能在官方 200 步内完成？

**设置**

- **样本**：与 [WP2](2026-10-01-c2-arm-wp2-gripper-geometry.md) 完全相同的 5 个开发样本（arm-dev-000 至 004）及其初始状态，每例只尝试一次。
  - 物体依次为番茄汤罐、果冻盒、大咖啡罐、午餐肉罐、饼干盒；测试组没有运行。
- **共用配置**：
  - 模型 `gpt-6-astra`，推理档位 medium；每次尝试最多 25 次工具调用（每次调用即一次 API 请求）；单次尝试，无跨尝试记忆；
  - 头部与手部 640×640 RGB-D、关节状态与相对里程计；每次调用都要写推理说明 `note`；
  - 手臂执行器 `arm-coordinated-v2-600`，与 WP2 相同；底盘控制器 v13；
  - 时限为非官方放宽条件：每条移动 ≤ 150 步，全程 ≤ 600 步，模型看到的预算也是 600 步。官方 200 步结果取同一次运行的前 200 步；
  - 成功判据为 MS-HAB 官方严格 Pick 五项：抓住目标、末端回休息位、关节回休息位、静止、累计力 < 5000。

| 设置 | 输入 | 提示或决策方式 | 输出或执行方式 | 真值使用 |
|---|---|---|---|---|
| V 可移动底盘 r1 | 目标类别文字、头部与手部 RGB-D、关节状态、相对里程计 | Astra；提示 `guidance-v4-mobile` | 末端工具与 `move_base` | 仅评分端 |
| WP2 | 同 r1 | r1 提示加夹爪几何段落（`guidance-v5-gripper-geometry`） | 同 r1 | 仅评分端 |
| official-state | WP2 的输入，加官方状态字段：目标位姿、末端位姿、目标点与抓持状态 | WP2 提示加官方状态字段的语义说明（`guidance-official-state-v1`） | 同 r1；执行器反馈另含执行阶段与连续 IK 失败数 | 官方状态字段进入模型输入；其余仅评分端 |

- **官方状态字段**：`obj_pose_wrt_base`、`tcp_pose_wrt_base`、`goal_pos_wrt_base` 与 `is_grasped`，取自 MS-HAB 官方环境每步返回的状态，坐标系为当前 `base_link`。
  - 提示说明了三点：物体位姿是模型原点，不是表面点、尺寸或抓取位姿；Pick 任务中的目标点为空；`is_grasped` 不代表整个任务成功。
  - 用户于 2026-10-01 批准，把这一条件作为新的主信息条件，与此前只用传感器的结果分开报告。它与 SAC 的完整观测并不相同。
- **不提供给模型或执行器的信息**：场景碰撞网格、导航网格、经真值筛选的候选和预计算的抓取位姿。评估端记录的接触信息也不进入模型输入。

**结果**

以下均为实际任务执行的官方严格 Pick 结果。

| 设置 | 200 步内严格成功（5 例） | 600 步内严格成功（5 例） |
|---|---:|---:|
| V 可移动底盘 r1 | 0/5 | 1/5 |
| WP2 | 2/5 | 2/5 |
| official-state | 2/5 | 5/5 |
| SAC 官方权重（参照） | 4/5 | — |

official-state 的 95% Wilson 区间：200 步为 11.8–76.9%，600 步为 56.6–100%。5 例是同一批已暴露的样本，与 r1、WP2 配对，但不是独立样本。SAC 只运行了官方 200 步。

| 实例 | 物体 | 成功步数 | 200 步内 | 600 步内 | WP2（600 步内） |
|---|---|---:|---|---|---|
| 000 | 番茄汤罐 | 213 | ✗ | ✓ | ✓ |
| 001 | 果冻盒 | 239 | ✗ | ✓ | ✗ |
| 002 | 大咖啡罐 | 123 | ✓ | ✓ | ✗ |
| 003 | 午餐肉罐 | 257 | ✗ | ✓ | ✗ |
| 004 | 饼干盒 | 145 | ✓ | ✓ | ✓ |

- 5 例在漏斗各级都通过，各级均为 5/5：路径检查通过、夹爪到目标 ≤ 5 cm、抓住目标、抓住后抬起 ≥ 5 cm、严格成功。
- 200 步内成功的样本换了：WP2 是 000 和 004，本轮是 002 和 004。000 由 WP2 的 137 步变为 213 步。

**每例的调用与受力**（列中数字为次数，累计力单位为 N）：

| 实例 | API 调用 | 路径检查 IK 无解 | 无效调用 | 闭合被拒 | 累计力（N） |
|---|---:|---:|---:|---:|---:|
| 000 | 16 | 1 | 0 | 1 | 170 |
| 001 | 20 | 2 | 1 | 0 | 0 |
| 002 | 15 | 2 | 0 | 0 | 797 |
| 003 | 21 | 2 | 0 | 0 | 360 |
| 004 | 13 | 1 | 1 | 0 | 147 |

- 2 次无效调用都发生在第一或第二轮，没有产生动作：004 是四元数不是单位长度，001 是调用格式无效。
- 闭合前对准被拒，WP2 中 5 例有 4 例出现，本轮只有 000 出现 1 次。

**模型如何使用官方状态**（依据各例的 `note`，并与动作和评估端记录核对）：

- **认目标**：003 第一轮就按官方目标位姿判断"罐子在前方 1.75 m"。r1 和 WP2 中都认错了目标，本轮没有再认错。
- **定位**：001、002、003 的 note 明确写着闭合位姿以"当前官方目标位置"为中心。深度图仍用来测物体顶面高度和家具边缘。对目标顶面的深度测量点，距目标表面 0.1–3.9 mm；测家具边缘的点不计入。
- **确认抓持**：5 例都在闭合后引用 `is_grasped=true`，然后才抬起。

**200 步内未完成的原因**：这一轮没有失败例。000、001、003 都是在 200 步之后才成功，而模型被告知的预算是 600 步。下表按工具统计步数的去向：

| 实例 | 底盘移动（次） | 底盘步数 | 转头步数 | 手臂与夹爪步数 | 总步数 |
|---|---:|---:|---:|---:|---:|
| 000 | 0 | 0 | 0 | 213 | 213 |
| 001 | 4 | 100 | 24 | 115 | 239 |
| 002 | 2 | 50 | 0 | 73 | 123 |
| 003 | 4 | 104 | 21 | 132 | 257 |
| 004 | 1 | 21 | 0 | 124 | 145 |

- **001、003**：目标在 1.69 m 和 1.75 m 外。模型每段最多前进 0.30 m，共前进约 1.08 m 和 1.04 m，每段前都重新测量家具边缘。note 给出的理由是：手臂遮挡了右侧通道，所以只走短段。003 另外用了 37 步把手臂移开，以露出前方视野。本轮 11 次底盘移动全部到位。
- **000**：底盘没有移动。第一次去预抓取点用了 95 步。下探时 `gripper_link` 碰到目标罐，采样力约 81 N，也碰到了旁边的午餐肉罐，罐子两次被推前约 3 cm，模型在 note 中如实记下了这一点。第一次闭合因对准未通过被拒（12 步）；抬升 19 cm 因参考时序不可行被拒，改为抬升 8 cm 后成功。按这些记录，000 归为执行偏离意图：掌部提前接触，推动了目标。
- **002 的非目标接触**：抬升和回休息位期间（第 100–123 步），`base_link` 与一个名为 `body` 的非目标物体有接触，采样力最高约 258 N。累计力为 797，没有超限，也没有影响成功。该物体的具体身份没有记录，接触点坐标在当前接口中不可用。

**结论**：加入官方状态字段后，同 5 例在 600 步内全部严格成功（WP2 为 2/5）。WP2 中的认错目标、执行器卡顿和非目标接触都没有再导致失败。但官方 200 步内仍只有 2/5，远处目标的分段底盘移动和下探时的接触是剩下的主要耗步来源。

**局限**：样本是同 5 个已暴露的开发样本，各只跑一次，与 WP2 配对但不独立。本条件向模型提供了仿真器给出的目标位姿与抓持状态，结果不能与只用传感器的结果直接比较，也不等同于 SAC 的完整观测。模型拿到的预算是 600 步，因此 200 步结果只是同次运行的前缀，不是模型在 200 步预算下的表现。

**链接**：
- 上一篇：[提示补充夹爪几何后的抓取复测（WP2）](2026-10-01-c2-arm-wp2-gripper-geometry.md) · [官方出生站位抓取能力测评（r1）](2026-10-01-c2-arm-capability-dev.md)
- 设计：官方状态输入与分阶段执行器修复（附件未发布）
- 诊断：结果、数据与媒体总入口（附件未发布）
- 演示清单：[official-state r1 演示与哈希](../media/c2-pick-arm-official-state-2026-10-01-r1/README.md)。在线录像经 CPU 后处理，不是动作重放。粉色虚线为已发出的目标点连线，青色为实测 TCP，橙色为底盘指令；叠加层为评估端 X-ray，接触页只供评估：

| 实例 | 同场景演示 | 轨迹 | 逐步记录 | 接触 |
|---|---|---|---|---|
| 000 成功（第 213 步） | [`astra-commanded-vs-executed-3d-demo.mp4`](../media/c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/astra-commanded-vs-executed-3d-demo.mp4) | [trajectory.html](../media/c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/trajectory.html) | [history.html](../media/c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/history.html) | [contacts.html](../media/c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/contacts.html) |
| 001 成功（第 239 步） | [`astra-commanded-vs-executed-3d-demo.mp4`](../media/c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/astra-commanded-vs-executed-3d-demo.mp4) | [trajectory.html](../media/c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/trajectory.html) | [history.html](../media/c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/history.html) | [contacts.html](../media/c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/contacts.html) |
| 002 成功（第 123 步） | [`astra-commanded-vs-executed-3d-demo.mp4`](../media/c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/astra-commanded-vs-executed-3d-demo.mp4) | [trajectory.html](../media/c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/trajectory.html) | [history.html](../media/c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/history.html) | [contacts.html](../media/c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/contacts.html) |
| 003 成功（第 257 步） | [`astra-commanded-vs-executed-3d-demo.mp4`](../media/c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/astra-commanded-vs-executed-3d-demo.mp4) | [trajectory.html](../media/c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/trajectory.html) | [history.html](../media/c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/history.html) | [contacts.html](../media/c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/contacts.html) |
| 004 成功（第 145 步） | [`astra-commanded-vs-executed-3d-demo.mp4`](../media/c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/astra-commanded-vs-executed-3d-demo.mp4) | [trajectory.html](../media/c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/trajectory.html) | [history.html](../media/c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/history.html) | [contacts.html](../media/c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/contacts.html) |

录制时间、结果与 SHA-256 见 [演示清单](../media/c2-pick-arm-official-state-2026-10-01-r1/README.md) 与 `docs/media/manifest.json`。

记录信息：
- 运行 ID：`arm-official-state-20261001-r1`；部署包 SHA-256 `52d2f0b7c0ce08c3805ea251a529e2f5a2ee07278de9b32f1bdeaca5dbf58ff2`。
- 用量：API 85 次，按返回 token 估算 USD 17.96；GPU1 1,408 进程秒；供应商实账待核；费用以财务子账为准。
