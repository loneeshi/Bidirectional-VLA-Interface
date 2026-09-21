# VLAs-as-Tools 方法对齐基线（活文档）

**用途**：本项目与 [Lei et al., arXiv:2605.13119v1](https://arxiv.org/pdf/2605.13119v1) 之间唯一的逐条对齐基准。任何"我们复现了 X / 偏离了 X"的表述以本文件为准。

**这不是**进度报告或计划。只记录：原文要求什么、代码现在是什么、证据在哪、补齐需要什么。

**维护规则**
1. 改动调用契约、反馈、调用粒度或评测协议的 PR，必须在同一提交更新对应行与 §10 修订记录。
2. "代码证据"必须是 `文件:行` 或可 grep 的符号，不写散文。
3. 状态只取 §0 词表的值。不确定写 `待查` 并在 §9 建条目。
4. **每一行必须注明它描述的是哪个配置（§1）。** 两个配置的调用契约不同，混写是本文件 09-20 首版最严重的错误来源。
5. 文件名不带日期。快照在 §10 记 commit，不复制成新文件。

**本轮核对基准**：工作树 `67588e3`（含未提交改动），2026-09-20。原文核对至 PDF 全文 17 页含附录 A–D。

---

## 0. 状态词表

| 状态 | 含义 |
|---|---|
| **对齐** | 机制存在、在该配置的运行路径上生效、语义与原文一致 |
| **部分** | 机制存在但作用范围、信号来源或粒度与原文不同 |
| **缺失** | 该配置的运行路径上不存在 |
| **冲突** | 存在对应物但与原文定义相左，需在报告中主动声明 |
| **不适用** | 依赖训练阶段；当前无训练，不构成偏离 |

`不适用` 仅用于 §5 的训练机制。推理期能生效却没生效的一律记 `缺失`。

---

## 1. 两个配置 —— 读本文件前必须先确定在说哪一个 ⚠️

仓库存在**两条互斥的运行路径**，调用契约完全不同。`run_coordinator.py:123-131` 的守卫明确规定 `--paired-ppo-episode` **禁止** `--tool-family-interface`。

| | **配置 A：paired PPO/SAC 目标工具** | **配置 B：tool-family 接口** |
|---|---|---|
| 启动开关 | `--paired-ppo-episode --goal-tools --organizer` | `--benchmark-episode --tool-family-interface` |
| 导航 / 操作 | 官方 PPO / 官方按物体 SAC | LightNav-0 / 官方按物体 SAC |
| 守卫 | `run_coordinator.py:125-131`（要求 `navigation_policy == 'official'`） | `run_coordinator.py:140-152`（要求 `lightnav`） |
| 运行脚本 | `scripts/run_ppo_sac_paired16.py` | `scripts/run_sac_interface_baseline.py`（`command():49` 硬编码 `--navigation-policy lightnav`） |
| 调用消息 | skill + `target_id`，**无** `tool_family`、**无** `instruction` | `mshab-tool-family/1`：skill + `tool_family` + `instruction` |
| GraspMonitor | **关闭**（`run_coordinator.py:397` `if 'pick' in skills and not args.goal_tools`） | 启用 |
| organizer scope | `goal_decomposition` | `benchmark_constrained_execution` |
| 运行状态（09-20） | **已运行**：16＋16 配对，固定程序 3 结束 / GPT 2 结束＋1 基础设施失败，其余 13＋13 未运行 | **未运行** |

> **配置 A 不使用 `mshab-tool-family/1` 契约。** 因此原文 §4.1.1 的 (g, z) 两字段调用消息在配置 A 中不是"部分实现"，是**不存在**。
>
> **待定（§9-D2）**：本周交付以哪个配置为准？1000 行预注册台账（`bind_sac_interface_manifest.py` / `run_sac_interface_baseline.py`）属配置 B，与已运行的配置 A 不是同一实验。

---

## 2. §4 开篇：两个设计选择（p.4）

> "The VLA tool is therefore not a standalone task policy, but a bounded executor that receives an agent-specified invocation and returns progress feedback."

| 条款 | A | B | 证据 |
|---|:--:|:--:|---|
| 工具是有界执行器，非 standalone task policy | **冲突** | **冲突** | 官方 PPO / SAC / LightNav-0 本身即 standalone 子任务策略，被有界调用包了一层 |
| 工具返回 progress feedback | **缺失** | **缺失** | §4 |
| 调用有界、由 agent 指定 | **对齐** | **对齐** | §3.6 |

**报告口径**：不得写"我们的工具是 VLA tool"，只能写"我们把 standalone 策略放进了有界调用契约中"。

---

## 3. §4.1.1 调用消息 C = G × Z（p.4–5）

| # | 原文条款 | A | B | 代码证据 / 说明 |
|---|---|:--:|:--:|---|
| 3.1 | 两字段：tool-family label ＋ scene-grounded instruction | **缺失** | **对齐** | A：`smoke_goal_tools.py:44` 构造的 SkillRequest 不带 `tool_family`/`instruction`。B：`coordinator.py:90,118,121`；`protocol.py:191-197` |
| 3.2 | g 从工具族 T={T_g} 选一个成员 | **部分** | **部分** | 均为异构后端，非一族共享骨干成员 |
| 3.3 | g 有实质选择空间 | **对齐** | **缺失** | **A：GPT 看见全部 20 个工具／目标组合并自行选择**，官方物体顺序仍强制但不向 GPT 暴露当前唯一正确调用。B：`protocol.py:194` 强制 `tool_family == skill`，而 admissible-call 集合每步只放行一个 |
| 3.4 | z 指定 object / relation / desired local effect | **部分** | **部分** | **A：结构化目标条件化——`target_id` 真正改变策略的物体／目的地特征并选择对应 SAC 权重；无自然语言指令条件化。** B：z 发出并校验，navigate 消费，pick/place 丢弃 |
| 3.5 | 分解使请求可检视 | **部分** | **对齐** | A 可检视 skill＋target，无 family／instruction 可检视 |
| 3.6 | 有界执行窗口 H_k，由 agent 指定 | **对齐** | **对齐** | `SkillRequest.max_steps`；`protocol.py:212`；`organizer.py:33` slice 钳制 |
| 3.7 | 可选短执行历史 h_t | **部分** | **部分** | `frame_stack=3` 为环境侧堆叠；原文标为 optional |
| 3.8 | s_{k+1}=U(s_k,c_k,τ_k,r_k) 累积 agent 状态（式3） | **部分** | **部分** | `run_coordinator.py:326,452,475`；history 全量累积并回传，但条目未存完整请求（缺 `instruction`、调用预算）→ c_k 未完整进入 s_{k+1} |

**3.4 的准确表述**（取代首版"目标被底层完全忽略"）：配置 A 有**结构化目标条件化**，没有**自然语言指令条件化**。这仍不能冒充原文的语言条件 VLA。

---

## 4. §4.1.2 反馈消息 R（p.5）

> "a binary completion signal can delay recovery" —— 原文明确把二值完成信号列为要被 p_t 取代的对象。

| # | 原文条款 | A | B | 代码证据 / 说明 |
|---|---|:--:|:--:|---|
| 4.1 | 主反馈＝连续进度 p_t ∈ [0,1] | **缺失** | **缺失** | `protocol.py:123-124` 字段存在，无生产者，恒 `None`/`"unavailable"` |
| 4.2 | 进度局部于调用 c_k，非全任务成功 | **对齐** | **对齐** | 反馈为逐目标／子任务级 |
| 4.3 | 避免反复查询大 VLM 的高成本 | **对齐** | **对齐** | slice 40 ＋ `--max-calls 40`；**但无高频监控对照臂，不得声称复现 Table 4 的调用成本优势** |
| 4.4 | 暴露停滞／偏离等中间执行状态 | **缺失** | **部分** | **A：GraspMonitor 关闭**（`run_coordinator.py:397`），只有原生完成反馈。B：`missed_grasp`/`grasp_lost`，来源为模拟器 `is_grasped` |
| 4.5 | 进度 chunk 与动作 chunk 对齐 | **缺失** | **缺失** | `tool_family.ProgressChunk` 仅 `eval_libero_family.py` 使用 |
| 4.6 | 阈值监控三分 advance/replan/continue | **缺失** | **部分** | `progress_monitor.py` 未被 `run_coordinator.py` import |
| 4.7 | 辅助头 p̂_t = ψ_ω(b_t) 挂 backbone 特征 | **缺失** | **缺失** | `learned_progress_available: False` |

**R 侧定位**（配置 A 的准确说法）：

> **有界工具调用 ＋ 目标相关的原生完成反馈，而非 progress-based replanning。**

**关于信号来源的措辞**（修正首版的"因果方向相反"，该说法过强）：差别在于**推理时反馈的来源与机制**——原文的 p_t 是执行器**从自身骨干特征预测**的量，我们用的是模拟器谓词**读数**。用模拟器谓词做监督或评估，与用它**替代**学习型进度反馈，是两回事；只有后者构成偏离。其实质后果是该反馈无法迁移到真机，而原文 §6 的 limitation 正指向真机扩展。

---

## 5. §4.2 TAPT（p.5–6）

无训练阶段，故 §4.2.1 训练单元、§4.2.2 residual adapters、§4.2.3 联合目标、附录 B 的 RL/GRPO 全部 **不适用**。

§4.2 中三条约束推理期或数据单元，不训练也已生效：

| # | 原文条款 | 状态 | 说明 |
|---|---|:--:|---|
| 5.1 | "TAPT follows the same pipeline as the agent–tool loop"：训练单元＝推理调用单元 | **部分** | 要求是**粒度明确且训练／推理一致**，不是把内部每个运动阶段都变成外部调用。见 §6 |
| 5.2 | 逐调用完成谓词 ψ_{z,g}(s) | **对齐（计算）／部分（命名）** | `goal_tools.py:116-120` 按请求目标分别调用 `_pick_check_success` / `_place_check_success` / `_navigate_check_success`。**不是三个工具算同一个布尔量。** 接口名 `benchmark_success`（`run_coordinator.py:447`）过于笼统，改名有助审计；**不得**把官方 pick 成功条件简化为 `is_grasped`（`goal_tools.py:100` 仅作 extra info），那会改变评测定义 |
| 5.3 | 推理期式(5)：z 条件化；选定残差路径在调用期间保持激活 | **部分** | z 条件化见 3.4（原文在 §3 式(2) 与 §4.2.2 式(5) **两处重复**此约束）；"调用期间不换后端"由 `runtime.execute(request)` 单请求单 skill 满足 |
| 5.4 | "g should control the execution path, not merely appear as an extra language token" | **对齐** | g 决定调哪个后端，比 LoRA 更强的路径控制 |
| 5.5 | residual adapters 的参数效率与保留预训练语义泛化 | **不适用** | 独立模型，非共享骨干。**Table 5 的 "+9% 参数 / 0.50×" 不可引用**；SAC 无预训练语义泛化，该动机对本项目为空 |

---

## 6. 调用粒度：两套 family 词表并存

| 路径 | families | 依据 |
|---|---|---|
| 接口／目标工具 | `navigate` / `pick` / `place` | `mshab_adapter.py:268`（源自 `uenv.task_cfgs`） |
| 数据 / LIBERO | `reach` / `grasp` / `move` / `release` | `progress_monitor.THRESHOLDS`；`prepare_two_day_s1_audit.py:42` |

**成立的部分**：两套词表不能混用。具体风险——`task_memory.object_memory` 含 `if ... family not in THRESHOLDS: continue`，一旦把 `navigate/pick/place` 喂进去会**静默丢空且不报错**。当前 `task_memory` 仅被 `eval_libero_family.py` 引用，故未暴露。应改为对不支持的 family 显式拒绝，不能套用 LIBERO 阈值。

**已撤回的部分**（首版推得过远）："一次 Pick 必须等于三次 invocation"、"粗粒度永远不能做 TAPT"。原文要求的是粒度明确且训练／推理一致，未规定外部调用必须细到运动阶段。

**决定（2026-09-20）**：**本批 baseline 保留 `navigate` / `pick` / `place`。** 官方 SAC 没有独立的 reach/grasp/move 控制接口，强行截断不能证明得到三个可靠工具。未来细粒度版本另立协议与实验，不混入本批。

---

## 7. 待补的工程项（不需 GPU、不需训练）

1. **history 存完整请求**：当前未存 `instruction` 与调用预算（§3.8）。配置 A 的结构化目标已由 `target_id` 保留，但记录仍不完整。
2. **`task_memory` 对未知 family 显式拒绝**，取代静默跳过（§6）。
3. **谓词接口改名**：`benchmark_success` → 按目标类型命名，便于审计（§5.2）。仅改名，不改计算。

**已撤回的提案**：首版建议"pick 用 `is_grasped` 实例化谓词"。该改动会把官方完整抓取成功条件替换为单一布尔量，**改变评测定义**，不得实施。

---

## 8. 评测协议与指标（§5、附录 D —— 非 §4，单列以免混淆）

| # | 原文 | 出处 | 现状 / 处理 |
|---|---|---|---|
| 8.1 | 每任务 50 trials | 附录 D p.17 | 配置 A 当前 16＋16 配对；50 集需约 2000 次调用，超 800 请求／USD 2 授权，须先对账 |
| 8.2 | Intervene Freq / Replan SR / Avg Calls | Table 4 | 已补本地代理指标并显式标注来源，缺失记 null。**原文正文与附录均未定义 Replan SR 的分母** → §9-Q1 |
| 8.3 | 效率对照臂：VLM 每 5 步直接监控 | Table 4 | `not_run_budget_and_protocol_gate`。不可仅改 `--organizer-slice-steps 5`：40 调用上限下仅 200 个物理动作（vs 1600），且每次调用重启执行器。需持久执行＋监控式查询＋等动作预算＋独立 API 授权。**当前不得声称复现 Table 4 的调用成本优势** |
| 8.4 | Table 3 第 5 行 interface-only = 80.2 / 80.4（低于 standalone 92.0 / 92.4） | p.8 | 机制仅适用于消费语言的执行器；配置 A 无语言条件化，**不适用**。当前批次结果已观察，**不得追认为预注册** |
| 8.5 | 指标口径 | Table 1/2/3 | TidyHouse 长程数**禁止**与单 pick 子任务 success-once@200 的数（84.3/72.6/44.3/33.3）并列或排名 |
| 8.6 | 调用数截断 | — | **已实现**：`classify()` 输出 `budget_terminated`。因预算终止的 episode 其 avg-calls 为右截断，不可与原文未截断均值并列 |
| 8.7 | 失败归类 | — | **已修正**：`classify()` 现要求 `model_rejection` 事件才把 ProtocolError 记为模型侧；其余 `error:` 一律 infrastructure_failure。注释明确"prior API calls do not prove this failure came from the model" |
| 8.8 | 配对样本量 | — | 固定程序 3 结束、GPT 2 结束。**配对集是两臂的交集（≤2）**，须按交集报告并显式写出 N，不得按各臂分别计数 |

**本批最准确的定位**：

> 在相同官方 PPO/SAC 执行器上，比较固定程序调度与 GPT 整目标工具调度。不验证 TAPT、学习型进度恢复或残差参数效率；无高频 VLM 监控对照。

---

## 9. 未决事项

**决策**

- **D2 本周交付以哪个配置为准**：已运行的配置 A（PPO＋SAC 目标工具，16＋16），还是配置 B（LightNav＋tool-family 接口，1000 行台账，未运行）。两者不是同一实验，结果不可合并。

**待查**

- **B1** 批次在 GPT seed 2 的 API／桥接请求路径停止，记为 `ProtocolError`，**原因未查明**。查明前不得报告完整面板 SR，也不得归因于模型能力。
- **B2** `run_sac_interface_baseline.py:49` 仍硬编码 `--navigation-policy lightnav`（配置 B）。若配置 A 为主线，该脚本与 1000 行台账的定位需重新声明。

**待问作者**（邮件草稿见 `sac-interface-audit-amendment-2026-09-20.md`，未发出）

Q1 Table 4 三指标的定义与分母｜Q2 进度头 feature tap 与阈值｜Q3 DROID-split 清单｜Q4 residual 插入层与秩｜Q5 直接监控基线在 5 步查询间是否保持执行器状态。

**来源存疑**

`progress_monitor.THRESHOLDS`（reach 0.9 / grasp 0.6 / move 0.9 / release 0.6）、drop 0.03、stagnation 10 步：**原文正文与附录 A–D 中均无**，已改标为本地设置。疑似源自 `cxliu0314/openpi` fork，精确出处未验证。

---

## 10. 修订记录

| 日期 | 工作树 | 变更 |
|---|---|---|
| 2026-09-20 (a) | `67588e3` | 建立本文件。依据 PDF 全文逐条核对 §4 |
| 2026-09-20 (b) | `67588e3` | **重大修正**。首版误把配置 B 当作唯一实现，而已运行的是配置 A。改动：(1) 新增 §1 双配置表，所有条款按配置分列；(2) 3.3 配置 A 下 g **有**选择空间（GPT 见全部 20 组合），撤回"无选择空间"；(3) 3.4 改为"有结构化目标条件化、无语言指令条件化"，撤回"目标被完全忽略"；(4) 4.4 配置 A 的 GraspMonitor **关闭**；(5) 5.2 谓词按目标类型分别计算，撤回"一个通用谓词"，并撤回 `is_grasped` 提案（会改变评测定义）；(6) §6 撤回"Pick 必须拆三次"，确定保留粗粒度；(7) 4.4 措辞由"因果方向相反"改为"反馈来源与机制不同"；(8) 8.6/8.7 记录 `classify()` 已实现 budget_terminated 与 model_rejection 归类；(9) 新增 8.8 配对样本量、§9-B1/B2 |

---

## 11. 已核实为准确、不必重查

- π₀.₅：LIBERO-Long standalone 92.4 → direct planner 80.4（−12.0）→ TAPT＋接口 97.2（+4.8）；RoboTwin 39.4 → 30.0 → 62.5（+23.1）。Table 1。
- OpenVLA-OFT 同栏掉 11.8 分。Table 1。
- 实验仅 LIBERO-Long / RoboTwin / CALVIN，**无 MS-HAB**，全为固定底座桌面任务、无底盘控制。§5.1、附录 D。
- RL 阶段要求采样有奖励方差（附录 B 式(8) 为 0/1 完成谓词）；全零成功策略无梯度信号。
- TAPT 起点是已在基准上训过的强策略，非从零协同提升。Table 1、Table 3。
- VLM 规划器可替换：四个强 VLM 间 LIBERO-Long 相差 0.6 分。Table 10。
