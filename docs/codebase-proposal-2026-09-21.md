# 代码库修改提案：为 feedback 系统这一周做准备

日期 2026-09-21。本文是提案，不是已执行的变更记录。涉及 GPU 的部分（S3 站位先验扫描）
需要单独授权后才启动；其余是纯 CPU 的结构调整。

本周目标（来自用户）：在 GPT 臂基础上改 feedback 系统，两条线——

- **A 线**：反馈「SAC 好操作的初始位姿」以提高 SR。难点是怎么找出高分位姿。
- **B 线**：整理喂给 GPT 的 information（image / trajectory history / 结构化信息）。
  难点是判断什么真正有效、怎么摘要。

## 0. 现状：pipeline 在哪，问题在哪

主循环不在 `src/`，在 `scripts/run_coordinator.py:454-525`，约 70 行：

```
scripts/run_ppo_sac_paired16.py        面板驱动：16 plan × 2 臂，子进程 / resume / 记账
  └─ scripts/run_coordinator.py        单 episode 主进程
       for index in range(max_calls):                          :454
           observation = adapter.observe()                     :465
           adapter.save_observation_images()                   :468
           request = coordinator.decide(observation, history)  :486   ← B 线入口
           result  = runtime.execute(request)                  :506
           history.append({...7 个字段...})                     :508   ← B 线数据源
              └─ SerialRuntime.execute()   src/bvi/runtime.py:33
                     skill.act() → env.step() → skill.feedback()      ← A/B 线反馈产出
```

三个具体缺口：

1. `coordinator.py:217` 把 `feedback_history` 原样 `list(history)` 塞进 prompt，**没有任何摘要**。
   `task_memory.object_memory()` 早就写好了按 object 折叠的逻辑，但从未接进这条路径。
2. `run_coordinator.py:508` 的 history 只有 7 个字段（call_id / skill / target_id /
   requirements / feedback / steps / elapsed）。trajectory 层面的量——走过的距离曲线、
   TCP 轨迹、停在哪、为什么停——一个都没进去。GPT 现在看不到「上一次 pick 在离物体
   0.4m 处超时」这种信息。
3. 图像走 `observation.images`（多模态 part），结构化信息走 JSON prompt，两条路径各自独立，
   **没有任何机制能单独关掉其中一条做对照**。所以「什么真正有效」现在无法回答。

第三点是最要紧的：不先解决它，A 线和 B 线改完都说不清涨的是哪一块。

## 1. 结构问题与三个方案

`src/bvi/` 53 个模块按实验批次命名（`s1_` `s2_` `native24_` `acdit_`），不是按角色。
当前在跑的只有 20 个 / 3,324 行，其余 32 个 / 6,523 行属于已冻结实验线。

一个事实先说清楚：**每轮实验都冻结了自己的 source zip**（如
`docs/results/standardized-teleport16-2026-09-21/source-v2.zip`），历史结果从各自的快照复现，
**不依赖 HEAD 的文件布局**。所以移动文件不会让已归档结果不可复现——这点我之前说重了。
移动的真实代价只有改动面。

### 方案 A：零移动 + 状态标注 + 防腐检查（推荐）

不动任何文件位置。做三件事：

1. 每个模块 docstring 第一行后加一行状态标记：
   ```python
   """Pure helpers for exact S1 training/inference preprocessing audits.

   STATUS: frozen 2026-09-19 — pi0.5 / S1-IA line, stopped. Not used by any current panel.
   """
   ```
   活的模块标 `STATUS: active — <所属子系统>`。
2. 新增 `src/README.md`：角色分层表 + 上面那张主循环调用链图。
3. 新增 `tests/test_module_registry.py`：断言 `src/bvi/*.py` 每个文件都有 `STATUS:` 行，
   且 `src/README.md` 的表格与文件系统一致。**索引写完就不会烂**，这是关键——
   纯文档的索引三周后必然过期。

改动面：53 个 docstring + 2 个新文件。零 import 变化，零测试破坏。

### 方案 B：把冻结模块移进 `bvi/frozen/`

`src/bvi/` 顶层从 53 降到 21，一眼看出哪些在跑。

改动面：**82 个文件**（33 个测试 + 49 个脚本）要改 import。都是机械替换，风险低但 diff 大，
在公开仓库里是一笔显眼的 churn。收益相对方案 A 只是"目录更短"。

### 方案 C：完全按角色分包

```
bvi/core/      protocol  runtime  logging
bvi/agent/     coordinator  bridge  providers
bvi/env/       mshab_adapter  nav_camera_env
bvi/skills/    goal_tools  teleport_skill  lightnav_skill  fetch_pi_skill  organizer
bvi/feedback/  progress_monitor  task_memory  + 本周新增
bvi/frozen/    其余 32 个
```

最干净，但改动面是方案 B 的全部再加上活模块的 import。**不建议现在做**——这周的时间应该花在
实验上，不是搬文件。等 feedback 子系统稳定了再一次性做。

**建议：现在做方案 A，本周新代码直接按方案 C 的目标位置放（见 §2），等这条线跑完再决定要不要补 B/C。**

## 2. 新增：`src/bvi/feedback/` 子包

新代码有明确的家，不再往 `src/bvi/` 顶层堆。已有的 `progress_monitor.py` 和 `task_memory.py`
**原地不动**，由新包 import——避免 import 破坏。

```
src/bvi/feedback/
    __init__.py       从 ..progress_monitor / ..task_memory 重导出，对外一个门面
    digest.py     新  history → 给 GPT 的结构化摘要（B 线核心）
    views.py      新  消融开关：按 channel 裁剪送进 prompt 的内容（B 线对照）
    trajectory.py 新  一次 invocation 的轨迹摘要（B 线数据源）
    spawn_prior.py新  站位先验查表 + 格式化（A 线消费侧）
```

### 2.1 `trajectory.py` — 补上缺失的数据

`ProgressGoalRLSkill` 现在每个 sim step 调一次 `progress_snapshot()`，拿到
`{distance, grasped}` 就丢掉了，只留最后一个 progress 标量。改成累积成一条 invocation-local
摘要：

```python
@dataclass(frozen=True)
class InvocationDigest:
    skill: str                  # navigate / pick / place
    target_id: str
    steps: int
    distance_start: float       # 起始 |base-target| 或 |tcp-obj| 或 |obj-goal|
    distance_end: float
    distance_min: float         # 最接近过多少 —— 比终值更有信息
    distance_at_min_step: int
    grasped_ever: bool
    stalled_steps: int          # 末尾连续 distance 变化 < eps 的步数
    end_reason: str             # step_limit / benchmark_fail / environment_horizon / ...
```

全部由 `progress_snapshot()` 已经在读的量导出，**不新增任何特权访问**，不调
`_pick_check_success` 等 native predicate。这点必须守住，否则 GPT 臂就变成 oracle 臂了。

`distance_min` 和 `stalled_steps` 是我判断最可能有用的两个：「靠到过 0.05m 然后退开」
和「一直停在 0.4m」是完全不同的失败，现在 GPT 两者都只看到一个 `timed_out`。

### 2.2 `digest.py` — 摘要层

接口一句话：

```python
def summarize(history: list[dict], profile: str) -> dict
```

`profile` 决定摘要粒度，是 B 线的实验变量：

| profile | 内容 |
|---|---|
| `raw_v0` | 现状：`list(history)` 原样。作为基线保留，保证能复现旧结果 |
| `object_v1` | `task_memory.object_memory()` 的输出：按 object 折叠，每个 family 只留最后一次 |
| `object_trajectory_v1` | `object_v1` + 每次调用的 `InvocationDigest` |
| `object_trajectory_prior_v1` | 再加 A 线的站位先验（见 §3） |

切 profile 不改任何 skill 或阈值，只改 prompt 里那一个字段。这样 B 线的消融是干净的。

### 2.3 `views.py` — 回答「什么真正有效」

从 `recovery_experiment.visible_history()` 提炼并推广。现在那个函数只能开关 progress 一项；
改成按 channel 裁剪：

```python
def apply_view(context: dict, images: tuple, view: FeedbackView) -> tuple[dict, tuple]
```

`FeedbackView` 四个独立开关：`images` / `trajectory` / `progress` / `structured_goals`。
关掉的 channel 从 prompt 和多模态 part 里整个消失，且**不改变 monitor 状态**
（这是原函数里那条注释守的性质，要保留）。

有了这个，「image / trajectory history / 结构化信息哪个真正有效」就是一个 2×2×2 的消融，
而不是一次改一堆然后猜。

## 3. A 线：站位先验

### 3.1 采样器已经有了

`teleport_skill.teleport_tidyhouse_navigation()` 里就是论文那套 spawn 采样：
目标 1.8m 半径内的 navigable 顶点、xy 噪声 σ=0.1 截断 0.2、旋转噪声截断 0.5、
最多 40 次碰撞检查重试（`SPAWN_LOC_RADIUS` / `SPAWN_XY_NOISE_*` / `MAX_SPAWN_ATTEMPTS`）。

缺的只有打分那一半。

### 3.2 新增 `scripts/scan_spawn_prior.py`

对每个物体类别，从上述分布采 N 个 spawn，每个跑一次官方 SAC Pick，记录：

```
{object_category, spawn_xy_wrt_object, spawn_yaw_wrt_object,
 success: bool, steps, failure_cause, min_tcp_object_distance}
```

`failure_cause` 直接用 `native_pick_audit.native_failure_causes()`（读
`cumulative_force_within_limit`，区分「撞力超限」和「够不着」）。

产出 `docs/results/spawn-prior-<date>/prior.json`：按物体类别分 bin 的
(相对位姿 → 经验成功率)，附样本数与 Wilson 区间。

### 3.3 三条硬约束（写死在 runner 里，不是靠自觉）

1. **先验只能在非评测 plan 上建。** 16 个评测 plan（seeds 0–11, 13, 14, 16, 19）的
   plan UID 进黑名单，runner 启动时校验，命中直接拒绝执行。否则就是在测试集上调参。
2. **在线只读，不试。** 评测时 GPT 拿到的是一张冻结的查表结果，不允许「现场试几个位姿挑最好的」
   ——那等于给了 GPT 一个 evaluator 探针，和 `evaluate()`/pointer 越界是同一类问题。
   先验文件在 episode 开始前加载并记 SHA-256，运行中只读。
3. **先验作为独立可消融变量**，不混进 feedback 协议。`object_trajectory_v1` 和
   `object_trajectory_prior_v1` 的差值，就是站位先验单独的贡献。

第 3 条是方法论上最要紧的：如果不拆开，审稿人会问「SR 涨了是 VLA-as-Tools 协议的贡献，
还是你手工调了交接分布」。而 teleport 21/80 vs 连续 PPO 14/80 已经把这个问题摆在桌面上了——
换个导航交接就多 7 个物体，三条臂完整任务 SR 却都是 0。

## 4. 度量：现在这个面板量不出改进

必须先说清楚，否则这周跑完也得不到结论。

当前面板：16 plan / 臂，完整任务 SR 全 0，唯一信号是完成物体数。Fixed 14/80 vs GPT 13/80——
**这 1 个物体的差没有任何统计意义**。更要命的是，两轮 teleport 只差评测器口径就是 21 vs 12，
接近一倍。口径噪声比想测的效应还大。

三条建议，按性价比排序：

1. **配对设计。** 同一个 plan、同一个 spawn seed，只切 feedback profile。配对比较比
   非配对强得多，16 个 plan 也能出信号。这是零额外成本的改进——只要保证两臂的
   `PYTHONHASHSEED` 和 spawn RNG 完全同步（`goal-tools-paired16` 已经在做 state SHA 校验，
   复用那套即可）。
2. **换主指标。** 完成物体数（0–5）比 0/1 的任务 SR 有信息量得多。更好的是
   **per-subtask 成功率**（每个 episode 20 个 native subtask），分母从 16 变成 320，
   同样的 GPU 预算下分辨率高一个量级。这个数据现在就在 events.jsonl 里，只是没被汇总。
3. **先 pilot 再全跑。** 先在 4 个 plan 上跑 `raw_v0` vs `object_trajectory_v1`，
   看效应量级再决定要不要全面板。省 GPU。

## 5. 执行顺序与验收

| # | 内容 | 类型 | 验收 |
|---|---|---|---|
| 1 | 方案 A：STATUS 标注 + `src/README.md` + registry 测试 | CPU | 全套测试通过；registry 测试能抓到漏标 |
| 2 | `feedback/` 子包骨架 + `trajectory.py` | CPU | `InvocationDigest` 单测；不新增特权访问的静态检查 |
| 3 | `views.py` + `digest.py`，`raw_v0` 必须与现状 byte 级一致 | CPU | 同一条 history 走 `raw_v0` 产出的 prompt 与旧代码逐字节相同 |
| 4 | per-subtask 汇总脚本（从已有 events.jsonl 重算，不跑新实验） | CPU | 三个已归档面板的完成物体数能被重算出来并对上 |
| 5 | `scan_spawn_prior.py` + 黑名单校验 | **GPU** | 黑名单命中拒绝执行的测试；先验文件含样本数与区间 |
| 6 | pilot：4 plan × 2 profile 配对 | **GPU** | 预注册指标与判据，跑之前冻结 |

1–4 是纯 CPU，可以立刻做，不消耗 GPU 预算，也不影响任何已归档结果。5–6 需要单独授权
（GPU 时长、API 请求额度、是否动用评测 plan 之外的场景）。

## 6. 不做什么

- 不删任何冻结模块。它们对应的报告还在仓库里，删了报告不可复现；标状态就够了。
- 不动 `src/bvi/` 现有文件位置（方案 B/C 推迟）。
- 不改任何 native predicate、pointer、`evaluate()` 调用路径。
- 不在评测 plan 上建先验。
- 不改旧结果的标签。`raw_v0` 存在的唯一理由就是保证旧 prompt 能逐字节复现。
