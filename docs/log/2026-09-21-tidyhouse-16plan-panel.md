# TidyHouse 16-plan 三设置对照（2026-09-20 → 09-21）

面向人读的实验小结。原始机器记录在 `docs/results/` 和 `runs/` 下，本文只做归纳，不产生新数字。

## 一句话结论

三种设置都把 16 个 episode 跑完了，**完整五物体任务成功率都是 0/16**。唯一有区别的是"平均完成了几个物体"，最好的一档也只到 1.3 个 / 5 个。也就是说：换调度方式（官方固定顺序 / GPT 工具调用 / teleport 导航）在当前执行器能力下没有把任务推过线，差别只体现在中途进度上。

## 面板

| 设置 | 完成 episode | 完整任务 SR | 完成物体 | 每集平均 |
|---|---|---|---|---|
| Fixed PPO + SAC（官方固定任务顺序） | 16/16 | 0/16 | 14/80（17.5%） | 0.875 |
| GPT + PPO + SAC（VLA-as-Tools 通讯协议） | 16/16 | 0/16 | 13/80（16.25%） | 0.813 |
| Teleport + SAC（官方固定任务顺序） | 16/16 | 0/16 | 21/80（26.25%） | 1.313 |

基础设施失败：三组均为 0。计划分母固定 16，不因中途失败缩小。

> **Teleport 行待核对。** 仓库里当前唯一同步到本地的 teleport 证据是
> [`docs/results/official-teleport16-2026-09-21/`](../results/official-teleport16-2026-09-21/README.md)，
> 它记的是 **12/80（15.0%，每集 0.75）**，逐 seed 明细加总也是 12。上表 21/80 来自之后
> 一轮更新的跑（对应 工作区 `runs/standardized-teleport16-source-v2-20260921.zip` 这一版源码），
> 该轮的 `panel-status.json` 尚未同步回仓库。在证据落盘之前，21/80 这一行不应被引用为
> 已归档结果。

## 三个设置分别是什么

**Fixed PPO + SAC** —— 官方基线。每一步都由官方评测器给出下一个子任务，导航用已发布的 PPO 策略，Pick/Place 用官方逐物体 SAC checkpoint。没有语言输入，没有模型调用。这是"上限参考"：调度完全正确时，执行器自己能走多远。

**GPT + PPO + SAC** —— VLA-as-Tools 协议臂。执行器和上面完全一样，换掉的只是调度：GPT-5.6 Luna 看到五个"物体→目的地"目标、20 个合法的技能/目标组合、相机图像和自己的执行历史，但**看不到**当前原生子任务索引、类型或下一步提示。物体顺序约束是公开的，不主张可以自由重排。每次调用 40 个动作 / 180 秒，每集上限 7000 动作 / 900 秒，GPT 有 40 次付费决策。

**Teleport + SAC** —— 论文口径的参考。用论文发布版上游 commit `4729821`，官方评测器对每个 Navigate 子任务直接 teleport，Pick/Place 仍是官方 SAC。它换掉的是导航，不是调度。

三组用的是同一批 16 个唯一 TidyHouse validation plan（seed 0–11、13、14、16、19）。

## 读这张表要注意的三件事

1. **Teleport 不与前两组同起点。** teleport 有意改变了导航交接的分布，初始物理状态不能声称与 PPO 两臂一致。所以 teleport 的 21/80（或 12/80）高出来，不等于"teleport 导航更好"，更可能是交接位姿分布不同。
2. **GPT 臂有一集是被预算掐掉的，不是原生失败。** seed 16 以 `request_exceeds_remaining_experiment_wall_budget` 结束，已完成 2 个物体。决策预算封顶（censoring）要单独报告，不能混进"模型不行"里。
3. **16 个 plan、零次完整成功 → 这是配对诊断，不是 benchmark 分数。** 论文口径的总体是 1000 个 rollout。在 0/16 这个位置上，14 / 13 / 21 的差异没有统计可分辨性。

## 逐 seed 完成物体（Fixed / GPT）

| seed | plan UID | Fixed | GPT | GPT API 请求 |
|---|---|---|---|---|
| 0 | val-557-0 | 0 | 0 | 20 |
| 1 | val-90-0 | 0 | 1 | 16 |
| 2 | val-147-0 | 2 | 2 | 27 |
| 3 | val-189-0 | 4 | 2 | 24 |
| 4 | val-553-0 | 0 | 0 | 10 |
| 5 | val-86-0 | 2 | 1 | 18 |
| 6 | val-682-0 | 0 | 0 | 6 |
| 7 | val-881-0 | 0 | 2 | 20 |
| 8 | val-17-0 | 0 | 0 | 6 |
| 9 | val-629-0 | 0 | 0 | 2 |
| 10 | val-586-0 | 1 | 2 | 13 |
| 11 | val-327-0 | 1 | 0 | 15 |
| 13 | val-850-0 | 1 | 0 | 18 |
| 14 | val-113-0 | 3 | 1 | 15 |
| 16 | val-334-0 | 0 | 2※ | 28 |
| 19 | val-110-0 | 0 | 0 | 7 |
| **合计** | | **14** | **13** | **245** |

※ 预算终止，非原生失败。API 请求列为最终 attempt 的计数；计入全部历史 attempt 后 GPT 臂累计 269 次。

Teleport 逐 seed（对应仓库里已归档的 12/80 那一轮）：`0:0, 1:0, 2:1, 3:3, 4:0, 5:2, 6:0, 7:1, 8:0, 9:0, 10:1, 11:1, 13:1, 14:2, 16:0, 19:0`。

## 开销

| | Fixed | GPT |
|---|---|---|
| 累计环境步 | 7,773 | 8,012 |
| 每集平均环境步 | 486 | 501 |
| 累计墙钟 | 880 s | 4,725 s |
| 每集平均墙钟 | 55 s | 295 s |
| 无效模型输出 | 0 | 0 |
| 重规划尝试 / 成功 | 0 / 0 | 0 / 0 |

GPT 臂每集慢约 5.4 倍，环境步数只多 3%——多出来的时间几乎全是规划开销和等待模型返回。实际 API 账单未核销（`pending_reconciliation`），预留额度不等于供应商出账。

训练更新 0，新租 GPU 0，RunPod 0。仅用 lab GPU1；GPU0 上的无关进程未动。

## 证据

- Fixed / GPT 面板：[`docs/results/goal-tools-paired16-2026-09-20/`](../results/goal-tools-paired16-2026-09-20/README.md)（逐 seed 事件与 summary 保留在工作区 `runs/gpt-audit/`，未随仓库发布）
- Teleport（已归档的一轮）：[`docs/results/official-teleport16-2026-09-21/`](../results/official-teleport16-2026-09-21/README.md)
- 对照口径与修复规则：[baseline protocol revision 2](../baseline-protocol-2026-09-20-v2.md)
- 两臂的设计与授权边界：[`ppo-sac-paired16-2026-09-20.md`](../ppo-sac-paired16-2026-09-20.md) · [`goal-tools-paired16-2026-09-20.md`](../goal-tools-paired16-2026-09-20.md)
- 录像：[`docs/media/goal-tools-paired16-2026-09-20-*`](../media/README.md)

## 接下来值得做的

- 把更新那一轮 teleport 的 `panel-status.json` 同步回仓库，确认 21/80 还是 12/80。两者差一倍，结论方向不同。
- seed 16 重跑一次不受 wall-budget 限制的 GPT episode，把 censoring 从分子里摘干净。
- 真正决定分数的是执行器（SAC Pick/Place），不是调度层。要让 SR 离开 0，下一步该动的是操作能力，不是再换一种调度。
