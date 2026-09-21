# S1 Level 3 同动作校准与首次分叉（2026-09-18）

## 结论

严格同起点的 S1-IA 与官方 SAC 在**第 1 个动作之后就已经分叉**，不是执行几十步后由物理随机漂移或累计力判据逐渐制造出的差异。

- 三个 fresh-process / fresh-env SAC 重放使用同一 GPU、同一保存快照和同一前 35 个动作；三份 `events.jsonl` 的 SHA-256 完全相同。15 维 qpos、TCP–目标距离和累计力的重复间最大漂移均为 **0**。
- 依照运行前冻结的规则，15 个 qpos 通道阈值全部由单位 floor 决定，为 `1e-4 m` 或 `1e-4 rad`。
- S1-IA 与 SAC 的 task、seed 2025、reference-state SHA 和恢复误差均通过严格配对；共同覆盖 35 步。
- 第 1 步有 14/15 个 qpos 通道越阈，只有 head tilt 未越阈。按 normalized error 最大者，主通道是底盘 yaw：IA `4.274809 rad`，SAC `4.112199 rad`，差 `0.162611 rad`，是阈值的 1,626 倍。

这一条把主嫌疑从“wrapper / GPU 数值漂移 / 力阈值”推进到**策略的第一个控制决策**。它仍不能区分策略内部的欠拟合、图像/state 条件、指令条件或随机动作头，但后续失败已经不是到第 20–40 步才第一次出现。

## 校准合同

| 项 | 固定值 |
|---|---|
| exact start | seed 2025 `reach-state.pt`, SHA `332928111d25f1cf03060cd760bb17d99c4204d5f0890d807ce24f3e00b20c5b` |
| SAC 动作源 | 49 步记录中的前 35 步，源 SHA `f68ef63fd460a382625339c29a0cc6854071543d10fb2aa7bbcce006c1d11d01` |
| 环境 | GPU backend，minimal shader，20 Hz，val `013_apple` |
| 重复 | 3 个串行 fresh process/env，每条 35 步，共 105 步 |
| 恢复门 | 每条 exposed state 最大误差 `2.3841858e-7`，要求不高于 `1e-5` |
| 状态 0 | qpos、TCP距离、累计力三项的跨重复 pairwise 最大差均为 0 |
| 阈值规则 | `max(1e-4 unit floor, 2 × 三重复/35步的最大 pairwise drift)`，使用严格 `>` |
| 实际阈值 | 所有通道 `1e-4`；校准漂移为 0 |

run01 在第 0 个动作前因只读身份记录使用 `controller.config` 而不是该运行时的 `controller.configs` 而 fail-closed；未生成阈值，GPU 正常释放。兼容修正后的 run02 在 48.237 秒内完成。失败证据与成功证据都保留。

## 第一步发生了什么

两条轨迹来自同一个保存状态，且 SAC 同动作重放在三次 fresh env 中逐值完全确定。S1-IA 与 SAC 的第一个 controller-normalized 命令却已经在多个关键通道方向相反：

| 通道 | S1-IA | SAC | 绝对差 |
|---|---:|---:|---:|
| shoulder pan | `+0.9884` | `-0.8022` | `1.7906` |
| upperarm roll | `+0.5089` | `-0.8235` | `1.3324` |
| base yaw | `+0.1987` | `-0.9985` | `1.1972` |
| wrist roll | `+0.2224` | `-0.9610` | `1.1834` |
| shoulder lift | `+0.6424` | `-0.4998` | `1.1422` |

这与第 1 步 qpos 的广泛分叉相符。不能把底盘 yaw 称为单一根因：它只是第 1 步 normalized error 最大的状态通道，另外 13 个通道也同时越阈。

## 35 步曲线

| 指标 | 第 1 步 S1-IA / SAC | 第 35 步 S1-IA / SAC |
|---|---:|---:|
| TCP–目标距离 | `0.9795 / 0.9539 m` | `1.1698 / 0.0101 m` |
| 累计力 | `0 / 0` | `5149.289 / 2145.612` |

SAC 在第 12 步开始产生有效接触力，第 23 步已到目标附近，最终第 49 步原生成功。S1-IA 到第 24 步才开始积累接触力，TCP 已经沿错误方向远离目标，并在第 35 步因累计力超过 5000 原生终止；没有抓取。

## 能说与不能说

可以说：

- 对这个严格同起点配对，第一次可测行为差异就在第 1 个动作；
- 当前 wrapper 没有改变已记录动作，同动作 GPU 物理重放又是逐值确定的，所以这次 IA/SAC 分离不能归因于 wrapper 或 fresh-env 数值漂移；
- 35 步后的力超限是早期错误控制的后果，不是最初分叉点。

不能说：

- 这是成功率估计；它仍是一组 seed 2025 的配对诊断；
- 14 个同时越阈通道中某一个是唯一根因；
- 已经区分了欠拟合、RGB/state 构造、条件指令或随机采样；
- 离线动作重建门已经全部通过。Level 1 的 yaw/torso family-conditioned 弱项仍在，新 VLA 训练继续冻结。

## 证据与资源

- 校准摘要：[summary.json](results/s1-level3-null-drift-2026-09-18-run02/summary.json)
- 冻结阈值：[thresholds.json](results/s1-level3-null-drift-2026-09-18-run02/thresholds.json)
- 首次分叉全曲线：[s1-level3-first-divergence-2026-09-18-run01.json](results/s1-level3-first-divergence-2026-09-18-run01.json)
- run01 失败摘要：[summary.json](results/s1-level3-null-drift-2026-09-18-run01/summary.json)
- 文件清单：[artifact-manifest.json](results/s1-level3-null-drift-2026-09-18-run02/artifact-manifest.json)
- 原始归档：`D:\AI\embodied intelligence\runs\s1-level3-null-drift-2026-09-18-run01-run02.tar.gz`
- 归档 SHA-256：`aa33c7624dfa422925b9ee07b62d46adcad215e1a9a1204584365f217f01e553`

本级实际执行 105 个 simulator actions；模型推理 0、训练 0、外部 API 0、新租 GPU USD 0。实验结束后实验室 GPU1 为 15 MiB / 0%。实验室费用未知；历史停止状态 Runpod 存储仍继续计费，本轮未刷新供应商账单。
