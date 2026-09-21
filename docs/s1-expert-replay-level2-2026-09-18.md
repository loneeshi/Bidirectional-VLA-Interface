# S1 Level 2 专家动作回放（2026-09-18）

## 结论

这次实验把结论推进到了动作 wrapper 之后，但**没有清掉完整物理链路**：

- 400 个已录制动作（CPU/GPU 各 200）通过 `FetchActionWrapper(stationary_head=True)` 后，最大改动均为 **0**。这与此前 BC A/B/C 的动作比特级一致互相印证，已录制动作没有在 wrapper 中被重新排列、缩放或裁剪。
- CPU metadata reset 虽然 UID、build、task plan 和 spawn 语义身份一致，但在 `state_index=0` 已与原轨迹不一致，因此 CPU 路径不能用于判断后续动作或物理是否可复现。
- GPU 路径对“当前版本 metadata reset 后新建的完整快照”完成了保存—重置—恢复，快照恢复最大叶误差为 **2.3841858e-7**。恢复后的观测在 `state_index=0` 全部落在预注册诊断阈值内。
- 同一 GPU 路径执行第 1 个专家动作后，物体相对底盘姿态已经出现 **0.008783 rad** 偏差，而机器人 qpos/qvel 和 TCP 仍接近数值精度；物体位置在第 2 步首次越阈，机器人 qvel 在第 17 步首次越阈，qpos/TCP 在第 18 步接触/抓取附近成组越阈。原轨迹从第 35 个动作起成功；回放抓到物体但 200 步内未成功。

因此，现有证据支持“**不是 wrapper 改坏了已录制动作**”，并证明当前 GPU 快照 round-trip 可用；它不支持“完整共享动作/物理链已经排除”。首个可见偏差位于 object-side dynamics，接触时才放大到机器人状态。残余差异至少落在原采集 actor 速度与隐藏物理状态未记录、PhysX contact cache / solver warm-start 未被 `get_state_dict()` 暴露、252-env GPU 采集与单 env 回放的数值顺序差异，以及采集与回放 runtime revision 差异之中；本实验不能再区分这些分支。

这不是 VLA 成功率实验，也不是论文协议复现。训练仍冻结。

## 实验合同

| 项 | 固定值 |
|---|---|
| 源轨迹 | `013_apple.h5`, `traj_0`, source env index 0, seed 2024 |
| 源 H5 SHA-256 | `03b29e86035d968df346c851067a052a75bd5a0f29da7792daa05858173f896e` |
| 动作 | 已录制 Fetch13 expert actions；每条路径最多 200 步 |
| wrapper | `mshab.envs.wrappers.FetchActionWrapper(stationary_head=True)` |
| CPU 路径 | metadata reset，CPU backend，minimal shader |
| GPU 路径 | metadata reset，GPU backend；保存并恢复本次重建快照 |
| 诊断阈值 | qpos `1e-4`；qvel `1e-3`；位置 `1e-4 m`；姿态 `1e-3 rad` |
| 训练 / 外部 API / 新租 GPU | 0 / 0 / USD 0 |

源 H5 没有保存原采集时的完整 simulator state、controller state、逐环境 RNG 或 force trace，也没有可用于核对原始物体线/角速度的 `env_states`。GPU 路径恢复的是**本次 metadata 重建的快照**，不是原始采集快照。当前重建快照中的 apple 13 维 actor leaf round-trip 误差为 `1.49e-8`；这只证明本次快照自洽，不能证明重建速度等于原 252-env 采集时的速度。

## 两条路径

| 路径 | 初态资格 | 首次越阈 | 200 步结果 | 回放力峰值 / 累计力峰值 |
|---|---|---|---|---|
| CPU reset | 不合格：状态 0 已分歧 | 状态 0，torso qpos `4.961e-4` | 未抓取、未成功 | `105.184` / `604.035` |
| GPU snapshot restore | 合格：状态 0 全通道在阈值内；快照恢复误差 `2.384e-7` | 状态 1，物体 quaternion `0.008783 rad` | 抓取过、未成功 | `115.054` / `801.457` |

GPU 状态 0 的各组最大误差为：qpos `5.722e-6`、qvel `6.461e-4`、位置 `2.289e-5 m`、姿态 `5.451e-6 rad`、抓取布尔值 0。动作与状态使用记录器的 `obs[0] --action[0]--> obs[1]` 语义。第 1 步回放与 source state 1 的 qpos/qvel/TCP 最大误差分别只有 `6.44e-6`、`7.74e-5`、`5.36e-6 m`，而与 source state 0 或 2 的对应误差大两个以上数量级，因此 off-by-one 已排除；完整物理判断仍受原始快照缺失限制。

旧元数据中的 `init_config_idx=26` 与当前运行时的数值索引不相等；该整数随数据/代码修订变化。更稳定的 UID、build、task-plan、spawn 均一致，且 GPU 状态 0 实测通过，因此没有把这一数值差异单独判为 reset 错误。

## 裁决和下一门

1. Level 1 的 pooled 结果排除了“全部活动通道整体接错”这一强假设，但 yaw 与 torso 的 family-conditioned 切片仍未过门。
2. Level 2 排除了“wrapper 修改已录制动作”，但专家轨迹没有被完整复现；不能据此启动新 VLA 训练。
3. Level 3 首次分叉工具已实现，但当前阈值不能由这次单轨迹、单转移的物体姿态偏差推导。下一最小门是在同一 GPU、同一保存快照、同一 SAC 动作序列上至少重复 3 次，先取得 15 维 qpos 的同策略 null-drift 包络并冻结阈值，再解释 IA-vs-SAC 首次分叉。现阶段只允许 plumbing/raw-error smoke，不发布科学分叉结论。

后续更新：该最小门已完成，三次同动作重放的漂移为 0，冻结阈值后的 IA-vs-SAC 配对在第 1 个动作后分叉。见 [S1 Level 3 同动作校准与首次分叉](s1-level3-first-divergence-2026-09-18.md)。

## 证据与资源收尾

- 结构化结果：[summary.json](results/s1-expert-replay-level2-2026-09-18-run01/summary.json)
- CPU 完整轨迹：[trace.jsonl](results/s1-expert-replay-level2-2026-09-18-run01/parent000-cpu_reset/trace.jsonl)
- GPU 完整轨迹：[trace.jsonl](results/s1-expert-replay-level2-2026-09-18-run01/parent000-gpu_snapshot_restore/trace.jsonl)
- 文件清单：[artifact-manifest.json](results/s1-expert-replay-level2-2026-09-18-run01/artifact-manifest.json)
- 原始归档：`D:\AI\embodied intelligence\runs\s1-expert-replay-level2-2026-09-18-run01.tar.gz`
- 归档大小：612,594 bytes
- 归档 SHA-256：`d9812cc88162fd306e34fb3faf248b77630bd3cf00932a754fd739061c815d5a`

两条路径总墙钟 153.034 秒；模型推理 0、训练 0、外部 API 0、新租 GPU USD 0。实验结束后实验室 GPU1 为 15 MiB / 0%。实验室费用未知；历史停止状态 Runpod 存储仍继续计费，未在本实验中刷新账单。
