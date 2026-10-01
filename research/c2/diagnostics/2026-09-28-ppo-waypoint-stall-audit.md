# #016 PPO 未到位：停止原因与障碍证据复核

复核日期：2026-09-28。范围：`real-handoff-pick-batch-navigation-20260927` 正式 20 例中的 5 个实际导航分支；不新增运行、API 或 GPU 作业。

## 结论

5 个分支的停止原因均为 `navigation_stalled`。没有因受力门、新增网格重叠、路线失效或官方终止而结束；归档导航采样的瞬时力、末时累计力均为 0。PPO 没有通过一个“暂停”信号退出，而是持续输出动作，由外层停滞检测终止调用。

停滞门检查 `平面距离误差 + 0.2 × 朝向误差（rad）`；只有比历史最好值降低超过 0.01 才记为进展，连续 30 个动作无此进展就停止。它不要求机器人完全静止。

| 起点／候选 | 导航动作数 | 采样最大力（N） | 最后30步距目标增加（cm） | 停止原因 |
|---|---:|---:|---:|---|
| 1／1 | 48 | 0 | 6.55 | navigation_stalled |
| 1／2 | 46 | 0 | 4.88 | navigation_stalled |
| 20／1 | 30 | 0 | 4.84 | navigation_stalled |
| 23／1 | 91 | 0 | 28.83 | navigation_stalled |
| 26／1 | 45 | 0 | 3.72 | navigation_stalled |

“最后30步距目标增加”用最后30个动作后状态中第一个与最后一个底盘世界XY计算，不包含这段之前的起点。位置23在这段中仍明显移动，说明停滞不等于原地停止。未记录到接触不能证明每个物理子步绝无接触，也不能排除模型看见障碍后保守或转向；这里能确认的是障碍安全门没有触发此次退出。

## 接口检查

`ppo_navigation_observation` 将路标XY及原目标物体高度放入合成 `goal_pos_wrt_base`，原物体位姿仍单独留在 `obj_pose_wrt_base`。它没有传入用户要求的终点yaw。到位门却同时要求位置误差≤0.08m、朝向误差≤0.12rad，连续3个动作满足。因此不能把这次失败全归结为PPO自身导航能力：合成路标输入、保留的物体位置与终点朝向要求之间存在任务适配问题。是否主要由视觉障碍、输入适配或控制输出造成，尚无对照实验可分离。

## 原始证据

归档根目录：`runs/real-handoff-pick-batch-navigation-20260927/`。

- 位置1：`main-chunk1/plan-001/decision-trace/events.jsonl`，rank1/rank2的 `candidate_navigation_action` 和 `candidate_navigation_end`。
- 位置20、23、26：`main-chunk2/plan-020`、`plan-023`、`plan-026` 下相同文件。
- PPO逐步动作与安全事件：各分支所在目录的 `adapter-events.jsonl`；5个分支共260个PPO动作，未出现 `pick_stage_ppo_collision_stop`。
- trial最终状态：各目录 `result.json`；汇总 `main-score.json` 的到位为0/5。
- 已核实 `preflight-code-v6.zip` 中 `src/bvi/pick_ppo_navigation.py` 与当前文件SHA-256一致；此次停止门与观察构造说明来自实际归档版本。

链接：[实验设计](../../../docs/design/real-handoff-batch-ppo-pick-experiment-design.md) · [导航实现](../../../src/bvi/pick_ppo_navigation.py)
