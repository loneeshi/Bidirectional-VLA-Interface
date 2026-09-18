# S1-IA 调用级在线评估预注册

2026-09-18 EDT，在查看本批结果之前冻结。checkpoint仅采用留出损失选择的best/855；不按在线表现改选。GPU1串行，内部3600秒、外层3660秒；API0、新租机0、lab费用未知，S1累计18小时不增加。

固定val种子2024–2028。先从上一轮GPU/minimal面板的完整初始状态恢复，逐叶最大误差<=1e-5。官方SAC每种子最多200动作生成起点，供独立调用诊断，不算VLA执行。reach从初始状态；grasp取首次TCP距离<=8cm且未持物；move取首次连续3帧持物。完整sim/controller快照保存，评估重新恢复和校验。未生成起点记not_evaluable，不换种子、不伪造原始H5专家状态。仿真隐藏接触缓存不可由序列化一致证明，move恢复后未持物须记基础设施/起点失败。

15个计划调用：每种子reach/grasp/move各一次。指令与训练窗口模板相同，策略只看两相机RGB+native24；SAC的特权state绝不传入VLA。每次调用重置模型RNG，执行每个预测chunk的首动作，不携带上次队列。reach最多120动作，以<=8cm且未持物完成；grasp最多60动作，以连续3帧持物完成；move最多120动作，以持物且原生Pick成功完成。保留原生终止、力限和剩余horizon，不暗中重置预算。

报告每族计划5、实际可评估数、成功数、策略失败、缺起点/恢复失败、步数和精确视频文件名。reach判据与训练几何分段对齐，不单凭该判据宣称开爪指令忠实度。独立调用能力不是GPT自主链；oracle/SAC准备不能当TAPT收益。发布首例且保留失败。S2仍未放行。

## 调用级结果（2026-09-18 EDT）

| family | 计划 | 可评估 | 成功 | 策略失败 | 无法评估 |
|---|---:|---:|---:|---:|---:|
| reach | 5 | 5 | 0 | 5 | 0 |
| grasp | 5 | 4 | 3 | 1 | 1 |
| move | 5 | 4 | 1 | 3 | 1 |

证据：[calls.json](results/s1-ia-calls-2026-09-18-run01/calls.json)。13次实际调用共481个动作/本地模型推理，API0。seed2024 grasp快照已原生终止，被前置检查拒绝；move起点未生成，两者不进入策略成功率分母。

reach五次全部因原生累计力超限终止（23/35/23/25/31动作），不是120步超时。这把下一步诊断重点放在接近过程的碰撞、底盘/手臂动作和力累计，不足以只指认闭爪时机。move2025力限失败，2027/2028持物120步但未达原生完成条件，2026成功26步。grasp2025/2026/2027均3步满足稳定持物，2028在60步内未持物。

重要限制：grasp从SAC接近/闭爪阶段保存的起点出发，三个3步成功可能部分继承接触和夹爪运动，不能据此宣称IA学会从远处抓取或优于普通SFT。需要同快照原模型/保持动作对照，才能分离残余动力学与新策略贡献；该补充评估本轮尚未启动。S2不自动放行，自主链未验证。

首个可评估调用及成功补充：

- [fetch-s1-ia-reach-seed2024-episode000-failed.mp4](media/s1-ia-calls-2026-09-18-run01/fetch-s1-ia-reach-seed2024-episode000-failed.mp4)
- [fetch-s1-ia-grasp-seed2025-episode000.mp4](media/s1-ia-calls-2026-09-18-run01/fetch-s1-ia-grasp-seed2025-episode000.mp4)
- [fetch-s1-ia-move-seed2025-episode000-failed.mp4](media/s1-ia-calls-2026-09-18-run01/fetch-s1-ia-move-seed2025-episode000-failed.mp4)
- [fetch-s1-ia-move-seed2026-episode000.mp4](media/s1-ia-calls-2026-09-18-run01/fetch-s1-ia-move-seed2026-episode000.mp4)

原始归档26,560,486字节已下载验SHA256：7f17419182e26322e4bfd80df2065b172b9bbb135969177bdd0ed7dcde9bd673。全部失败和SAC起点记录保留。外层进程退出，GPU1复核15MiB/0%。server/result.json保留最后waiting_for_next_client快照，它不表示服务仍在运行。新增租机0、API0，lab费用未知；历史云存储本轮未重核。
