# Fetch 进度时点修复与配对复测 — 2026-09-17 UTC

**主线第 2 步：旧检查点的时点兼容修复、决策记录与精确重放通过；反馈校准和任务成功仍未通过。** 两组均没有抓住物体。修正时点只把 reach 切换从第16步推迟至第17步，距离从24.12cm降到23.44cm，仍远于8cm边界。因此，不能再把主要误判归结为“少执行了一步”。

## 原文与作者代码的实际契约

[论文 §4.1.2／§4.2](https://arxiv.org/html/2605.13119v1)将进度作为调用内反馈，近1表示接近完成；未规定我们这里的 `+1` 时间偏移。以下是固定作者 fork `f4eb160ba52b22c1e85fe432de59c24bbbac6187` 的代码事实，不等于完整论文实现已公开或跑通：

- [data_loader.py:675–679](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/src/openpi/training/data_loader.py#L675)：`(frame_index + offset) / max(episode_len-1,1)`，offset从0开始。第一个输出对应当前帧。
- [eval_progress_inference.py:347–356](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/scripts/eval_progress_inference.py#L347)：明确记录 `current_target` 与 `current_pred`。
- [pi0.py:380–406](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/src/openpi/models/pi0.py#L380)：从当前观测的 pooled prefix 特征和时间偏移嵌入生成进度序列。我们的 AC-DiT 移植读取 action-token hidden，仍是显式的架构差异。
- [独立评估器:331–446](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/examples/libero/openvla_eval_port/run_libero_eval_openpi.py#L331)：在执行新动作前检查进度。但其标量提取与服务器向量输出尚有集成缺口，切换后还存在重新加入旧动作的代码路径；没有照抄这些问题。

我们的 step1199 检查点已经用 `(t+j+1-start)/(end-start)` 训练，不能通过改名变成作者的“当前帧”模型。本次保留其监督含义，加入明确的 `post_action_position_v1` 契约：在 t 预测，执行第一个动作，在 t+1 消费第一个进度值。每次预测仍只更新一次原监视器；原阈值、两次阈值条件、停滞／回退逻辑不变。

**这个值仍是动作前作出的预测，不是动作后观测重新估计的进度。** 此桥接只修旧检查点的消费时点，不声称已完成作者同构进度头或新的TAPT训练。未知语义检查点会拒绝加载；旧检查点的兼容例外绑定已验证SHA。

## 实际复测

同一 apple Pick、val seed2025、同一检查点、200动作上限、无GPT／导航、无新训练。两组分别有300秒进程上限，合计实耗146秒。所有13个原生控制通道保留；原生终止与安全限制优先于进度切换。

|观测|旧消费时点|修正后|
|---|---|---|
|reach 阈值切换|第16步；24.12cm；预测0.940975|第17步；23.44cm；同一预测0.940975|
|grasp 阈值切换|第18步；未持有；23.05cm|第20步；未持有；21.63cm|
|停止|第26步，move进度回退|第27步，move进度回退|
|曾抓住／原生成功|否／否|否／否|

不是成功率对照：每种模式只有一个episode。可建立的局部结论是：**在这个固定起点，时点修复没有消除远距离完成误判。** 余下问题包含进度目标与物理边界的校准、错误交接状态覆盖，以及AC-DiT进度头移植差异；各因素的贡献尚未全部分离。

## 配对和重放证据

- 旧模式精确复现历史26步的动作、qpos、qvel、extra、info和全部进度事件。新增记录没有改变这条轨迹。
- 分岔前9次reach决策（步骤0、2…16）的原始观测／历史／仿真与控制器状态、模型输入、模型随机数、动作输出和进度输出逐项相同。
- 每次决策保存原始输入、CPU模型张量、Python／NumPy／Torch CPU／CUDA随机数、输出和SHA清单；可重放模型输入，不必由录像猜状态。
- 按预定规则选每组的首个决策、最后一个reach决策、首个grasp决策，共6个；重新加载同一模型并恢复随机数，**6/6动作与进度逐位一致，最大误差0**。这是模型输入重放验证，不是重放仿真成功。
- 最终本地全套检查162项通过、3项跳过；时点／契约回归12项通过，服务器CPU记录与时点检查10项通过。原生模型参数和原阈值未调整。运行后追加的检查点契约守卫只补充拒绝未知语义，实际执行源码保存在原始归档的 `executed/`。

## 产物

- [旧模式：legacy-pre-action-fetch-tapt-pick-episode000-failed.mp4](media/fetch-tapt-timing-2026-09-17-run01/legacy-pre-action-fetch-tapt-pick-episode000-failed.mp4)
- [修正后：post-action-fetch-tapt-pick-episode000-failed.mp4](media/fetch-tapt-timing-2026-09-17-run01/post-action-fetch-tapt-pick-episode000-failed.mp4)
- [配对结果](results/fetch-tapt-timing-2026-09-17-run01/comparison.json)、[精确重放](results/fetch-tapt-timing-2026-09-17-run01/replay.json)、[有界运行命令](results/fetch-tapt-timing-2026-09-17-run01/batch.json)。
- [时点桥接](../src/bvi/progress_timing.py)、[决策记录](../src/bvi/decision_trace.py)、[在线执行器](../scripts/run_lab_fetch_tapt_online.py)、[重放脚本](../scripts/replay_fetch_tapt_decisions.py)、[配对比较脚本](../scripts/compare_fetch_timing_runs.py)。

完整637,610,348字节归档含决策张量与随机数，保存在本机 `D:/AI/embodied intelligence/runs/fetch-tapt-timing-2026-09-17-run01.tar.gz`；SHA256 `3490c9a7e68b6ab128ff667636deaa8563f72dc6f2e79ce01c40768370bd9eeb`。公开仓库提供脚本、摘要、日志及录像，不把大体积张量加入Git。

## 下一验收门与资源

优先在已经能够精确重放的错误调用状态上校验进度监督与物理边界，准备带明确版本的当前观测进度训练数据／配置，补实际交接和失败状态覆盖；再做TAPT更新和相同起点复测。底座保持冻结。DROID、GRPO及LightNav真实交接仍未完成，不能扩大声称为全流程复现。

两组运行及33秒精确重放均结束，GPU1回到15MiB。API0次、新租机USD0、实验室价格未知；历史40GB停止云盘仍计费。本轮没有进行新的TAPT训练。
