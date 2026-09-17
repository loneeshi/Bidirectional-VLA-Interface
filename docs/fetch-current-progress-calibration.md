# Fetch 当前观测进度校准 — 2026-09-17 UTC

主线第2步。本批检验当前帧标签和补充调用起点监督，是否缓解此前的提前完成误判。运行结果以本页后续结果表及原始 JSON 为准；代码完成不等于任务成功。

## 固定的监督含义

作者 OpenPI fork `f4eb160ba52b22c1e85fe432de59c24bbbac6187` 的 [data_loader.py](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/src/openpi/training/data_loader.py#L675) 从 offset=0 开始监督当前帧进度。我们将经过完成证据验证的教师调用记为观测 `[start,end]`、动作 `[start,end)`：

`progress[j] = (t+j-start)/(end-start)`，只监督 `t+j <= end` 的进度。

只有两个预测动作都在已记录区间内时，才使用原生动作目标。末端样本没有完整动作窗口时，只更新进度头，不复制最后一个动作或虚构静止动作。失败轨迹结束不是完成端点。

新契约为 `current_observation_v2`，在线在执行动作之前消费 index0。旧1199更新检查点继续保留动作后语义；未经新监督更新的权重不能被重新命名成当前帧检查点。阈值、停滞／回退监视器和原生终止条件均保持原值。

## 新数据与划分

教师仍来自完整轨迹级20/5划分的已采集SAC示范。新增冻结策略调用数据固定为：训练 scene=train seeds3000–3003，验证 scene=train seeds3020–3021。教师与所有衍生调用按父轨迹共同划分。2025是已知失败回归，2030是预先固定的额外测试，均不入训练。

补充监督只覆盖每次调用的首个观测，局部时间定义为0；未来进度和全部动作都屏蔽。这是**重建的调用起点时间监督**，不是作者公开的负样本数据，也不是“物理失败=0”的分类标签。只有真实记录到的错误交接才计入覆盖，不从录像外观推断。

本批为原生起点下的策略调用交接，**尚未覆盖 LightNav 导航交接**。释放族仍只有2条训练、1条验证调用，覆盖不足保留为限制。

## 训练与比较边界

从已有1199更新的四族LoRA与进度头热启动；原生AC-DiT和底盘专家冻结。LoRA rank8/alpha8，微批量1、有效batch8，最多200更新或1200秒更新/周期验证时间，外层2400秒保护包含准备与收尾。

保留原生动作学习目标，进度损失权重0.1。教师可执行窗口仍用原生训练路径；只有进度的样本从实际去噪推理获得隐藏特征，切断骨干／LoRA梯度，仅更新进度头。**进度头仍读取AC-DiT动作token特征，作者实现读取观测prefix；这是明确的移植差异，并未完成作者同构结构复现。**

固定留出起点／中点／验证完成端点和所有留出调用起点，使用真实推理进度MSE与原生动作损失选择更新后的检查点。零更新权重只作损失参照，不参与新契约检查点选择；测试成功率不参与挑选。

检查点冻结后，固定val seeds2025、2030，各最多200动作／300秒，无GPT、无导航、无SAC接管。此测试用于进度校准和局部操作诊断，不能代表GPT＋LightNav＋AC-DiT完整任务成功。

## 复现入口

- 数据采集：[collect_fetch_handoff_anchors.py](../scripts/collect_fetch_handoff_anchors.py)
- 当前帧标签：[fetch_current_labels.py](../src/bvi/fetch_current_labels.py)
- 有界校准：[calibrate_fetch_tapt_current.py](../scripts/calibrate_fetch_tapt_current.py)
- 固定在线检查：[evaluate_fetch_current_calibration.py](../scripts/evaluate_fetch_current_calibration.py)
- 先前时点修复与失败证据：[fetch-progress-timing-fix.md](fetch-progress-timing-fix.md)

实验室GPU1；不调用模型API、不新增租机。实验室计费未知，不填为0。历史Runpod停止存储另记账。


## 数据采集结果

6/6预定轨迹完整执行，动作数依次为23、21、26、70、35、36；均未达到原生Pick成功。未根据成功或失败替换种子。

训练数据实际包含2次远距离grasp起点（12.91cm、26.62cm），以及1次未持有的move起点。留出数据中也有错误grasp（42.22cm）与未持有move。物理值只作覆盖审计，不作为运行时进度规则，也不直接定义学习标签。

[逐交接数值](results/fetch-current-handoffs-2026-09-17-run01/handoff-audit.json) · [6份录像的准确文件名索引](media/fetch-current-handoffs-2026-09-17-run01/README.md)

示例错误交接录像：[train-seed3003-fetch-tapt-pick-episode000-failed.mp4](media/fetch-current-handoffs-2026-09-17-run01/train-seed3003-fetch-tapt-pick-episode000-failed.mp4)。这是冻结模型的数据采集失败，不是校准后的评估录像。
