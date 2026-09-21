# Fetch 当前观测进度校准 — 2026-09-17 UTC

**主线第2步：当前帧标签、错误交接数据和118次真实TAPT更新已完成；反馈校准与在线Pick成功验收仍未通过。** 固定两起点均未抓住物体。以下分开报告离线改善与在线失败。

## 固定的监督含义

作者 OpenPI fork `f4eb160ba52b22c1e85fe432de59c24bbbac6187` 的 [data_loader.py](https://github.com/cxliu0314/openpi/blob/f4eb160ba52b22c1e85fe432de59c24bbbac6187/src/openpi/training/data_loader.py#L675) 从 offset=0 开始监督当前帧进度。我们将经过完成证据验证的教师调用记为观测 `[start,end]`、动作 `[start,end)`：

`progress[j] = (t+j-start)/(end-start)`，只监督 `t+j <= end` 的进度。

只有两个预测动作都在已记录区间内时，才使用原生动作目标。末端样本没有完整动作窗口时，只更新进度头，不复制最后一个动作或虚构静止动作。失败轨迹结束不是完成端点。

新契约为 `current_observation_v2`，在线在执行动作之前消费 index0。旧1199更新检查点继续保留动作后语义；未经新监督更新的权重不能被重新命名成当前帧检查点。阈值、停滞／回退监视器和原生终止条件均保持原值。

## 新数据与划分

教师仍来自Pick、Place每任务20训练/5验证的完整轨迹级划分。实际有效调用为训练59段、验证11段；缓存观测不能当作独立轨迹数。新增冻结策略调用数据固定为：训练 scene=train seeds3000–3003，验证 scene=train seeds3020–3021。教师与所有衍生调用按父轨迹共同划分。2025是已知失败回归，2030是预先固定的额外测试，均不入训练。

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

6/6预定轨迹完整执行，动作数依次为23、21、26、70、35、36；其中train3001、validation3020原生Pick成功，其余4条失败。未根据成功或失败替换种子。初版文字误写为全部失败，已按原始result.json纠正，原始日志与录像文件名未变。

训练数据实际包含2次远距离grasp起点（12.91cm、26.62cm），以及1次未持有的move起点。留出数据中也有错误grasp（42.22cm）与未持有move。物理值只作覆盖审计，不作为运行时进度规则，也不直接定义学习标签。

[逐交接数值](results/fetch-current-handoffs-2026-09-17-run01/handoff-audit.json) · [6份录像的准确文件名索引](media/fetch-current-handoffs-2026-09-17-run01/README.md)

示例错误交接录像：[train-seed3003-fetch-tapt-pick-episode000-failed.mp4](media/fetch-current-handoffs-2026-09-17-run01/train-seed3003-fetch-tapt-pick-episode000-failed.mp4)。这是冻结模型的数据采集失败，不是校准后的评估录像。


## 命令（实验室已固定环境）

以下目录均位于 `~/bvi-research`。六轨迹采集和校准各只执行一次；已存在的输出目录会拒绝覆盖。

```bash
# 第一阶段：冻结策略采集；每条150秒/80动作，上界960秒。
timeout -k 15s 960s envs/acdit/bin/python collect_fetch_handoff_anchors.py \
  --output /home/pshuai/bvi-research/runs/fetch-current-handoffs-2026-09-17-run01

# 第二阶段：完成采集后才允许启动；准备/验证/更新/收尾外层上界2400秒。
timeout -k 15s 2400s envs/acdit/bin/python calibrate_fetch_tapt_current.py \
  --gpu-index 1 --output runs/fetch-current-calibration-2026-09-17-run01 \
  --handoff-dir runs/fetch-current-handoffs-2026-09-17-run01 \
  --max-updates 200 --max-seconds 1200

# 第三阶段：最终训练门通过、GPU释放后，固定检查best.pt。
timeout -k 15s 630s envs/acdit/bin/python evaluate_fetch_current_calibration.py \
  --checkpoint /home/pshuai/bvi-research/runs/fetch-current-calibration-2026-09-17-run01/best.pt \
  --output /home/pshuai/bvi-research/runs/fetch-current-progress-eval-2026-09-17-run01
```

本批用 `finish_fetch_current_calibration.py` 自动衔接已经运行的训练与第三阶段，不会再启动训练。若该等待器已运行，不重复执行第三阶段。检查点中的 `updates` 是本轮额外更新数，`warmstart_updates=1199` 单独记录。


## 最终校准结果

本轮实际追加118更新，更新与周期验证耗时1200.54秒，到达预定时限即退出。按固定留出联合损失选中追加step20的检查点；step50、step100均未超过它。选择不使用两个测试种子的成功率。

四族各168个参数张量更新，进度头更新；原生参数与缓冲区哈希在20步及最终审计均不变。实际学习进度头没有换成规则，阈值未改。终点和无动作标签样本只训练进度，不伪造控制目标。

| 留出指标 | 零更新参照 | 选中step20 |
|---|---:|---:|
| 原生动作损失 | 0.003822 | 0.003756 |
| 真实去噪推理进度MSE | 0.060945 | 0.028392 |
| 联合分数 | 0.009916 | 0.006595 |
| grasp调用首帧平均预测（目标0） | 0.603572 | 0.223639 |
| move调用首帧平均预测（目标0） | 0.371310 | 0.173466 |
| 教师grasp完成端点平均预测（目标1） | 0.926237 | 0.787893 |

起点误报降低，但完成端点预测也有下降，不能只凭总体MSE宣布反馈可靠。reach和move的教师当前帧MSE均变差；[逐族数据](results/fetch-current-calibration-2026-09-17-run01/selected-validation.json)完整保留。此批同时调整标签、补充样本并继续TAPT更新，是组合修正，没有单独隔离每个因素的因果贡献。

[训练报告](results/fetch-current-calibration-2026-09-17-run01/result.json) · [全部更新](results/fetch-current-calibration-2026-09-17-run01/updates.jsonl) · [父轨迹与标签清单](results/fetch-current-calibration-2026-09-17-run01/split-manifest.json)

选中检查点SHA256：`a721c20f9b95e797792dda651f6c3e18fd28e3e62984000e59757d0e2a32ca4f`。1199为热启动历史更新数，20为选中版本本轮追加数；最终118步恢复点另外保存。

## 固定在线诊断结果

同一选中检查点、val场景、200动作上限；两条进程正常完成，没有基础设施截断。它们都未抓住物体，原生Pick完成为0/2。这是小样本诊断，不是benchmark成绩，也没有GPT或LightNav参与。

| 起点 | reach切换 | 最终结果 |
|---|---|---|
| 2025，已知回归 | 第26步，进度0.90394，距物体15.17cm | 第31步原生累计力失败；累计统计5175.18，未抓住 |
| 2030，预先固定额外起点 | 第18步，进度0.93795，距物体30.57cm | 第26步grasp进度回退至0.16874，距物体8.17cm，未抓住；请求高层重规划后停止 |

2025对照前一轮时点修正版本：reach从第17步/23.44cm，变为第26步/15.17cm，仍未达到8cm教师边界。2030没有配对的旧权重在线对照，不能用它计算新旧成功率差异。

2030结束时原生`fail=false`；固定诊断执行器在学习进度回退请求时结束，因此没有测试GPT能否恢复。2025则是实际原生安全失败。二者的失败原因不同。

[逐项结果](results/fetch-current-progress-eval-2026-09-17-run01/outcomes.json) · [实际启动命令](results/fetch-current-progress-eval-2026-09-17-run01/batch.json) · [动作裁剪审计](results/fetch-current-progress-eval-2026-09-17-run01/clipping-audit.json)

日志中存在76/41个生成chunk通道裁剪记录；未执行的排队动作可能在切换时丢弃，不能将该数字当成执行动作数。旧2025版本也有59次裁剪，裁剪机制本轮未改。这提示控制饱和值得继续分析，并不单独证明接口映射有bug。

准确录像：

- [seed2025-fetch-tapt-pick-episode000-failed.mp4](media/fetch-current-progress-eval-2026-09-17-run01/seed2025-fetch-tapt-pick-episode000-failed.mp4)
- [seed2030-fetch-tapt-pick-episode000-failed.mp4](media/fetch-current-progress-eval-2026-09-17-run01/seed2030-fetch-tapt-pick-episode000-failed.mp4)
- 旧2025对照：[post-action-fetch-tapt-pick-episode000-failed.mp4](media/fetch-tapt-timing-2026-09-17-run01/post-action-fetch-tapt-pick-episode000-failed.mp4)

## 结论与下一验收门

本轮组合修正仍未解决提前完成误判，不能据此把剩余问题归因于单一因素。当前监督和执行时点已对齐，真实错误调用起点已进入训练，但远距离reach完成误判仍存在；当前头还不能承担可靠的技能完成判断。

下一优先项是保持动作策略固定，对[独立观测特征进度头](fetch-observation-head-followup.md)做离线配对验证，明确去掉动作／底盘去噪隐藏特征依赖。该文是尚未实施的源码方案，不能称作者同构结构已经复现。同时仍需真正LightNav交接和带恢复动作的教师数据；本轮首帧零进度锚点没有教会策略恢复动作。DROID、GRPO和完整GPT＋LightNav＋AC-DiT链仍未完成。

代码与标签本地197测试通过、4项因依赖跳过，实验室Torch CPU15项检查通过。实际训练和两条在线执行另有上表证据。GPU1最终回到15MiB，训练和自动复测进程均退出；后续归档只用CPU。


## 归档与资源收尾

完整训练归档8,622,769,847字节已下载并核验SHA256 `1929a80816848ccaef4da8be665ba0917caf7830700af7f71d3e2efd14d26a79`。全部394个归档文件与远端冻结文件一致，其中375个为缓存样本；打包后生成的完成标记另存，详见[备份记录](results/fetch-current-calibration-2026-09-17-run01/backup.json)与[逐文件核验](results/fetch-current-calibration-2026-09-17-run01/archive-verification.json)。采集和在线评估原始归档亦已核验。训练报告result.json保留训练结束时状态，后续在线结果独立保存，不改写历史记录。

[最终资源核验](results/fetch-current-calibration-2026-09-17-run01/resource-close.json)：GPU1于2026-09-17 03:07:09 UTC为15MiB/0%，训练、评估与归档核验进程均退出。API0、新租用GPU费用USD0；实验室计费未知。历史40GB停止Runpod存储仍按账本估算约USD0.266667/日计费。
