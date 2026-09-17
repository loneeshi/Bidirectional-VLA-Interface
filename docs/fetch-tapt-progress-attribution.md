# Fetch TAPT 进度误判归因 — 2026-09-17 UTC

主线第 2 步：1199 次 SFT 更新已完成，本次检查训练后的反馈正确性。**确认了进度监督与在线决策的时点错位，以及错误交接后技能输入超出训练覆盖；本次留出教师状态对照未提供证据，将本轮误判主要归咎于训练／推理特征路径差异。** 首个 reach 在 24.1 cm 处输出高进度的精确原因仍未完全闭合，不把已发现的时点缺陷当作全部原因。

本次未训练、未修改底座／检查点／阈值、未调用 GPT，未运行新的任务成功率评估。已有失败录像保留：[fetch-tapt-pick-episode000-failed.mp4](media/fetch-tapt-online-pick-2026-09-17-run01/fetch-tapt-pick-episode000-failed.mp4)。

## 同观测配对结果

使用原模型选择验证集的全部 33 个缓存时点：来自 **4 条不同教师轨迹、11 个调用片段**，reach/grasp/move/release 分别 9/9/12/3 个时点。身份与原训练报告逐项一致。每条路径重新设置 seed 8181；图像、点云、状态、指令、工具族和 step1199 检查点相同。每个观测运行 9 条诊断路径，总计 297 次路径调用。

| 路径 | 有效输出（最多两个）的进度 MSE，先按样本再平均 | 第一个输出 MAE |
|---|---:|---:|
|训练路径：教师动作加噪，固定随机时间 458|0.017445|0.090377|
|训练路径：教师动作加噪，固定时间 200|0.015221|0.090301|
|真实五步去噪推理|0.016908|0.093100|

这些是相对**局部时间比例标签**的误差，不是物理完成率。该表的进度损失不能与旧报告的 0.006199 **动作＋0.1×进度联合损失**直接比较；本次随机数也不是原验证流的重放。

实际推理时间序列为 `[999, 799, 599, 400, 200]`。进度头读的是最后一次 DiT 前向、最终 scheduler 更新之前的 hidden，**不是 t=0，也不是对返回动作重新前向的特征**。

受控干预结果：

- 把推理底盘分支的加权视觉条件改回训练路径的原始条件，第一输出平均绝对变化仅 **0.0000925**，最大 **0.000471**。训练侧相反干预同样很小。这个真实代码差异在本批状态上影响很小，不能据此修改原生 baseline。
- 固定训练时间 200、相同噪声和原始底盘条件，把教师动作换成模型生成动作后，第一输出平均绝对变化 **0.01127**，最大 **0.09190**。说明进度对动作输入有敏感性，但没有出现普遍抬高；平均变化为 −0.00524。
- 固定原始底盘条件，教师动作单步训练路径与实际去噪路径第一输出平均绝对差 **0.01079**，最大 **0.08983**。在本批教师观察和该随机种子上，未见普遍进度误差扩大；不能排除路径差异与在线异常状态的交互。
- 两条主要路径均在 21 个低于标签阈值的时点上出现 **0 个单帧标签阈值误报**。这不代表在线零误报：稀疏起／中／末观察不能回放“两次连续超过阈值”的监视器，也不能代替物理完成判定。

分族真实推理进度 MSE：reach 0.002538、grasp 0.008176、move 0.023971、release 0.057965。release 仅一个验证片段，不能据此推断稳定性。

## 已确认问题 1：动作后的预测，被当成动作前的完成

[训练标签](../scripts/gate_fetch_tapt_training.py)对第 j 个输出使用：

```text
target[j] = (t + j + 1 - start) / (end - start)
```

因此第一个输出表示**下一动作执行之后**的局部进度。最后动作前的观测 `t=end-1` 已被标为 1。[在线执行器](../scripts/run_lab_fetch_tapt_online.py)却先读取第一个输出，触发时清空动作队列、切换工具族，相关动作尚未执行。

留出数据直接说明了这个语义问题：

|reach 最后动作前的观测|标签|真实推理进度|当前 TCP–物体距离|
|---|---:|---:|---:|
|样本 2|1.0|0.9822|13.77 cm|
|样本 11|1.0|0.9876|11.85 cm|
|样本 20|1.0|0.9632|11.38 cm|

三者都还未满足 8 cm 的 reach 边界，模型却可以非常准确地预测训练标签。**拟合标签正确不等于当前状态已经完成。** 训练集最后动作前的距离也为 8.08–14.69 cm；这一问题不是单个验证样本偶然现象。

此外，标签是到有证据端点的局部经过时间比例，不是完成概率。沿用 reach 0.9、grasp 0.6 等阈值，必须核对原文的进度语义与执行时刻；仅复制阈值不能保证这个 AC-DiT 移植具有相同含义。CPU 理想时间预测器检查甚至在两个训练 reach 片段上、尚距物体 10.3/8.7 cm 时满足了两次阈值条件；这是标签／决策语义的诊断，不是新的策略成绩。

该缺陷可造成提前终止，但**不能独自解释本次 24.1 cm 的全部提前量**，修复后是否能完成 reach 必须重测。

## 已确认问题 2：错误交接造成后续技能输入不在训练覆盖内

|工具族|实际 SFT 缓存中的状态覆盖|失败在线调用|
|---|---|---|
|grasp|57 个训练时点距物体 0.95–7.74 cm；9 个验证时点也不超过 7.77 cm|第16步在 **24.1 cm** 处启动；进度 0.623；第18步在23.1 cm、未持有时进度0.639，超过0.6两次而切换|
|move|57/57 个训练时点、12/12 个验证时点均已持有物体|第18步在**未持有**状态启动；随后进度回退，在第26步请求重规划|

本次 SFT 缓存不包含这些 grasp 远距离起点，也不包含 move 未持有状态。失败原始轨迹虽保留在数据归档中，并未因此自动成为进度头的失败／不可完成监督。数据覆盖缺口不等于已经证明模型完全没有相关能力。

这支持“reach 提前结束 → grasp 在未覆盖的远距离起点仍给高进度 → 未持有进入 move”的错误传播链；不能据此断言增加某个负样本就一定能修复。第26步回退本身不是抓取成功，也不是新增的成功恢复能力。

## 排查过但没有证据作为主因的项目

- [进度头](../src/bvi/acdit_tapt.py)确实取最后两个 action token，没有错取 state/context token。
- 训练和在线均输入相邻两个仿真帧 `[t-1,t]`；工具切换保留观测历史。
- 本轮训练和在线均使用单条原样指令、`padding=False`。未发现本轮训推语言 padding 不一致；在线脚本的 padded-batch 注释已过时，应单独修正文档，不当作故障证据。
- 四族由明确 family 选择，检查点键／形状严格加载。本次所有可训练张量的前后哈希相同，没有参数更新。
- 上游原生 runner 的 raw/weighted mobility 条件差异已经通过本机及服务器同一源码 SHA 验证：`a90ce91ae96b28c6a47e9f29fa652a992513bb69976ec8e242ceab4b74c4f4f4`。本次仅用进程内诊断干预，未修改上游源码。

## 下一步验收门

1. 先对照作者进度标签和执行循环，统一“当前观测进度”与“预测动作后进度”的契约，加入端点动作尚未执行时不能误当已完成的回归检查。保留现有失败证据，不以调阈值掩盖问题。
2. 在相同 seed2025 诊断中记录每次决策的完整观测、点云和随机数状态；确认时点修复后 reach 是否仍在远距离输出高完成度。本轮没有保存原失败步骤的完整模型输入，无法用旧日志做精确同输入反事实。
3. 再根据结果补真实交接及失败／回退状态的 TAPT 监督、单独评估物理边界误报。冻结原生 AC-DiT 底座；不把 SAC 或仿真判据接管算作学习进度通过。
4. 反馈正确性通过后再接 GPT＋LightNav 连续链。DROID、GRPO、真实导航交接覆盖仍未完成。

这是验证集中、单扩散随机种子的有界诊断；该集合已参与检查点选择，不能作为独立测试集泛化成绩。当前只能排低某些故障假说，不能宣称完整根因或在线修复已验证。

## 复现与证据

- [GPU 配对诊断脚本](../scripts/audit_fetch_tapt_progress.py)、[CPU 标签审计](../scripts/audit_fetch_progress_labels.py)、[汇总及一致性校验](../scripts/summarize_fetch_progress_audit.py)。
- [原始结果](results/fetch-tapt-progress-audit-2026-09-17-run01/result.json)、[297 次路径读数](results/fetch-tapt-progress-audit-2026-09-17-run01/paired.jsonl)、[标签与物理状态](results/fetch-tapt-progress-audit-2026-09-17-run01/labels.json)、[分族汇总](results/fetch-tapt-progress-audit-2026-09-17-run01/summary.json)、[SHA256 清单](results/fetch-tapt-progress-audit-2026-09-17-run01/manifest.json)。

```bash
# Existing lab environment, idle GPU1 required. Use a NEW output directory.
cd ~/bvi-research
timeout -k 15s 900s env PYTHONUNBUFFERED=1 envs/acdit/bin/python \
  audit_fetch_tapt_progress.py --out runs/fetch-tapt-progress-audit-YYYY-MM-DD-runNN
# Use the same new batch directory for the CPU audit.
envs/acdit/bin/python audit_fetch_progress_labels.py \
  --out runs/fetch-tapt-progress-audit-YYYY-MM-DD-runNN
# Local repo, after downloading the three raw JSON/JSONL files:
python scripts/summarize_fetch_progress_audit.py docs/results/fetch-tapt-progress-audit-2026-09-17-run01 \
  --training-report docs/results/fetch-tapt-sft-2026-09-16-run01/final-result.json
```

GPU 审计耗时259秒，低于900秒上限；进程已退出，GPU1回到15 MiB。原始三份文件下载前后 SHA256 一致。API0次、新租算力USD0，实验室计费未知；历史两台Runpod仍EXITED，40GB存储约USD0.266667/天继续计费。


## 后续验证

[时点兼容修复与精确配对复测](fetch-progress-timing-fix.md)已经完成：旧轨迹精确复现，修正后仍在23.44cm切换reach，两组均未抓住物体。时点缺陷不是主要误判的充分解释。6个关键决策从记录输入／随机数重放得到逐位一致的动作和进度。
