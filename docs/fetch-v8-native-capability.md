# Fetch V8 原生能力门：0/5，本周停止 π₀.₅ TAPT

2026-09-17 UTC。按用户插入的原生能力门执行；主线仍在第2步，未进入Fetch π₀.₅工具族训练。

## 固定结果

| 种子 | 面板 | 动作数 | 曾抓住 | 原生成功 | 累计力终值 |
|---|---|---:|---|---|---:|
| 2024 | 主 | 20 | 否 | 否 | 5031.87 |
| 2025 | 主＋辅助 | 24 | 否 | 否 | 5956.08 |
| 2026 | 主 | 36 | 否 | 否 | 5857.30 |
| 2027 | 主 | 25 | 否 | 否 | 5037.69 |
| 2028 | 主 | 37 | 否 | 否 | 5946.08 |
| 2030 | 辅助 | 40 | 否 | 否 | 5001.20 |

**主结果0/5；辅助2025/2030为0/2，2025复用，不按7个独立样本计数。** 六条均在抓取前超过原生累计力上限5000，基础设施完整、没有GPT调用或学习进度中断。6唯一episode、182个实际动作/模型前向；服务器277.36秒。

## 协议与证据边界

- 同9/16的val任务集、种子2024–2028、200动作上限、20Hz、CPU物理与原生成功/安全谓词。5cm是TCP归位阈值；完整成功还需持有、robot_rest、静止、累计力合格。
- 每条reset qpos及head/hand RGB/depth逐项完全匹配历史。旧AC未保存完整simulator状态，所以不声称与其全状态完全配对；本批完整状态已留存。
- 冻结已有Fetch V8，71预训练参数严格加载；workspace224/hand128、state30起点相对XY、原分位数归一化、BF16、每预测执行1动作、head8/9归零不变。AC原生每次执行2动作；披露这一策略自身协议差异，不将此表冒称单变量模型架构比较。
- 无新prefix进度头、无工具族残差训练、无提示词搜索、无SAC接管。BF160.004201训推进度差仍是未来学习头验收风险，本轮没有进度头，因此不能用它解释本轮原生力失败。
- 原生结果说明当前冻结策略在这些起点缺乏可用控制表现；没有接触分解/干预证据，不能直接断言是某个关节、归一化或特定碰撞导致全部失败。

## 止损与下一条主线

已执行预设≤1/5止损：本周停止π₀.₅ TAPT参数更新（含20更新检查）。远端hold为STOP_PI05_TAPT_THIS_WEEK，真实结果文件和hash代入训练入口也被拒绝；训练更新0。这个时间边界不是“SFT永远不能改善能力”的结论。

固定教师50条采集完成，Pick21/25、Place18/25，失败保留。[错误交接8案例](results/fetch-pi05-handoff-replay-2026-09-17-run01/README.md)已全部通过独立双次回放与真实workspace输入验证；验证输入就绪和学习进度能正确报告失败仍是两道不同验收。

后续恢复已工作的GPT＋SAC单物体链与LightNav＋SAC参照；原控制和协调配置冻结，再单列学习反馈对照。SAC不是VLA、附加进度头不是作者TAPT，不把工程交付称完整VLM＋两种VLA方法复现。

## 原始记录与演示

- [完整批次报告](results/fetch-v8-native-capability-2026-09-17-run02/batch.json)、[全部文件哈希](results/fetch-v8-native-capability-2026-09-17-run02/archive-manifest.json)。逐seed事件和起止图像保留；原始输入NPZ与simstate保存在完整归档。
- [首次0动作基础设施失败](results/fetch-v8-native-capability-2026-09-17-run01/batch.json)：缺少bvi导入路径，修复并独立重试；不计入成功率。
- 完整run02归档SHA256：`943ff415cbda3dc077faaaa0264ff254e4a4769570e97caee2b6988d61e8fe7f`，17,277,519字节，241个成员逐文件已校验。

| 视频文件名 | 结果 |
|---|---|
| [fetch-v8-native-pick-seed2024-episode000-failed.mp4](media/fetch-v8-native-pick-2026-09-17-run02/fetch-v8-native-pick-seed2024-episode000-failed.mp4) | 原生失败 |
| [fetch-v8-native-pick-seed2025-episode000-failed.mp4](media/fetch-v8-native-pick-2026-09-17-run02/fetch-v8-native-pick-seed2025-episode000-failed.mp4) | 原生失败 |
| [fetch-v8-native-pick-seed2026-episode000-failed.mp4](media/fetch-v8-native-pick-2026-09-17-run02/fetch-v8-native-pick-seed2026-episode000-failed.mp4) | 原生失败 |
| [fetch-v8-native-pick-seed2027-episode000-failed.mp4](media/fetch-v8-native-pick-2026-09-17-run02/fetch-v8-native-pick-seed2027-episode000-failed.mp4) | 原生失败 |
| [fetch-v8-native-pick-seed2028-episode000-failed.mp4](media/fetch-v8-native-pick-2026-09-17-run02/fetch-v8-native-pick-seed2028-episode000-failed.mp4) | 原生失败 |
| [fetch-v8-native-pick-seed2030-episode000-failed.mp4](media/fetch-v8-native-pick-2026-09-17-run02/fetch-v8-native-pick-seed2030-episode000-failed.mp4) | 原生失败 |

## 费用

API0、新增租机USD0。实验室GPU1使用已登记，服务费未知，不能填0；V8退出后实测15MiB/0%，随后错误交接渲染也已结束，GPU1再次确认为15MiB/0%。历史40GB停止Runpod存储继续约USD0.266667/日，供应商账单未在本批刷新。
