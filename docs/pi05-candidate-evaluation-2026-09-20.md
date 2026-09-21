# π₀.₅ 两日候选实验评估

**阶段2验收结论：离线候选改善成立，但同初态原生回归基线与候选均0/5，G1未通过。** 实际完成日期2026-09-19 UTC，早于计划交付日；文件名保留2026-09-20以对应原计划。

## 诊断问题与合同

16个固定输入审计没有找到独立证据支持的输入或时点缺陷，因此执行路径B。保持原best855的相机、state24、动作头、normalizer与单共享LoRA；冻结所有非LoRA参数和dtype，关闭progress/family banks。新AdamW以常量1e-4开始；最多100更新或3600秒。父轨迹0/1的全部85有效调用行，reach34/grasp8/move43，不能视为85个独立任务。

实际完成100更新、800个microbatch暴露、170次离线评估前向，总墙钟1533.50秒。保存checkpoint100与完整训练状态，独立CPU恢复71叶通过。首动作和有效chunk使用相同行、相同RNG、相同有效mask；误差单位是裁剪到[-1,1]的控制器归一化动作，排除两个head通道。611个有效chunk action存在重叠，不能当作独立样本。

| 范围 | 有效动作数 | 原best855 RMSE | 诊断100 RMSE | 相对降低 |
|---|---:|---:|---:|---:|
| all / first_action | 85 | 0.399533 | 0.292585 | 26.8% |
| all / valid_chunk | 611 | 0.379795 | 0.269611 | 29.0% |
| reach / first_action | 34 | 0.480006 | 0.366207 | 23.7% |
| reach / valid_chunk | 250 | 0.489565 | 0.359375 | 26.6% |
| grasp / first_action | 8 | 0.556956 | 0.423046 | 24.0% |
| grasp / valid_chunk | 21 | 0.573236 | 0.403982 | 29.5% |
| move / first_action | 43 | 0.275043 | 0.172877 | 37.1% |
| move / valid_chunk | 340 | 0.250377 | 0.159954 | 36.1% |

冻结门全部通过：首active11和chunkactive11均至少改善20%；reach yaw由0.484381降至0.370849，torso由0.510635降至0.336687，也通过各自不恶化超过10%的guard。[原始判定](results/two-day-delivery-2026-09-20/diagnostic-decision.json)与[独立复算](results/two-day-delivery-2026-09-20/diagnostic-decision-review.json)一致。

这支持“本预算下，既有共享LoRA可改善已见训练控制映射”。它不证明动作已经足够准确、泛化成立或原生任务成功。训练H5图像不是可恢复的完整接触快照，因此没有伪造训练短闭环成绩。历史训练的默认freeze filter曾允许非LoRA参数；本轮显式收窄为20个LoRA叶，不将该差异宣布为历史失败根因。

## 唯一正式候选

`s1-bounded-candidate-2026-09-19-run01` 于2026-09-19 01:55:28 UTC从**原best855**初始化；没有从诊断checkpoint继续。上限500更新/9000秒，包括加载、编译、开发评估及保存。原始完整train6835行，冻结reach有效yaw/torso和move运动窗口采样，dev8父轨迹20/21只用于预注册选点。[采样与选点规则](results/two-day-delivery-2026-09-20/candidate-preregistration.json)、[启动记录](results/two-day-delivery-2026-09-20/candidate-launch.json)。

实际完成500更新，总墙钟6508.4385秒，含加载、编译、评估及保存。最终选择best/250。score为首动作与有效chunk RMSE分别除以原best855基线后的比值均值，必须小于1，且reach yaw/torso首动作误差各不恶化超过10%；仅用dev8，不用在线成功率选点。

| 开发点 | score | 首动作RMSE改善 | 有效chunk RMSE改善 | reach guards / 选点 |
|---|---:|---:|---:|---|
| step100 | 0.9513896488 | 3.42% | 6.30% | 均通过 |
| step250 | 0.9258189853 | 10.09% | 4.75% | 均通过；最终选中 |
| step500 | 0.9182959712 | 14.19% | 2.16% | yaw恶化20.66%，不合格 |

step500虽有更低综合score，但reach yaw RMSE从0.4114715升至0.4964638，超过冻结10%上限，不能替换step250。[独立原始数组复算](results/two-day-delivery-2026-09-20/candidate-selection-review.json)与[最终裁决](results/two-day-delivery-2026-09-20/candidate/candidate-decision.json)一致。新CPU进程用135.65秒顺序完成两次完整恢复：51个非LoRA叶的字节、shape、dtype与原best855完全一致，20个LoRA叶均改变；零GPU、零前向。选中参数树SHA256为`7ff8ae1572bfbc7430605918d19a96903bbd4813cb6dd821632a9bfe4e951919`。[完整候选6.634GB备份](results/two-day-delivery-2026-09-20/candidate-full-backup.json)已完成。

这些证据支持固定开发集离线改善。本轮同时包含续训、固定采样及LoRA-only范围，缺少等预算对照，不能唯一归因于重采样。

## 原生与后续门

两组采用同一原生初态、seed2024—2028、minimal shader/GPU backend、200动作上限及原生成功谓词，使用与IA训练窗口一致的oracle指令。逐例result/events、完整备份与模型参数身份均核验；初态、normalizer及协议源码哈希一致。[原始配对摘要](results/two-day-delivery-2026-09-20/oracle-paired-comparison.json)。

| seed | 基线/候选动作数 | 基线/候选最近TCP物距(m) | 原生结局 |
|---|---:|---:|---|
| 2024 | 22 / 23 | 0.718 / 0.727 | 均累计力失败 |
| 2025 | 34 / 34 | 0.719 / 0.659 | 均累计力失败 |
| 2026 | 30 / 32 | 0.451 / 0.593 | 均累计力失败 |
| 2027 | 25 / 19 | 1.137 / 1.131 | 均累计力失败 |
| 2028 | 40 / 55 | 0.175 / 0.401 | 均累计力失败 |

基线r3完成5/5，0成功、5策略失败，151次推理、254.0845秒；候选r1完成5/5，0成功、5策略失败，163次推理、266.8217秒。两组均无基础设施失败或未运行例。所有10例始终为reach，未抓持、未切指令；最近距离均大于冻结0.08m grasp切换阈值。候选最近距离2例改善、3例退步，未观察到原生成功提升，不能称泛化改善或架构失败。

首执行动作发生clip的步数为基线142/151、候选163/163，clip标量352/487。此为饱和现象，不建立累计力失败因果；时长不同也使总量不可直接当控制效果比较。首步、5/10/20步TCP及物体base-frame端点变化均保存在摘要，不能解释为世界坐标物体移动。候选2027仅19步，step20明确缺失。

r1基线因venv路径解引用缺NumPy在0.867秒退出；r2因normalizer文件/目录合同在6.887秒退出。两次均零推理、五例未运行，独立保留为基础设施失败，未并入原生策略分母。修复经37项实验室CPU测试通过，最终有效协议为基线v4/候选v3，历史冻结文件仍保留。

G1失败，无成功例可进入独立复跑。复跑、LightNav交接(G2)、GPT链(G3)及Place均未运行，不另开训练或API试验。

### 唯一下周建议（未执行）

进行一次有界的**底座运动通道干预诊断**：固定step250和同五个原生初态，各配对正常执行与仅将底座平移/yaw指令置零，保持其他动作、RNG、指令与原生终止不变，每例上限20动作，不学习、不搜缩放系数、不选checkpoint。比较最大共同步数（不超过19）内的累计力、TCP物距变化及具体接触link，并记录世界坐标base/TCP/object位姿。若力下降而接近也受损，只说明该干预下移动与接触的权衡；若力不降，则削弱简单底座指令解释。它用于区分剩余不确定性，不是已证明根因、策略修复或原协议新成绩。

历史固定整句0/10的原始launch/metadata实际对应普通S1 best6000，不能归给IA best855。历史身份核查与原始文件见[更正回执](results/two-day-delivery-2026-09-20/historical-identity-audit/receipt.json)。任何oracle新结果不能直接和不同checkpoint、不同指令协议的历史0/10计算提升。

本次离线诊断不产生新仿真录像。历史best855的reach失败例为 [fetch-s1-ia-reach-seed2024-episode000-failed.mp4](media/s1-ia-calls-2026-09-18-run01/fetch-s1-ia-reach-seed2024-episode000-failed.mp4)，SHA256 `665f9731ece50d696f89a576f6f951f83179e88b547677edfe9e0de0fdbd66d1`。它来自SAC准备的独立调用起点，不是此次诊断结果或自主完整抓取。

## 复现与开销

[新进程CPU复核说明](two-day-reproduction-2026-09-20.md)、[小型原始诊断证据](results/two-day-delivery-2026-09-20/diagnostic/)、[完整资产备份回执](results/two-day-delivery-2026-09-20/asset-backup.json)。S1和S2完整第二副本已校验；新诊断备份状态单列。新租GPU与外部API为0；实验室费用未知，历史Runpod停止存储继续计费，不能记为免费。账单与资源收尾另见finance ledger。
