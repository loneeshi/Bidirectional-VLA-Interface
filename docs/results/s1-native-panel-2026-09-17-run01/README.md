# S1 原生能力评估：0/10，未通过

SFT完成不等于能力通过。固定10个val起点，200动作上限，沿用原生终止；9个在第199动作原生失败结束，seed2031在25动作因累计接触力5043.03超过5000结束。全部ever_grasped=false，无基础设施报错。没有修改阈值或按测试选择检查点。

官方pi05_base目标域普通LoRA SFT：150训练父轨迹，50验证父轨迹，6835更新。按150个固定留出帧动作loss选best/6000，loss0.124100，最终6835步0.125258。TAPT尚未训练，不能归因于进度头。

| seed | 动作数 | 原生成功 | 曾持物 | 视频（完整文件名见链接） |
|---|---:|---|---|---|
| 2024 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2024-episode000-failed.mp4) |
| 2025 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2025-episode000-failed.mp4) |
| 2026 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2026-episode000-failed.mp4) |
| 2027 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2027-episode000-failed.mp4) |
| 2028 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2028-episode000-failed.mp4) |
| 2030 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2030-episode000-failed.mp4) |
| 2031 | 25 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2031-episode000-failed.mp4) |
| 2032 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2032-episode000-failed.mp4) |
| 2033 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2033-episode000-failed.mp4) |
| 2034 | 199 | 失败 | False | [录像](../../media/s1-native-panel-2026-09-17-run01/fetch-s1-native24-pick-seed2034-episode000-failed.mp4) |

参数/归一化哈希见native-capability.json及server-metadata.json。所有逐步事件、episode报告及失败录像已归档；原始观测请求与初始状态保存在本地.runtime/backups/s1-native-panel-2026-09-17-run01.tar.gz及服务器原目录，备份SHA已核验，见archive-receipt.json。未删除服务器检查点。

评估进程已退出，GPU1=15MiB/0%。API调用0、新租机USD0；lab费用未知，历史停止云存储费用持续且本轮未刷新账单。S2不放行。

下一步先用现有离线观测/动作日志审计动作语义与分布、接近/对齐/闭爪失败阶段，再按plan v2核算一次特权state诊断臂。此次未自动启动诊断训练或扩预算；不能仅凭loss下降排除接口问题或断言视觉是唯一瓶颈。
