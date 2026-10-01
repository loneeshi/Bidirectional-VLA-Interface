# 高清视觉接口 r4：渲染修复通过本次检查，第二次转头未过门

2026-09-30 UTC。用户 resume 后，独立授权 GPU1 串行5/55/58，3×300=900进程秒，API0，旧余额不转入。部署SHA256 b58156626b5d303cc904d5dd747da3de947fb2a6ea35e124c23392bec9e02d40；452项CPU测试通过。

## 所在阶段 → 推进 → 阻塞 → 下一道门

所在阶段：高清视觉接口/G2准备。推进：5号从原快照恢复，实际2048传感器读取和77步的直接评估相机录制未触发状态变化检查；媒体交付完整。第一条look通过。

阻塞：第二条look在40步上限返回未到位。第一条目标pan +0.30，用37步，最终误差0.002367rad；第二条目标pan -0.30，用40步，实测-0.075517rad，误差0.224483rad、pan速度-0.152557rad/s。不是渲染误判：未达到0.02rad/0.03rad/s门槛。累计力0，官方fail=false，剩123步；不能称为官方Pick失败。runner把接口断言归为infrastructure_censored，原标签保留。

代码证据：VisualTools.look复用settling_action，动作限幅±0.25；两段记录均触及该限幅。应检查头部控制是否不适当地继承停稳限幅，而非放宽40步或到位标准。此次不修改控制、不自动重跑。

两次已完成观察的头/手相机都有目标像素，G2 P95约1.45–1.56mm，外参最大差≤5.061e-7。仅说明这些帧满足门槛；未完成回位及55/58，不能宣布整套G2/接口通过。

下一道门：CPU审查头部控制，冻结修订与新的有界复测。尚不进入Astra试点。

## 结果与媒体

- [完整结果](remote-results/results/plan-005-visual-interface/result.json)
- [原始批次回执](remote-results/usage-ledger.json)
- [完整执行历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-ed6d11d5/delivery/history.html)
- [online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-ed6d11d5/delivery/online-command-vs-executed.mp4)
- [trajectory.html](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-ed6d11d5/delivery/trajectory.html)
- [全部交付与哈希](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-ed6d11d5/delivery/README.md)

这是API0脚本化接口检查，不是Astra在线决策。look没有模型给出的TCP路径，视频对应时段明确标为无metric path；不伪造预测轨迹。视频共78帧（初始+77步）。侧栏视频中的strict Pick failure字样只是来源布尔值false的通用渲染标签，在本API0条件下不代表进行过抓取试验；以本回执为准，后续标签应按接口条件区分。

## 用量和停止

GPU1 54.594382889568806进程秒，API0，环境77步；55/58未启动。GPU1收尾15MiB/0%、无计算进程，GPU0未触碰。实验室项目收费0依据用户确认、无发票；其他历史未知费用不变。未用845.405617秒关闭，不转移。财务ledger-eef-visual-interface-20260930r4.json已结算。没有新增严格Pick成功，整体实验未完成。
