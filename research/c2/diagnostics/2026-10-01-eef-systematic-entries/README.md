# C 系统对照：真实入口与停稳后规划核对

所在阶段：第 2 步，失效归因与接口验收。推进：9/9 入口完整恢复、前缀与官方计数核对完成，27 个在线视频交付。阻塞：共同规划器仍拒绝 7/15 条末端路径，4 条时间推进参考超过 60 步。下一门：解决移动组与公共规划/参考预算的不兼容；当前不能启动四条件物理比较或 Astra 试点。

## 实际运行

- r1 启动漏传 `MS_ASSET_DIR`，环境创建前失败，物理动作 0/API 0；回执保留于 [r1-preenvironment-failure.json](r1-preenvironment-failure.json)。新 runner 固定资产目录，补充入口独立恢复与失败停止测试，没有修改任何历史版本。
- r2：GPU1 串行 3 个位置进程、各最多 300 秒，共 397.509984 进程秒；API 0，416 个物理步（冻结前缀 380、停稳 36）。各入口从原首次交接快照独立重建环境。
- 第 58 号历史 turn 8 保留 122 个已消耗步，入口停稳 2 步，剩余 76 步。低位 58 的前缀 140 步、停稳 2 步，剩余 58 步。累计力未清零。
- `raw-entry.pt` 是停稳前的完整配对入口，保存官方计数、控制器和 RNG；未来四条件必须从此恢复并分别支付停稳步数。`settled-entry.pt` 仅用于这次核对，不得拿来跳过入口停稳。
- [入口摘要](entry-summary.json)、[进程回执](process-receipts.json)、[源码/部署冻结](r2-deployment-manifest.json)、[CPU 测试](cpu-tests.json)。完整相关回归 547 项通过、3 个其他 torch 模块跳过；新增入口与汇总专项 4 项通过（汇总测试是在 547 回归之后新增并单独验证）。

## 停稳后仍存在的协议阻塞

15 条末端路径中 8 条规划接受、7 条拒绝：5/58/61 三条地面位姿与 58 历史 turn 8 无法在公共参考生成的 200 步上界内参数化；三个小幅修正返回 IK 搜索失败。IK 搜索失败不是不可达性证明。

三个桌面移动分别需要 87/95/96 个参考点；61 的竖直移动需要 62 点。C/D 每运动步推进一点，分别至少需要 86/94/95/61 个运动步，尚未计入口停稳，已经超过冻结的 60 步规格。5/58 的竖直参考在真实停稳后为 45/55 点，与直接归档重建不同；不能继续把归档状态当作真实入口。

上述是实际入口停稳后进行的只读规划核对。没有执行 18×4 移动，没有四条件物理得分，没有执行五段严格抓取序列；没有新 Astra 成功，不代表 G0 通过。不修改位姿、速度或步数门来消除这些拒绝，也不把这次准备阻塞记成已完成的系统对照失败。

## 媒体和完整档案

9 个入口均有在线原视频、同步 TCP 曲线视频、指令/实际轨迹页面；共 27 个视频，569 个原始文件 SHA-256 已核对。视频展示的是冻结动作前缀回放和实测停稳，不是 Astra 预测，不伪造模型计划曲线。场景覆盖是评估专用，X-ray 标签遵循既有渲染器。

完整快照、逐步轨迹和原始录像帧保存在本地 `runs/eef-systematic-entries-20261001-r2/`，服务器归档也保留。[收集回执](collection-receipt.json)、[原始文件哈希](all-results-sha256.json)、[全部媒体准确路径与哈希](media-index.json)。以下链接逐入口提供原视频、TCP demo 和分析页面：

- handoff-005: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-005-00407866/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-005-00407866/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-005-00407866/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-005-00407866/delivery/trajectory.html)。
- handoff-058: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-058-449e091b/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-058-449e091b/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-058-449e091b/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-058-449e091b/delivery/trajectory.html)。
- handoff-061: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-061-79eabcef/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-061-79eabcef/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-061-79eabcef/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-handoff-061-79eabcef/delivery/trajectory.html)。
- low-005: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-005-ee3d6597/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-005-ee3d6597/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-005-ee3d6597/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-005-ee3d6597/delivery/trajectory.html)。
- low-058: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-058-79d75d00/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-058-79d75d00/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-058-79d75d00/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-058-79d75d00/delivery/trajectory.html)。
- low-061: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-061-eade9fdb/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-061-eade9fdb/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-061-eade9fdb/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-low-061-eade9fdb/delivery/trajectory.html)。
- r2-005-turn-006: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-r2-005-turn-006-bea0b4ed/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-r2-005-turn-006-bea0b4ed/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-r2-005-turn-006-bea0b4ed/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-r2-005-turn-006-bea0b4ed/delivery/trajectory.html)。
- r3-005-turn-005: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-r3-005-turn-005-0f1af4cb/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-r3-005-turn-005-0f1af4cb/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-r3-005-turn-005-0f1af4cb/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-r3-005-turn-005-0f1af4cb/delivery/trajectory.html)。
- retry58-058-turn-008: [online-raw.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-retry58-058-turn-008-cec484cb/delivery/online-raw.mp4)、[online-tcp-demo.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-retry58-058-turn-008-cec484cb/delivery/online-tcp-demo.mp4)、[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-retry58-058-turn-008-cec484cb/delivery/online-command-vs-executed.mp4)、[trajectory.html](../../../../docs/media/c2-pick-systematic-entries-2026-10-01-retry58-058-turn-008-cec484cb/delivery/trajectory.html)。

## 费用与资源

r1 4.227134 秒、r2 397.509984 秒，合计 401.737118 GPU1 进程秒，API 0。两个独立额度均已关闭，剩余不转入。实验室项目收费 0（用户确认、无发票）；本次没有供应商 API 费用，历史待核账单与停止的 RunPod 存储费用继续保留原状态。本阶段 GPU1 已无计算进程；GPU0 的既有 VLLM 未干预，见 [资源回执](resource-final.json)。
