# 演示与图像证据

## C2-r2.5 基线 20 例（2026-10-07）

20 例各一次物理尝试，严格 Pick 成功 1 例（005）。视频是运行时在线录制，不是重放。粉色虚线是 Astra 下发的目标路径，青色是实测末端轨迹，橙色是底盘指令。叠加层和渲染相机只用于评估，没有进入模型请求。每个目录另有 `online-tcp-demo.mp4`、`online-command-vs-executed.mp4` 和 `online-raw.mp4`。失败归因见[报告](../failure-attribution/2026-10-08-c2-r25-baseline20/report.html)。

- baseline-000，265 步，失败（模型放弃）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-000/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-000/trajectory.html)
- baseline-001，377 步，失败（模型放弃）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-001/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-001/trajectory.html)
- baseline-002，286 步，失败（累计力超限）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-002/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-002/trajectory.html)
- baseline-003，176 步，失败（累计力超限）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-003/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-003/trajectory.html)
- baseline-004，253 步，失败（模型放弃）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-004/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-004/trajectory.html)
- baseline-005，245 步，成功：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-005/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-005/trajectory.html)
- baseline-006，330 步，失败（累计力超限）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-006/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-006/trajectory.html)
- baseline-007，141 步，失败（操作员停止）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-007/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-007/trajectory.html)
- baseline-008，487 步，失败（模型报告完成，严格判定失败）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-008/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-008/trajectory.html)
- baseline-009，262 步，失败（模型放弃）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-009/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-009/trajectory.html)
- baseline-010，678 步，失败（运行时限截断）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-010/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-010/trajectory.html)
- baseline-011，532 步，失败（累计力超限）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-011/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-011/trajectory.html)
- baseline-012，31 步，失败（累计力超限）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-012/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-012/trajectory.html)
- baseline-013，505 步，失败（模型放弃；基础设施重试 attempt-1）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-013-attempt-1/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-013-attempt-1/trajectory.html)
- baseline-014，414 步，失败（模型报告完成，严格判定失败）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-014/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-014/trajectory.html)
- baseline-015，624 步，失败（累计力超限）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-015/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-015/trajectory.html)
- baseline-016，663 步，失败（模型放弃）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-016/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-016/trajectory.html)
- baseline-017，312 步，失败（累计力超限；基础设施重试 attempt2）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-017-attempt2/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-017-attempt2/trajectory.html)
- baseline-018，431 步，失败（累计力超限）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-018/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-018/trajectory.html)
- baseline-019，774 步，失败（预算用完）：[astra-commanded-vs-executed-3d-demo.mp4](c2-r25-baseline20-2026-10-07-baseline-019/astra-commanded-vs-executed-3d-demo.mp4) · [trajectory.html](c2-r25-baseline20-2026-10-07-baseline-019/trajectory.html)

## 官方状态字段复测 5 例（official-state，2026-10-01）

[同场景目标路径 / 实际轨迹演示清单](c2-pick-arm-official-state-2026-10-01-r1/README.md)：同 5 例，模型另得官方目标位姿、末端位姿与抓持状态；严格 Pick 200 步 2/5、600 步 5/5。原在线录像的 CPU 后处理，无动作重放；颜色含义同下。清单内另有每例的轨迹、逐步记录与接触分析页。实验说明见[日志](../log/2026-10-01-c2-arm-official-state.md)。

- arm-dev-000 成功（第 213 步）：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-3e756bde/delivery/astra-commanded-vs-executed-3d-demo.mp4)
- arm-dev-001 成功（第 239 步）：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-4beb91ee/delivery/astra-commanded-vs-executed-3d-demo.mp4)
- arm-dev-002 成功（第 123 步）：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-eecf43b2/delivery/astra-commanded-vs-executed-3d-demo.mp4)
- arm-dev-003 成功（第 257 步）：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-691cb7ae/delivery/astra-commanded-vs-executed-3d-demo.mp4)
- arm-dev-004 成功（第 145 步）：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-4d31155e/delivery/astra-commanded-vs-executed-3d-demo.mp4)

## 夹爪几何复测 5 例（WP2，2026-10-01）

[同场景目标路径 / 实际轨迹演示清单](c2-pick-arm-wp2-2026-10-01-r1/README.md)：提示追加夹爪几何后的同 5 例，严格 Pick 2/5。原在线录像的 CPU 后处理，无动作重放；颜色含义同下。实验说明见[日志](../log/2026-10-01-c2-arm-wp2-gripper-geometry.md)。

- arm-dev-000 成功：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-8a8902b7/delivery/astra-commanded-vs-executed-3d-demo.mp4)
- arm-dev-001 失败（模型放弃）：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-90194498/delivery/astra-commanded-vs-executed-3d-demo.mp4)
- arm-dev-002 失败（累计力超限）：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-2c00e1ec/delivery/astra-commanded-vs-executed-3d-demo.mp4)
- arm-dev-003 失败（抓错物体后报告完成）：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-83a042d7/delivery/astra-commanded-vs-executed-3d-demo.mp4)
- arm-dev-004 成功：[astra-commanded-vs-executed-3d-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-f17e6eca/delivery/astra-commanded-vs-executed-3d-demo.mp4)

## 可移动底盘 r1 5 例

[同场景目标路径 / 实际轨迹演示](c2-pick-arm-dual-path-2026-10-01-mobile-r1/README.md)包含 `arm-dev-000` 至 `arm-dev-004` 的 5 个录像，成功 1 例、失败 4 例。原在线录像的 CPU 后处理，无动作重放。粉色虚线为 Astra 已发出的目标点连线，青色为真实 TCP，橙色为底盘指令；叠加标注为评估端 X-ray。

## 历史媒体

- [固定底盘成功例 online-tcp-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-000-V-7af6e6c7/delivery/online-tcp-demo.mp4)。
- [可移动底盘成功例 online-tcp-demo.mp4](c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-f7ff8757/delivery/online-tcp-demo.mp4)。
- [seed4-fixed-base-extended-001-head-hand-2x-failure.mp4](c2-pick-2026-09-25-seed4-extended/seed4-fixed-base-extended-001-head-hand-2x-failure.mp4)：修改时间预算的失败诊断。
- [seed4-fixed-base-sac-001-head-hand-2x-failure.mp4](c2-pick-2026-09-25-seed4-fixed-base/seed4-fixed-base-sac-001-head-hand-2x-failure.mp4)：固定底盘失败诊断。
- [seed4-oracle-assistant-001-retry2-head-hand-slow4x-failure.mp4](c2-pick-2026-09-24-seed4-oracle/seed4-oracle-assistant-001-retry2-head-hand-slow4x-failure.mp4)：失败诊断，精确文件以清单为准。

文件、录制日期、结果和 SHA-256 见 [manifest.json](manifest.json)。完整在线录像、传感器帧和原始轨迹仍在本地保留。
