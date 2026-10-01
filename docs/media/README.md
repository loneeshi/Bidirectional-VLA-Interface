# 演示与图像证据

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
