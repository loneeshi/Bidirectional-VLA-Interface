# C2 图像证据

本目录仅发布实验中 GPT 实际看到的 SAC **调用前**候选拼图。每格是从相同冻结交接直接设置对应底盘站位、执行一帧全零稳定动作后的相机图像；它不展示机器人移动过程，也不展示 SAC 执行结果。

| 状态 | 头部相机拼图 | 手部相机拼图 |
|---|---|---|
| seed9 | [seed9-fetch_head.png](c2-pick-2026-09-23/seed9-fetch_head.png) | [seed9-fetch_hand.png](c2-pick-2026-09-23/seed9-fetch_hand.png) |
| seed19 | [seed19-fetch_head.png](c2-pick-2026-09-23/seed19-fetch_head.png) | [seed19-fetch_hand.png](c2-pick-2026-09-23/seed19-fetch_hand.png) |

原始实验工作区和已归档的 A2 证据包没有这批候选 SAC 的视频文件，因此当前不发布“执行录像”。上周主实验的整集视频属于另一轮实验，不能代替此处的配对站位录像。

发布文件的 SHA256 与冻结 GPT 输入的对应图像一致，见[媒体清单](manifest.json)。
