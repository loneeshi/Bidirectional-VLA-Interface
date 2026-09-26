# C2 图像证据

## seed4 固定底盘延长时限诊断（2026-09-25 UTC）

[seed4-fixed-base-extended-001-head-hand-2x-failure.mp4](c2-pick-2026-09-25-seed4-extended/seed4-fixed-base-extended-001-head-hand-2x-failure.mp4)：同一物理起点仅把剩余 Pick 动作从 39 补到 200。左侧头部、右侧腕部，均为每个动作执行前的真实 128×128 图像，放大两倍；20 Hz 原控制以 10 fps 回放。固定底盘、无新碰撞力，但始终未碰到或抓住桌上的碗，修改时限条件下失败。此视频不是原始 benchmark 成功或失败的独立评分；逐步数值见[单项诊断](../../research/c2/diagnostics/2026-09-25-fixed-base-extended/seed4-fixed-base-extended-horizon.md)。SHA-256 见[媒体清单](manifest.json)。

## seed4 固定底盘 SAC 失败轨迹（2026-09-25 UTC）

[seed4-fixed-base-sac-001-head-hand-2x-failure.mp4](c2-pick-2026-09-25-seed4-fixed-base/seed4-fixed-base-sac-001-head-hand-2x-failure.mp4)：左为头部、右为腕部，使用每个 Pick 动作执行前的真实 128×128 图像，仅放大两倍；原控制频率 20 Hz，视频以 5 fps 慢放四倍。底盘前进和转向指令均为物理零速，实际最大平移漂移 0.006 m；39 步后官方严格 Pick 失败，从未碰到或抓住碗。结尾卡片来自官方评分，最后动作后没有相机帧。SHA-256 见[媒体清单](manifest.json)。

## seed4 Oracle–SAC 失败轨迹（2026-09-24 UTC）

由该批逐动作保存的真实 `fetch_head` / `fetch_hand` 图像并排制作；每张图是对应 Pick 动作**执行前**的相机观测。原控制频率 20 Hz，视频以 5 fps 慢放四倍。结尾失败卡片来自官方评分，最后一次动作后的相机图像未记录。两条轨迹均未完成严格 Pick。

| 轨迹 | 视频文件 | 结果 |
|---|---|---|
| 第一次有效轨迹 | [seed4-oracle-assistant-001-retry2-head-hand-slow4x-failure.mp4](c2-pick-2026-09-24-seed4-oracle/seed4-oracle-assistant-001-retry2-head-hand-slow4x-failure.mp4) | 第 4 步底盘碰沙发；之后助手检查点超时，最终第 39 步时限失败。 |
| 修订后共驾轨迹 | [seed4-oracle-assistant-002-head-hand-slow4x-failure.mp4](c2-pick-2026-09-24-seed4-oracle/seed4-oracle-assistant-002-head-hand-slow4x-failure.mp4) | 助手在第 0/10/20/30 步的指令均执行；第 14 步碰沙发，第 37 步累计力超限。 |

原始逐动作图像和动作、受力记录保留在 seed4 诊断证据包；视频哈希见[媒体清单](manifest.json)。

## 先前 C2 图像证据

以下 C2 A3 图像是实验中 GPT 实际看到的 SAC **调用前**候选拼图。每格是从相同冻结交接直接设置对应底盘站位、执行一帧全零稳定动作后的相机图像；它不展示机器人移动过程，也不展示 SAC 执行结果。

| 状态 | 头部相机拼图 | 手部相机拼图 |
|---|---|---|
| seed9 | [seed9-fetch_head.png](c2-pick-2026-09-23/seed9-fetch_head.png) | [seed9-fetch_hand.png](c2-pick-2026-09-23/seed9-fetch_hand.png) |
| seed19 | [seed19-fetch_head.png](c2-pick-2026-09-23/seed19-fetch_head.png) | [seed19-fetch_hand.png](c2-pick-2026-09-23/seed19-fetch_hand.png) |

原始实验工作区和已归档的 A2 证据包没有这批候选 SAC 的视频文件，因此没有这组候选 SAC 的执行录像。上周主实验的整集视频属于另一轮实验，不能代替此处的配对站位录像。上面的 seed4 视频也属于另一项诊断。

发布文件的 SHA256 与冻结 GPT 输入的对应图像一致，见[媒体清单](manifest.json)。
