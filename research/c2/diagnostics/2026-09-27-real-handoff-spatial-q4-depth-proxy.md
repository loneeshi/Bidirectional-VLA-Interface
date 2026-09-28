# 阶段 1：Q4 原始深度简单规则

五阶段主线仍在第 2 步。本项是阶段 1 的离线非模型对照，不执行移动、SAC、GPU 渲染或模型 API。规则先写入[冻结标签方法](../docs/real-handoff-spatial-stage1-label-methods.md)，再在冻结 20／40 例上评分；实现见[深度几何函数](../../../src/bvi/offline_spatial_depth_geometry.py)与[CPU 评分器](../../../scripts/score_real_handoff_spatial_q4_depth_cpu.py)。

## 输入与来源

- 每例只读取归档的官方 `fetch_head-depth-raw.npy`、由 URDF 和关节角计算的相机相对底盘位姿。逐例校验原始深度哈希；不把世界物体位姿、碰撞网格或 Q4 标签送入特征计算。
- 头部相机 128×128、纵向 FOV `2` rad、安装于 `head_camera_link`，来自固定 Fetch 源码 SHA-256 `cfa07e1dac21bd75f9f63f31742f4d06dd374ab46eb439b5ab367700100a4b39`。本地[标定审计](2026-09-27-real-handoff-spatial-majority-and-depth-readiness.md)已验证 60／60 例相对位姿。
- 固定的远端 ManiSkill `sensors/camera.py` 将 FOV 传为 `fovy`，源码 SHA-256 `cc8946310612d0da106d3192c2acbae6d94d0f8c8d2cf629b4221e0f09f06da9`；`render/shaders.py` 的 minimal shader 将深度设为 `-Position.z` 毫米，源码 SHA-256 `83655c0128b2acfee7543bff325998f427986cdafb403bd405d0a3b83323a90d`；`utils/sapien_utils.py` 给出 SAPIEN→OpenCV 固定轴变换，源码 SHA-256 `466cce0db68a01493454163c83740578aa197c1b1521367798ce4d5ea0cbf405`。这些源码在本项是只读核对，不改变冻结渲染。
- Q4 真值来自冻结标签文件 SHA-256 `53609a5e4e88795b762843649bb5ffd8bc4a50d440a4e484483df174ee0422ec`；标定回执 SHA-256 `19d82f63a51505005fb7024ac348ea8b517d30613c7ebd40222a6f1d796b02b8`。

规则将可见深度点转到底盘坐标，按固定三方向、低障碍高度带和底盘半径代理取最近距离，再以 Q4 的 0.20／0.50 m 边界分档。没有合格可见点的方向标为不可评；不依据真值填补。它不是连续碰撞检测，也不保证分离地面、机器人本体、目标物体与其他场景物体。

## 测试集结果

| 方向 | 冻结方向数 | 深度可评 | 深度正确 | 同样本“远”正确 | 深度不可评 |
|---|---:|---:|---:|---:|---:|
| 正前 | 40 | 33 | 27 | 21 | 7 |
| 左前 | 40 | 40 | 6 | 27 | 0 |
| 右前 | 40 | 3 | 3 | 3 | 37 |
| 合计 | 120 | 76 | 36 | 51 | 44 |

开发集对应的可评数为正前 16／20、左前 20／20、右前 3／20；没有根据开发或测试成绩更改规则。左前 40 个可评答案全部被规则判为 `near`；右前 37／40 不可评，因此右前的 3／3 不能代表该方向可靠。与“远”多数类的差距只在同样本 76 个方向上比较：36／76 对 51／76。不能把 36／76 写成全部 120 个方向的准确率，也不能把未观测方向填为远。

逐例回执在本地忽略路径 `runs/real-handoff-spatial-stage1-20260927/q4-depth-proxy-60.eval-only.json`，SHA-256 `ccc02fe66912280b9fe6ff986a5152dea5a85f9ba5cd131a5fce092eabc873f8`。Q1／Q3 仍缺非特权目标像素掩码；这份结果仅补 Q4 简单规则。模型主结果、特权消融及闭环效用均尚未测得。模型 API 实际请求 0；下一停止门是独立费用授权。
