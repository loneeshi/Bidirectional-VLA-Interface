# 高清视觉接口 r5：通过（API 0）

所在阶段：C2 视觉闭环前的接口验收。推进：453项CPU回归通过，5/55/58三例接口与G2均通过。阻塞：Astra试点尚未授权、尚未运行。下一道门：独立三例视觉试点授权。

修复：录像改为直接读取评估相机，保持完整状态严格相等检查；主动头部通道使用官方±1范围与每步±0.25斜率限制，不再借用保持通道±0.25限幅。v13底盘、40步look门、官方200步及成功判据均不变。保留r2/r3/r4原结果。

|案例|第一次look|第二次look|return_to_rest|总步数|
|---|---:|---:|---:|---:|
|5|19|26|2|47|
|55|19|26|2|47|
|58|27|26|2|55|

两次look均到位，最大头部残差0.002330 rad。回位满足官方ee_rest、robot_rest、is_static；官方关节容差允许头部仍偏约0.298 rad，不声称精确回到头部关键帧。只读检查通过，2048实际传感器图像与序列化请求尺寸检查通过。

G2：头部最坏帧P95=0.002649 m、最大0.003065 m，手部P95=0.001471 m、最大0.002133 m；外参最大绝对误差8.687e-7。有效目标分割像素头11912、手53100，在这些已采样目标像素中无效深度比例均0；不推广为全图无效率。参考为同渲染器表面位置，非独立深度噪声测量。

这是脚本化接口验证，不是Astra运行，不是严格Pick试验，不更新任何Pick分母；任务整体未完成。侧栏通用视频模板把strict_pick_success=false显示成failure，这是展示限制，不能解读为本批Pick失败；推荐下列命令/实测叠加视频。命令线是指令，不是Astra物理预测，本批没有Astra指令。

- 5号：[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-7856f66d/delivery/history.html)；[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-7856f66d/delivery/online-command-vs-executed.mp4)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-7856f66d/delivery/trajectory.html)。
- 55号：[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-3e212116/delivery/history.html)；[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-3e212116/delivery/online-command-vs-executed.mp4)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-3e212116/delivery/trajectory.html)。
- 58号：[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan058-812ca75d/delivery/history.html)；[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan058-812ca75d/delivery/online-command-vs-executed.mp4)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan058-812ca75d/delivery/trajectory.html)。

部署SHA-256：`f51020b001302e12156497bc657042e6afd2cdfb3eb34a83737bf3288119efb0`。源码/参数/测试哈希见cpu-gates.json、authorization.json；原始记录见remote-results/，汇总result-summary.json，媒体哈希已核验并注册docs/media/manifest.json。

资源：GPU1串行3进程，157.1087709031999进程秒，149环境步，API0。实验室项目收费0（用户确认，无发票）；已核验GPU1无进程，742.891229秒未用额度关闭，不转入下批。历史Runpod存储账单未刷新，费用仍单列未知。
