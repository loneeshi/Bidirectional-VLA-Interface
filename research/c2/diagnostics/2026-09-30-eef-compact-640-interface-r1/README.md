# 640视觉接口 r1：三例通过（API0）

所在阶段：C2视觉工具接口门。推进：5/55/58三例真实640 RGB-D、转头回位、只读检查与双相机G2通过。阻塞：尚无640 Astra模型任务结果。下一道门：独立低预算模型试点。

|案例|look1|look2|return_to_rest|环境步数|
|---|---:|---:|---:|---:|
|5|19|26|2|47|
|55|19|26|2|47|
|58|27|26|2|55|

四张图确为640原尺寸，FOV保持。回位按官方ee_rest/robot_rest/is_static，不声称头部精确回零。视频采集保持状态相等检查。这里只验证接口，不是Pick试验，不新增任何Pick分母或Astra能力成功。

G2：头部最坏帧P95 0.007559828m、最大0.007645356m；手部P95 0.001630236m、最大0.002082039m。外参最大绝对误差8.686972e-7，均过原门槛。采样目标像素头1168/手5189，所采样目标像素无效深度比例0；不推广为全图比例。采用同渲染器表面参考，不是独立深度噪声验证。与2048接口头部最坏帧P95 2.649mm相比，本次7.560mm更大；这不是严格受控精度实验，但差异必须保留。

## 媒体

下列online-command-vs-executed.mp4为在线采集的场景叠加（不是重放）。本批命令由脚本给出，不是Astra预测。通用侧栏online-tcp-demo.mp4把strict_pick_success=false显示为failure是既有展示限制，应按本README的“接口测试，未测Pick”解释；推荐下列命令/实测场景叠加版本。

- 5号：[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-45534624/delivery/online-command-vs-executed.mp4)；[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-45534624/delivery/history.html)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-45534624/delivery/trajectory.html)。
- 55号：[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-bf8f349d/delivery/online-command-vs-executed.mp4)；[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-bf8f349d/delivery/history.html)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-bf8f349d/delivery/trajectory.html)。
- 58号：[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan058-9f6a0a93/delivery/online-command-vs-executed.mp4)；[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan058-9f6a0a93/delivery/history.html)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan058-9f6a0a93/delivery/trajectory.html)。

视频、原始轨迹、传感器、请求与指令均在remote-results及docs/media保留；哈希逐项核对并写入媒体索引。原始授权包SHA见authorization.json对应CPU deployment-manifest.json。最终回执visual-interface-proof.json带resolution640/context_version，旧2048不能替代。

GPU1串行3进程，共114.47278877720237进程秒、149环境步、API0。资源最终核验所有本批PID退出，GPU1无compute进程。实验室项目收费0（用户确认，无发票），未用785.527211秒关闭；历史API未知费用/存储继续单列，不转入后续预算。
