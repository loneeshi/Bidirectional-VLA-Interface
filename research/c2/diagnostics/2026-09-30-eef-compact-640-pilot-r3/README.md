# Compact640 r3：两例正常结束未成功，一例网络删失

所在阶段：C2主视觉开发试点。推进：有界补发已恢复5号一次连接重置，并取得5/55完整失败结局。阻塞：58号首请求重置，补发出现SSLError，当前分类器不重试该类错误。下一道门：确认SSL异常类别并补齐传输测试；分析5号IK等待与执行时长。不启动新批次。

冻结清单3例：0成功、2正常失败、1删失。完成子集0/2，仅作描述；不是完整0/3任务失败率。三例均已暴露开发案例，不推断泛化或学习收益。

|案例|工具调用|API发送含补发|环境步数|结局|
|---|---:|---:|---:|---|
|5|10|11|151|give_up，未抓住|
|55|8|8|31|give_up，未抓住|
|58|0|2|0|网络删失|

## 逐例证据和归因

5号：头部定位，顶部路径被拒后改斜向路径通过；开夹爪4步，两次move_to各60步、位置残差0.040625/0.113616m，手臂IK等待26/52步；return_to_rest27步到位，剩49步时give_up。模型turn6承认预抓取差4.1cm并用手部相机复测；turn8称第二次接近又超时且差11.4cm，保留回位预算。hindsight：“both arm approaches timed out despite an accepted robot-only path check”。归为执行偏离意图：路径静态通过但动态追踪未到位；具体IK等待根因尚未定位，不能自动归因躯干或碰撞。无闭合抓取、无底盘指令。环境步数151=停稳9+运动142。

55号：look31步到位，之后仅定位/只读IK检查和give_up。目标约1.52m远，IK拒绝；随后主动量桌角与椅子，发现一次取点落到低处表面后重新测量。hindsight认为两表面水平距离约0.36m小于底盘直径0.576m，未找到安全替代路线。归因暂为无法判定：这些点不能证明所有路线都不可行，也不能确认该点间距就是整车通道净空。模型避让判断有传感器依据，但没有实际探索其余路线；不称环境必然阻止任务。无move_base，未回休息位即放弃；停稳3+转头28步。

58号：初始请求ConnectionResetError10054，补发SSLError(errno1)，无有效模型指令。日志没有SSL library/reason标识，无法区分EOF、握手或其他TLS异常。当前只重试ConnectionError/TimeoutError，因此只用了1次补发即停止；不能声称所有网络错误重试已覆盖。无hindsight，整体任务不可判定。

## 重试与费用

5号turn006第1次重置、第2次成功；wire哈希相同、attempt ID不同，只执行1次locate_point。未知费用没有因补发成功释放。总21次发送，18有效、3未知，工具18次。实际计数与本地账本/远端回执一致。USD上限来自授权JSON，没有在实现中硬编码金额。

本轮已知保守估算USD2.4490375，未知预留USD1.4206875，实际账单未知。剩余USD11.1302750及GPU剩余额度关闭、不转入。GPU1 560.740894进程秒、182环境步；实验室项目收费0（用户确认、无发票）。PID1763975/1764114/1765410/1766590均退出，GPU1无compute进程；GPU0他人进程未动。历史未知API及RunPod存储另列。

## 媒体和档案

5/55均本次在线录像，非重放；命令曲线是模型指令，不是物理预测。58只有初始帧，0动作，未伪造运动视频。完整请求/响应/note/推理摘要、逐步轨迹保留，媒体哈希全部通过。

- 5号：[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-941271d8/delivery/history.html)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-941271d8/delivery/trajectory.html)。[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-941271d8/delivery/online-command-vs-executed.mp4)。
- 55号：[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-5e7e5289/delivery/history.html)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-5e7e5289/delivery/trajectory.html)。[online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan055-5e7e5289/delivery/online-command-vs-executed.mp4)。
- 58号：[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan058-966f97a5/delivery/history.html)；[交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan058-966f97a5/delivery/trajectory.html)。

部署SHA e5e4aef01a66856c1393589aa9d5bcf4d2d3ff75aeb1ea0c7075d26f50f719c5；启动前20项专项测试通过，前轮480项回归通过。
