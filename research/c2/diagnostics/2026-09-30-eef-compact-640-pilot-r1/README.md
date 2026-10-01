# Compact640 pilot r1 — 网络删失，未完成三例

所在阶段：C2 主视觉闭环开发试点。推进：5号8次有效响应、64环境步，完整历史和在线视频已收齐。阻塞：第9次请求连接重置，供应商结果未知。下一道门：先排查连接稳定性；任何重跑另行授权。

5号 api_censored；55/58未运行。无新增严格Pick成功；不能报告完整0/3任务失败。接口通过与模型任务完成是两回事。

## 关键执行证据

- turn0/1：头部像素定位两次；turn2顶部抓取check_path拒绝，0环境步。
- turn3改成水平接近，check_path通过；这不保证实际时长或无碰撞。
- turn4张开夹爪4步（停稳3步）；turn5末端运动60步（停稳2步、运动58步），未到位，位置残差0.087376m、旋转残差0.598648rad，工具步数上限触发。还有136官方步。
- turn6模型note明确承认“pregrasp stopped 8.7 cm short”，使用手部相机locate_point成功，表面点[0.821739,-0.136400,0.982218]m。不能说本例只看头部图。
- turn7根据手部测量再检查路径。turn8未收到有效响应，没有执行其动作。
- 没有move_base指令；里程计平移0.005666m为漂移，不是主动靠近。未闭合抓取、未return_to_rest、无hindsight；网络删失不额外调用反思。

归因：本例整体任务结果无法判定，停止原因是传输错误而非官方任务终止。turn5按note期待到位但实测未到，属于“执行偏离意图”；尚未定位控制滞后/接触等根因，不能自动归因躯干或模型。不能据此评价抓取策略最终成败。模型确实根据反馈承认残差并换手部视觉复测。

错误证据：URLError -> ConnectionResetError errno10054。它无法区分本机网络/代理/服务端连接问题；不能断言OpenAI故障。较小图片未消除该问题。无自动重试。

## 费用和资源

API9次意图：8有效、1未知。已知用量保守估算USD1.0384875；未知请求保留USD0.69155，不是实际账单。剩余授权USD13.2699625关闭，不自动转入。GPU1 210.695151进程秒；64环境步。实验室项目收费0（用户确认、无发票）。批次PID1759521/1759660均退出，GPU1无compute进程；GPU0其他用户作业未动。历史未知API及RunPod存储另列。

## 媒体与完整档案

在线记录，不是回放；命令曲线表示模型命令，不是物理预测。网络删失标签保留。

- [online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-8ccda183/delivery/online-command-vs-executed.mp4)
- [完整执行历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-8ccda183/delivery/history.html)
- [交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-8ccda183/delivery/trajectory.html)

原始请求、provider响应、note与推理摘要在broker-records；remote-results保留传感器、动作和状态。所有交付哈希通过media-hash-check.json。冻结部署SHA fb9ee5536e1082347798c184cb050c8a977a033275c4a55da96b99c48ac4e6c7。
