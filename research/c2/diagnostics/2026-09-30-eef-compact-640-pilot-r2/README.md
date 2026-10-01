# Compact640 pilot r2 — 重试仍因连接重置删失

所在阶段：C2主视觉闭环开发试点。推进：5号11次有效响应、75环境步；斜向预抓取和抓取位姿均到位，闭合未抓住，模型开始重新定位。阻塞：第12次请求URLError / ConnectionResetError 10054，供应商结局未知。下一道门：排查传输稳定性；不自动重试。

5号api_censored，55/58未运行；无新增严格Pick成功。官方final fail=false，尚余125步，不是官方任务失败；不能报告完整0/3。旧r1原判和未知费用全部保留。

## 执行证据与归因

- 头部locate_point后，顶部抓取check_path拒绝；模型改斜向方案通过。check_path理论下界9步不代表真实执行时长。
- 张开4步；预抓取53步（停稳2、运动51，手臂IK等待5包含在执行内），残差9.320mm/0.02597rad；下降8步（停稳2、运动6），残差9.254mm/0.01899rad；闭合10步（停稳2、运动8）。总75步，停稳9、运动66。
- turn7/10使用手部相机locate_point。turn7 note声称罐体位于张开手指之间，并以新测量准备下降；turn9闭合后指距0，stable=false。turn10 note明确说“the grasp missed”，重新量罐顶准备纠正。
- 评估端is_grasped=false、累计力0，与模型没抓住的判断一致。无move_base、look、return_to_rest；底盘漂移0.005361m不是主动移动。
- 对局部未抓住的根因归为“无法判定”：模型曾相信手指已对准，实际两个move_to在现有到位容差内，但尚未做指尖/罐体几何分析，不能确定是抓取偏置判断、到位残差还是其他因素。任务整体因网络删失，不能判断后续纠正是否成功。无done/give_up、无hindsight；未因网络删失额外收费反思。

URLError只给出连接重置10054，不足以区分本机网络、代理或OpenAI端。传输没有自动重试。此次仅放行新的匹配批次ID/账本；控制器、提示、相机和传输实现保持原冻结值。CPU12项通过。

## 资源费用

API12次意图：11有效、1未知。已知保守估算USD1.5927250，未知预留USD0.7844875；实际账单未知。授权剩余USD12.6227875关闭、不转入。GPU1 220.650403进程秒，实验室项目收费0（用户确认、无发票）。PID1761771/1761910均退出，GPU1无compute；GPU0其他用户进程未动。历史未知API与RunPod存储继续单列。

## 完整交付

视频为本次在线记录，不是回放；命令线是模型目标，不是物理预测。保留网络删失标签。

- [online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-ee07a50f/delivery/online-command-vs-executed.mp4)
- [完整执行历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-ee07a50f/delivery/history.html)
- [交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-ee07a50f/delivery/trajectory.html)

原始请求、响应、note及推理摘要在broker-records，动作传感器在remote-results；交付文件哈希全部核对。冻结包SHA 1cbc3dcb9c7cd86e11b278b923787bd12c47c015fd2b1888dea8dac17c2b1e4d。
