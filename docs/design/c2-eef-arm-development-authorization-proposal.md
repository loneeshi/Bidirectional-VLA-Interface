# 官方出生站位抓取测评：修订二开发组授权提案

**重新提交，未批准。旧USD1800/500次方案已撤回；以主设计修订二、已决定事项17为准。**

开发组：arm-dev-000, arm-dev-001, arm-dev-002, arm-dev-003, arm-dev-004。原前5例覆盖middle/high，无需替换；原开发10例没有low层，不挪用测试样本。测试30行逐字段保持不变，以清单canonical SHA验证。高度仍是初始碰撞网格最低world-z的支撑面估计。

清单SHA256 `55372a36bc3549ac936bdfed06b0011350ff197f7b135c9e6d107f94c4b95fb8`。

执行器协调v2源码SHA不变；每移动150/总600非官方条件、内部601/外层600补偿reset，五项判据与累计力不变。首次获批运行必须核验，不符合即停，不修改计数。SAC官方200；200/600是同次600步尝试前缀。

P入口、特权白名单、P提示、配对汇总已移除。locate_point仅在评估端记录返回点到目标碰撞形状三角形表面的最短无符号距离（米），当前base_link转目标局部坐标；内部点仍到表面计算距离，不能用AABB内部零距离替代。无有效返回点记NA。诊断独立保存，不写入工具结果、history或请求。

640 r1–r3历史40个逻辑请求，旧重试使实际发送为42次；37份唯一已知usage、5次发送usage未知（含补发）。输入中位数10169、最大14676，未知不当0，重复归档不重复统计。第25轮3万输入+4000输出按普通输入$10/百万及输出$50/百万为$0.50，125次$62.50。按全缓存写入$12.5/百万保守算则$71.875，可能提前触及$65；该额度不保证125次用满。保留逐次实际wire预留，已计/未知预留+新预留将超$65即发送前停批；不自动加钱，不改模型/提示/图像或截断历史。不以26.9万文字安全上限设批次额度。

历史Astra实测进程耗时（包括失败/删失，非600步耗时保证）：

| 历史批次 | case | 进程秒 | 返回码 |
|---|---:|---:|---:|
| r1 | 005 | 210.695 | 1 |
| r2 | 005 | 220.650 | 1 |
| r3 | 005 | 362.425 | 0 |
| r3 | 055 | 166.040 | 0 |
| r3 | 058 | 32.276 | 1 |

资源硬上限：125次API、USD65；每V1800秒、script600秒、SAC300秒，共13500秒（3小时45分）。每例V、SAC、script各一次，共15独立进程，同UID/index/seed核对初态。所有可能计费发送均计入125，每V最多25发送/25工具，输出4000token。未知响应停批核账、不重发，不继承旧额度。

脚本600步严格成功>=4/5才允许提出测试组提案；未达即停交用户决定，不修补。测试30例本次不运行。每次录像、同步TCP曲线及分析页、完整轨迹和媒体索引/哈希规则保持。开发结果只进diagnostics，测试正式日志需按预注册完成后用户确认。

关节空间候选141–147参考点比末端直线87–96更多，不合理。远端目标IK构型或连续关节未按最短方向绕均为待查假设；当前不修、不重跑候选。

API只走实验室服务器代理，密钥规则保持执行器计划第三节第1条：~/.config/bvi/openai.env，目录700/文件600，只运行时读，不复制或打印。获批后再确认调度许可、GPU1空闲、源码/资产哈希和真实初态。旧授权模板失效，新模板status=not_authorized。

[CPU回执](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-10-01-arm-capability-revision2-cpu/README.md) · [冻结清单](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-10-01-arm-capability-revision2-cpu/roster.eval-only.json) · [历史实测凭据](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-10-01-arm-capability-revision2-cpu/historical-usage-and-process-time.json) · [源码哈希](https://github.com/loneeshi/Bidirectional-VLA-Interface/blob/19bb15f07ffa6e464e614419b2156cbb5f87680d/research/c2/diagnostics/2026-10-01-arm-capability-revision2-cpu/deployment-freeze.json)

批准后入口为冻结包内run_arm_capability_batch.py，参数authorization/roster/code/lab-root。缺媒体、接口错误、资源触界或未知响应即停批，不能自动重跑。当前不运行、不发布GitHub。
