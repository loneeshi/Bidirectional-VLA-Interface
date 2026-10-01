# 站位就绪抓取开发组：revision 2，r3 完成并停止

当前主线：阶段 2，接口执行与失败归因。冻结的 5 例、V/SAC/脚本共 15 个进程完成，无缺失或删失；脚本严格成功 0/5，未达到 4/5 测试准入门槛。测试组未启动，不进入新一轮修补。下一门槛是用户对失败开发门槛的后续决定；此结果不代表 TAPT 训练完成或原生基准复现。

## 执行器 CPU 验证

相关 CPU 回归 41 项通过，见 [cpu-regression.xml](cpu-regression.xml)。修正了历史快照共享引用导致的请求审计中断，并使重试额度携带前次 API 消耗；协调 v2 的物理执行器、guidance、开发组顺序均未改变。冻结源码见 [deployment-freeze.json](deployment-freeze.json)；目标 IK 关节空间候选桌面参考点 141–147 多于末端直线 87–96 的问题仍只记录，不修。

V/脚本为非官方时间条件：每条移动 150 步、总计 600 步，判据仍是官方五项；同时统计同一次运行首次满足严格成功是否在 200 步内。固定 MS-HAB 的 Pick 内部时限为 601，用于抵消 reset 时 evaluate 的递减，外层严格限制 600。SAC 原生参照只运行官方 200 步，未延长或重跑。

## 脚本参照与 Astra V

| 条件 | 严格成功 k/n | 95% Wilson 区间 | 首次严格成功 |
|---|---:|---|---|
| 脚本，200/600 步 | 0/5 | 0–43.45% | 无 |
| Astra V，200 步 | 0/5 | 0–43.45% | 无 |
| Astra V，600 步 | 1/5 | 3.62–62.45% | arm-dev-000，第 217 步 |
| SAC，官方 200 步 | 4/5 | 37.55–96.38% | 第 41、24、39、56 步 |

P 已移除，不存在 P 配对统计。脚本成功子集为空，分母 0，比例和区间不定义，不能将其报告为 0% 的 Astra 成功率。

| 实例 | 类别 / 原高度层 | V：终止 / 实际步数 | SAC | 脚本 |
|---|---|---|---|---|
| 000 | tomato_soup_can / middle | 严格成功 / 217 | 力限制，5 步 | front 被接受，但初始移动 reference_timing_infeasible，5 步 |
| 001 | gelatin_box / middle | give_up / 144 | 成功，41 步 | no_accepted_path，0 步 |
| 002 | master_chef_can / high | give_up / 214 | 成功，24 步 | no_accepted_path，0 步 |
| 003 | potted_meat_can / middle | give_up / 393 | 成功，39 步 | no_accepted_path，0 步 |
| 004 | cracker_box / middle | give_up / 122 | 成功，56 步 | front 被接受，但 reference_timing_infeasible，5 步 |

漏斗按实例分别计数：V 的 check_path 接受、TCP 到包围盒 ≤5 cm、is_grasped、抓住时上升 ≥5 cm、官方严格成功依次为 5/5、1/5、1/5、1/5、1/5；脚本依次为 2/5、0/5、0/5、0/5、0/5。SAC 不调用 check_path，该项为不适用，其余四项均为 4/5。完整机器可读定义和统计在 [summary.json](summary.json)。

五例来自五个不同场景，同场景的三个条件相关，不能把 15 个进程当作 15 个独立实例。Wilson 区间为描述性区间，未校正场景聚类。middle 层 4 例、高层 1 例；五个“类别 × 高度”单元各 n=1，无法据此推断分层泛化。沿用冻结前五例，未用 C2 测试集快照替换。

## 失败归因与定位诊断

按照预注册规则，V 和脚本都失败的 001–004 标为“执行器不可行”；这是所冻结脚本的操作性标签，不能证明所有固定底盘抓取都无解。000 的脚本失败而 V 成功，直接说明该标签不是普遍不可达证明。

定位诊断在评估端计算每个 locate_point 到目标碰撞三角形表面的最小距离，不是到包围盒的距离，未进入任何请求。模型笔记是自述，以下结论结合执行轨迹与距离检查。

- **000**：模型先记录 “inclined approach passed”，整段移动收到 reference_timing_infeasible 后拆分中间位姿，再闭合、提升、回到 rest。除一次有意查询座椅外，目标定位误差约 0.2–0.9 mm；第 217 步严格成功，支持闭环恢复有效。
- **001**：模型记录两路深度把目标放在约 `[1.67, 0.23, 0.43]` m，并认为超出固定底盘水平可达范围。两次定位到真实目标表面的距离约 2.1、0.7 mm，不能归因为认错物体。脚本也没有接受路径；现有证据支持可达性限制，尚未证明不存在其他抓取方案。
- **002**：最初试探的红罐点距真实目标约 0.406 m，随后改为蓝色罐，误差约 0.3–1.0 mm。hindsight 记录 “fingers closed empty with stable=false” 和四次插入修正无 IK 解。空抓与定位修正均可核对，但无法确定剩余失败是细部姿态还是固定底盘可达性：机制归因为未定。
- **003**：模型称多次朝向 IK 失败，接受的接近仍把目标留在指尖外。四次查询点全部距真实目标约 2.241–2.246 m，显示持续错误的目标位置信念；另有 150 步耗尽后仍差约 0.238 m 的执行偏差。错误信念和执行受限同时存在；脚本失败使其不进入“脚本成功而模型失败”的主归因子集。
- **004**：五次定位误差均 ≤2.4 mm，目标定位正确。模型 hindsight 记录 “closure reported zero finger separation and stable=false”，抬起时盒子仍在柜上，并称无效调用耗尽预算。CPU 审计确认全批 13 次调用因非单位 xyzw 四元数被拒绝，见 [invalid-validation.cpu.json](invalid-validation.cpu.json)。这属于接口参数拒绝，不应计为 IK 无解；具体空抓原因仍未定。此时不再修接口或重跑。

SAC 的原生动作包含底盘移动：成功的 001、002、003、004 相对出生位置最大位移分别为 1.0820、0.5154、1.0640、0.2197 m。因此 SAC 4/5 不能直接作为固定底盘手臂能力上界。V 没有 move_base；003 仍有约 0.2167 m 物理底盘漂移。官方 Pick 出生站位不能自动保证固定底盘手臂可达。逐例笔记、轨迹统计在 [case-evidence.json](case-evidence.json)。

## 资源与证据

累计 API 103/125 次（r3 新增 102，前次 1），依据实际返回 token 估算 USD 19.817925/65；供应商实账待核，不当作实际现金支付。没有未知预留。GPU1 累计 2,136.2828/13,500 秒，约 35.6 分钟，完成后核实空闲；GPU0 的他人进程未触碰。未使用的 USD 45.182075、22 次请求及 GPU 额度随失败门槛关闭，不转为其他实验授权。实验室计算收费按既有用户确认记 0，历史 Runpod 存储状态未刷新，持续费用另行待核。见 [closeout.json](closeout.json)、[api-ledger.json](api-ledger.json)、[batch.json](batch.json)；外部财务账和主线日记同步记录。

12 个有物理步数的进程均保留在线录像、同步生长 TCP 场景曲线、命令与执行对照及分析轨迹，共 36 个视频；另外 3 个脚本进程为 0 物理步，不要求录像。所有第三方镜头和评估坐标只用于分析，没有进入模型请求。原始 result 与用于渲染的 rendering-result 分开保留，渲染版路径重写不改变原结果身份。见 [media-verification.json](media-verification.json) 和 [media-index.json](media-index.json)。

成功实例录像：[online-tcp-demo.mp4](../../../../docs/media/c2-pick-arm-2026-10-01-arm-dev-000-V-7af6e6c7/delivery/online-tcp-demo.mp4)；其 [trajectory.html](../../../../docs/media/c2-pick-arm-2026-10-01-arm-dev-000-V-7af6e6c7/delivery/trajectory.html) 是评估端分析。其余在线录像精确文件与哈希见 [媒体清单](../../../../docs/media/manifest.json)。

原始归档 `raw-run.tar.gz` 在本目录本地保留（721,441,352 字节，SHA-256 `bd6140866038a11feb642804664da1de0784d663503622e0a6b54b4142d0c88a`），服务器原运行目录亦保留；大归档不进入 Git。冻结运行源码 zip 已保留并提交。此次只作本地提交，不发布 GitHub，不写入 docs/log/。
