# 同五例 Astra V-mobile：r1 启动回执

## 结项检查：五例完成，600 步严格成功 1/5

批次及所有 worker 已退出，GPU1 核实空闲；五例均为有效完成、无删失，各自录像/轨迹媒体生成通过。整轮 600 步严格成功 1/5（95% Wilson 3.62–62.45%），200 步前缀 0/5（0–43.45%）。没有测试组授权，未重跑或修补。主线仍为阶段 2 的工具接口与失败归因；不是 TAPT 完成或原生基准复现。下方启动状态为历史回执，以本结项节为最新状态。

| 实例 | 物理步数 | 官方严格成功 | 终止 | 最大底盘位移 |
|---|---:|---|---|---:|
| 000 | 226 | 是，首次第226步 | official_success | 0.014 m（未主动移动底盘） |
| 001 | 252 | 否 | give_up | 0.890 m |
| 002 | 180 | 否 | cumulative_force_limit | 0.249 m |
| 003 | 228 | 否 | done | 0.003 m（主动原地转向） |
| 004 | 184 | 否 | give_up | 0.135 m |

check_path 接受 5/5，TCP 到目标包围盒≤5 cm为3/5，目标 is_grasped、抓住时升高≥5 cm、严格成功均为1/5。003 的手指 stable=true 不是目标 is_grasped：官方目标没有被抓住。模型 done 也不替代官方评分。

同五例旧固定底盘 V600 是1/5（000，第217步），新移动 V600仍1/5（000，第226步）；配对成功不一致为0例。五例均已暴露，本轮同时改变底盘可用性、明确参数块与错误反馈，不能归因单一因素或声称总体泛化收益。原固定底盘脚本0/5的失败门槛保持，不借新条件追认通过。每类/高度单元只有1例（middle4、高层1），不同场景五个，同场景新旧条件相关；区间是未经聚类校正的描述性 Wilson 区间。

## 关键失败归因

全部依据模型 note/hindsight，再与官方目标定位距离、实际动作及结果核对；rationale是自述。每次定位到目标碰撞表面的距离仅在评估端记录，查询家具边缘的点不能误计为认错目标。

- **000 成功**：第一次闭合空抓，模型记录“zero finger separation with stable=false”，随后重新打开、加深插入，第二次闭合 stable=true，返回 rest 后第226步官方成功。目标定位误差约0.1–0.5 mm；0.390 m的一次点是有意量支撑面。无需主动底盘移动。
- **001**：三次各0.30 m底盘指令都 arrived=true，实际前进0.296、0.295、0.295 m，无累计力。目标定位约0.45–2.73 mm，接近后达到目标附近，但两次闭合零指距/unstable。hindsight承认“both grasp closures reached zero finger separation with stable=false”，图像中盒子仍在沙发。定位和底盘到达有支持，空抓的具体几何原因未定；不是仍被固定底盘够不到，也不能简单归因于导航失败。
- **002**：模型用柜侧点 y=-0.333 m（略超0.288 m底盘半径）推断前方可以行驶，并说明左前方仍有遮挡。0.25 m命令实际0.246 m，平移残差4.49 mm、角残差0.030082 rad略超0.03，返回 residual_after_settle；没有放宽阈值。后来手臂移动触发累计力5230.78>5000，官方终止。目标定位误差约0.32–0.52 mm；模型计划与环境接触风险有关，但当前证据不能仅凭累计力确定碰到哪一物体/部位，接触机制未定。
- **003**：查询的“目标”点距真实目标约2.248–2.276 m（另一次支撑面点约1.881 m）。模型原地转-2.23 rad并对这个错误目标抓取，第三次闭合 stable=true，hindsight声称“两路相机显示仍持有 potted meat can”，调用done。官方目标 is_grasped=false、TCP也从未接近目标包围盒。明确归为错误目标信念和错误成功判断；非零指距和稳定反馈不能确定抓住了哪一个物体，也不能据此认定真实目标抓取成功。
- **004**：底盘0.14 m命令到位，实际0.131 m。目标点误差主要1.4–2.3 mm（首次8.0 mm，一次0.342 m点有意量桌边），TCP达到目标附近，但两次闭合空抓，之后两次 close_alignment_failed。hindsight承认空抓与对齐失败，返回rest后give_up。实际执行对齐未按模型预期完成；空抓几何原因未定，不能用到位误差小证明指尖接触正确。

完整模型笔记、工具反馈、定位和力值见 [case-evidence.json](case-evidence.json)，分母与成功时间见 [summary.json](summary.json)。

## 经费与媒体收尾

API108/125次，返回token估算USD22.1865/65，供应商实账待核；无未知/未完成预留。GPU1进程累计1663.3683/9000秒，约27.7分钟；已退出且GPU1空闲，GPU0他人进程保留。未使用USD42.8135、17次请求、7336.63秒余量关闭，不转移至后续实验。历史Runpod存储状态未刷新、费用仍单列待核。见 [closeout.json](closeout.json) 与 [api-ledger.json](api-ledger.json)，外部财务账同步。

五例都保留在线原始录像、同步生长TCP场景曲线、命令/执行对照与分析轨迹，共15个视频。第三人称镜头、碰撞几何与场景坐标仅用于评估，没有进入请求。原始运行归档 `raw-run.tar.gz` 在本目录本地及服务器保留，673,953,074字节，SHA-256 `8a09f58418a23fe52c0c3d5d879db2e4f8eb30822ba3d16c10b04344b6799043`；大归档不进入Git。视频的精确路径/哈希见 media-verification.json 与媒体总索引。开发诊断不写 docs/log/，不发布GitHub。

五例同步TCP在线录像（分析镜头）：

- 000：[online-tcp-demo.mp4](../../../../docs/media/c2-pick-arm-2026-10-01-arm-dev-000-V-mobile-f7ff8757/delivery/online-tcp-demo.mp4)
- 001：[online-tcp-demo.mp4](../../../../docs/media/c2-pick-arm-2026-10-01-arm-dev-001-V-mobile-3d307714/delivery/online-tcp-demo.mp4)
- 002：[online-tcp-demo.mp4](../../../../docs/media/c2-pick-arm-2026-10-01-arm-dev-002-V-mobile-dd1988c7/delivery/online-tcp-demo.mp4)
- 003：[online-tcp-demo.mp4](../../../../docs/media/c2-pick-arm-2026-10-01-arm-dev-003-V-mobile-79ff0e6a/delivery/online-tcp-demo.mp4)
- 004：[online-tcp-demo.mp4](../../../../docs/media/c2-pick-arm-2026-10-01-arm-dev-004-V-mobile-b8a8af13/delivery/online-tcp-demo.mp4)

主线阶段 2：允许底盘的模型工具执行诊断。用户在新提案之后明确“启动”，已登记新 USD65 / API125 / GPU1 9000 秒额度，旧余额转入 0；只跑 arm-dev-000..004，各一个 V-mobile 进程。测试组及开发集 2 不运行。

冻结包 SHA-256 `590063d9988a51814f311caa784d2d51565ba276883092a8ecad1404e07ebab2`；149 项 CPU 回归和服务器 CPU 部署检查已通过。启动时重新核实 GPU1 空闲、108 源文件/1046 资产哈希、清单及五个 r3 初态绑定。见 [授权](authorization.json)、[启动回执](launch.json)、[新额度提案](../../../../docs/design/c2-eef-same-five-mobile-authorization-proposal.md)。

批次 PID 1900102，首例 worker PID 1900232。首例已初始化并发出首个模型请求，尚无最终结果；启动不代表抓取成功。GPU1 正在使用，GPU0 的他人 PID672548 未触碰。API 每个发送均由服务器 broker 持久登记，未知/未完成响应保留预留，不当作实际已知消费或供应商账单，见 [启动时账务快照](launch-accounting.json)。GPU 用时在进程结束后结算；首例运行中不能把已结束进程累计 0 秒当作实际用时 0。

服务器输出目录 `/home/pshuai/bvi-research/runs/arm-mobile-five-20261001-r1/`；部署目录 `/home/pshuai/bvi-research/deploy/arm-mobile-five-20261001-r1/`。批次自动串行运行至五例完成或预注册停止原因，不自动重试/修补。每例上限1800秒，整轮600步，最多25发送；正常官方失败继续下一例，未知API、接口异常、超预算或缺媒体则停批。

有物理动作的尝试由运行时直接录制并保存逐步轨迹，结束后生成 TCP 同步轨迹视频和分析页；当前尚无完成的视频，不能用回放冒充在线录像。后续文件与哈希归档于本诊断和媒体索引。供应商实账待核；历史 Runpod 存储状态未刷新、费用另行待核。仅本地记录，不发布 GitHub。

## Scene demos with issued and actual paths

[Five preferred scene demos](../../../../docs/media/c2-pick-arm-dual-path-2026-10-01-mobile-r1/README.md) overlay Astra issued goal connectors and actual TCP trajectories in the original online camera recording. Pink denotes goals, cyan measured TCP, orange base commands. CPU postprocessing only; original recordings and results are preserved.
