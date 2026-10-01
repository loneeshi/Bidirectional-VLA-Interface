# C2 工具冒烟独立授权提案（2026-09-29 UTC，待批准）

主线第 2 步 → v3 r2 的四项 CPU 硬门通过 → 未运行物理冒烟，实验室调度许可待确认 → 用户批准本提案后才启动。

本提案仅申请工具冒烟。依据[修订 1 设计](real-handoff-eef-tool-smoke-e1-pilot-design.md)与 [CPU r2 回执](../diagnostics/2026-09-29-eef-tool-smoke-cpu/resume-r2/README.md)。E1、Astra API、G0 阶段 B/C、IK7 重跑不在范围内。

## 样本与被冻结的实现

| 交接 | plan UID | 教师实测位姿帧 |
|---|---|---|
| 5 | tidy_house-sequential-val-102-0 | 2/5/7 |
| 8 | tidy_house-sequential-val-105-0 | 6/11/17 |
| 11 | tidy_house-sequential-val-108-0 | 8/16/24 |

全部属于冻结开发集，从原交接快照恢复，无 SAC 新推理。三个快照的字节哈希在[部署清单](../diagnostics/2026-09-29-eef-tool-smoke-cpu/resume-r2/deployment-manifest.json)的 manifest 绑定与[服务器核对](../diagnostics/2026-09-29-eef-tool-smoke-cpu/resume-r2/server-preflight.json)中，运行前再次核对。

实现 `eef-tool-v3-smoke-r2`，IK8，权重、容差、加密粒度、夹爪时序与步数均按 CPU 冻结参数。仅修复正则候选违反硬约束时的回退与排序；v1/v2 保留，A1 模型不重拟合。

- 九个位姿矩阵文件 SHA-256：`c6dae6326cd28d0fd3ca144785240e1e429e1bcc1fb710f5697cf02c4703c944`。
- 部署 ZIP SHA-256：`ebdcc5ac5da19a1f824682ddf1fa194d6fdfa35b76eec0d1fd6c17380ce41498`。
- 待批准授权记录 SHA-256：`c688a3bb7f5e07975d1855f6d757391d8acb666892de8274b9292b427869dc4f`。
- 本地包：`runs/eef-tool-smoke-20260929-r2/deployment.zip`；远端拟解包目录：`/home/pshuai/bvi-research/runs/eef-tool-smoke-20260929-r2`。

待批准记录在包内是 pending_user_approval，不可执行。批准后在包外生成正式授权，引用上述 ZIP 哈希与批准消息；不更改已冻结代码/位姿/模型。额外确认调度许可后才将 scheduling_clearance_confirmed 置 true。

## 冻结运行顺序与验收

GPU1 串行，第 5 → 8 → 11 号各最多一个进程，无自动重试。每例在同一官方 Pick episode、同一累计力和剩余时限中依次执行：

| 指令 | 判据 | 步数上限 |
|---|---|---:|
| 0 不可达 check_path（g 沿 base_link +x 1.5m） | 拒绝；qpos、环境状态与步数不变 | 0 |
| 1 张开 | 指令值 1.0 生效；不等待 | 1 |
| 2 三点 check_path | 接受；只读 | 0 |
| 3 三点 move_eef_chunk | 中间关节误差旋转≤.05rad/躯干≤.01m；末端≤.01m/.05rad；指距≥.08m | 24/44/58 |
| 4 闭合 | 闭合前≤.002m/.02rad、躯干每步位移<.001m；对准≤10步；闭合后等待≤5步，稳定不是验收门 | 15 |
| 5 实测末端沿 base_link +z .10m | 朝向不变；末端≤.01m/.05rad | 20 |
| 6 关节空间 return_to_rest | 官方 ee_rest、robot_rest、is_static 同时真；不调用末端 IK | 60 |

总上限 120/140/154 步，同时不能超过恢复后官方剩余 Pick 步数（≤200）；所有夹爪、对准、等待、收尾都计步。原控制器、累计力<5000、官方成功判据不变。底盘不动，不增加场景碰撞过滤。

单指令失败，记本例失败，跳过后续运动指令；未官方终止时只检验回休息位。该进程收尾后停批，报告证据与未运行样本，不补跑、不换 IK7。基础设施删失单列。三例全部指令通过且无力/时限终止才是冒烟通过。严格 Pick 单列，不作为冒烟门，也不归因于 Astra。

躯干规则按修订 1：失败后停下，给出躯干/关节/末端/接触证据，用户决定归因、是否锁定；**不自动锁定、不自动 IK7 重跑**。力超限保存评估端接触身份（末物理子步采样，不是全部子步积分），不能把接触名单输入工具或自动归因于控制器。

## CPU 证据与风险

原 43 项继续通过；v3 共 61 项，加运行器授权防护 10 项。四硬门通过，G1=108/108，坐标最大矩阵误差1.78e-15，第11回归明确拒绝跳解。

A1 非门槛预测：第5号31步但抬升跳解失败；第8号64步参考检查全满足；第11号接近58步达上限，含收尾76步，未完成整套。**物理冒烟可能首例就停止**；本申请没有把参考失败调掉，依据设计只将其作为风险。模型无接触/抓取物理；CPU门、真实接口检查、严格Pick与任务完成分别报告。

## 资源、启动门与记账

仅实验室 GPU1 `GPU-b7ebba23-7824-7601-df32-be55628936c3`；最多3进程，每进程≤180秒（含初始化/恢复/退出），总≤**540进程秒**。监督器串行预留每例180秒，内层173秒退出、外层174秒后终止/强杀并记实际秒数。API请求/token/USD均0，策略推理/训练0。其他授权未用余额不转入。实验室向项目收费按用户确认记0，无发票；逐进程秒仍记账。历史RunPod持久盘独立跟踪，不新增租赁资源。

已重读finance README和ledger（主账读取编码gb18030，SHA-256 `2045f1c09c0ba095bf9a41c1460a4a7571427e2122b702cca281b64f4e2dfc57`）。本轮准备GPU/API/仿真均0，正式GPU额度尚未批准。独立子账 `BVI-research-plan-2026-09-14/finance/ledger-eef-tool-smoke-20260929.json` 记待批准申请，不记作已授权支出。

只读服务器检查：GPU1空闲15MiB/0%，GPU0既有进程未触碰；目录可写、636GB可用；官方MS-HAB提交、sequential_task/BaseEnv源码与三个快照哈希一致。未发现squeue/qstat，**尚不能确认实验室直接调度规则**。请求批准实验范围，并确认可按既有方式直接串行使用GPU1（如有新调度要求请指定）。凭证只在ignore的本地配置，未入包。

实际启动前再次读finance/ledger、核对GPU1/存储/权限/调度规则/全部包哈希与CPU源路径；每个批准进程内先验证快照恢复、官方控制器、FK坐标与动作映射，再发第一步。前置失败照计进程秒并停止，无额外免费重试。新Linux运行器只有CPU编译/授权防护结果，尚无真实部署通过回执。

结果只进 `research/c2/diagnostics/<运行UTC日期>-eef-tool-smoke/`，保存全部命令、qpos/qvel、完整末端姿态、夹爪、官方指标、评估端接触、逐进程回执及来源哈希。不进docs/log及两份索引，默认不录视频。收尾核对GPU1进程状态，核算实际用量，剩余额度关闭。

冒烟3/3通过后，才另外提交E1授权提案（最多60次API请求、token/USD估算与5400进程秒），该后续资源不由本提案授权。


## 批准与关闭回执（2026-09-29 UTC）

用户批准了本提案及直接串行GPU1调度。实际监督器在CPU预检中因选错lightnav解释器（缺gymnasium）退出；GPU子进程0、GPU进程秒0、API0、仿真0，三例均未运行。原540秒额度关闭，无自动重试。已有acdit环境的独立CPU官方源码/快照检查通过，但不是物理结果。见[诊断](../diagnostics/2026-09-29-eef-tool-smoke/README.md)和[待批准环境修正重试](real-handoff-eef-tool-smoke-runtime-retry-proposal.md)。
