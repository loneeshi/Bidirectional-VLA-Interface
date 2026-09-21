# AC-DiT Apple Pick：中期训练核查与原生评测门

本报告面向 2026-09-20 交付，核查发生在 2026-09-19 UTC。**capability validation not completed（能力验证未完成）**。训练仍在运行，不能把当前底盘权重用于全身 Pick 评测。

## 已取得的事实

实验 `acdit-apple-bounded-20260919-run01` 位于服务器 `/home/pshuai/bvi-research/runs/acdit-apple-bounded-20260919-run01`。18:57 UTC 只读快照为 stage1=158 更新；18:58 UTC 下载的训练日志/状态覆盖 163 更新。两个不同时间快照均保留，没有拼成同一时刻。stage2 状态文件不存在，未进入全身正式训练。最终收尾只读状态另见结果目录 `closing-state.json`。

| 项目 | 当前证据 | 能说明什么 |
|---|---|---|
| Stage1 | 快照163次更新，3917 microbatches；FP32、batch24、LR1e-4 | 底盘训练在推进，不等于抓取能力 |
| Stage2 | 尚未开始；正式更新0 | 没有合格全身候选 |
| 保存点 | latest manifest generation4 / update145；best开发分数0.2371478677，对应update145 | 文件存在及训练内首更新 round-trip 检查已记录；本轮未独立加载权重 |
| 开发选择 | 6条validation记录的16个(parent,step)身份完全一致，均在dev split；选非EMA active-channel first-action RMSE最小者 | 没有用在线五例成功挑点 |
| 数据 | 743唯一成功轨迹，593 train / 150 dev，63场景；26995/7193帧 | 从manifest和collection独立核对父轨迹与场景隔离；本轮未重新扫描大型H5 |
| 采集 | 1184次尝试、229928仿真动作/教师调用 | 既有累计数，不是本轮新增或策略成功率 |

H5记录哈希为 `9c068bdcd223c8c6c6a8437d6ad8cfa7d81d8f7dde589c06a62e6670460464e9`。完整父轨迹列表、语言embedding哈希、初始化资产哈希及源文件身份均见[证据索引](results/acdit-apple-delivery-2026-09-20/README.md)。

## 运行时限与保存边界

已从实际进程确认 `timeout --signal=TERM --kill-after=20 63700` 包裹现有 supervisor；不是本轮启动的新进程。TERM 名义时点约 2026-09-20 10:53:54.817 UTC，最多20秒 kill grace；合同截止 10:56:27.989 UTC。没有延长时限、重试或新增自动唤醒。stage1 当前 deadline 为 9/19 22:01:08.992 UTC。训练脚本在阶段末预留600秒退出更新循环，仍需完成开发评估和保存，因此 deadline 不是更新完成量承诺。

本轮不改 ongoing training。当前原子 latest manifest 只说明已发布一对文件名；轮换文件与best仍可在后续保存时更新。本轮仅 stat 文件并复制小日志，没有在训练中复制/哈希多GB活动权重，不能声称已独立校验最新 checkpoint 或完成完整备份。权重、optimizer、成功与失败H5第二副本均未在本轮建立。

## 阶段转移代码审查

服务器源码副本由 pinned commit `90ad00a926f34da04816ed9c3312aaf3bc845b7f` 的 archive 建立，本身无可用Git元数据；本轮远端git查询返回128，故以 `assets/source.json` 的来源记录、完整39文件源码zip与逐文件哈希确定实际版本，不伪造干净Git状态。

`ACDiTRunner.__init__` 从 stage1 `module` 提取 DiT，严格加载并冻结 mobility head；训练器另对 lang/img/state/lift3d/in_context 五组 mobility 输入投影逐组 strict load并冻结，复制 `lift3d` 权重。点云encoder复制与其参数是否可训练是两件事，不宣称全部冻结。stage2尚未执行，本轮只确认代码路径，未取得实际stage2 transfer hash、训练后冻结参数一致性或全身加载证据。

## 评测入口与五例面板

旧 `scripts/run_lab_acdit_native.py` 硬编码 `src/AC-DiT`、社区 stage2 checkpoint-25000 与 mobility checkpoint-30000；旧 batch入口还复用历史seed2024的结果。两者都不能直接当作新候选的五例评测命令。本轮未运行它们，也没有修改训练或为了无候选分支搭建新集成。

已核查旧 AC-DiT seeds2024–2028 的 `result.json`：reset_info记录原生谓词，但没有完整 build_config/task_plan/init_config 身份。旧π₀.₅两日面板使用 IA oracle + minimal GPU；旧AC-DiT使用whole-episode + CPU/default，不能据相同seed就宣称同条件或因果提升。

保留计划分母5，逐例见 `regression-panel.csv`，当前五例全部 `not_run`，success为null，不写0/5。seed名单已登记；完整场景/初态面板**尚未冻结**，因为没有全身候选，且历史映射不能直接确认。候选合格后必须在看任何策略结果前，从真实plan/spawn和无策略reset核实身份并冻结五例；若历史不可恢复，明确命名新的开发面板，不能补造映射。

本轮停止能力分支的直接理由：没有完整可加载whole-body候选。其后还需合格的新候选入口、train/eval输入合同核对、全身fresh-process strict load，以及单独的原生评测资源授权。现有18小时授权只覆盖训练，未发现覆盖本轮后续五例评测的明确授权；旧π₀.₅评测额度不移用。尚不能给出经验证的精确新入口命令，因此本轮不提交空泛的GPU许可请求。进入该分支前，应提交实际可审查命令、冻结五例、可选一次成功fresh-process复跑、GPU/时间上限及停止机制。

原生成功、策略失败、基础设施失败、timeout、not_run须分别保留。仍沿用200动作和首次原生终止，不使用SAC预抓、瞬移或放宽力阈值。成功复跑单列，不进入五例分子。LightNav→Pick还须可重复成功和独立额外预算（最多3例）；本轮未运行。Place、π₀.₅新实验、正式100×3、trained TAPT均未完成。

## 已执行验证与未覆盖风险

CPU快照验证通过：24份下载文件哈希、593/150场景隔离、固定16个dev身份、连续163更新loss/grad有限、9个Python文件语法。`tests/test_acdit_contract.py` 9 passed（0.13秒）；初次因迁移后缺 `PYTHONPATH=src` 收集失败，设置正确路径后通过。未运行GPU加载、全身推理、simulator reset或native rollout。因此没有新增原生任务成功，也不能以静态检查替代输入数值等价验收。

费用：本轮新增GPU实验0、runtime付费模型API0、新租/恢复RunPod0；既有GPU1训练继续，实验室费用未知。停止云盘历史40GB/约USD0.266667每日仍是待对账历史记录，本轮未刷新供应商状态/余额，不能当作已核实当前消费。编码助手会话不是机器人runtime API调用。
