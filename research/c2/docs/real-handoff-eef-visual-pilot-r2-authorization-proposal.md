# 高清视觉试点 r2 重试授权提案（待批准）

r1在5号第三次API发生URLError后停止：前两调用有效、环境0步；55/58未启动。不能判断第三次是否计费，不能据此判断Astra任务能力。原记录与USD1.6134未知预留保留。

本提案申请独立r2：5、55、58各一次，从同一冻结交接快照和独立新会话开始，H0、不注入r1历史。完整2048 RGB-D、guidance-v3、Astra medium、25工具调用/200官方步/最多一次终止反思、无目标真值的条件全部不变。三者仍是已暴露开发诊断。r1/r2分别列明，不抹除删失或把挑选后的结果冒充首轮结果。

资源申请：GPU1串行3×1800=5400进程秒；最多78次API；USD1010上限；旧余额不转入。原[视觉提案](real-handoff-eef-visual-pilot-authorization-proposal.md)中的完整历史费用算法继续适用，未新增模型/提示/分辨率变量。实际用量、未知预留与供应商实账分开。

[CPU463项通过](../diagnostics/2026-09-30-eef-visual-pilot-r2-cpu/README.md)。只改变日志与授权绑定：URLError保存底层类型/错误码；监督器API计数汇总；r2独立账本。尚未证明网络故障已消除；遇到未知结果再次停止整批，不自动重试。普通任务失败按冻结顺序继续。模型成功与接口成功分开报告。

部署：`runs/eef-visual-20260930-pilot-r2/deployment.zip`；SHA-256 `001db69e90ab01291ca56d82597193187c48a2711dd9c784290a7f24e42a70cc`。待批准文件`authorization.launch.pending.json`，状态pending，账本拟为`ledger-eef-visual-pilot-20260930-r2.json`。完整源码哈希见[freeze.json](../diagnostics/2026-09-30-eef-visual-pilot-r2-cpu/freeze.json)。已过接口的控制/传感器代码完全未改。

批准后读取finance README/ledger、确认GPU1空闲与磁盘/权限/调度、核验远端源码和运行时CPU门。以新root启动，绝不重启r1或重复其请求ID。API凭证不上传；代理记录请求前预算承诺。网络未知时保留删失、核验进程退出后结账。

每例返回完整执行历史、在线视频、命令与实际轨迹叠加及交互轨迹；零动作/异常片段如实标注。结果只进diagnostics，媒体进docs/media，不发布GitHub。等待明确批准后才运行本批GPU/API。
