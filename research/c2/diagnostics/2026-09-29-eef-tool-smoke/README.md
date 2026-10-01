# C2 工具冒烟：部署前置停止（2026-09-29 UTC）

主线第 2 步 → 用户批准后上传冻结包并启动 CPU 监督器 → lightnav 环境缺 gymnasium，CPU 源码预检退出 → 更正环境的独立重试提案待批准。

## 结果与归因

没有启动 GPU 子进程、创建仿真、恢复 episode 或执行动作。三例 5/8/11 全部未运行，严格 Pick 没有可评分样本，不能记为 0/3 任务失败。冒烟尚未通过，不能申请 E1。CPU 监督器入口 traceback 见 [controller.log](controller.log)，结构回执见 [run-receipt.json](run-receipt.json)。

本次启动配置误选 `/home/pshuai/bvi-research/envs/lightnav/bin/python`，缺少 gymnasium。此前确认“解释器存在”不足以证明依赖可用；这是部署配置错误，不是物理躯干、IK 或执行器失败。没有自动锁定躯干、换 IK7 或再次启动冒烟。

## 授权与资源收尾

原批准范围3×180=540进程秒、API0；正式授权和冻结部署包均已核对。原授权保留为 [authorization.json](authorization.json)，关闭记录为 [authorization.closed.json](authorization.closed.json)。远端授权也关闭，防止误启动。因在GPU子进程预留前退出，GPU子进程数与GPU进程秒都是0；CPU监督器壁钟/退出码没有持久回执，保留null，不编造。API请求/token/USD0，策略推理/训练/仿真动作0。

GPU1实测15MiB、0%、无计算进程，监督器PID已不存在；GPU0其他用户进程未触碰。未用540秒关闭，不自动跨批转入。实验室收费按用户确认0、无发票；历史RunPod存储独立保留，本轮未核实时账户。没有视频或其他媒体；结果不进docs/log及两份索引。

## CPU 环境修正准备

已有acdit解释器可导入gymnasium/numpy/scipy/torch/PIL；没有安装软件、修改冻结执行器或启动GPU。GPU隐藏的独立CPU核对通过：正确官方MS-HAB源码、BaseEnv及三个快照的CPU反序列化，cuda_initialized=false。见 [environment-correction-cpu-gate.json](environment-correction-cpu-gate.json) 与 [corrected-environment.json](corrected-environment.json)。此检查不是真实环境恢复或部署成功。

修正重试仅更换启动解释器和独立输出目录，原包、位姿、参数、步数、样本、停止规则不变。见 [待批准计划](retry-plan.pending.json)和[独立重试提案](../../docs/real-handoff-eef-tool-smoke-runtime-retry-proposal.md)。新的GPU运行须用户另行批准，不复用已关闭授权。
