# 工具冒烟环境修正重试提案（2026-09-29 UTC，未批准）

主线第 2 步 → 修正运行环境的 CPU 源码/快照核对通过 → 原试点在 GPU 子进程前删失，未有物理结果 → 申请独立重试，用户批准后才启动。

## 前批停止与唯一修正

[前批诊断](../diagnostics/2026-09-29-eef-tool-smoke/README.md)：监督器误选缺gymnasium的lightnav解释器，CPU导入阶段退出。GPU子进程0、GPU进程秒0、API0、仿真0；三例未运行，不能记为严格Pick失败。原540秒授权关闭，不转入本批。原代码、参数、教师/位姿、记录和失败门全部保留；不自动锁躯干或换IK7。

唯一运行修正：解释器改为 `/home/pshuai/bvi-research/envs/acdit/bin/python`，Python3.11.13，二进制SHA-256 `55e8f2734d0e882df2013e591fc2cd91fc31a4fc4d82dd0e78326af8fd3abdc1`；工作目录改为 `/home/pshuai/bvi-research/runs/eef-tool-smoke-20260929-r2-retry01`，防止覆盖原失败证据。不存在pip安装或执行器改动。

该解释器实际导入gymnasium0.29.1、numpy1.26.4、scipy1.17.1、torch2.7.0+cu128、Pillow12.3.0；GPU隐藏的独立CPU官方源码检查和5/8/11快照CPU反序列化均通过，CUDA未初始化。这不是仿真恢复成功。来源回执见[CPU检查](../diagnostics/2026-09-29-eef-tool-smoke/environment-correction-cpu-gate.json)和[环境记录](../diagnostics/2026-09-29-eef-tool-smoke/corrected-environment.json)。

## 全部冻结条件与资源

继承[已批准原提案](real-handoff-eef-tool-smoke-authorization-proposal.md)的所有样本、指令0–6判据、24/44/58接近步数、闭合15/抬升20/回休息60、官方剩余步数、累计力与严格Pick判定，以及失败停止/不重试/接触仅评估端的规则。

- 原部署包SHA-256：`ebdcc5ac5da19a1f824682ddf1fa194d6fdfa35b76eec0d1fd6c17380ce41498`，包不重建。
- 原位姿文件SHA-256：`c6dae6326cd28d0fd3ca144785240e1e429e1bcc1fb710f5697cf02c4703c944`。
- v3 r2全部硬门108/108及61核心/10防护测试保持，A1模型不重拟合。
- 原模型风险不变：5号参考抬升跳解、11号参考接近超时；不得调整阈值为过门。
- 新独立额度：GPU1 `GPU-b7ebba23-7824-7601-df32-be55628936c3`，串行最多3进程（5→8→11），每进程≤180秒，总≤540进程秒，API请求/token/USD0。旧额度转入0。费用按用户确认实验室项目0（无发票），按进程实测记秒；无新RunPod资源。

前置导入/恢复失败或任一冒烟指令失败后停批，无重试。未官方终止时只作设计规定的回休息检查。每个GPU进程初始化/恢复/退出计入180秒；监督器维持原173/174秒内外超时保护。原监督器在GPU子进程前的CPU预检若失败也停止，记录CPU失败与无GPU子进程事实，不宣称完成实验。

启动前再次重读finance与ledger，核对GPU1空闲、存储/权限、用户已确认的直接串行调度规则、解释器二进制/依赖版本、包与所有文件哈希，以及CPU源码/快照检查。正式授权在包外生成，引用本提案和批准消息，不编辑包内代码。使用独立目录，保留原账与证据。新方案仍未获授权，当前可执行额度0。

结果只写research/c2/diagnostics，逐例报告通过/失败/删失/未运行及实际进程秒，收尾核对GPU1；不进入docs/log及两份索引，默认不录视频。仅全部3例物理冒烟通过才另行申请E1；本提案不授权Astra API或E1。


## 执行与关闭（2026-09-29 UTC）

用户持续执行目标授权本具体重试。第5号真实冒烟在闭合前对准失败；三点移动、关节回休息通过，8/11未启动。实际35.481647进程秒/540、API0，剩余关闭。包/位姿/参数未改；无自动锁躯干或IK7重跑。详见[诊断](../diagnostics/2026-09-29-eef-tool-smoke-runtime-retry01/README.md)。全设计未完成，E1未启动，下一步待用户躯干决定。
