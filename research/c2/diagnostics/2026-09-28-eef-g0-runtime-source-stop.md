# G0 run01：动作 0 前的源码身份门停止

2026-09-28 UTC。主线仍第 2 步接口检验；本轮没有执行 Pick。

用户批准 [G0 提案](../../../docs/design/real-handoff-eef-g0-authorization-proposal.md)后，
独立分账记录 GPU1 总计7200进程秒、单进程180秒、最多55进程、API0。
本地16项CPU测试通过，服务器三个纯CPU检查与运行器导入通过；随后启动
串行控制器 PID1616386。首个 plan-005-sac0 在源码检查门退出，批次没有自动重试。

| 指标 | 数值 |
|---|---:|
| 已尝试独立进程 | 1 |
| GPU1进程墙钟秒 | 4.178846336901188 |
| 仿真动作 | 0 |
| SAC推理 | 0 |
| API请求 | 0 |
| 实验室项目收费 | 0 |

收费0来源为用户确认、无发票；进程秒包括本次失败启动，不意味着实际执行了GPU计算。
严格 Pick 未测，不写成0%成功率；G0/G1/G2实际轨迹验收均未完成。

## 原因与修复

运行器加载了虚拟环境editable安装中的AC-DiT内附MS-HAB，而启动命令没有
明确把冻结官方源码目录置于导入路径最前。启动前只检查了能导入，未检查
加载路径和哈希，这是本次启动检查的遗漏。

- 错载 SHA-256：`a4230a26514f83c371e2a2c24a9440b7044fcdde675c16df2849a6717ec1c54c`。
- 冻结官方 SHA-256：`47fd4999f7326568a550890caa48eac070901525373fac667ff56384eeb2a6be`。
- 官方源码实际仍匹配冻结SHA与commit `e9ff3d23496d38e4431c8d913e147ffa007f7f72`。
- 已新增 `scripts/check_eef_g0_runtime_cpu.py`：任何环境导入前固定官方MS-HAB
  和ManiSkill；在控制器准入GPU子进程前核对两套源码、commit、资产根、task
  plans及全部11快照CPU反序列化。程序检查CUDA未初始化。
- 修复后的纯CPU门已通过：11快照均通过哈希与结构检查，ManiSkill BaseEnv
  SHA为 `09a176d2a81924ae3b0234405bab2d45cd861a250c37ea52f12b6994b5753208`。
  本地16项CPU测试通过。此处不是仿真状态恢复通过。
- 同时把运行器收尾接到adapter.close()。原run01部署与授权文件保留，
  新候选run02授权状态为pending_user_approval，不能执行。

本批收尾：GPU1 15MiB、0%、无计算进程；控制器已退出，GPU0未触碰。
已用4.178846秒，剩余7195.821154秒；55进程额度剩54个。历史RunPod存储未刷新。
独立账本已核对回执，并以新增usage记录链接到总账，保留所有历史条目。

原始证据位于canonical repo：
`runs/real-handoff-eef-g0-20260928-run01/usage-ledger.json`、
`results/plan-005-sac0/result.json`、`results/plan-005-sac0-receipt.json`、
`runtime-cpu-repair-gate.json`。这些文件不进入模型请求。

下一步：[源码路径修复重启修订](../../../docs/design/real-handoff-eef-g0-source-repair-amendment.md)。
原提案规定失败保留、没有自动重试；本轮保持停止，等待修订批准。
