# 离线空间理解：GPU1 两例先导首次启动诊断

## 授权与执行

用户于 2026-09-27 回复“批准”，范围为位置 5、8 各一次快照导出；GPU1 两进程串行，单次至多 180 秒，累计至多 360 秒；不执行 PPO、SAC、训练或模型 API。授权原文范围与参数见[先导提案](../docs/real-handoff-spatial-stage1-gpu1-authorization-proposal.md)及[机器可读授权](../docs/real-handoff-spatial-stage1-gpu1-pilot-authorization.json)。

| 位置 | 进程 | 用时（秒） | 快照恢复 | 图像 | 几何包 | 结果 |
|---|---:|---:|---|---|---|---|
| 5 | 1 | 4.127386 | 未开始 | 未生成 | 未生成 | 启动防护停止 |
| 8 | 0 | 0 | 未开始 | 未生成 | 未生成 | 未启动 |

位置 5 的快照 SHA-256 为 `e56a12e95d6a4b630a9c5bae36ebaea6b73a952104e17f38c4767b57789e847a`，位置 8 为 `6cf3378cc558f08c473b32f394601782c2dcb20e6be2e64b75217fd9810318e0`；启动前均已在实验室再次验证。首进程退出码 1，日志 SHA-256 为 `16bd225d6b8d5f6ae3ddf38a0b7ebe4f38c41428be40e4b9a28c54568ff2529b`。本地原始回执位于 `runs/real-handoff-spatial-stage1-20260927/gpu1-pilot-attempt-001/`，实验室原始回执位于 `/home/pshuai/bvi-research/runs/real-handoff-spatial-stage1-20260927/results-pilot/`。

## 原因和修正

首次启动的 `PYTHONPATH` 指向 `/home/pshuai/bvi-research/src/official-mshab-runtime`，而官方 `mshab` Python 包实际上位于该目录下的 `mshab/`。工作脚本导入到的 `sequential_task.py` 未通过冻结 SHA 检查，故在创建环境前中止。已核对正确文件的 SHA-256 为 `47fd4999f7326568a550890caa48eac070901525373fac667ff56384eeb2a6be`，其仓库提交为 `e9ff3d23496d38e4431c8d913e147ffa007f7f72`。执行器现增加了在子进程前核对导入路径、源码 SHA 与提交的预检；后续启动命令须把 `.../official-mshab-runtime/mshab` 加入 `PYTHONPATH`。

原提案写明每个位置一次独立快照恢复，并在任一验收门失败时停止。位置 5 的首次进程已计入授权进程数，因此没有把剩余额度自动用于重试或位置 8。修正后的重试需要明确的一次协议修订；本次没有空间理解实验结果。

修正后的 `source-pilot-v2` 已做一次不创建子进程的远端预检：使用正确 `PYTHONPATH` 时，官方源码 SHA 与提交检查通过，执行器随后按预期拒绝复用已存在的首次输出目录。这证明启动路径修正有效；尚未验证图像、碰撞包或高分辨率渲染。

等待重试授权期间，CPU 检查补齐了 `source-pilot-v3`：逐张校验官方头/手相机 PNG 的尺寸与编码，单独生成只含相机、模态、尺寸和哈希的可信传感器清单；深度原始 NPY、16 位毫米 PNG 与给模型看的固定 8 位展示 PNG 分别记哈希和转换规则。渲染和几何导出后各复查一次仿真状态；GPU1 空闲门额外检查计算进程和剩余磁盘。相关本地测试通过；这些 CPU 检查不等于实际渲染验收。

用[重试范围](../docs/real-handoff-spatial-stage1-gpu1-pilot-retry-proposal.json)做远端 `--dry-run` 通过：两份快照 SHA、官方运行时 SHA、GPU1 UUID、空闲进程清单和磁盘余量均已复核；未创建输出或子进程。用原授权文件做相同干跑被拒绝，避免把初次授权重复用于新批。用户后来批准该修订并运行一次，见[重试 001 诊断](2026-09-27-real-handoff-spatial-gpu1-pilot-retry-001.md)。

后续 CPU 代码审计还发现几何包的 `source_scope` 必须为模板提取器已定义的 `real_handoff_batch`；先导导出器已修正该字段，以 `call_id=spatial-stage1-gpu1-pilot` 保留本批来源。Q1 标签器现在与 Q4 一样，对关键实体缺失碰撞形状返回 `unavailable`；向一份归档几何注入缺失沙发形状的只读测试得到 `unavailable`，没有修改原始包。Q1 的现有 20 份有包样本均来自失败交接且均为有界搜索无见证，40 份成功交接尚缺同快照几何；这批已有标签不能代表 60 例的二元可达率。

## 资源收尾

GPU1 收尾占用 15 MiB、利用率 0%，无本批计算进程；GPU0 的既有进程未触碰。模型 API 0、策略动作 0、RunPod 新资源 0。逐进程秒与费用分类记在研究计划 `finance/ledger-real-handoff-spatial-stage1-gpu1-pilot-20260927.json`。
