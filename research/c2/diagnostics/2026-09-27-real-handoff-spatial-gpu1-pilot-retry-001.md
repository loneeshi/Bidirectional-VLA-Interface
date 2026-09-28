# 离线空间理解：GPU1 两例导出重试 001

## 授权和结果

用户回复“批准”后，按[重试修订](../docs/real-handoff-spatial-stage1-gpu1-pilot-retry-proposal.json)仅在实验室 GPU1 串行尝试位置 5、8：至多两个进程，单进程 180 秒，合计 360 秒，策略动作与模型 API 均为 0。启动前干跑确认官方 MS-HAB 源码、两份快照 SHA、GPU1 空闲和磁盘余量。

| 位置 | 实际进程 | 进程秒 | 快照恢复 | RGB-D | 几何包 | 结果 |
|---|---:|---:|---|---|---|---|
| 5 | 1 | 4.027745 | 未开始 | 未生成 | 未生成 | 资产路径错误 |
| 8 | 0 | 0 | 未开始 | 未生成 | 未生成 | 停止门未启动 |

位置 5 的启动命令将 `MS_ASSET_DIR` 设为 `/home/pshuai/bvi-research/assets/data`。ManiSkill 的 `ASSET_DIR` 定义会再追加 `data`，结果指向不存在的 `assets/data/data/scene_datasets/.../all.json`。进程退出码 1，发生在环境构造、仿真动作、渲染和几何导出之前。逐进程日志 SHA-256 为 `51da8ff3a0f6d978eedb57e2d9a0549276874c2588aa3f2a21004c7aca26828a`；原始日志、回执和收尾状态保存在本地忽略目录 `runs/real-handoff-spatial-stage1-20260927/gpu1-pilot-retry-001/`，实验室原始目录为 `/home/pshuai/bvi-research/runs/real-handoff-spatial-stage1-20260927/results-pilot-retry-001/`。

## 修正和 CPU 验证

正确的环境变量为 `MS_ASSET_DIR=/home/pshuai/bvi-research/assets`，从而 `ASSET_DIR` 等于真实的 `assets/data`。控制器现于创建子进程前校验 `ASSET_DIR` 的绝对路径、官方 `all.json` 是否存在，以及它是否匹配冻结清单 SHA。远端 `--dry-run` 用正确路径通过，仍显示 GPU1 15 MiB、0%、无计算进程；用错误路径做同样干跑则明确拒绝并指出 `assets/data/data`。两次干跑均未创建输出或 GPU 子进程。

本次批准的重试在位置 5 遇到停止门后结束；位置 8 未启动，剩余时间不自动转入新批。未得到空间理解的图像或几何样本。进一步 GPU 尝试需新的授权与新的输出目录；全 60 例导出和模型 API 仍未授权。

待批的[下一次范围](../docs/real-handoff-spatial-stage1-gpu1-pilot-retry-002-proposal.json)另做了更完整的 CPU 干跑。控制器在创建子进程前还核对两例 episode 配置哈希、目标物体与官方任务计划的对应、官方抓取姿态资产的存在及 SHA，以及固定的 Fetch 自碰撞过滤源码。位置 5 的 episode／抓取资产 SHA-256 分别为 `a14e341a9942dd0299bb8b5b9e721e53f329f73a924fa3ec181d7c239a292df7`、`39838afd6ff1377c1f324d115b2a13cad99ded43841ef6d60683dfece3a958b5`；位置 8 分别为 `32a6037bb93c7aa933478611ac80332b54253b928b7d29cbda757de80f732e1c`、`2882d8077c8ef83fe4e5694df693c57dba20debfd97be2673ff1335db05bd7ee`。远端干跑通过，GPU1 为 15 MiB、0%、无计算进程，未创建输出目录或仿真子进程。此检查只排除静态资产错误，不证明后续恢复、渲染或几何导出必成功。

Q5 评分还需要官方可通行地面三角网格；原碰撞包没有归档该文件。现已在先导导出器加入只读复制与源／副本 SHA 核对，并在启动前检查文件存在。位置 5 的官方网格 SHA-256 为 `4f185177981b21cfa373e4244855fef91f3fbd2226b2e04095e1b9ee4dc337ab`，位置 8 为 `4202dec1b4379eb1428998b1b5e772b0a5eb8d05013d8b06af4b1d786085418b`；新一轮远端 CPU 干跑通过，GPU1 仍 15 MiB、0%、无计算进程。网格仅存于评分端几何目录，主条件传感器清单不会读取。此静态复制尚未经历真实导出进程验收。

## 资源和费用

本次 GPU1 进程时间 4.027744762599468 秒，与首次 4.127386145293713 秒分开记账；模型 API 0 次、策略动作 0、RunPod 新资源 0。实验室向项目收费 0（沿用用户确认的实验室政策），历史停止的 RunPod 持久存储继续单独跟踪。GPU1 收尾为 15 MiB、0%、无计算进程，GPU0 未操作。见[独立账本](../../../../BVI-research-plan-2026-09-14/finance/ledger-real-handoff-spatial-stage1-gpu1-pilot-20260927.json)。
