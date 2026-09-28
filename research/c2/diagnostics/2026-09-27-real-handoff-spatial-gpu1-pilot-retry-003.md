# 离线空间理解阶段 1：GPU1 图像导出 retry-003

## 授权与执行

用户在看到[位置 5 单例提案](../docs/real-handoff-spatial-stage1-gpu1-pilot-retry-003-proposal.json)后回复“批准”。本次仅从冻结真实交接清单位置 5 恢复快照，导出官方头部／手部 RGB-D 的 128、256 px 版本及同快照物理碰撞包。实验室 GPU1 最多一个进程、120 秒；模型 API、策略动作和新 RunPod 资源均为 0。位置 8 和全 60 例不在授权范围内。

远端批准文件 SHA-256 为 `d01eb32900b346c773d95361f2656d237a90e847c748f1c99f710a430fe14263`。执行前干跑核对官方 MS-HAB／ManiSkill 源码、任务和资产、位置 5 快照、上次回执及 GPU1 空闲状态；第一次干跑因 `PYTHONPATH` 指向已安装的 AC-DiT 依赖而被官方源码哈希门拒绝，未创建 GPU 子进程。修正为冻结的官方 `src/official-mshab-runtime/mshab` 与 `ManiSkill` 路径后干跑通过。

| 位置 | GPU1 进程 | 用时（秒） | 128 px RGB-D | 256 px RGB-D | 同快照碰撞包 | 状态门 | API／策略动作 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 48.075921 | ✓ | ✓ | ✓ | ✓ | 0／0 |

128 px 的官方 Pick 输入哈希与冻结普查记录完全一致。两个分辨率分别从原快照恢复，恢复、相机读取及导出后的最大状态误差均为 `2.384185791015625e-7`，低于预定 `1e-5` 门；读取相机未触发任务评估的状态修改。八个供模型显示的 RGB／深度 PNG 在本地逐一通过结果清单 SHA-256 校验；原始毫米深度 `.npy` 与 16 位 PNG 也保留。`result.json` 的状态为 `completed`，几何包没有导出碰撞形状错误。

碰撞包完整导出 127 个场景体和机器人连杆信息；其 `observed_missing_witness_inputs` 标记表示它本身还不足以独立证明完整抓取 IK／动态避碰见证，并非本次图像导出失败。基于该包和同快照官方地面网格，在本地 CPU 生成了消融 B 专用的 `map.json`、`height.npy`、`map.png`；地图覆盖门通过，来源与图像结果哈希绑定。该真值图不得进入视觉主条件的模型输入。

## 回执与边界

远端原始目录为 `/home/pshuai/bvi-research/runs/real-handoff-spatial-stage1-20260927/results-pilot-retry-003/`；本地忽略目录为 `runs/real-handoff-spatial-stage1-20260927/results-pilot-retry-003/`。远端归档与本地下载的 tar SHA-256 同为 `71ff674b9f6915dab001e3d8ceb02adf3f36c3fdfa4718e6d75fc42d0823e669`；逐进程回执 SHA-256 为 `f3f3d47ef3fa04f0f07890439e1c860d129e9ef17b7445817e8e2160a933fd8e`，日志 SHA-256 为 `3d69d11a10e052561641b93e389121d4f9fb7b739950c5e210a6915b278afa4e`。收尾 GPU1 为 15 MiB、0%、无计算进程；GPU0 未操作。

本授权范围已用完一次进程，未用的 `71.924079` 秒不转入下一批。新增 GPU1 实耗 `48.075920935720205` 进程秒，模型 API 0 次／费用 0；实验室对本项目收费按用户此前确认记 0，历史 RunPod 停机持久盘仍单独计费。详见[独立账本](../../../../BVI-research-plan-2026-09-14/finance/ledger-real-handoff-spatial-stage1-gpu1-pilot-20260927.json)。

本次证明位置 5 的冻结图像、几何、地图和无评估相机读取链路可完成；它不提供 GPT 空间理解分数、候选站位、导航到位或严格 Pick 成功。下一验收门是补足其余冻结交接的图像和标签，再按冻结条件评测模型；相应 GPU 批与付费 API 尚未获本次授权。
