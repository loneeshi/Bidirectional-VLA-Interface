# S1 官方数据 pilot 与 S2 分段预检

主线第 2 步：S1 数据准备通过一部分接口检查，尚未训练；S2 仅做 CPU 分段预检。

## 本轮实测

- 官方 SetTable 苹果 Pick：父轨迹 0–19 训练、20–24 验证，共 889 / 230 个动作样本。导出 Parquet 的全部 state/action 与原始 H5 精确一致；42 项相关 CPU 测试通过。
- 原生观测是 qpos12 + qvel12 = 24 维。MS-HAB `SequentialTaskEnv._get_obs_agent` 去掉机器人底盘的三个关节，不能直接复用旧 V8 的 30 维输入。head/hand RGB 均为原生 128×128，动作 13 维；特权物体/目标/持物字段仅用于标注。
- 每条导出至首次原生 success/fail 转移（含该动作），完整原始 H5 保留。SFT 不伪造终点零动作；S2 完成端点观测仍在原始 H5 中，后续标签构建必须读取它。
- Pick 有 reach/grasp/move 各 25 个区间。Place 25 条全部为官方成功示范，但现有 15 cm 分段条件仅得到 move/release 各 13 个（训练 9、验证 4）。另外 12 条在持物时最小目标距离约 15.17–18.82 cm，未满足分段条件，保留原轨迹和排除证据，不把它们算成策略失败，也未放宽阈值。
- Pick 转换 39.60 秒，Place 审计 22.42 秒；两进程已退出。CPU 转换速度不是 GPU 训练吞吐。

## 下一道门

1. 为新 S1 接入同一原生 24 维训练/推理输入，训练集独立计算归一化，验证相机与动作变换；旧 V8 归一化不得复用。
2. 完成 S2 当前帧标签及终点/失败 mask；错误交接状态单列验证，不能以官方成功示范覆盖替代。
3. 依计划运行 100 更新吞吐门，再决定扩大至 150 或 300 条。当前仅导出 20/5 pilot，尚无新 S1 检查点，也未通过 10 起点 ≥3/10 能力门；不得启动 S2 参数更新。
4. C 线真实 plan UID 与多 episode 执行器仍待完成，不能报告 C 评估已开始。

## 可复现命令（实验室，CPU）

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src/Bidirectional-VLA-Interface/src \
 envs/openpi/bin/python convert_official_fetch.py \
 --source data/fetch-tapt-source --output runs/NEW_PICK_BATCH --task pick
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src/Bidirectional-VLA-Interface/src \
 envs/openpi/bin/python convert_official_fetch.py \
 --source data/fetch-tapt-source --output runs/NEW_PLACE_AUDIT --task place --audit-only
```

源数据 revision、H5/JSON SHA256、完整父轨迹划分和分段证据见本目录 manifest；导出数据仍在实验室 `runs/s1-official-pilot-2026-09-17-run01/`，本地已保存报告，未声称完整数据已备份。扩大训练集时固定保留 pilot 验证轨迹，避免泄漏。

本轮 GPU 工作 0、API 0、新租机 USD0；实验室费用未知。GPU1 末次查询 15 MiB / 0%。历史 Runpod 40GB 停止存储费用继续，当前未刷新供应商账单。
