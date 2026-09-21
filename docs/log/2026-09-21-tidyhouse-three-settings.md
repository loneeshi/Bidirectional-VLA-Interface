# TidyHouse 16-plan 三设置实验记录

| 设置 | 完成 episode | 完整任务 SR | 完成物体 | 每集平均 |
|---|---:|---:|---:|---:|
| Fixed PPO + SAC（官方固定任务顺序） | 16/16 | 0/16 | 14/80（17.5%） | 0.875 |
| GPT + PPO + SAC（VLA-as-Tools 通讯协议） | 16/16 | 0/16 | 13/80（16.25%） | 0.813 |
| Teleport + SAC（官方固定任务顺序） | 16/16 | 0/16 | 21/80（26.25%） | 1.313 |

三组完整任务成功率都是 0/16。14、13、21 表示中途完成的物体数，不是完整任务成功。

---

## 实验 #001

- **日期**：2026-09-20 至 2026-09-21
- **要回答的问题**：在调度顺序完全正确时，官方 PPO 导航和逐物体 SAC 操作能否完成五物体 TidyHouse？
- **run 名称 / ID**：`goal-tools-paired16-20260920:fixed`
- **代码版本**：MS-HAB commit `e9ff3d23496d38e4431c8d913e147ffa007f7f72`；面板 SHA-256 `5fea051d078bf7c347fbddb997aeda12afa1324313ed3ffaa86f7cd83c6b5029`
- **环境**：Python 3.11.13 / PyTorch 2.7.0+cu128 / ManiSkill + MS-HAB / Quadro RTX 8000 48 GiB（GPU1）

### 配置（评测参数）

| 项 | 值 |
|---|---|
| 任务 & 数据划分 | MS-HAB TidyHouse sequential validation |
| plan roster / episode 数 | 16 个唯一五物体 plan；seeds 0–11、13、14、16、19 |
| 导航策略 | 官方 PPO |
| Pick / Place 策略 | 官方逐物体 SAC |
| 调度器 / 通讯协议 | 官方固定任务顺序；无语言模型 |
| observation / policy state | RGB-D；官方 42D policy state |
| skill horizon / episode 上限 | Navigate 500；Pick/Place 200；每集 7,000 步 / 900 秒 |
| 模型调用与费用边界 | 0 次 API 调用；费用 0 |
| 随机种子 | 0–11、13、14、16、19 |
| 其它关键设置 | 训练更新 0；串行执行；基础设施失败不计作任务失败 |

### 结果

| 指标 | 值（均值 ± population SD，N=16） | 备注 |
|---|---|---|
| 完成 episode | 16/16 | 基础设施失败 0 |
| 完整任务 SR | 0/16（0%） | 没有一集完成五个物体 |
| 完成物体 | 14/80（17.5%） | 固定分母 80 |
| 每集完成物体 | 0.875 ± 1.218 | 范围 0–4 |
| 环境步数 | 总计 7,773；485.8 ± 301.9 / 集 | |
| API 请求 | 0 | |

### 观察与结论

- **观察**：即使调度顺序完全正确，官方执行器也没有完成任何一集完整任务。
- **异常**：无基础设施失败；结果是原生任务失败，不是启动或适配器失败。
- **结论**：假设不成立。该设置只能作为“正确调度下执行器能走多远”的冻结基线，不能当作成功上限。

### 下一步 / TODO

- 无。当前公开范围不训练、不调参；保留为冻结对照。

---

## 实验 #002

- **日期**：2026-09-20 至 2026-09-21
- **要回答的问题**：保持 PPO/SAC 执行器不变，只把调度换成 VLA-as-Tools，是否能提高完整任务或物体完成率？
- **run 名称 / ID**：`goal-tools-paired16-20260920:gpt`
- **代码版本**：MS-HAB commit `e9ff3d23496d38e4431c8d913e147ffa007f7f72`；面板 SHA-256 `5fea051d078bf7c347fbddb997aeda12afa1324313ed3ffaa86f7cd83c6b5029`
- **环境**：Python 3.11.13 / PyTorch 2.7.0+cu128 / ManiSkill + MS-HAB / Quadro RTX 8000 48 GiB（GPU1）

### 配置（评测参数）

| 项 | 值 |
|---|---|
| 任务 & 数据划分 | 与实验 #001 相同的 16 个 TidyHouse validation plan |
| plan / 初始状态配对 | plan UID 与 Fixed 一致；初始状态 SHA-256 必须匹配 |
| 导航策略 | 官方 PPO |
| Pick / Place 策略 | 官方逐物体 SAC |
| 调度器 / 通讯协议 | `gpt-5.6-luna`；VLA-as-Tools goal protocol |
| 可见信息 | 五个物体目标、合法技能/目标、图像和执行历史；不提供原生下一子任务提示 |
| skill / episode 上限 | 40 动作 / 180 秒每次调用；40 次决策；每集 7,000 步 / 900 秒 |
| API 边界 | 2,048 输出 token；512,000 输入 bytes；每集预留上限 USD 0.05 |
| 随机种子 | 与实验 #001 相同 |
| 其它关键设置 | 训练更新 0；执行器、成功判定和分母不变 |

### 结果

| 指标 | 值（均值 ± population SD，N=16） | 备注 |
|---|---|---|
| 完成 episode | 16/16 | 基础设施失败 0 |
| 完整任务 SR | 0/16（0%） | 与 Fixed 相同 |
| 完成物体 | 13/80（16.25%） | 比 Fixed 少 1 个 |
| 每集完成物体 | 0.813 ± 0.882 | |
| 环境步数 | 总计 8,012；500.8 ± 237.5 / 集 | |
| API 请求 | 最终 attempts 共 245 次 | 实际供应商账单未核销 |

### 观察与结论

- **观察**：GPT 调度没有带来完整任务成功；物体完成数从 14 降到 13，这 1 个物体差异不足以支持优劣结论。
- **异常**：seed 16 在完成 2 个物体后因剩余 wall-budget 不足终止，应视为预算截尾，不应写成原生策略失败。
- **结论**：在当前 PPO/SAC 执行器上，VLA-as-Tools 调度没有改善这组 16-plan 诊断结果；不能据此推断所有 VLA 调度无效。

### 下一步 / TODO

- 无。下周模型调用和训练更新均为 0，不追加付费重跑。

---

## 实验 #003

- **日期**：2026-09-21
- **要回答的问题**：把 Navigate 替换为标准化 teleport、保持官方 SAC 与固定调度，能否提高操作阶段的中途进度？
- **run 名称 / ID**：`standardized-teleport16-20260921`
- **代码版本**：MS-HAB commit `e9ff3d23496d38e4431c8d913e147ffa007f7f72`；teleport source commit `4729821db3fc94a2470cfd625e6f8ab439f01478`；面板 SHA-256 `76367f66a7cae6b25aa46ece7fb37038cf2412d037e005818b44dacdf6c90576`
- **环境**：Python 3.11.13 / PyTorch 2.7.0+cu128 / ManiSkill + MS-HAB / Quadro RTX 8000 48 GiB（GPU1）

### 配置（评测参数）

| 项 | 值 |
|---|---|
| 任务 & 数据划分 | 与实验 #001 相同的 16 个 TidyHouse validation plan |
| plan / 初始状态配对 | plan UID 与 Fixed 一致；绑定 Fixed 初始状态 SHA-256 |
| 导航策略 | standardized teleport |
| Pick / Place 策略 | 官方逐物体 SAC |
| 调度器 / 通讯协议 | 官方固定任务顺序；无语言模型 |
| observation / policy state | 与 Fixed/GPT 相同的评测 runtime 和相机设置 |
| skill horizon / episode 上限 | 原生 horizon；每集 7,000 步 / 900 秒 |
| 模型调用与费用边界 | 0 次 API 调用；费用 0 |
| 随机种子 | 与实验 #001 相同 |
| 其它关键设置 | 训练更新 0；只替换 Navigate 执行方式 |

### 结果

| 指标 | 值（均值 ± population SD，N=16） | 备注 |
|---|---|---|
| 完成 episode | 16/16 | 基础设施失败 0 |
| 完整任务 SR | 0/16（0%） | 与另外两组相同 |
| 完成物体 | 21/80（26.25%） | 比 Fixed 多 7 个 |
| 每集完成物体 | 1.313 ± 1.310 | |
| 环境步数 | 总计 5,950；371.9 ± 261.7 / 集 | |
| API 请求 | 0 | |

### 观察与结论

- **观察**：Teleport 组完成了更多物体，但仍没有完整任务成功。
- **异常**：Teleport 改变了每次导航后的交接位姿分布；即使 episode 初始状态绑定一致，操作策略接收到的导航后状态也不同。
- **结论**：该设置说明改变导航交接可以增加部分进度，但不能证明 teleport 或某个学习导航策略“更强”，也没有解决完整任务失败。

### 下一步 / TODO

- 无。该结果作为参考臂冻结，不与旧 paper-release teleport 数字混用。

---

## 负结果备忘

| 试了什么 | 结果 | 可能原因 / 避免重复方式 |
|---|---|---|
| 正确固定调度 + 官方 PPO/SAC | 0/16 完整任务 | 调度正确不足以弥补操作链中的失败 |
| GPT 调度替换固定调度 | 0/16；13/80，未优于 Fixed 的 14/80 | 当前瓶颈不只在调度；不要继续仅换 prompt 重跑 |
| Standardized teleport 替换 PPO 导航 | 0/16；中途进度增至 21/80 | 交接分布改变但完整操作链仍失败；不能写成导航策略胜出 |
| 旧 paper-release teleport 面板 | 与标准化轮口径不同 | 不进入三组对照表，不再维护其原始记录 |

机器数据见 [`../results/tidyhouse-16/summary.json`](../results/tidyhouse-16/summary.json)。
