# 离线空间理解：GPU1 两例导出重试 002

## 范围与执行

用户在看到 [retry-002 修订](../docs/real-handoff-spatial-stage1-gpu1-pilot-retry-002-proposal.json)后回复“批准”。授权仅覆盖真实交接清单位置 5、8 的冻结快照导出：GPU1 串行最多两个进程，每进程至多 180 秒，累计至多 360 秒；模型 API、策略动作和新 RunPod 资源均为 0。旧两次先导的剩余额度没有转入。

执行前 CPU 干跑通过官方 MS-HAB 源码、任务计划、两份快照、episode 配置、目标物体、抓取资产、自碰撞过滤源码及官方网格的核对。GPU1 起始 15 MiB、0%、无计算进程。授权文件、控制器、导出器在远端的 SHA 分别为 `a0a8506108fda8b8e12a0fb45225fabb9bccbf692548b34bdd10ed8509fe986f`、`39f045f967c43f4a67556cefad32519d980618994308dda336b2aa582eeead6f`、`feb96c07d888bc9fb2afda4cea0bb05c6f84c7fc528018c3a8210cf9dbcfdf67`。

| 位置 | 进程 | GPU1 进程秒 | 快照恢复 | 128 px 官方输入 | RGB-D 归档 | 几何包 |
|---|---:|---:|---|---|---|---|
| 5 | 1 | 27.757747 | 通过 | 哈希通过 | 0 | 0 |
| 8 | 0 | 0 | — | — | 0 | 0 |

位置 5 已创建环境并恢复同一冻结交接；恢复误差和 `subtask_pointer` 检查通过，随后 128 px 的 `adapter.observe().policy` 与普查时记录的官方 Pick 输入 SHA 相同。源码复核表明 `adapter.observe()` 返回的是适配器缓存的 policy 输入，不能据此断定本次新渲染的 RGB-D 相同。接着调用 `adapter.uenv.get_obs()` 读取 RGB-D，模拟器状态相对快照的差值超过预设 `1e-5`，导出器抛出 `sensor rendering changed handoff simulator state`。现有日志未记录差值大小或具体状态字段，不能断言是物体位姿、传感器缓存还是其他字段变化。图像保存前即停止，不能把本次传感器读数当作已验证的无动作输入。

控制器按停止门没有启动位置 8。原始位置 5 日志 SHA-256 为 `56f71b245098cf9997d53674bb28b8e8ad312ce5286671194d5876195c79830f`；回执、日志、事件和未完成的 `result.json` 在本地忽略目录 `runs/real-handoff-spatial-stage1-20260927/gpu1-pilot-retry-002/`，远端源目录为 `/home/pshuai/bvi-research/runs/real-handoff-spatial-stage1-20260927/results-pilot-retry-002/`。尚无可信 128／256 px 成对图像、同快照几何包或 Q1／Q4 标签。

## 下一步诊断

需要在传感器读取前后分别记录状态差异的字段路径、数值和最大误差，并确认差异是否发生在第一次官方输入读取还是第二次 `get_obs()`。在未区分物理状态和可重建的观察缓存前，保持 `1e-5` 状态守卫，不放宽阈值。修改导出器并通过 CPU 测试之后，新的 GPU 进程须另立有界修订；本次 332.242253 秒未用额度不能自动续跑。

已在本地导出器加入第一次官方输入读取后、第二次 RGB-D 读取后的逐字段差异摘要，并在失败前持久化；状态阈值保持 `1e-5`。16 项相关 CPU 测试通过。原冻结文件和 v2–v4 修订未覆盖；v5 输入代码修订 SHA-256 为 `9254ebdf9055b68320aa067e61ef6e52d08e40140fca9f3574637d6e47e1c30b`，只更新导出器源码哈希，60 例清单、条件和 Q2／Q3 标签未改。此修复仅改进下一次失败归因，还没有在 GPU 仿真中验证。

### 事后源码归因

随后核对远端固定版本的 ManiSkill `sapien_env.py`（SHA-256 `09a176d2a81924ae3b0234405bab2d45cd861a250c37ea52f12b6994b5753208`）：无参数 `get_obs()` 先调用 `get_info()`，后者调用任务 `evaluate()`；MS-HAB `sequential_task.py`（SHA-256 `47fd4999f7326568a550890caa48eac070901525373fac667ff56384eeb2a6be`）的 `evaluate()` 会增加累计接触力、更新 subtask 指针并递减剩余步数。旧导出器正是无参数调用 `get_obs()`，因此它并非纯读取相机。源码还表明 `_get_obs_sensor_data()` 是 `get_obs()` 使用的传感器渲染链路，且不调用 `get_info()`。本地已改为直接读取该传感器链路，保留读取前后 `1e-5` 状态守卫；合成接口测试验证不会调用 `get_obs()`／`get_info()`。此次运行没有记录变动字段，所以源码链路解释了一个明确的污染机制，但尚未实测证明它是唯一差异来源。后续 GPU 验收仍必需。

## 资源与费用

本次 GPU1 27.757747244089842/360 进程秒，API／策略动作／新 RunPod 资源均 0。实验室向项目收费沿用用户已确认的 0；历史停止的 RunPod 持久盘继续单列。收尾 GPU1 15 MiB、0%、无计算进程，GPU0 未操作。见[独立账本](../../../../BVI-research-plan-2026-09-14/finance/ledger-real-handoff-spatial-stage1-gpu1-pilot-20260927.json)。
