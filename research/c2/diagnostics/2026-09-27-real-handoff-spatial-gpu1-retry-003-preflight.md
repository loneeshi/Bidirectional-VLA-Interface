# 阶段 1 图像导出 retry-003：运行前检查

## 修正

retry-002 的位置 5 在无参数 `get_obs()` 后触发状态一致性门。固定版本源码表明 ManiSkill `get_obs()` 先调用 `get_info()`，MS-HAB 的 `evaluate()` 会修改任务计数和累计力；详见[上次诊断](2026-09-27-real-handoff-spatial-gpu1-pilot-retry-002.md)。新导出器改为调用同一官方相机渲染链路 `_get_obs_sensor_data()`，不调用任务评估；官方 ManiSkill `sapien_env.py` 的 SHA-256 锁定为 `09a176d2a81924ae3b0234405bab2d45cd861a250c37ea52f12b6994b5753208`。快照恢复与渲染前后仍须满足 `1e-5` 状态误差门，不因源码判断而放宽。

## 待批范围

[retry-003 提案](../docs/real-handoff-spatial-stage1-gpu1-pilot-retry-003-proposal.json)只覆盖位置 5 的一条冻结交接：实验室 GPU1 单进程上限 120 秒、累计上限 120 秒，API、策略动作、新 RunPod 资源均为 0。遇到任何运行错误即停止，不启动位置 8。此前未用额度不转入。提案状态仍为 `pending_user_approval`，尚未创建 GPU 子进程。

## CPU 与远端静态验收

- 本地阶段 1 相关测试 36/36 通过；冻结 v8 的 27 项源码 SHA 与当前文件逐项一致，60 例清单及标签未变。v8 SHA-256 `e54c015e57145b2b6ed18a6a10609a7074f42ffeeca0f3a9c667e9e62a2b131f`。
- 已将固定源码复制到远端新目录 `stage/source-pilot-v8/`，不覆盖 retry-002 的归档目录。控制器、导出器和 v8 清单在远端的 SHA 分别为 `fe2a969554fcec875f2a27f69515e9bfb6a4b4affa8a63143fc8c39348891f96`、`707ee92d66db4394dbcd53e4fe6e83e99823255678aabd6d442c3d5f46667507`、`e54c015e57145b2b6ed18a6a10609a7074f42ffeeca0f3a9c667e9e62a2b131f`，与本地提案一致。
- 远端 `--dry-run` 接受待批提案但不创建输出或 GPU 子进程。它核对官方 MS-HAB 源码、ManiSkill 相机源码、前次回执、位置 5 快照、episode 配置、目标和任务计划、抓取资产、自碰撞过滤源码、官方网格、资产根目录和磁盘余量。位置 5 快照 SHA-256 为 `e56a12e95d6a4b630a9c5bae36ebaea6b73a952104e17f38c4767b57789e847a`，官方网格 SHA-256 为 `4f185177981b21cfa373e4244855fef91f3fbd2226b2e04095e1b9ee4dc337ab`。干跑时 GPU1 15 MiB、0%、无计算进程；GPU0 有他人进程，未操作。

此静态验收不能证明新相机读取对物理状态无影响，也不能证明 128／256 px 图像或几何导出成功。只有批准后的单例 GPU1 运行能检验这点。全 60 例 GPU 批和两个模型的 API 预算仍未授权。

后续用户已批准并完成位置 5 单例执行；结果见[retry-003 执行诊断](2026-09-27-real-handoff-spatial-gpu1-pilot-retry-003.md)。本页以上内容保留为运行前状态。
