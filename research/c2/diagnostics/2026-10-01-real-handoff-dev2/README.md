# 开发集 2：20 个有效首次交接已冻结

所在阶段：第 2 步，失效归因与接口验收。推进：24 个候选得到 20 个有效交接，原生 SAC 单次严格成功 13/20；20 快照通过 CPU 反序列化与哈希核对。阻塞：C 系统对照入口/参考时间协议未验收。下一道门：C 完成后才可运行开发集 2 Astra；本批 Astra 尝试 0。

## 冻结与边界

- 依原 UID 字典序执行；排除旧 64 次普查尝试及已核实曝光，首 30 候选已冻结。尝试前 24 个后达到 20 有效交接即停，剩余 6 个未启动。3 个导航失败、1 个初始化时导航已满足，单列且不补跑旧实例。没有基础设施删失。

- 原生 PPO/SAC 观察、检查点及 500/200 步不变；每个有效交接只运行一次 SAC，成败不影响入选；测试集未动。新交接不参与本轮执行器调试。

- 评估端 1.14 m 保守筛分：界外 18 例、界内 2 例。这个标签不是抓取可达性证明；目标包围盒几何不进入 Astra 请求。

- 分层 SAC：界外 11/18；界内 2/2。整体 13/20（65%）只代表本次单次原生 SAC 参考，不能当作 Astra 成功，也不能推断复跑稳定性。

- [冻结 20 个交接](frozen-dev2.eval-only.json)、[原 30 候选](roster.json)、[UID 顺序核证](uid-order-proof.json)、[逐例摘要](summary.json)、[快照 CPU 检查](snapshot-cpu-check.json)。CPU 反序列化不是物理恢复一致性验证，后续使用前仍须检查恢复状态与官方剩余步数。

## 媒体与档案

23 个有动作的尝试全部保留在线视频、TCP 轨迹叠加和分析页面；1 个零动作初始化案例明确无动作，不伪造视频。下表链接为每例完整媒体索引，文件名均列于索引。它们是原生 PPO/SAC 参考，不是 Astra；SAC 未输出末端轨迹，不人为补造 Astra 预测曲线。叠加坐标与第三人称视图只用于评估。

第一个样本的 CPU 媒体任务误重复处理，原视频 SHA-256 与第一次成功回执完全一致；重复处理的错误回执保留，交付状态已更正，无物理重跑。收取档案曾超过 SSH 60 秒等待；等待远端打包结束后收取同一档案并校验，没有重跑实验。

- [收取回执与完整原始档案哈希](collection-receipt.json)、[媒体索引](media-index.json)、[所有原始文件哈希](remote-results/full-results-manifest.json)。原始在线 PNG 完整保存于实验室 `/home/pshuai/bvi-research/runs/real-handoff-dev2-20261001-r1/full-results.tar`；本地收取包包含全部快照、逐步记录、结果与生成视频，未替代或删除远端原始帧。

|候选|原普查位置索引|任务 UID|状态|层|SAC 严格成功|完整媒体|
|---:|---:|---|---|---|---|---|
|0|64|tidy_house-sequential-val-156-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan064-1ba61bc0/delivery/README.md>)|
|1|65|tidy_house-sequential-val-157-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan065-c7de2a80/delivery/README.md>)|
|2|66|tidy_house-sequential-val-158-0|completed|outside|False|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan066-78338d78/delivery/README.md>)|
|3|67|tidy_house-sequential-val-159-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan067-66190437/delivery/README.md>)|
|4|68|tidy_house-sequential-val-16-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan068-aa4f6c08/delivery/README.md>)|
|5|69|tidy_house-sequential-val-160-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan069-ec43c327/delivery/README.md>)|
|6|70|tidy_house-sequential-val-161-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan070-986e8fbc/delivery/README.md>)|
|7|71|tidy_house-sequential-val-162-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan071-f80ba93b/delivery/README.md>)|
|8|72|tidy_house-sequential-val-163-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan072-ea42c6bd/delivery/README.md>)|
|9|73|tidy_house-sequential-val-164-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan073-b32fd6d7/delivery/README.md>)|
|10|74|tidy_house-sequential-val-165-0|navigation_failed|—|不适用|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan074-8b110e1f/delivery/README.md>)|
|11|75|tidy_house-sequential-val-166-0|completed|inside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan075-9fb10805/delivery/README.md>)|
|12|76|tidy_house-sequential-val-167-0|navigation_failed|—|不适用|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan076-98968cd1/delivery/README.md>)|
|13|77|tidy_house-sequential-val-168-0|completed|outside|False|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan077-2b39d83c/delivery/README.md>)|
|14|78|tidy_house-sequential-val-169-0|completed|outside|False|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan078-eb274ab1/delivery/README.md>)|
|15|80|tidy_house-sequential-val-170-0|completed|outside|False|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan080-04b306f6/delivery/README.md>)|
|16|81|tidy_house-sequential-val-171-0|completed|outside|False|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan081-2cadd736/delivery/README.md>)|
|17|82|tidy_house-sequential-val-172-0|completed|outside|False|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan082-27cbac79/delivery/README.md>)|
|18|83|tidy_house-sequential-val-173-0|navigation_failed|—|不适用|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan083-30329605/delivery/README.md>)|
|19|84|tidy_house-sequential-val-174-0|completed|outside|False|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan084-3c1f0717/delivery/README.md>)|
|20|85|tidy_house-sequential-val-175-0|navigation_already_satisfied_at_reset|—|不适用|无动作，无视频|
|21|86|tidy_house-sequential-val-176-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan086-cbdd6c7e/delivery/README.md>)|
|22|87|tidy_house-sequential-val-177-0|completed|inside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan087-79b9b23f/delivery/README.md>)|
|23|88|tidy_house-sequential-val-178-0|completed|outside|True|[视频、轨迹与回执](<D:/AI/Embodied Intelligence/VLA as Tools/repo/Bidirectional-VLA-Interface/docs/media/c2-pick-dev2-2026-10-01-plan088-f04614ed/delivery/README.md>)|

## 资源与验收

24 个 GPU1 串行进程，实耗 **1494.218746 / 5400 进程秒**；4219 个环境动作，API 0。未用 **3905.781254 秒**已关闭，不转入 C 或模型试点。实验室收费按用户确认 0、无发票；[GPU1 收尾核查](resource-final.json)显示无计算进程。历史停机 RunPod 存储本轮未查询，费用未知。
数据生成与 CPU 档案检查完成；原生 SAC 严格 Pick 13/20；Astra 机器人实验未启动；C 和研究整体均未完成。未发布 GitHub。