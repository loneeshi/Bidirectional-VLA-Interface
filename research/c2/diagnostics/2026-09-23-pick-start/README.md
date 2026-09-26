# 2026-09-23 Pick 监督诊断归档

本批结果从正式 `docs/log/` 移出。运行时给 GPT 的候选仅有 ID 与未标明语义的 `[x,y,yaw]` 路线；没有逐候选相对目标几何、实际到达率或机械臂可达性。seed9 中 GPT 将只改变底座朝向的 C5 描述为改变高度。因而这批数据保留为接口和失败归因诊断，不能作为 GPT 几何选位能力的正式证据。

- [#009 第 0 步接口验收](2026-09-23-pick-start-interface-gate.md)
- [#010 seed9 历史交接](2026-09-23-seed9-online-recovery.md)
- [#011 独立起点探索](2026-09-23-independent-pick-recovery.md)
- [运行前冻结设计](../../docs/pick-start-supervision-recovery-experiment-design.md)

原始轨迹和逐次模型请求继续保存在独立运行目录；支出与未知账单保留在项目财务账本。归档只变更文档位置和结论层级，不改写已发生的动作、请求或官方严格 Pick 评分。
