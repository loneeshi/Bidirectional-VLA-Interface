# 工程降级基线：历史配置锁定与下一验收门

2026-09-17：仅本地CPU核查，无GPU、API、训练。π₀.₅本周训练止损继续生效。

| 历史条件 | 高层与底层 | 已有结果 | 待验证范围 |
|---|---|---|---|
| organizer006 normal | GPT-5.6 Luna＋PPO导航＋按物体SAC操作 | 8请求、229动作、首物体链通过 | 实验室完整链复测；当前源码与历史版本不同 |
| B11 | oracle调度＋LightNav-0导航＋按物体SAC操作 | 4调用、278动作、首物体链通过；API0 | 原服务参数/调用上限补证据；实验室完整链复测 |
| 后续组合 | GPT＋LightNav-0＋SAC | 尚无该组合成功证据 | 单列新实验，不能合并以上两条成功结论 |

两条旧条件均seed1、TidyHouse sequential val、首计划`tidy_house-sequential-val-90-0`、GPU物理、depth堆叠3、stationary_head。历史环境均启用`navigate.ignore_arm_checkers=true`；这是已有工程配置，不是本轮新增修改，但不能称未改官方评估。原生Pick200步五种子测试是另一协议，不能与这个单物体链直接合并成功率。

Organizer历史40动作调用切片、650总动作上限、900秒、workspace/wrist给GPT、原导航head给PPO。原授权16请求/$0.32仅作为历史证据，不自动复用于新实验。可参考`scripts/run_coordinator.py --organizer`及`docs/coordinator-status.md`启动模板；远端路径与API桥须重新核实。

B11保留fetch_nav增补相机、waypoint_velocity、250导航预测上限、1200总动作上限和900秒。记录中存在方向恢复指令，旧成功中恢复调用0；不能将其删除后仍称完全相同配置。目标指令已逐字核对`configs/lightnav-seed1-disambiguated.json`。B的metadata未保存max_calls和skill_wall_seconds，不能从实际4调用反推所有启动参数。`--dry-run`在该runner中仍运行真实仿真，但绕过GPT。

[冻结清单](results/fallback-baseline-audit-2026-09-17/freeze-manifest.json)绑定两份原始metadata和当前/历史运行源码SHA。历史源码存在变化，须审查后才能声称行为不变。此清单不是可直接执行的新实验配置。

下一步按顺序：核对实验室完整链资产、checkpoint与GPU物理运行环境；补全历史参数并审查源码差异；先复测原有两条件。学习反馈先只记录，不接管控制：同一轨迹记录预测、原规则反馈与原生判据，报告错误完成/回退/停滞。现有AC-DiT头读取其动作token，不是独立通用观测头，也没有在SAC轨迹分布验证；不因SAC成功就认为迁移头可靠，不恢复AC或π基座训练。通过反馈门后才单列控制接入及GPT＋LightNav＋SAC组合。

该路线是工程闭环对照；SAC不是VLA，独立附加学习反馈不能称作者TAPT。8个验证输入已就绪，但尚无学习反馈验收通过结果。
