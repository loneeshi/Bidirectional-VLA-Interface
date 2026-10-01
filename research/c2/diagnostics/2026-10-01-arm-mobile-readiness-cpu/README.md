# 同五例允许底盘条件：当前代码 CPU 审计

## 后续修复回执：新移动条件已冻结，运行额度待确认

用户随后要求修复并启动。已新增独立 eef_mobile_contract / eef_mobile_api 和移动 broker、worker、串行 batch，历史 arm 模块不改。开放 move_base，每次请求发送显式 robot_specifications；非单位四元数等错误返回可修正的参数说明，无自动归一化；启动时调用 validate_base_runtime。五例仅跑 Astra V-mobile，新 worker 与 r3 实测初态逐例核对。手臂与底盘物理控制器保持原冻结实现。

149 项 CPU 回归通过，见 [mobile-regression.xml](mobile-regression.xml)。包括允许底盘历史的请求审计、无效四元数反馈后下一次合法请求、移动后旧深度拒绝与新观测可用、抓取 last_goal 刷新、80 步耗尽、官方终止即时停、速度/频率错误拒绝、请求和 USD 触界前拒绝、未知发送不重试。原多轮历史快照和定位隔离回归亦通过。

服务器 CPU 部署检查通过：108 源文件与 1046 资产哈希、5 个 r3 初态绑定、正确 assets/data 路径、禁止未授权模板、密钥权限检查，CUDA 初始化 false、API/物理步 0，GPU1 检查时空闲。密钥内容未读取。见 [server-mobile-deployment-cpu.json](server-mobile-deployment-cpu.json)。跨系统初态绑定使用明确 LF 序列化，冻结 zip 用确定性时间戳；最终运行包 SHA-256 为 `590063d9988a51814f311caa784d2d51565ba276883092a8ecad1404e07ebab2`，见 [mobile-freeze.json](mobile-freeze.json)。

部署准备完成，具体 [新预算提案](../../../../docs/design/c2-eef-same-five-mobile-authorization-proposal.md) 为五进程、API最多125次、USD65硬上限、GPU1最多9000秒；[authorization.template.json](authorization.template.json) 仍为 not_authorized。旧剩余额度不转入；新上限确认后才运行 launch-after-budget-approval.py。目前没有新增计费运行。下文为修复前审计与设计依据，其中“未修改运行接口”描述的是当时状态，以本节为最新回执。

主线阶段 2：模型工具接口和失败归因。用户要求在同样五例允许 Astra 底盘执行，并核查参数和代码。已完成只读 CPU 请求/验证链审计，API 0、物理步 0，不涉及模型或 GPU 运行；固定底盘 r3 结果与冻结执行器未修改。本记录不是新实验授权。下一门为独立移动条件接口实现、CPU 验证及用户批准的新资源提案。

## 已核实事实

1. `eef_guidance_v4 → v2 → METHOD_GUIDANCE` 实际生成的提示已包含机器人常量：底盘水平保守半径 0.288 m、直径 0.576 m、前沿 0.287 m；这只包住刚性底盘，不包住伸出的手臂。还有 0.5 m/s、1 rad/s、先转后直行、不能横移、到位误差 0.02 m / 0.03 rad、每调用 80 步，以及来自机器人 URDF 的肩部与手臂长度上界。这些不是场景真值。CPU 生成当前完整请求确认这些文本全部存在。
2. 固定底盘 r3 由 `eef_arm_contract` 从 schema 删除 move_base，并在提示和 validate_call 同时禁止。执行器继承的 v13 底盘能力仍存在，但不能只改一句提示便通过当前 schema、审计器和 broker。必须使用独立的 V-mobile 条件与新授权，保留旧请求版本不变。
3. 历史 #016 的 PPO 输入只提供 waypoint XY，却用终点 yaw 判定到达，是曾经发生的输入/验收不一致。当前 v13 不使用该 PPO：它从模型 turn_rad 得到 heading，并在实际里程计循环及 arrival 判断中使用 heading。因此该历史缺陷不能直接归因给当前底盘工具。另有旧 E1-B v13 未提供机器人几何与方法指导的版本，之后已由 guidance-v1 补入，当前继承了该指导。

## 当前需要处理的问题

| 问题 | 当前证据 | 新条件的最小处理 |
|---|---|---|
| 无效参数原因丢失 | 非单位四元数在 parse_response 中变成 call=None / invalid_tool_call，worker 只能生成笼统反馈。r3 共 13 次：002 为 1 次，003 为 4 次，004 为 8 次。 | 新版本返回受控、仅基于请求参数的错误代码与改正说明；原始失败调用保留在评估档案，不能作为有效历史调用执行。禁止自动修改模型位姿。 |
| 基座合法命令超过调用时限 | 最大组合 π rad + 1.5 m，即使零加减速/零等待也至少 123 步；底盘调用只允许 80 步。 | 明确 80 步只是一调用上限，先转后走分阶段、接近时短距离分段；不承诺合法命令必到位。保留实际位移、残差、settle/motion 步数反馈。 |
| 部署未校验底盘实际动作映射 | worker 只调 validate_official_controller，未调已有 validate_base_runtime。底盘动作缩放按固定 ±1 m/s、±3.14 rad/s 和 20 Hz 写死。 | 移动条件初始化必须核实映射、速度缩放与频率；不符合即停止，不能启动后猜测修正。 |
| check_path 通过不保证实际运动完成 | 只验证 IK/连续性及理论下界；执行阶段才调用 coupled_schedule，可返回 reference_timing_infeasible。000/004 的脚本实际受此限制。 | 维持现有执行器；明确该反馈与到位门限。新 CPU 测试覆盖通过 check 后执行拒绝/步数耗尽可被模型理解，不把它包装为可执行保证。单独优化执行器超出本轮最小改动。 |
| 移动使旧位置和最后抓取位姿失效 | locate_point 返回点在源观测的当前 base_link，odom 为尝试起点坐标；现有 v13 move_base 退出会重置 last_goal 为实测 TCP，深度工具也有旧观测保护。 | 对新移动条件加集成回归：转动/前进后必须刷新感知再定位，不能直接复用旧点；验证实际 current-base FK、close 的目标和 prior/history 一致。无需传场景真值。 |

上述前三项由 [audit_arm_mobile_readiness_cpu.py](../../../../scripts/audit_arm_mobile_readiness_cpu.py) 在 CPU 上直接检查或复现；[audit.json](audit.json) 记录当前提示哈希、错误计数与最低时间计算。当前未修改运行接口，也未声称移动控制已在这五例通过物理测试。

## 新条件建议

只运行同一冻结清单 arm-dev-000..004 的 Astra V-mobile，各一次，使用与 r3 相同 UID、spawn_index、seed、官方 Pick 初态和五项判据。每轮 600 步非官方条件，并报告首次成功步数与 200/600 步结果；工具最多 25 次。原生 SAC r3 结果仅作为已有上下文，不新增 SAC/脚本进程。沿用协调 v2 手臂与 v13 底盘，不修关节空间候选、不加入 PPO、navmesh 或真值碰撞筛选。

建议新提示版本 guidance-v4-mobile，所有 base 参数单独放在明确的机器人规格块，并写清各工具上限：arm 150、base 80、look 40、gripper 15、rest 150、total 600。该块从固定机器人模型和控制器配置校验生成，不从目标/家具碰撞形状推导。保留底盘 footprint 不包含伸出手臂、深度遮挡不等于自由空间、家具边缘优先停车参考的说明。

预算在新接口冻结后提交，旧批未用额度已关闭，不自动转移。若仍采用每轮最多 25 次，则五例最多 125 次发送；若采用每进程 1800 秒，则五例 GPU 上限 9000 秒。USD 硬上限需用户批准，实际预留逐次按请求计算；可使用 r3 的 102 个新请求、估算 USD19.7510125 为历史规模参照，不能将其当作本轮费用保证。

这五例均已暴露，是开发诊断，不能算独立测试结果。原固定底盘脚本 0/5 门槛失败保持不变；新移动条件不能声称通过旧门。测试 30 例和开发集 2 不动。继续保存每次物理运行的原始录像、同步 TCP 曲线、轨迹与 rationale/termination，评估端定位诊断不进请求。结果只写 diagnostics；不发布 GitHub。
