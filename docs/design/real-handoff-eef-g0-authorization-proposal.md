# G0：真实交接末端执行器上限运行提案

日期：2026-09-28。**待批准，不构成 GPU 或 API 授权。**

当前五阶段主线仍在第 2 步的失效归因与接口检验。本周转向
[末端工具闭环与上下文重试设计](../../research/c2/docs/real-handoff-eef-tool-loop-icl-design.md)。
本提案只验证执行器是否能够复现 SAC 的动作能力；没有 Astra 请求，
即使回放严格成功，也不构成 Astra 能力或任务整体完成。

## 冻结来源与样本

使用既有 60 交接普查按排序后行号 `i % 3 == 0` 冻结的 20 开发交接；
其中首次 SAC 严格成功的 11 个全部纳入。不是按 plan_position 取模。
不访问测试集轨迹来选择或调试样本。

| plan_position | plan_uid |
|---|---|
| 5 | tidy_house-sequential-val-102-0 |
| 8 | tidy_house-sequential-val-105-0 |
| 11 | tidy_house-sequential-val-108-0 |
| 17 | tidy_house-sequential-val-113-0 |
| 29 | tidy_house-sequential-val-124-0 |
| 32 | tidy_house-sequential-val-127-0 |
| 35 | tidy_house-sequential-val-13-0 |
| 38 | tidy_house-sequential-val-132-0 |
| 45 | tidy_house-sequential-val-139-0 |
| 48 | tidy_house-sequential-val-141-0 |
| 52 | tidy_house-sequential-val-145-0 |

逐例快照 SHA-256、归档成员名、字节数和冻结来源哈希在
[机器清单](../../research/c2/diagnostics/2026-09-28-eef-g0-cpu/readiness.eval-only.json)。
11 个归档成员均已核验字节哈希。**仅证明文件完整；运行时恢复仍待验收。**

## 运行前门与资源

- 重新读上级 AGENTS.md、finance/README.md 与 ledger.json；本地财务 ledger
  当前为 GB18030 编码，按原编码读，不覆盖或重写历史。
- 2026-09-28 本轮只读 SSH：GPU1 UUID
  `GPU-b7ebba23-7824-7601-df32-be55628936c3`，15 MiB、0%，没有该卡计算进程；
  GPU0 被他人占用，未触碰。工作目录可写，官方源码可读，余量约 636 GB，quota 无限制报告。
- 登录 PATH 没有 sbatch/qsub；未确认实验室的预约、共享卡或排队规则。
  **启动前必须确认 GPU1 可按本提案使用**，不能从空闲或无调度命令推断许可。
- 启动时重查 GPU1 UUID、占用和调度许可；占用超过 1 GiB 或出现其他进程则不启动。
- 凭证和 known_hosts 已验证被 git ignore；不打印、不写入报告。
- 工具运行器绑定 live controller：模式 `pd_joint_delta_pos`、`use_target=False`、
  正确 joint names、动作切片、归一化及上下限；不匹配则动作 0 前退出。
  固定动作排列 arm7 / gripper1 / body3(head_pan,head_tilt,torso) / base2。
  arm/body 每步物理增量限幅 ±0.1（旋转 rad、躯干 m）；夹爪是绝对 mimic
  命令 [-0.01,0.05] m，对应 normalized [-1,1]。
- 工具回放底盘动作始终为零。使用 `stationary_base=True, stationary_torso=False,
  stationary_head=False` 包装器；普通末端移动保持头部实测角度，只有
  return_to_rest 主动复原头部。旧 runner 的 stationary_head=True 会抹掉这两个通道。
  SAC 来源重跑保留原 baseline 包装器配置，分别记录配置，不改变历史基线。
- 从同一快照恢复物理、控制器、RNG、任务指针、剩余步数、累计力及计时器，
  核对初始 qpos/qvel、物体状态、观测；不得 reset 力或额外增加时限。
  恢复失败单列基础设施删失，不伪装 Pick 失败。

## 固定运行顺序（最多 55 个独立进程）

逐例按上表顺序，所有失败都留档，没有自动重试或补样本。

1. 每个交接 SAC 从原快照重跑两次（22 个进程），保持原策略随机性，记录种子/RNG。
   每次按官方剩余 horizon，最多 200 环境步；成功立即停止。
2. 两次都失败的交接保留为自然复跑不稳定，无成功教师可回放，不补第三次。
   有成功的交接只用**第一个**成功重跑作为教师，不择优。
3. 对每条教师，分别从同快照做 8 DOF 与 arm-only 7 DOF 末端关键帧回放
   （最多 22 个进程）；7 DOF 躯干保持起点，头部两者都不参与 IK。
4. 每例再做一个原教师关节动作回放参照（最多 11 个进程），诊断快照/随机性/
   关键帧和 IK 分配导致的损失。严格得分与工具回放分开。

## 逐步采集、压缩与执行

每个动作前后都记录环境步号、完整 15 维 qpos/qvel、官方原始和实际动作、
夹爪命令与实测指距、URDF FK 的 base_link TCP 位置及 xyzw 单位四元数、
头/手 RGB-D、内参、URDF 相机外参。TCP 为 gripper_link，启动时用官方
agent.tcp 作**评估端**核对；有非单位 TCP offset 时先停止并修正定义。
官方 success、robot_rest、ee_rest、is_static、is_grasped、累计力等单独写
eval-only 文件，不流入未来请求或示范反馈。世界坐标仅用于评估端核对。

关键帧保留首末帧、夹爪开合事件及事件前后帧；其余按最大直线位置重建
误差 0.01 m、SLERP 朝向误差 0.05 rad 压缩，不依赖目标真值或接触身份选帧。
保留顺序、原时间戳和夹爪保持区间。若压缩轨迹超出已冻结 horizon，则失败，
不得为了通过改时限或跳过 waypoint。两种 IK 使用完全相同关键帧。

执行器默认加密上限每段 0.02 m / 0.1 rad；每点以前一点 IK 解为初值，
起点为实测 qpos；位置残差 ≤0.002 m、朝向残差 ≤0.02 rad。仅用机器人
URDF、关节限位和实测角度，不加载场景网格或 FCL。无解只报搜索未找到，
不声称不可达。实际动作每步依据最新实测 qpos 计算，再限幅和归一化。
规划时给出动作限幅下最少步数估计；它不是动力学完成时间保证。
出现官方 terminated/truncated 立即停止，不继续关夹爪或回休息位。

抓后调用 return_to_rest：qpos[3:-2] 回官方 rest 关键帧（躯干 0.386 m），
底盘保持、夹爪保持最后命令；最多额外 3 个停稳步，均计入原 horizon。
当前程序侧 0.002 的关节目标容差只是控制停止条件，不能替代官方 robot_rest
容差或 is_static。记录加载到的官方阈值（源码默认 grasping 容差 0.6），
最后由官方五项联合判定严格成功。不可用 IK 回 TCP 位置代替关节回原位。

## G0 / G1 / G2 通过标准（批准时冻结）

- 固定分母 11，逐例列出 SAC 两次结局、关节回放、8 DOF、7 DOF；另报至少
  一次 SAC 复跑成功的 N_ref。报告所有删失与缺教师，不改分母隐藏失败。
- G0 前置：至少 9/11 有成功教师；恢复身份、官方 controller 与固定底盘门
  全部通过。关节回放成功 ≥ ceil(0.9 × N_ref)，否则先归因重放不稳定。
- G0 主门：8 DOF 严格回放成功 ≥ ceil(0.9 × N_ref)，且相同教师的
  return_to_rest 结束时官方 robot_rest 全部成立；不满足即停在执行器诊断，
  不进入 API。7 DOF 作为配对消融，不是放行主门，结果不得混成 Astra 成绩。
- 统计每条 SAC 完整轨迹躯干 min/max、peak-to-peak、累计 |Δq|、离休息位
  最大偏差；报告 8/7 DOF 躯干差别及严格得分配对表。
- G1：全部教师实测 TCP 位姿均以已知到达 qpos、前一帧 qpos 分别测试；
  此外按实际 waypoint 顺序从交接 qpos 调用 check_path。已到达点接受率
  必须 100%，残差在上述限内，check_path 调用不得产生 env.step 或改 qpos。
  自点种子自洽通过不代替连续路径门；失败报告种子/残差/限位原因。
- G2：CPU 已完成合成坐标/无效深度测试与 60 相机 FK 外参核对，但尚无
  目标表面误差验收。G0 采集图像后，评估端用真值分割选表面像素，绑定
  depth 单位、光轴与相机 convention，反投影误差 p95 ≤0.01 m、有效像素
  比例 ≥95%；头/手分开报。未过不进入主条件；阈值变更必须另写修订。
- G3 暂无请求，后续复用请求审计器，扩展必须适配允许的 IK 与结局反馈，
  不通过删掉旧审计规则来绕过主条件禁止信息。

## 拟申请用量与停止门

仅实验室 GPU1，**总进程墙钟秒 ≤7200，单进程 ≤180 秒，最多 55 个进程**。
每次 CUDA 初始化和所有失败进程也计时；每次启动前预留完整单进程上限，
累计预算不足不启动。外层 timeout 与运行器内部计时双守卫，进程退出前核实
子进程已释放。超过累计上限、恢复门失败、controller 不匹配、GPU 被占用
或调度许可不明时停止。旧实验余额不转入。API 0 次、训练更新 0。

实验室项目收费按用户本轮确认记为 0（无发票），仍逐进程记录秒数与来源；
无新增租机/存储购买。历史 RunPod 存储费用另列，本轮未刷新其状态或账单。
当前本提案授权余额未知；批准后建立独立财务分账并挂入总账，在每个进程
执行前登记预留、执行后登记实际用量，结束重查 GPU1，记录继续占用状态。
7200 秒只是申请上限，没有运行或收费发生。

批准后先完成独立 G0 runtime runner/逐步 trace 接线与零动作预检，再按固定
清单执行。CPU 核心与测试已准备，真实控制器接线与完整恢复仍是运行前门。
