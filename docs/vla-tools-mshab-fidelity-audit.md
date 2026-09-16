# MS-HAB：VLA-as-Tools 方法完整性与替代策略审计

审计日期：2026-09-16。仅检查本地代码、历史日志及公开一手来源；未修改 baseline 机制，未租 GPU、训练或调用实验 coordinator API。

## 结论与证据边界

当前 MS-HAB 系统是工具调用接口迁移，加上原始 LightNav-0 和普通 Fetch LoRA π₀.₅；并非完整 TAPT 迁移。LIBERO 已完成的小规模工具族训练不能当作 Fetch 权重或导航训练。现有失败不能证明完整 VLAs-as-Tools 方法无效，也不能据此断言补齐方法就必然成功。

MSHAB013c 的单通道干预仅说明三个被测试的替换方式不足以救回抓取。混合策略本身可能产生不协调，不能把它解读成已唯一锁定某种模型内部缺陷，或证明重训一定优于接口修复。

## 实际训练账目

| 后端 | 本项目已执行 | 未执行 |
|---|---|---|
| LightNav-0 | 加载官方权重、推理与 Fetch 动作接口适配；检查点 revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` | MS-HAB SFT、导航工具族 TAPT、学习进度头、RL |
| Fetch π₀.₅ V8 | SAC 示范上的普通 LoRA 模仿学习；2,000 updates；1,228 帧、24 source runs | 按 reach/grasp/move/release 选择残差银行、Fetch 学习进度、DROID invocation-aligned 中间阶段、GRPO |
| LIBERO π₀.₅ | 四组真实 LoRA＋作者进度头；动作损失＋进度 MSE；20 条完整训练示范、5 条验证示范；后续语言变体适配 | 完整 DROID 中间阶段、原文规模与全任务评估、RL |

V8 的24个来源包含重复起点和恢复尾段，并非24个独立任务分布。对 SAC 生成的动作做监督拟合仍是 IL/SFT，不能称为 π₀.₅ 自己经过 SAC 或 GRPO 训练。

可追溯证据：`v8-final-status.md`、`analysis/mshab012/training-support.json`、`results/workspace-recovery-training-data.json`、`tapt-libero-run01.md`、`tapt-libero-run02.md`、`mshab-tool-interface-migration.md`。MSHAB011 实际 metadata 明确记录 `residual_family_adapters=false`、`learned_progress_available=false`。

## LightNav 打转：已有日志的新审计

来源：本地 `runs/mshab011/final-extracted/runs/mshab011-attempt5/events.jsonl`；SHA256 `c1dd2f342976db98a6241833dc8e8d03fd0d623159ee84de796a4b6acbbe36c9`。这是单场景历史运行，不是新增对照。

- 9 次导航调用、66 次 LightNav 推理、330 个仿真控制步。
- 260/330 步（78.8%）满足审计定义 `abs(command_v)<0.01 m/s` 且 `abs(command_w)>0.3 rad/s`。该阈值只用于描述日志，不加入控制逻辑；搜索时转向本身未必错误。
- 相邻日志位姿累计绝对转角约17.92 rad，约2.85圈的绝对转动量；不是同方向绕圈次数。平面累计路程约1.83 m。
- 实际模式为 `waypoint_velocity`、相机 `fetch_nav`。不能用早期 `position_tracker` 的先转后走逻辑或128像素 head camera解释这一轮。
- 第一次模型预测本身已经接近纯转向。必须分别审计模型输出与底盘实际跟随，不能把全部转向归于 tracker。
- 第一非零 waypoint 选择逻辑与官方固定取首行存在代码差异，但本轮66次均选择第0行，所以该差异不能解释本轮打转。

### 待验证的复现问题

1. **历史生命周期。** `LightNavSkill.start()` 每次工具调用都执行 `client.reset()`；同一导航目标每40步预算到期重试也清空历史。官方协议允许在同一 session 中更新文字指令，reset用于 episode/目标边界。清空旧动作队列与清空观测历史是两件事。两者是否必须一起清空，原文没有给出 LightNav 的规定，需独立配对验证。
2. **场景指令不稳定。** 同一罐子的九条 GPT 指令交替写 dining table、kitchen countertop、hallway floor，并保留内部对象 ID。它们在文字上不一致；是否指向错误实例仍需对照输入图像。`instruction`被原样传递不等于满足论文的 scene-grounded 语义要求。
3. **学习反馈尚未迁移。** 当前没有导航学习进度，也没有基于其停滞/回退的事件触发；40步调用上限是我们的工程设置。`stop/visible`不是论文的调用局部连续进度。
4. **作者实现边界。** 作者 LIBERO evaluator 的问题重规划分支还要求 `idx > 0`，且进度阈值仅定义操作 primitive。不能不加说明地把它套到首个 navigate 调用或照搬 grasp/release 阈值。
5. **控制与视角。** 继续核对轴向、相机外参、速度饱和、实际偏航、原生终点朝向。官方文档说 waypoint 不自带时间基准；记录中的 `waypoint_dt_s=0.1`是显示设置，不能据此断言我们的0.25秒执行窗口有单位错误。

9次调用中近原地转向步数依次为40、15、35、30、35、40、35、25、5。预测首行选择、指令、位姿和实际速度均可从上述JSONL重算。运行时导航源码SHA256 `ae8ef6c25308d7c42beb6163214e067371b55f8374fe38b7ee0462bce27b1a97`，与仓库 `5f74883:src/bvi/lightnav_skill.py`及当前文件规范化LF后的字节一致。

## 原文训练与本项目的关系

[VLAs-as-Tools](https://arxiv.org/html/2605.13119v1) §4、§5.1、附录A/B描述调用窗口、工具族残差路由、动作与进度联合监督，以及基于调用局部成功的GRPO。论文区分DROID-split中间训练和目标域适配；SFT与RL分别是实验设置，不能把所有表格都理解成同一条必须叠加SFT和RL的配方。完整IL分支也不等于必须加RL。

附录B要求后续工具从前序工具实际产生的边界状态训练，并讨论更新策略后刷新起点池。对本项目尤其重要的是导航到达姿态、角速度、手臂状态与抓取训练起点的匹配。该机制提供可检验方向，不是本轮已确认的因果解释。

LightNav自身论文包含ER中间训练、SFT/DAgger和在线RL；这与我们有没有给它做MS-HAB工具对齐训练是不同问题。DROID是操作数据，不能直接用于LightNav导航动作SFT；将TAPT扩展到它需要导航调用数据、明确的动作token监督和进度标签。这属于异构骨干上的方法迁移，原论文没有LightNav或MS-HAB实验。

作者OpenPI fork含有可复用实现，但不是已经核实完备的论文发布包。固定版本 `f4eb160ba52b22c1e85fe432de59c24bbbac6187` 的训练脚本使用本地数据路径、8设备默认设置；脚本batch/步数与论文摘要配置存在差异，需要在训练锁定清单中明确选择依据。完整split、训练权重、RL配置尚未核实可用。不得直接把脚本默认值当作完整论文复现配置。

## MS-HAB 操作替代候选

| 候选 | 一手来源确认 | 可用性与适配判断 |
|---|---|---|
| SG-VLA | MS-HAB专用多视角RGB-D、13维移动操作VLA，含空间辅助监督 | 最相关的研究候选。作者仓库/项目页仍写Code coming soon，模型权重未发布。公开ZIP已检查为22个LaTeX/图形文件，没有Python实现。不能当即接入。13维数量一致也不代表通道/控制语义一致。 |
| AC-DiT | 官方仓库有MS-HAB数据生成、训练和评估代码；移动操作扩散策略 | 当前更可审计的工程替代。README链接的mshab_checkpoints是官方教师，不是已核实AC-DiT训练权重；本次未找到可直接推理的AC-DiT权重发布。点云/语言嵌入/控制接口需适配。其协调机制属于另一种方法，应单列实验，不能混入TAPT复现。 |
| AnchorVLA | 作者论文声明MS-HAB移动操作与GitHub链接 | 本次该链接及GitHub API均无法取得公开仓库，API返回404；这不证明永远没有代码，只表示当前不能认定可部署。 |
| MobileWAM | 作者论文报告MS-HAB实验，使用世界动作模型 | 原文仍称代码待发布；本次未核实可用实现和权重。 |
| MS-HAB SAC/DP/BC | 官方代码与检查点 | 可用于正对照与示范，不能标作语言条件VLA。 |

SG-VLA的73%是其报告的操作设置平均值，不能与我们的全链单episode直接比较，也不是其Pick成功率。替换为SG-VLA辅助损失或AC-DiT协调头属于更换方法，不是修复VLAs-as-Tools遗漏。

## 建议执行顺序（尚未启动付费工作）

1. 固定当前baseline的代码、权重、相机和控制配置；保留历史失败。先完成LightNav session/指令/控制与Fetch state/action数值接口审计。归因对照每次只改一个被审计变量，并明示是复现修正还是方法扩展。
2. 写出逐项方法锁定表：原文条款、作者代码位置、本地实现、训练数据与checkpoint证据、缺失材料。保留原生success；把普通GPT包装、完整TAPT SFT与后续RL条件分开。
3. 先补齐Fetch的调用对齐SFT：按完整轨迹划分数据，覆盖真实LightNav交接状态；再分段、训练工具族残差与进度头，保留动作学习目标。对失败/停滞状态验证进度行为，不能只用成功段时间标签证明可检测失败。
4. 若坚持完整原文IL流程，补回或取得可核验的DROID-split中间阶段，再做目标域IA-SFT；缺少该阶段只能报告删减版。DROID到Fetch必须分别保留对应本体的状态/动作变换和归一化。
5. 原始能力具备后，单列原文式GRPO迁移：调用级起点池、局部二值完成谓词、固定时限、组内采样、后续刷新边界池。若各组均零成功，没有有效奖励差异，应停止诊断数据/起点，不能默认加长训练可解决。具体π₀.₅ likelihood与RLinf实现必须锁定，不能用SAC训练替代GRPO并保持同名。
6. LightNav先验证上述忠实性问题，再评估导航TAPT迁移可行性；其独立骨干、动作token和进度训练独立记录。共享的是调用/反馈协议，不是π₀.₅权重。
7. 依次报告单调用、导航→抓取、完整首物体链和固定多初始状态评估；同场景训练材料不能再冒充未见验证成绩。

完整数据和原文训练规模不能从现有单A6000短时LoRA结果外推成本；需另做数据量、吞吐、显存和剩余额度核算。本次新增实验GPU/API消费均为0，不代表原有存储持续费用为0。

## 2026-09-16 后续核验

用户已选择AC-DiT操作主线。随后找到第三方 `JJho1314/AC-DiT-MSHab-Reproduction` 的mobility/main两组公开权重，并下载校验；这更新了上文“本次未找到”的检索范围，仍不等于作者官方发布或本项目已成功推理。来源、哈希、通道、真值权限、实验室服务器路径和训练验收门见 [AC-DiT主线记录](acdit-mshab-mainline.md)。历史π₀.₅结果与基线机制保留。

## 一手来源

- [VLAs-as-Tools正文与附录](https://arxiv.org/html/2605.13119v1)
- [作者OpenPI fork](https://github.com/cxliu0314/openpi/tree/f4eb160ba52b22c1e85fe432de59c24bbbac6187)
- [LightNav原文](https://arxiv.org/html/2608.30935v1)、[部署约定](https://github.com/lightorigins/LightNav-0/blob/main/docs/DEPLOYMENT.md)、[协议](https://github.com/lightorigins/LightNav-0/blob/main/docs/PROTOCOL.md)
- [SG-VLA项目](https://trs07170.github.io/SG-VLA/)、[作者仓库](https://github.com/TRS07170/SG-VLA)
- [AC-DiT仓库](https://github.com/PKU-HMI-Lab/AC-DiT)
- [AnchorVLA论文](https://arxiv.org/abs/2604.01567)、[论文声明的仓库](https://github.com/jason-lim26/AnchorVLA)
- [MobileWAM论文](https://arxiv.org/abs/2608.04657)
- [MS-HAB官方仓库](https://github.com/arth-shukla/mshab)
