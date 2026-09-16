# AC-DiT 操作主线：接口审计与迁移门

2026-09-16 UTC。新主线为 GPT coordinator + LightNav-0 导航 + AC-DiT 操作。保留历史 π₀.₅ 配置、检查点与失败结果。**用户已提供实验室服务器这一计算路径，不再新租或恢复付费 Runpod GPU。** SSH信息已配置在本地忽略文件，域名可解析，但22端口超时，尚未认证；服务器GPU、磁盘和调度规则仍未核实，需先接通获授权的校园网络/VPN或管理员指定访问路径。

当前完成：公开代码审计、两个社区检查点下载及哈希验证、张量元数据检查、接口检查代码、本地测试。**尚未完成实际模型严格加载或仿真成功，也未开始 Fetch TAPT、DROID 中间训练或 GRPO。**

## 来源与检查点

作者代码固定：[PKU-HMI-Lab/AC-DiT@90ad00a926f34da04816ed9c3312aaf3bc845b7f](https://github.com/PKU-HMI-Lab/AC-DiT/tree/90ad00a926f34da04816ed9c3312aaf3bc845b7f)。13个关键文件哈希见 [source lock](../configs/acdit_source_lock.json)。Windows 全量 checkout 因 vendored 文档中含冒号的文件名失败；目前本地是源码审计快照，完整安装应在实验室 Linux 服务器进行。

权重来自 [JJho1314/AC-DiT-MSHab-Reproduction](https://huggingface.co/JJho1314/AC-DiT-MSHab-Reproduction/tree/f57e782c6a152c5ada83a33d5c29273c857003fd)，revision `f57e782c6a152c5ada83a33d5c29273c857003fd`。这是**第三方社区复现，不是作者官方权重**。社区报告的成绩不作为本项目成绩。

| 文件 | 实际字节数 | SHA256已验证 |
|---|---:|---|
| `stage1_mobility_head/checkpoint-30000.pt` | 2,042,274,179 | `da3a1d036f2ff6f73c57ec1d2b46cb9fa6de7a2c3cd607ec705cc611160dc94c` |
| `stage2_all7_baseline/checkpoint-25000.pt` | 4,811,606,217 | `9020d7848b31db90a34756a6333215c8e181ea2f74a86b5dc3074c2a4d7e25d7` |

受限元数据解析分别得到1,621和2,264项tensor。主策略输出层形状128×2048，语言入口2048×1152，状态及mask入口2048×256，真值上下文入口2048×18；mobility对应hidden size为1024。这与源码配置的主要维度相符，**不等于全部state_dict严格匹配**。两份文件各有324项LIFT3D LoRA tensor，没有在其之外发现LoRA，也未发现以tool_family/progress命名的tensor；不能把LIFT3D视觉LoRA误称为TAPT工具族残差。

运行前仍需严格加载两个检查点、锁定SigLIP/LIFT3D等依赖，禁止缺权重时随机初始化后生成“接入成功”结果。机器可读证据见 [checkpoint-audit.json](results/acdit-interface-2026-09-16/checkpoint-audit.json)。

## 接口忠实性检查

| 项目 | 作者实现与本项目边界 |
|---|---|
| qpos | 上游观测去掉前三个根关节，evaluator索引针对12维观测；机器人原始qpos是15维。新代码按关节名称映射，不混用索引。 |
| state | arm7、右指位置1、head pan/tilt/torso3、世界坐标vx/wz2。保留物理值；不用π₀.₅ quantile/z-score；不把单指位置翻倍成总开度。 |
| base state | 上游训练取世界坐标线速度x和角速度z；不能悄悄换成底盘前向速度。动作的底盘前向语义另外核验。 |
| action | 128维中取 `[0,1,2,3,4,5,6,10,125,126,127,100,102]`，每次2×13。动作已是归一化控制器命令；只在环境边界裁剪，并记录原始值和裁剪索引。不修改训练标签。 |
| 控制器 | `pd_joint_delta_pos`：arm/body delta、gripper absolute、base forward/yaw velocity。在线还须核验类型、顺序、物理范围和20Hz频率。 |
| 图像 | 两帧，每帧head/hand/空第三相机槽；保留原pad/SigLIP处理。初始旧图像为空、旧点云复制当前点云。 |
| 点云 | 1024×6世界坐标米单位XYZ和RGB[0,1]；沿用上游围绕base的裁剪和随机采样，不改成底盘坐标或FPS。 |
| 真值权限 | goal3＋grasp1＋object pose7＋TCP pose7共18维，显式标为privileged baseline；缺字段拒绝。不依据注释猜测重排四元数。 |
| 指令 | 原evaluator随机读预存语言嵌入。新调用边界保存原始UTF-8指令与哈希，传给编码器；真正SigLIP编码再进入实际 `MSHabModel.step` 尚待在线验收。 |

代码：[acdit_contract.py](../src/bvi/acdit_contract.py)。当前原生边界拒绝传入`tool_family`，因为尚没有训练工具族适配器；也不输出虚构学习进度。未来TAPT运行时需另行注册并验证。

控制权采用串行调用：LightNav导航时控制底盘并保持操作姿态；AC-DiT操作时取得**整个Fetch**控制权，保留其底盘—机械臂协调。不可让LightNav同时写底盘，或把AC-DiT底盘输出清零后仍称为原生策略。

导航沿用 [前轮审计](vla-tools-mshab-fidelity-audit.md)：同目标重试时历史清空、同目标文字地点变化、未迁移导航学习进度仍是待验证问题。本轮没有修改导航机制。后续固定起点、目标、模型采样与仿真种子，逐变量对照历史生命周期和指令稳定性，记录模型waypoint和实际速度。清空动作队列不自动意味着清空观测历史；不加随机脱困或SAC接管冒充复现修正。上轮66次均选第0行waypoint，选首个非零行的差异不能解释该轮打转。

## 按用户顺序推进

1. **先过接口与原生能力门。** 实验室服务器只读体检→完整源码/依赖锁定→双检查点严格加载→`set_table/pick/013_apple`单episode，200步，记录实际初始状态，遇任意原生终止停止。先核验该社区模型覆盖的任务，再测试LightNav交接与旧罐子任务，不能默认它们同分布。
2. **Fetch TAPT SFT。** 用真实LightNav交接起点收集示范；先按完整父轨迹/导航rollout划分train/val，再分reach/grasp/move/release调用片段。训练四套确定性路由残差和进度头，保留AC-DiT原生扩散动作目标；单独验证所选残差更新、队列清理、进度触发与任务结果。失败片段不能按录像/轨迹结束标完成。原生Pick还含抬起、回收、静止要求，不能直接充当每个primitive的局部完成谓词。
3. **完整IL另建有来源的分支。** 优先取得作者可核验DROID-split材料；缺失则记录重建、观测和本体差异。完整顺序必须是DROID中间训练→Fetch适配：第二步先跑目标域小规模试验，之后补回DROID时需要重新衔接Fetch训练，不能倒置两阶段后声称全流程。AC-DiT额外需要点云和18维上下文，不能直接复用π₀.₅的DROID权重/归一化。
4. **GRPO独立实验。** 固定调用级起点、局部完成谓词、奖励、时限、组采样和边界池刷新。先验证扩散策略可用的概率比/优化目标实现，不能拿SAC或别的骨干训练脚本换名。全失败且奖励无差异时停止扩大训练，先解决起点和基础控制能力。
5. **导航独立工具对齐。** LightNav使用独立导航数据、动作监督、适配器和进度标签；共享调用协议，不能用DROID机械臂动作标签直接训练导航。没有学习进度时明确反馈来源。

AC-DiT骨干上的TAPT是**方法迁移**，不是原文π₀.₅数值复现。AC-DiT自身mobility/main两阶段训练也不等于TAPT或GRPO。各阶段分别验收，不因换了骨干而省略失败结果。

## 本地命令与结果

在BVI仓库根目录运行，Windows使用`.venv/Scripts/python.exe`替换`python`：

```bash
python scripts/preflight_acdit.py --weights-dir ../../AC-DiT-checkpoints --report ../../runs/acdit-interface-2026-09-16/preflight.json --fetch-weights
python scripts/inspect_checkpoint_metadata.py ../../AC-DiT-checkpoints/stage1_mobility_head/checkpoint-30000.pt --output ../../runs/acdit-interface-2026-09-16/mobility-metadata.json
python scripts/inspect_checkpoint_metadata.py ../../AC-DiT-checkpoints/stage2_all7_baseline/checkpoint-25000.pt --output ../../runs/acdit-interface-2026-09-16/policy-metadata.json
python -m pip install pytest==8.3.5
python -m pytest -q tests
```

下载器不调用GPU/API；元数据解析器不执行任意pickle全局对象、不读取tensor存储、不分配GPU。`preflight.json`明确保留`runtime_verified=false`。[参考配置](../configs/acdit_mshab_reference.json)不是已接入主协调器的启动配置。

新增接口与元数据安全测试11项。全量检查曾因缺pytest失败，补依赖后146通过、1跳过；已有websocket兼容性弃用警告单列保留，不作为AC-DiT新故障。尚无新仿真演示。

## 费用与服务器切换

Runpod插件18:35UTC读数：累计USD14.514049619，USD20授权剩余上限USD5.485950381，未扣待出账费用，不是账户余额。本轮GPU/API新增消费0；当前无运行GPU，两个历史EXITED Pod的40GB存储约USD0.266667/日。MSHAB013供应商读数USD0.854164029已取代原估算并关联调整，不重复相加。

用户随后要求不再租服务器。后续使用实验室服务器，先确认GPU型号/显存、磁盘配额、共享或独占、VPN/跳板与Slurm规则。SSH凭证仅放Git忽略的本地配置，不能提交。旧Runpod存储在资料备份与清理完成前继续记账。
