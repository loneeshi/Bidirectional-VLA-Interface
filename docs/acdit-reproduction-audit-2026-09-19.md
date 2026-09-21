# AC-DiT 33.3% 复现核查（2026-09-19）

当前目标：按作者 AC-DiT 两阶段、七任务训练配方，争取复现 Set-Table Apple Pick **33.3±1.9%**。该数字是单任务分数，七任务均值为55.6%。不是 π₀.₅＋SAC、单 Pick 特化、TAPT 或在线 RL。用户已授权一次、实验室GPU1最多18小时；未授权扩时、新租GPU或付费模型API。**GPU训练尚未启动。**

## 1. 是否有作者权重

2026-09-19实查：GitHub当前HEAD为`90ad00a926f34da04816ed9c3312aaf3bc845b7f`，与本地审计源码一致；Releases为空；完整文件树中唯一`.pt/.pth/.bin/.safetensors/.ckpt`是`data/empty_lang_embed.pt`，它不是策略权重。README与项目页没有给出作者训练后的AC-DiT主模型或mobility head下载。公开[权重请求issue #8](https://github.com/PKU-HMI-Lab/AC-DiT/issues/8)目前没有回复。

因此准确结论是：**在已核查官方渠道中没有找到作者发布的训练后权重**，而非断言任何地方绝对不存在。README下载的`arth-shukla/mshab_checkpoints`是MS-HAB的SAC/PPO/BC/DP模型；`gen_data.py`实际用其中SAC/PPO采集示范。RDT-170M、RDT-1B基础预训练权重另有公开发布，需要在其上完成AC-DiT训练。

证据：[官方仓库](https://github.com/PKU-HMI-Lab/AC-DiT)、[Releases](https://github.com/PKU-HMI-Lab/AC-DiT/releases)、[本次API回执](results/acdit-reproduction-audit-2026-09-19/official-release-check.json)。

## 2. 旧0/5的正确解释

旧权重并非完全来源不明：下载源是`JJho1314/AC-DiT-MSHab-Reproduction`，固定revision `f57e782c6a152c5ada83a33d5c29273c857003fd`，两个文件的SHA256已核验并记录在[历史来源锁](../configs/acdit_mshab_reference.json)。它是第三方社区复现，**没有作者背书或论文checkpoint身份验证**。本次访问该仓库README固定revision返回404，不能据此抹去已有下载和哈希证据，也不能推断它当时未存在。

旧实验只证明“指定社区checkpoint，在本项目FP32、固定五起点、遇任意原生终止即停的协议下得到0/5”。不能据此认定作者AC-DiT无效，也不能单独归因于本项目管线。既有strict-load、观测映射和渲染核验仍各自有效。即使真实成功率为1/3且五例独立同分布，0/5概率也约13.2%；实际两组协议还不相同。

## 3. 可核查的训练配方

| 项目 | Mobility阶段 | AC-DiT主阶段 |
|---|---|---|
| 作者入口 | `scripts/finetune_mshab_base_only.py` | `scripts/finetune_mshab_acdit.py` |
| 初始化 | `robotics-diffusion-transformer/rdt-170m` | `robotics-diffusion-transformer/rdt-1b`＋本次训练的mobility checkpoint |
| 任务 | 同一七任务集合 | 同一七任务集合 |
| 预测目标 | 底盘两个通道 | 全身13通道，嵌入128维统一向量 |
| 更新上限 | 30,000 | 20,000 |
| 每卡batch | 24 | 16 |
| 梯度累积 | 1 | 1 |
| 原实验硬件 | 8×A800 | 8×A800 |
| 有效batch | 192 | 128 |
| 学习率 | 1e-4，constant | 1e-5，constant |
| 精度/分布式 | bf16，DeepSpeed ZeRO-2 | bf16，DeepSpeed ZeRO-2 |

共同设置：AdamW β=(0.9,0.999)、weight decay0.01、clip1；两帧视觉历史、head/wrist RGB＋空第三槽；SigLIP384；1024×6原生点云和LIFT3D；真实base velocity和关节状态；privileged18上下文；action chunk2，推理执行两步；image augmentation、state noise SNR40；不使用π₀.₅分位数归一化。

“全量”应指完整AC-DiT训练配方，而不是把所有预训练编码器全部解冻。主阶段冻结mobility DiT及预训练SigLIP部分，训练主DiT及作者指定适配模块；LIFT3D包含LoRA。CPU实查独立参数总数：stage1 591,595,138、可训练162,529,730；stage2 1,825,218,818、可训练1,245,288,642。它不是只更新一个小动作头。

源码：[mobility入口](https://github.com/PKU-HMI-Lab/AC-DiT/blob/90ad00a926f34da04816ed9c3312aaf3bc845b7f/scripts/finetune_mshab_base_only.py)、[主入口](https://github.com/PKU-HMI-Lab/AC-DiT/blob/90ad00a926f34da04816ed9c3312aaf3bc845b7f/scripts/finetune_mshab_acdit.py)。硬件与有效batch的重要性由[作者公开回复](https://github.com/PKU-HMI-Lab/AC-DiT/issues/6#issuecomment-4190626956)确认；同issue另有4×3090复现者报告等效batch仍得0，属于第三方观察，不能单凭该报告定位原因。

## 4. 开跑前必须解决的复现差异

1. **数据数量冲突。** 论文§4.1.1写每任务1000条成功示范，共7000条；公开`HDF5MSHABDataset`第71行硬编码100，七任务只采样前700条。不能把“硬盘有1000条”和“loader实际使用1000条”混为一谈。以论文规模为复现目标时必须显式改为1000并记录这项代码差异；未获作者解释前不能声称这是作者产生33.3%的精确配置。
2. **现有数据不兼容。** 已下载的官方Apple Pick/Place H5没有`base_linear_vel`、`base_angular_vel`、`pointcloud/xyzrgb`，还缺另外五任务。深度不等于AC-DiT的点云输入；不能以目标动作替代真实底盘速度，不能补零冒充原生观测。应使用作者fork、官方SAC/PPO与train task plans重采七任务，保留真实速度及原生裁剪/采样点云。重新采集不等于获取论文原始示范。
3. **训练量与单卡时间。** 原两阶段合计8,320,000个采样训练样本（含重复），其中主阶段2,560,000。microbatch1时维持原有效batch需累积192/128；单纯累积24/16会缩小有效batch八倍。18小时完成全部两阶段要求平均128.4训练样本/秒，尚不计采集、验证和保存；目前没有GPU实测吞吐支持此目标。不能先把步数和batch缩小再称完整复现。FP32是现有Turing卡适配，与作者bf16不同。
4. **评测终止不一致。** 作者`eval_mshab.py`第234–239行只有`terminated/truncated`且`info['success']`为真时才设置done；失败终止不立即停止。旧项目runner遇任何原生终止即停。作者代码同时随机抽取任务语言嵌入；旧runner固定第一条指令。应分别报告“作者代码协议”和“原生首次终止协议”，保留首个失败时刻，不能偷偷替换口径。论文只说明200步成功，未细述这个失败后继续行为。
5. **100×3口径。** 33.3±1.9%来自每任务100episodes×3次评估及均值/标准差；不是5例，也不自动等于3次重新训练。作者脚本env首次reset硬编码2024，三次运行的完整seed列表、论文选点方法未给出。必须预注册模型/环境种子、连续reset语义、每次分母和最终选点规则；不得按测试成功率挑checkpoint。
6. **其他代码差异需留痕。** EMA注释称未使用，但训练代码确实建立并更新EMA，采样使用非EMA模型；图像历史mask在t0将当前帧也置无效；作者采样诊断使用训练数据而非独立dev。上述项不能静默“修正”后仍宣称逐字复现。GPU单卡适配还需修复`.module`假设、硬编码设备、失败时无限重采等执行问题，并分别记为工程修正。

参考：[论文](https://arxiv.org/html/2507.01961v2)、[数据loader](https://github.com/PKU-HMI-Lab/AC-DiT/blob/90ad00a926f34da04816ed9c3312aaf3bc845b7f/data/hdf5_mshab_dataset.py)、[评测入口](https://github.com/PKU-HMI-Lab/AC-DiT/blob/90ad00a926f34da04816ed9c3312aaf3bc845b7f/scripts/eval_mshab.py)。

## 5. 执行顺序与当前状态

1. 固定源码、encoder、RDT初始化及SAC/PPO教师版本；补齐七任务观测完整的训练数据。
2. 完成所有任务的数据读取、动作/state映射、语言绑定、实际样本数、冻结范围和checkpoint恢复检查。
3. 冻结论文规模/公开代码规模差异表，按原有效batch核算真实训练与评测所需时长。单卡GPU测时须计入授权资源范围，不能擅自加卡、延长18小时或消耗另一轮训练次数。
4. 资源与配置一致后，只执行一次stage1→stage2；从原始RDT初始化，不将社区主权重作为论文初始化替代品。stage2用这一次的stage1输出。
5. 按预注册的100×3 Apple Pick协议比较33.3±1.9%，同时报告原生首次终止分数；若声称完整七任务复现，还必须给出全部七任务100×3分数。

已完成：原始RDT两权重2.789GB下载与SHA核验；两阶段CPU构造和Transformer权重加载；官方数据转换/增广/两阶段collator合计16批CPU检查。CPU构造审计对LIFT3D中无条件`.cuda()`的常量构造做了局部CPU重定向，不含模型forward，不能当GPU兼容性证明。旧官方H5的1000条/63场景审计和小BC诊断仅作为既有数据证据，不冒充新七任务数据。

未完成：观测完整的七任务训练集、单卡GPU训练吞吐/显存验证、与18小时预算一致的完整训练配置、300例正式评测。单Pick采集、单Pick限时训练、五例评测的准备草案已经被用户全配方目标覆盖，禁止直接启动。`delivery-2026-09-20.md`和旧实验归档未修改。

费用：新增租机/付费模型API为USD0；实验室费用未知。历史停止Runpod存储仍持续计费，本次没有刷新供应商账单或账户余额。
