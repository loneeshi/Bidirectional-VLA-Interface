# TAPT009：修复后的固定在线对照

状态：20个episode已完成并归档，临时GPU已删除，所有API费用已逐调用估算。

|组名|底层权重|协调器|
|---|---|---|
|standard-character-fix|官方 pi05_libero|原协调器，规则反馈|
|tapt-old-character-fix|TAPT007 step1800|原协调器，学习进度|
|tapt-old-memory-recovery|TAPT007 step1800|对象记忆与恢复约束，学习进度|
|tapt-new-memory-recovery|TAPT008 step200|对象记忆与恢复约束，学习进度|

所有组使用字符数校验修复、同一任务的前5个官方初始化、环境seed7、20等待步、520动作步预算及每episode20次GPT请求。每组重新启动模型服务以恢复相同的模型随机种子。GPT使用已有gpt-5.6-luna配置，图像与请求逐次存档并记账。

前两组重测原来发生协议中断的配置；第二、三组比较高层修复组合；第三、四组比较语言适配训练。协调器修复是工程扩展，不能把其收益全部归为论文TAPT。父/新权重均在本轮评估前选定，不依成功率换检查点。

内部上限：GPU及60GB临时盘$2，API$1，最多400请求，单GPU资源最多3小时。独立停机保护截止2026-09-15 23:10:50 UTC。无新增持久卷，不新增充值。

本批录像归档目录预定为 `docs/media/tapt-libero-2026-09-15-run03/<组名>/`；每组保留全部原始episode及失败标记。评估成功与否以LIBERO原生判定为准。

## 运行中断与恢复

第三组episode002在190步发生SSH响应发布失败：API已正常返回，本地缓存与远端response.upload.json的SHA一致，但未原子改名为response.json。此episode保留为基础设施失败，不覆盖、不重发API。修复仅为缓存响应传输增加最多3次重试，118项本地测试通过。恢复时按完整learned_progress记录数推进模型随机数状态，继续episode003/004和第四组；环境reset序列同步推进。最终报告保留这一偏离。

## 最终结果

|配置|原生成功|
|---|---:|
|standard-character-fix|5/5|
|tapt-old-character-fix|5/5|
|tapt-old-memory-recovery|4/5|
|tapt-new-memory-recovery|5/5|

字符/字节校验中断在本轮未复发。工具族学习进度闭环已取得原生任务成功；第三组保留1次SSH基础设施失败，不从分母删除。GPT输出有随机性，且第三组发生过服务恢复，因此不能把前后差异全部归因于某一个修复；新权重未展示成功率优势。固定单任务/5初始化的小样本不代表完整LIBERO或论文成绩，也不证明MS-HAB适配。

[全部原始录像及文件名](media/tapt-libero-2026-09-15-run03/README.md) · [逐episode日志与汇总](results/tapt-libero-2026-09-15-run03/evaluation.json)。本批GPU/盘估算$0.6292，199次API估算$0.043884；已完成备份校验和资源删除。

核验：四组初始状态哈希逐episode一致，20段原始录制的帧数等于执行动作步数。新权重组学习阈值29次、学习回退2次，共31次学习事件参与闭环；旧权重仅校验修复组共29次学习事件。证据见[verification.json](results/tapt-libero-2026-09-15-run03/verification.json)。
