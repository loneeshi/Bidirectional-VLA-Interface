# 单任务 history 与每例媒体交付：CPU 回执

2026-09-30 UTC。所在阶段：C2 高清视觉接口准备。

推进：guidance-v3 每轮加入模型自述 task_state、完整历史索引与上一动作前后对照，保留样本独立响应链、四路原尺寸2048图和旧轮上下文。v13 控制参数不变；没有力或进展不足事件中断，也没有跨尝试记忆。

新视觉 Astra/API0 runner 逐步保存在线帧；批处理在GPU子进程退出后调用CPU交付脚本，生成完整历史HTML、同步末端轨迹demo、轨迹HTML/SVG、哈希与README。成功、失败和删失全部进入交付链；缺失或编码失败明确报告并阻止继续批次，不自动重放。收回结果后，用 finalize_eef_visual_attempt.py --attempt <目录> --register-media docs/media 注册准确路径及哈希。

验证：[cpu-tests.json](cpu-tests.json)记录 **450 passed**。包括原395项、视觉41项、已有可视化6项及新history/media8项。新增测试覆盖25轮历史请求与预算、坐标跨界、禁止字段注入、模型笔记校验、逐帧记录、API删失、素材缺失和合成帧实际编码/媒体索引。合成帧仅为CPU测试夹具，临时视频不作为实验demo。语法检查及18份源码/依赖文件哈希见 [source-sha256.json](source-sha256.json)。本地补齐 imageio 2.38.0，imageio-ffmpeg 0.6.0 已有。

阻塞：实际GPU录制、渲染耗时和现场G2未运行；尚不能宣称物理接口通过。新源码不同于旧部署包，旧提案已标明需重新冻结。预算见 [budget.json](budget.json)：沿用原估算费率和计费公式，三例78请求上限估算USD1000.8816，未授权；最大规划上下文1,014,624，小于1,050,000。超过文本或上下文门时明确停止，不删除历史或降分辨率。

下一道门：重新冻结部署包及依赖，运行前核对实验室与finance，提交新接口授权范围。此轮 **GPU 0、API 0、仿真0、无新增研究服务消费**。没有新的严格Pick成功，研究任务整体未完成。未发布GitHub。改动在工作区，其他已有未提交工作保留。

设计与收集命令：[单任务history与媒体修订](../../docs/real-handoff-eef-single-attempt-history-and-media.md)。每次真实实验最终必须返回history、demo和trajectory三个准确链接；无运动或缺素材须明确说明。
