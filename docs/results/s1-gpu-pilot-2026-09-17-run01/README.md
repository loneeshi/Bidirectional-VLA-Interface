# S1 GPU pilot：100次更新完成

美国东部时间2026-09-17 12:47启动、12:52前完成。用户取消按日期等待限制，后续按依赖门推进。

- 官方pi05_base初始化，忠实native24/head-hand RGB，动作13，普通目标域LoRA SFT，非S2 TAPT。
- microbatch1/accumulation1，100更新，总266.54秒；稳定吞吐0.5503样本/秒，包含预处理/设备传输/训练/日志，排除前10热身。
- 依据0.5–1分支立即扩展150训练/50验证，目标1epoch。medium CPU转换PID1459928，900秒外层限制；长训未启动。
- 训练检查点在实验室 `/home/pshuai/bvi-research/runs/s1-gpu-pilot-2026-09-17-run01/checkpoints/pi05_fetch_native24_s1/official-pilot/99`，已保存，尚未全量下载到本机或作新进程恢复验证。
- GPU1末次15MiB/0%，API0/新租机0，lab费未知；历史云存储继续计费。

此结果通过训练执行/吞吐门，不证明抓取能力。尚无10起点成绩，S2更新仍禁止。step日志中的进度字段为禁用进度损失时的接口占位，不是学习进度预测。
