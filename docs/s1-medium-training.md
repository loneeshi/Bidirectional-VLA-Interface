# S1：150父轨迹单epoch训练

按已测0.5503样本/秒选择medium分支。训练150条/6835帧，留出50条/2033帧，原生RGB与state24，训练集独立归一化。来自官方pi05_base的新普通LoRA SFT，不混入旧V8、不进行TAPT进度训练。

`train_native_s1_epoch.py`复用作者初始化、action loss和优化器；micro1/accum1，固定seed7 permutation遍历一次，共6835更新。每1000更新及最终保存latest和验证；每条留出轨迹固定起点/中点/末pre-action帧，共150样本，用action loss均值选择best。最终test成功率不参与选点。best/latest各保留一份，含优化器状态；resume依step恢复固定permutation和训练随机序列，累计墙钟不得重置。

实验室输出：`/home/pshuai/bvi-research/runs/s1-medium-epoch-2026-09-17-run01`；日志`s1-medium-epoch-run01.log`。outerPID1461011，内部18000秒/外层18600秒，单GPU1。正式进程已启动，GPU首步及checkpoint保存仍须看真实日志，不因执行器实现而宣称通过。训练完成后仍须十起点原生Pick≥3/10才能启动S2。

本轮API0/新租机0，lab服务费用未知，原Runpod停止存储持续计费。项目时间America/New_York，按依赖门推进无日历等待。
