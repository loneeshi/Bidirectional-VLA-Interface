# 实验日志

`docs/results/` 和 `runs/` 里的 `panel-status.json` / `events.jsonl` 是给程序读的：完整、可校验，但看不出发生了什么。这个目录放对应的人读版本——每个面板一篇，讲清楚跑了什么、结果是多少、哪些数字不能直接引用。

规则：日志只归纳已归档的证据，不产生新数字；每条结论都要能指回 `results/` 或 `runs/` 里的具体文件；证据还没落盘的数字明确标为待核对。

| 日志 | 覆盖 |
|---|---|
| [2026-09-21 TidyHouse 16-plan 三设置对照](2026-09-21-tidyhouse-16plan-panel.md) | Fixed PPO+SAC、GPT+PPO+SAC、Teleport+SAC 在同一批 16 个 TidyHouse validation plan 上的配对结果 |
