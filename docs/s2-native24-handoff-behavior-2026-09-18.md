# B′ native24 wrong-handoff shadow 结果（2026-09-18）

离线行为子门 **failed**。五种组合在两条 grasp 负例序列均于第二次预测触发 `learned_threshold`，当帧物理完成谓词为 false；共 10 个组合×序列失败。C-M0/M1、closed loop 和追加训练继续冻结。完整行为准入仍为 `not_evaluable`。

## 输入与执行

validation parents 3020/3021 各双重精确重放，另执行 108 个 simulator actions，得到四条连续十帧 native24 窗口：3020 grasp 8–17、move 11–20；3021 grasp 22–31、move 25–34。窗口在已锁边界之前，分类仅描述末帧，不能传播到全部帧；它们不是实际 family invocation 后的十次部署调用。40 帧逐帧物理完成谓词均为 false。独立来源验证通过，manifest SHA-256 `b05e22296244c68bf85c6f84d7cc98fa06fd4ca756b30970d75f436e2dc1af50`。

冻结 monitor：grasp/release 0.6、reach/move 0.9、连续两击；drop 0.03；至少十次预测且增长小于 0.03 判 stagnation。模型仅 shadow 查询，使用相同逐帧 RNG/noise。run04 在 268.583 秒完成 200 次前向和 200 次 queue clear，optimizer、simulator、外部 API 均为 0。前三次启动因辅助模块路径和 Python 环境缺失而在模型前退出，证据均保留，前向次数均为 0。

## 裁决

| 组合 | near-grasp 前两次 progress | far-grasp 前两次 progress | grasp 首事件 | 两条 move 首事件 |
|---|---|---|---|---|
| H0/B0 | 0.645548, 0.655924 | 0.672936, 0.666904 | 第2次错误完成 | 第10次 stagnation |
| H20/B0 | 0.771435, 0.787088 | 0.740375, 0.734636 | 第2次错误完成 | 第10次 stagnation |
| H0/B20 | 0.644463, 0.658710 | 0.670614, 0.666773 | 第2次错误完成 | 第10次 stagnation |
| H20/B20 | 0.805034, 0.820110 | 0.776347, 0.770983 | 第2次错误完成 | 第10次 stagnation |
| H20/仅move B20 | 0.771435, 0.787088 | 0.740375, 0.734636 | 第2次错误完成 | 第10次 stagnation |

integrity、coverage、RNG identity、routing、queue clear 全通过；negative false completion 失败。独立 CPU 重算的全部 checks 与原汇总完全一致。首事件后 grasp 其余八次预测仅作诊断，不再作为 runtime monitor 输入。move 的 stagnation 只证明信号机制触发，不能证明物理停滞或回退正确。

A 已把 grasp/release action regression 定位到 B20 bank；B′ 另证明换回 B0 也不能修复当前 grasp 完成判定。H0 是诊断 step0 初始化的 progress head，不能当成 S1 具备已训练 progress 能力。H20 在这两条负例上提高 progress，但本实验不足以区分标签目标、初始化、训练支持度和阈值迁移各自的因果贡献。

静态代码显示 progress 训练目标是调用内线性时间比例 `index/(length-1)`（`scripts/train_native_s2.py`），而行为门用物理完成评价终止信号。两者并不等价；这构成下一项语义对账的明确线索，尚不能据此宣布唯一根因。不得根据这四条已看过的序列调阈值再宣布通过。

## 下一步

先做零模型、零仿真的 progress 合同审计：对齐训练标签、初始化、teacher-forced 与 sampled-action progress 路径、既有阈值来源及错族负例覆盖。基于独立 dev 数据冻结单因素修复与新的 held-out 验收集，再申请所需的有界计算。当前 selective-bank 候选未通过行为门，不进入 C；M2/D 和更多 optimizer updates 保持原授权边界。

## 证据与资源

- [运行结果](results/s2-native24-handoff-behavior-2026-09-18-evidence/s2-native24-handoff-behavior-2026-09-18-run04/result.json)、[汇总](results/s2-native24-handoff-behavior-2026-09-18-evidence/s2-native24-handoff-behavior-2026-09-18-run04/summary.json)、[独立重算](results/s2-native24-handoff-behavior-2026-09-18-evidence/s2-native24-handoff-behavior-2026-09-18-run04/independent-validation.json)。
- adjudication 加载前后 SHA 不变：`996992c6d3cc01f717ce2c6070e5dca783fbec5a739d036c73682edfeede11be`。
- 序列归档：`D:\AI\embodied intelligence\runs\s2-native24-handoff-sequence-2026-09-18-run01-evidence.tar.gz`，1,450,308 bytes，SHA `7ef21c1f23e618ac0591b31c9468cd852b64cb912eb80c49908cd37e9db7ad02`。
- 行为归档（含三次预检失败）：`D:\AI\embodied intelligence\runs\s2-native24-handoff-behavior-2026-09-18-run04-evidence.tar.gz`，500,129 bytes，SHA `ed989bbd53c325c3cac683c5e7365466c1a90194588c857f1cac1403c77228fe`。均已下载核验。
- 进程均退出，GPU1 收尾 15 MiB / 0%；GPU0 既有任务未触碰。新租机和外部 API 支出为 0，实验室费用未知；历史 Runpod 存储继续计费，本轮未刷新账单或余额。未新增 native success 或 TAPT 完成。
