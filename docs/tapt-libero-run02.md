# TAPT008：协调器修复与语言适配试验

状态：代码修复已通过 116 项本地测试；真实训练已超过 20 次更新，四个工具族更新审计通过。step-0020 验证联合损失 0.0022359，高于本轮父权重 0.0021415，尚无改善结论；尚未在线评估。

## 独立变化

- 协议：默认 `schema-characters`，修复 Unicode 字符/字节不一致；历史 run01 必须显式指定 `--instruction-limit-mode legacy-bytes`。
- 可选协调器 `--coordinator-mode memory-recovery`：每对象保存各工具族最近反馈，所有完成状态仍标为未知；低进度 reach 超时后请求 grasp 会被拒绝并反馈给 GPT，不替换其指令或执行隐式技能。拒绝仍占请求预算。
- 训练：从 run01 的 step-1800 权重初始化四组 LoRA 和进度头，重新初始化优化器，最多 400 次更新或 1800 秒。保留冻结骨干、原动作损失、0.1 进度损失、微批量 1/有效批量 8。每个工具族至多更新 100 次。

## 数据与选择

沿用原先按完整轨迹划分的 20 条训练、5 条验证；不增加演示数量，不使用评估轨迹训练。仅训练侧增加两个手工审核的同义指令，原指令仍以 1/3 概率采样；不加入左右位置、成功或失败等未经标注的条件。验证侧保持原指令，单独随机数流使语言采样不改变动作窗口采样。

进度标签仍为片段内时间代理。没有新增失败恢复数据，本次不是恢复能力训练，也不是严格按论文新增实验。保存父检查点哈希、优化器重置声明和语言变体清单。验证包括未更新的父权重（新批次 step-0000）；训练检查点按验证联合损失选择，禁止用任务成功率挑权重。

## 分离归因的评估顺序

1. 原 step-1800 + 字符修复 + legacy 协调器。
2. 原 step-1800 + 字符修复 + memory-recovery 协调器。
3. 新选权重 + 相同 memory-recovery 协调器。

固定同一任务、初始化、520 步/20 请求和相机配置。第 1→2 项用于分析高层修复，第 2→3 项分析本次训练变化。本次启动阶段 API 预算为 0；以上在线 GPT 对照尚未执行，不能报告改善。

## 命令

在已固定作者 fork 的 Python 环境中运行：

```bash
python train_libero_family.py --data /workspace/tapt/data \
  --checkpoint "$MODEL" --output /workspace/tapt/evidence/training \
  --warmstart-adapters /workspace/tapt/parent-adapters.pkl \
  --language-augmentation --steps 400 --seconds 1800
```

修复后的评估入口（另需模型服务、评价锁和已授权 API 桥接）：

```bash
python scripts/eval_libero_family.py --mode tapt --output "$NEW_OUTPUT" \
  --instruction-limit-mode schema-characters --coordinator-mode memory-recovery \
  --authorization-id "$AUTHORIZED_EVALUATION_ID"
```

资源：单 A6000 $0.53/h，60GB 临时盘约 $0.008333/h，无持久卷。内部 GPU/盘上限 $2、资源最长 3 小时，不新增充值。独立官方插件停机保护已设至 2026-09-15 22:10:24 UTC，正常路径在备份、校验后提前删除。

环境补充：本轮仅搭建训练环境，固定作者 fork 与官方基础权重。作者模型模块导入 pytest，补装 `pytest==8.3.5` 后启动成功；原始数据25文件及父权重SHA均验证通过。
