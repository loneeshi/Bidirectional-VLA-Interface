# 两日实验的复核与继续入口

本文件覆盖 2026-09-19—20 计划的受限诊断。当前结果以 [交付索引](results/two-day-delivery-2026-09-20/manifest.json) 为准；存在代码入口不代表相应 GPU 实验已执行。重跑会产生新实验，必须使用新的输出目录并重新登记资源上限。

## 固定身份

- 基线：实验室 `/home/pshuai/bvi-research/runs/s1-ia-epoch-2026-09-18-run01/best/855`，参数树 SHA256 `7854919f17de40ea8c62ee966327904a61503cfe2c0eb1715e3f30ae6e72892e`。
- 原始数据：`data/fetch-tapt-source/pick/013_apple.h5`，SHA256 `03b29e86035d968df346c851067a052a75bd5a0f29da7792daa05858173f896e`。
- 数据 manifest：`runs/s1-official-medium-2026-09-17-run01/manifest.json`，SHA256 `27e4769b8fbd0d610da773f900d50f7056eb29529f8006a3eb530aaf73a29b2e`。
- normalizer：`runs/s1-official-medium-norm-2026-09-17-run01`；`norm_stats.json` SHA256 `0f6264129ebbe89f8adee0c0caa2f204314674e5a95c882e1710413ee734e8d2`。
- OpenPI：干净作者版本 `f4eb160ba52b22c1e85fe432de59c24bbbac6187`。本地与实验室项目 checkout 不相同，因此同时保存 [source/runtime lock](results/two-day-delivery-2026-09-20/source-runtime-lock.json)、源快照和逐文件哈希。
- 仅 GPU1 UUID `GPU-b7ebba23-7824-7601-df32-be55628936c3`；不操作 GPU0。实验室 Python 为 `envs/openpi/bin/python`。凭证不属于实验包。

上述相对路径均相对于实验室 `/home/pshuai/bvi-research`。完整权重的本地第二副本、大小及字节校验见 [备份回执](results/two-day-delivery-2026-09-20/asset-backup.json)。字节一致与框架加载校验分别记录。

## 只读复核诊断

下面在一个新的 CPU 进程中重新计算配对指标、验证冻结合同并恢复保存的参数树；不运行前向、训练或仿真。原 decision 文件不可覆盖，复核另起文件名。

本轮已将仓库 `scripts/verify_s1_bounded_diagnostic.py` 上传至实验室根目录并核对源哈希；重新部署时需要先复制该文件到下面命令引用的位置，或把命令中的脚本路径换成同版本仓库路径。无需复制凭证。

```bash
cd /home/pshuai/bvi-research
export PYTHONPATH="$PWD/src/Bidirectional-VLA-Interface/src:$PWD:$PWD/src/openpi-author/src"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu envs/openpi/bin/python verify_s1_bounded_diagnostic.py \
  --run-dir runs/s1-bounded-diagnostic-2026-09-19-run01 \
  --contract runs/two-day-s1-input-audit-2026-09-19-run01/execution-contract.json \
  --verify-checkpoint \
  --output runs/s1-bounded-diagnostic-2026-09-19-run01/decision-independent-review.json
```

入口：[`verify_s1_bounded_diagnostic.py`](../scripts/verify_s1_bounded_diagnostic.py)。只完成指标比较而未恢复完整 checkpoint，不足以放行候选。基础设施失败、证据缺失和真正未达到指标门分别编码；不能把未运行写成策略失败。

诊断使用固定两条成功训练父轨迹的 85 行，同一行、同一 RNG、同一动作 mask 做 before/after。每个预测只比较 active11 非 head 通道；保存完整 chunk 和执行 clip 后结果。正式门为首动作与有效 chunk RMSE 都至少改善 20%，且 reach yaw/torso 首动作 RMSE 均不恶化超过 10%。这是训练映射改善证据，不是泛化或原生 Pick 成功。

## 最终候选的独立CPU复核

候选实际完成500更新、6508.4385秒，最终best/250。step500综合score更低但reach yaw恶化20.66%，违反10%守门，未获选。已完成的新CPU进程核验耗时135.65秒，顺序完整恢复两棵参数树，51个非LoRA叶字节/shape/dtype一致、20个LoRA叶改变，零前向；[裁决](results/two-day-delivery-2026-09-20/candidate/candidate-decision.json)为`verified_candidate_ready`。这只放行冻结原生回归，不是原生成功。

以下为独立复核的完整CPU命令；采用新输出名，保留原裁决。脚本及同目录的`verify_s1_bounded_diagnostic.py`需与[CPU验收记录](results/two-day-delivery-2026-09-20/candidate-verifier-cpu-preflight.json)一致。

```bash
cd /home/pshuai/bvi-research
export PYTHONPATH="$PWD/src/Bidirectional-VLA-Interface/src:$PWD:$PWD/src/Bidirectional-VLA-Interface/scripts:$PWD/src/openpi-author/src"
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
  envs/openpi/bin/python src/Bidirectional-VLA-Interface/scripts/verify_s1_bounded_candidate.py \
  --run-dir runs/s1-bounded-candidate-2026-09-19-run01 \
  --source-checkpoint runs/s1-ia-epoch-2026-09-18-run01/best/855 \
  --source data/fetch-tapt-source/pick/013_apple.h5 \
  --dataset-manifest runs/s1-official-medium-2026-09-17-run01/manifest.json \
  --normalizer runs/s1-official-medium-norm-2026-09-17-run01 \
  --audit-panel runs/two-day-s1-input-audit-2026-09-19-run01/audit-panel.json \
  --verify-checkpoint \
  --output runs/s1-bounded-candidate-2026-09-19-run01/candidate-decision-independent-review.json
```

## 原生oracle面板命令与执行状态

基线r1在0.867秒内因解释器symlink解引用到缺NumPy的基础Python退出；r2在6.887秒内因normalizer传入文件而服务要求目录退出。两次均零推理、五例未运行，分别保留[首轮故障备份](results/two-day-delivery-2026-09-20/oracle-baseline-r1-infrastructure-backup.json)和[第二轮故障备份](results/two-day-delivery-2026-09-20/oracle-baseline-r2-infrastructure-backup.json)，不计策略失败。修复保留venv可执行文件绝对路径但不解引用，normalizer按既有服务合同验证目录及provenance；实验室37项CPU测试通过。

基线r3于2026-09-19 04:03:01 UTC启动，监督PID1614344，沿用首次启动的04:25:08 UTC绝对截止时间，启动剩余1326秒；最终5/5完成、0成功、5策略失败，151次推理、254.0845秒。候选r1于04:08:32 UTC启动，监督PID1618354，最终5/5完成、0成功、5策略失败，163次推理、266.8217秒。有效两组均无基础设施失败或未运行例。全部10例在reach阶段触发累计力限制；[G1裁决](results/two-day-delivery-2026-09-20/g1-decision.json)未通过，不运行复跑、导航交接或GPT/Place链。

以下为[归档基线argv](results/two-day-delivery-2026-09-20/oracle-launch/s1-oracle-base-20260919-r3-launch.json)的历史执行命令；已有输出目录不得重跑覆盖。外层timeout限制剩余预算，内部上限仍为2100秒。

```bash
cd /home/pshuai/bvi-research/src/Bidirectional-VLA-Interface
timeout \
  --signal=TERM \
  --kill-after=15 \
  1326 \
  flock \
  -n \
  /home/pshuai/bvi-research/.gpu1-s1-ia.lock \
  /home/pshuai/bvi-research/envs/openpi/bin/python \
  /home/pshuai/bvi-research/src/Bidirectional-VLA-Interface/scripts/run_s1_oracle_panel.py \
  --checkpoint \
  /home/pshuai/bvi-research/runs/s1-ia-epoch-2026-09-18-run01/best/855 \
  --normalizer \
  /home/pshuai/bvi-research/runs/s1-official-medium-norm-2026-09-17-run01 \
  --output \
  /home/pshuai/bvi-research/runs/s1-oracle-base-20260919-r3 \
  --model-python \
  /home/pshuai/bvi-research/envs/openpi/bin/python \
  --sim-python \
  /home/pshuai/bvi-research/envs/acdit/bin/python \
  --reference-panel \
  /home/pshuai/bvi-research/runs/s1-native-render-aligned-2026-09-18-run01 \
  --protocol-manifest \
  /home/pshuai/bvi-research/src/Bidirectional-VLA-Interface/docs/results/two-day-delivery-2026-09-20/oracle-baseline-frozen-v4.json \
  --variant \
  baseline \
  --execute
```

基线v4 SHA256为`0fedf722d338b47e5485981dffffed9142436b0f5ec696937cfd5186722d68fe`；候选v3为`cde0dc5930ab66b9e78f174e0abe7031c950a9c665c80856d9a2fc509563fe47`。旧冻结文件保留，不替换其历史身份。normalizer参数必须是上述**目录**，不能是checkpoint assets中的`norm_stats.json`文件。

以下为[归档候选argv](results/two-day-delivery-2026-09-20/oracle-launch/s1-oracle-cand-20260919-r1-launch.json)的历史执行命令。实际使用flock与内部2100秒监督，没有额外外层timeout。执行环境不得继承`JAX_PLATFORMS=cpu`或`JAX_PLATFORM_NAME=cpu`；以下argv不包含CPU限定前缀。

```bash
cd /home/pshuai/bvi-research/src/Bidirectional-VLA-Interface
flock \
  -n \
  /home/pshuai/bvi-research/.gpu1-s1-ia.lock \
  /home/pshuai/bvi-research/envs/openpi/bin/python \
  /home/pshuai/bvi-research/src/Bidirectional-VLA-Interface/scripts/run_s1_oracle_panel.py \
  --checkpoint \
  /home/pshuai/bvi-research/runs/s1-bounded-candidate-2026-09-19-run01/best/250 \
  --normalizer \
  /home/pshuai/bvi-research/runs/s1-official-medium-norm-2026-09-17-run01 \
  --output \
  /home/pshuai/bvi-research/runs/s1-oracle-cand-20260919-r1 \
  --model-python \
  /home/pshuai/bvi-research/envs/openpi/bin/python \
  --sim-python \
  /home/pshuai/bvi-research/envs/acdit/bin/python \
  --reference-panel \
  /home/pshuai/bvi-research/runs/s1-native-render-aligned-2026-09-18-run01 \
  --protocol-manifest \
  /home/pshuai/bvi-research/src/Bidirectional-VLA-Interface/docs/results/two-day-delivery-2026-09-20/oracle-candidate-frozen-v3.json \
  --variant \
  candidate \
  --best-json \
  /home/pshuai/bvi-research/runs/s1-bounded-candidate-2026-09-19-run01/best.json \
  --execute
```

只做CPU preflight时，可对上述脚本及参数去掉`--execute`，并使用`CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`前缀。**不能在这个CPU命令上直接加回`--execute`当作GPU执行命令**：worker会继承JAX CPU限制，必须将CPU preflight与历史GPU执行命令分开。这里的执行命令是已完成实验记录，不授权新一轮运行；重跑需另选输出目录并重新登记预算。

## 固定协议与后续门

[`train_s1_bounded_candidate.py`](../scripts/train_s1_bounded_candidate.py) 的 `--preflight` 只读数据。实际 worker 另查诊断 decision、真实 CPU 恢复证明、证据哈希和原 best855 身份。最多 500 更新或 9000 秒，采用新 optimizer、既有共享 LoRA 和冻结采样；dev8 选择 checkpoint，不能使用在线成功率选点。具体采样与开发规则见 [候选预注册](results/two-day-delivery-2026-09-20/candidate-preregistration.json)。

[`run_s1_oracle_panel.py`](../scripts/run_s1_oracle_panel.py) 默认仅 preflight；显式 `--execute` 才会启动独立监督的模型服务和五个 fresh 仿真进程。它要求已冻结的源文件、模型、normalizer 和原生初态哈希；候选须带真实开发集选点证据。原模型与候选都使用同一 `ia_oracle_train_windows_v1` 协议和种子 2024—2028，每例最多 200 动作、300 秒，全批最多 2100 秒。

协议使用仿真距离/抓持谓词切换 reach/grasp/move 指令，所有低层动作由 π₀.₅ 生成。每次只执行预测 chunk 第 0 项，余下 9 项丢弃；切换不重置模型 RNG。原生成功和失败条件不变。它称为 oracle-assisted 调用串接诊断，不能改名 GPT 自主链，也不能和历史固定整句结果直接计算提升。

`eval_native_s1_oracle.py` 只允许记录的 `initial-state.pt` 配对，拒绝已有抓持或终止状态。单例成功仍须在新进程、同配置复跑后才可能通过 G1；复跑不加进原五种子分子。未通过 G1 不启动 LightNav 交接，API 额度在本轮冻结为 0，Place 无能力证据时不接入。

## 证据保护

输入审计原始归档在 Windows `D:\AI\embodied intelligence\runs\two-day-s1-input-audit-2026-09-19-run01.tar.gz`。训练前源码快照在仓库 `.runtime/two-day-delivery-20260919/source-snapshot.zip`；该时间点后的修改须通过交付快照另存。候选完整6.634GB第二副本已保存并核验，见[候选备份回执](results/two-day-delivery-2026-09-20/candidate-full-backup.json)。完整权重保存在 Windows `D:\AI\embodied intelligence\runs\two-day-assets-2026-09-19\`，不放进 Git。

原始 JSON、NPZ、训练记录及保存失败的证据均保留。结果目录新增索引时不改变原始文件；任何更正单独记录。复核时先检查 source/runtime 身份、artifact manifest 和备份回执，再解释指标。
