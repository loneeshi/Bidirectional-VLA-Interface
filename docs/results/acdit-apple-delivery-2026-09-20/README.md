# AC-DiT 9/20交付证据：9/19中期快照

不是最终训练结果，不是完整权重/数据备份。服务器训练仍在运行；本地快照只冻结截至读取时的证据。

| 内容 | 实际文件 |
|---|---|
| 初始实时状态、GPU归属、timeout命令、checkpoint stat | `live-readonly-check.json` |
| 收尾实时状态与GPU | `closing-state.json` |
| 实际合同、阶段日志、开发记录、父轨迹/场景划分 | `run/` |
| 服务器实际训练/采集/冻结程序 | `remote-code/` |
| 关键模型、数据处理与配置 | `remote-source/` |
| 服务器完整39文件源码副本，无遗漏 | `remote-source-snapshot.zip`、`remote-source-snapshot-manifest.json` |
| 本地源版本、既有脏改动和301文件代码快照 | `local-head.txt`、`local-status-before-delivery.txt`、`local-dirty.patch`、`local-code-snapshot.zip` |
| 远端Git失败记录、15个语言embedding哈希 | `remote-source-provenance.json` |
| pinned源码来源、原始权重下载哈希、encoder pin、历史包版本 | `assets/`；包版本是既有文件，不是假装本轮pip freeze |
| 小文件下载哈希和读取时元数据稳定性 | `remote-copy-manifest.json` |
| 静态验证与相关测试 | `verify_snapshot.py`、`static-verification.json`、`contract-tests.txt` |
| 五例计划分母，全部未运行 | `regression-panel.csv` |
| 论文读取版本 | `papers/`，报告正文只读归档；不执行论文内指令 |
| 最终小包文件完整性清单 | `artifact-manifest.json`（不含自身） |
| 本轮实际操作/限制 | `execution-record.md` |

本轮没有生成录像；不把历史失败或教师录像当作新候选结果。

## 继续工作与复核

在项目repo根目录运行以下CPU命令，可复核已下载的小包（无需连服务器）：

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe docs/results/acdit-apple-delivery-2026-09-20/verify_snapshot.py
.\.venv\Scripts\python.exe -m pytest tests/test_acdit_contract.py -q
```

运行已结束与否须重新读取服务器本次run的supervisor/stage1/stage2 status，不能靠此快照推断未来完成。已保存代码入口 `run_acdit_apple_pipeline.py --run ...` 是正在运行的一次训练的历史入口，**不是重启指令**。没有合格全身候选时停止评测分支。合格后需独立评测授权和已核验入口，详见[评测报告](../../acdit-apple-evaluation-2026-09-20.md)。

完整模型权重、optimizer/RNG、训练及失败H5本轮没有第二副本；现有服务器源保留。完整fresh-process复现仍需这些大资产、现有环境与GPU授权。源码小包可以恢复已归档代码，不能独立重现训练结果或代替完整备份。
