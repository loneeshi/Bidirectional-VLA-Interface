# Astra 原题库完整评测：传输停止记录

记录日期：2026-09-28 UTC。状态：开发集未完成，测试集未启动。

## 发生了什么

第 89 次请求 `position-042-gpt-6-astra-highres_rgbd_p1-scene` 在 03:38:09 UTC 遇到 `SSLError: SSLV3_ALERT_BAD_RECORD_MAC`。没有收到提供商响应或响应 ID，是否执行及收费未知。发送器和后续控制器均退出，退出码均为 1；没有重发或替换该题。

| 项目 | 计划请求 | 已记录意图 | 有效答复 | 传输未知 |
|---|---:|---:|---:|---:|
| 开发集 | 135 | 89 | 88 | 1 |
| 测试集 | 235 | 0 | 0 | 0 |
| 额外稳定性 | 30 | 0 | 0 | 0 |

88 份有效答复包括此前经十进制边界修复恢复的 1 份；原始记录和修订回执都保留。

## 证据与评分

归档根目录为 `runs/real-handoff-spatial-stage1-20260927/`。

- 未知意图与错误回执：`api-astra-full-output.eval-only/position-042-gpt-6-astra-highres_rgbd_p1-scene/intent.json`、`transport-error.json`。
- 后续控制器：`astra-full-continuation-003/stopped.json`，原因为 `existing batch has censored/unknown outcome`。
- CPU 复核：`astra-development-ssl-stop-q1q4-verified.eval-only.json`，记录 89/135 个意图，SHA-256 `7da0647f7dc6c43f9c1ed185a1568e625a7f1415da35c5383d395f43dc734a4f`。只复核 Q1–Q4 与接口删失，Q5 未在本次重新计算；没有新增 API 请求。
- 一次初始 CPU 检查使用了不存在的输出目录，产生 0 个记录的报告 `astra-development-ssl-stop-q1q4.eval-only.json`；该文件不是评测结果，以上经过路径核实的报告才是本记录依据。

这是不完整开发集的状态记录，不报测试成绩、抓取 SR 或正例识别能力。原题库 Q1/Q2 尚无经过验收的正例；正例来源 GPU 预检单独待批。

## 传输检查与恢复提案

本地 Python 3.12.14、OpenSSL 3.5.8；进程环境没有代理变量。仅凭这些检查无法定位错误发生在客户端、网络中间环节还是服务端，不把 SSL 错误归因于模型。没有关闭 TLS 验证，也没有用付费请求测试网络。

原批准提案要求“传输结果未知立即停止”。拟单独修订为：保留此意图及 USD 0.13 预留，不重发、不替换；只继续原清单剩余 311 个未发送意图，总上限仍为 400 次／USD 55。提示词、题目、输入、标签和评分阈值不变。原始有效分母最多为 399，删失单列；开发集接口与评分验收须允许这 1 份已明确登记的删失。此修订等待用户批准，尚未执行。

## 财务

88 份响应合计输入 80,255、输出 72,202 token，已知用量按核实标准价估算 USD 4.41265；未知请求另外保留 USD 0.13，实付账单未核。新增 GPU1 进程秒、RunPod 资源均为 0。原授权未发送请求余额为 311；不能把未知请求记作免费或把 USD 55 减已知估算当成已核现金余额。

分账：项目 `finance/ledger-real-handoff-spatial-stage1-astra-full-20260928.json`。历史 RunPod 持久存储费用未刷新。

链接：[批准提案](../docs/real-handoff-spatial-stage1-astra-full-api-proposal.md) · [数值边界修复](2026-09-28-spatial-probability-boundary-repair.md)
