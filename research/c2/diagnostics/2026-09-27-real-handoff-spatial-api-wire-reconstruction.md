# 阶段 1：开发集模型请求临发前重建审计

五阶段主线仍在第 2 步。本项没有发出模型 API 请求。原先 60 份开发集请求在生成时经过[冻结泄漏审计器](../../../src/bvi/offline_spatial_audit.py)，发送器在网络前核对固定 manifest 和逐文件哈希；本次把[发出前预检](../../../scripts/preflight_real_handoff_spatial_api_dev.py)加固为：从冻结清单、原普查、可信相机 manifest 和提示词重新生成受审计的内部请求及提供商 JSON，再与将要发送的文件逐字节比较。消融 B 仍必须绑定同快照真值图且显示 `PRIVILEGED ABLATION`；其余四条件不含真值图。

本地 CPU 预检重建并审计 **60／60** 份已准备请求：四个开发集位置 5、8、20、23；两个模型各 30 份；总 JSON 字节数 `6,718,826`。冻结清单 SHA-256 为 `ed800316c145acdbb5d36347d877de856f6e2406eec8897e997789231f434d62`。重新生成的每份内部请求哈希与 manifest 相同，提供商请求字节与原文件相同。预检报告 `network_requests=0`、`authorization_status=not_checked_not_implied`；这项通过不构成付费授权。

合成测试覆盖修改待发送字节、可信来源改变但旧 manifest 和旧请求仍存在等拒发情形；相关 16 项测试通过。该加固不改变冻结提示词、题目、图像、标签或任何已准备请求的字节，也不增加计划请求数。下一停止门是用户确认开发集 API 的独立 USD 3.50 上限、其与既有项目额度的关系及账户支付条件；[财务账本](../../../../BVI-research-plan-2026-09-14/finance/ledger-real-handoff-spatial-stage1-api-20260927.json)目前仍记录授权请求数 0、实际请求数 0。
