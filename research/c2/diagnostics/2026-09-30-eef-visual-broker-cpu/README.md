# 视觉API代理CPU补充验收

所在阶段：接口已通过，补齐本地模型请求代理。推进：58项视觉相关CPU测试通过，其中6项代理新增测试覆盖请求前持久化预留、重复交付不重发、未知传输不重试、预留后中断、预算/审计拒绝不联网、未授权拒绝和请求绑定。仅用假响应；没有实际供应商API连通性或计费验证。

新模块eef_visual_broker.py负责逐请求持久化，run_eef_visual_broker.py提供有界本地轮询、授权/源码检查、独占锁及SSH交付。凭证只在本地读取，不上传服务器。旧E1B2仅复用SSH Connection类，不调用旧启动器和旧预算。

API0、GPU0。此前实测控制源码不变，r5接口回执仍保留；新增代理不冒称已经完成端到端物理模型试点。部署/授权文件另存with-broker，旧包保留。CPU回执见cpu-receipt.json。

阻塞：USD1010及GPU5400秒独立阶段授权尚未收到。下一道门：授权后执行启动前资源/账本/源码核验，启用代理并运行冻结三例；任何未知传输停止而不重试。

## 启动前进一步修复

发现模型待批准模板残留接口r5账本文件名，已改独立ledger-eef-visual-pilot-20260930-r1.json，并冻结model/medium/资源约束。新增启动器提供本地preflight-only，验证授权与pending一致、包/源码哈希和专用账本；实际启动前核验资源、调度、远端源码及CPU runtime gate。此前无物理实验或API受此配置错误影响。

60项视觉相关测试通过（代理/启动/范围8项）。最终包deployment-launch-ready.zip，授权模板authorization.launch.pending.json，哈希见launch-preflight-receipt.json。旧pending包保留但不再用于启动。没有运行新GPU或API；模型试点仍待批准。
