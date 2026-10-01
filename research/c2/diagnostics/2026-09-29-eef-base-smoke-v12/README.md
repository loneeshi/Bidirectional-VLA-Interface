# C2 底盘冒烟v12：第20号通过，监督器记录错误后停止

所在阶段：物理底盘冒烟未完成。冻结五例只完成第20号，23/5/8/11未启动；不是5/5通过，也没有新的严格Pick成功或Astra能力结果。API0。

## 观测结果

|第20号独立恢复动作|环境步|位置误差m|角度误差rad|通过|
|---|---:|---:|---:|---|
|左转0.1rad|12|0.002914920|0.028461552|是|
|前进0.1m|15|0.018763220|0.009626150|是|

两次都未被官方终止，累计力门通过；最大里程计与真值差约1.41e-7m／3.43e-7rad。运行时控制周期确认为0.05s，即20Hz。分别从同一冻结快照恢复；没有路线预检或碰撞过滤。完整数值见[summary.json](summary.json)及[原始结果](results/plan-020-smoke/result.json)。

## 停止原因与修复边界

冻结子进程汇总只提供smoke_passed和records，未提供顶层strict_pick_success；冻结监督器在输出进度时使用`result['strict_pick_success']`，触发KeyError。错误发生在第20号完整结果、用量和进程回执已经落盘之后，父监督器进入停止分支，其余四例没有启动。见[原始控制日志](controller.log)和[用量记录](usage-ledger.json)。这是漏测的汇总接口错误，不是底盘、躯干或碰撞失败。

CPU修复新增独立r2监督器与窄化授权的子进程入口；进度信息改用可缺省的Pick字段，不伪造值。**v12控制器、速度、动作、门槛和已运行代码全部保留原样。**实际第20号结果加入回归，264项测试通过，见[修复测试](repair-cpu-tests.json)。新包仅本地冻结，未上传、未启动。

## 资源与费用

实耗1个GPU1进程、47.275746054947376进程秒，27环境步、API0。实验室收费0为用户确认、无发票；原900秒额度关闭，未用852.7242539450526秒不转入新授权。[财务回执](finance-receipt.json)。

[最终资源核对](resource-final-check.json)：监督器1687840及子进程1687979均退出；GPU1 15MiB/0%、无计算进程，GPU0他人进程未触碰。[RunPod只读核对](runpod-resource-final-check.json)：两个历史实例均EXITED，各保留20GB存储，无network volume；存储费用继续单列，未刷新账单、未删除资源。

## 下一道门

待用户批准[剩余四例续跑提案](../../docs/real-handoff-eef-base-smoke-remaining-authorization-proposal.md)：23/5/8/11，共4×180=720进程秒、API0；不重复第20号。新包SHA-256 `a183a3249b03c1944566cc0fdeef842c751d7f99575575e4c31681f68b2a04ab`，见[部署清单](repair-deployment-manifest.json)。旧与新批次分别记账，同版v12五例验收结果合并前核对控制器哈希。任一未通过即停止；全部通过后才申请E1-B。

未录视频。结果仅在diagnostics，不进docs/log或索引，不发布GitHub。
