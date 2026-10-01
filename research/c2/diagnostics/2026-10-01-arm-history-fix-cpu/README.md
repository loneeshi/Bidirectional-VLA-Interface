# 多轮历史快照修复（CPU）

主线第2步：按用户“继续修”修复r2的可变历史引用。感知接口复制history和previous_state；worker保存独立bundle/call、执行反馈及上一状态快照。后续追加或修改历史不会改变已经审计并归档的请求。请求内容、权限字段和反馈白名单保持原规则。

40项相关CPU测试通过。新回归实际调用感知、dispatch和build_wire，连续25轮覆盖有效locate_point、旧图recall_observation、过期定位拒绝，全程零物理动作；逐轮对比全部旧归档，并验证修改嵌套笔记/点坐标/上一TCP状态不会污染快照。此前测试fixture自行深复制历史，遗漏了真实感知接口的共享引用。

服务器修复包导入通过，资产根目录正确，CUDA未初始化、API0、物理步骤0。相对r2仅run_arm_capability_case.py与eef_arm_sensing.py发生运行时变化；协调v2执行器、guidance、清单和评分均未变。新冻结包及hash见deployment-freeze.json；旧包和失败归档不覆盖。CPU结果不代表物理抓取成功，也未验证修复后真实GPU工具循环。

本轮未重跑开发组。脚本/SAC未运行，Astra无有效抓取结果，4/5门未评估。原预算累计API1、估算USD0.0669125（供应商实账待核），GPU1累计29.570134秒；本轮增量0。GPU1复核空闲，历史Runpod存储未查询。无物理运行，不产生录像。下一验收门为修复包实际多轮运行；恢复须绑定新包并沿用剩余124次API、估算USD64.9330875、GPU13470.429866秒。

仅本地提交，不发布GitHub。
