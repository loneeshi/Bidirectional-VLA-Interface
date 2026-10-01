# 高清视觉单次试点授权提案（guidance-v3，待批准）

当前阶段：接口已通过，模型试点未启动。本提案不继承旧API或GPU余额。

## 已过前置门

[接口r5回执](../diagnostics/2026-09-30-r5-eef-visual-cpu/README.md)：453项CPU测试通过；5、55、58三例真实2048渲染、主动转头、只读检查、官方回位、双相机G2通过。47/47/55环境步，API0。接口通过不是严格Pick成功。

## 冻结条件

顺序5、55、58，全部为已暴露开发案例，61与测试集不使用。每例恢复冻结交接快照、独立响应链，H0单次，gpt-6-astra medium，guidance-v3；25工具调用、200官方步、最多一次终止反思（无动作），合计最多78请求。4张头/手RGB-D均实际2048×2048、原尺寸detail=original；完整轮内历史保留，每轮状态对比、history index、note和task_state。无跨尝试记忆，不引入力或进展不足事件反馈。

只给目标语义、机器人传感器、本体与相对里程计；不给目标真值位置、接触身份、世界位姿、网格。主动look、locate_point和check_path可用；只读工具零环境步，运动沿用通过接口的控制参数。保留最多60步回位指导。评估数据与请求分离；同类别歧义单列，不用真值消歧。

## 资源与费用

GPU1串行3×1800=5400进程秒。API最多78次，单响应最多4000输出token。申请USD1010总上限，旧余额0。这是保守预留，不是预计实花。

每张图ceil(64×64×1.2)=4916输入token，4张19664；每轮新增非图文本采用14336 UTF-8字节上界，另留1024封装token。第t轮输入预留为(t+1)×35024+t×4000；反思按第26轮连同图片保守预留。最大输入1010624，加输出4000仍低于1050000上下文。跨样本绝不接续历史。

按输入USD10/M、输出USD50/M，保守输入采用cache-write USD12.50/M；超过272K输入时全请求输入×2、输出×1.5，不假定缓存折扣。完整三例最坏预留USD1000.8816，向上申请1010。[官方模型价格](https://developers.openai.com/api/docs/models/gpt-6-astra)；[官方图像规则](https://developers.openai.com/api/docs/guides/images-vision)，2026-09-30核对。逐轮明细见[budget.json](../diagnostics/2026-09-30-r5-eef-visual-cpu/budget.json)。实际usage逐请求记录，最终以供应商账单核对；不把预留当消费。

请求前检查实际序列化白名单、输入包络、历史响应链、累计预留与上下文；越界明确停止，不删历史、不降清晰度。未知传输结果停止整个批次并保留删失，不自动重试；无效参数占工具次数但不执行动作。普通任务失败继续冻结后续案例。官方终止立即停止动作，反思仅发送允许的结果和终止类别。

## 冻结与启动门

部署SHA-256：`449f1103c5ec02a6ffa565e5c4da399e5f171e91ecbac59c7235aeb90c7c7604`。完整源码、接口proof与待批准授权哈希见[pilot-freeze.json](../diagnostics/2026-09-30-r5-eef-visual-cpu/pilot-freeze.json)。本地部署包runs/eef-visual-20260930-pilot-r1/deployment.zip，授权文件authorization.pending.json尚未生效。运行前再次读取finance README与ledger、核对GPU1占用/存储/权限/调度，核实部署和源码哈希，检查SDK初始化与本地API代理的无网络预检。凭证不上传服务器。任一前检失败不发请求；不得未经记录替换冻结代码。

## 报告与停止边界

逐例报告严格Pick、官方终止/模型done或give_up/工具上限、首次底盘距离、分段靠近、look/locate使用、停稳/底盘/手臂/夹爪/回位耗步。失败引用关键note及hindsight，对照实际传感器与动作，按AGENTS四类归因；删失单列。只报告开发诊断，不单独归因高分辨率，不修改旧分母。

每例交付完整history.html、在线录像online-raw.mp4、online-command-vs-executed.mp4和trajectory.html；命令路径不叫物理预测。异常退出也交付已完成片段与缺失说明。结果仅diagnostics，媒体仅docs/media并核对索引和哈希，不发布GitHub。

三例完成后再写H0独立重试与H1自身历史重试的独立提案，当前不启动重试。等待用户对本提案的明确批准后才能启动本批GPU/API。


## 启动链路补齐（2026-09-30，仍待批准）

本地代理补充完成，[CPU回执](../diagnostics/2026-09-30-eef-visual-broker-cpu/README.md)58项通过，其中6项新增；不改变r5实测控制器。最终待批准包改为`runs/eef-visual-20260930-pilot-r1/deployment-with-broker.zip`，SHA-256：`2ead6de87e712a0cdf30a5225040ee168eb3123da7aece8453478fc6c8d4fd84`；对应`authorization.with-broker.pending.json`，替代上文无代理的旧待批准包，旧包保留。资源、样本、提示与金额不变；API0/GPU0，尚无启动授权。


## 最终启动冻结（替代上述待批准包，不改变实验条件）

补齐专用启动器，并修复模板残留接口账本绑定。最终部署：`runs/eef-visual-20260930-pilot-r1/deployment-launch-ready.zip`，SHA-256：`caaea611d5ebdc16a1235cde8dfeae6c1ddc937c3df8a4dd292189aab75ff9a6`；唯一当前授权模板`authorization.launch.pending.json`（仍pending）。模型账本独立为`ledger-eef-visual-pilot-20260930-r1.json`。60项视觉相关CPU测试通过，见[补充回执](../diagnostics/2026-09-30-eef-visual-broker-cpu/launch-preflight-receipt.json)。

批准后将同一模板另存authorization.json、记录批准证据并创建专用预算账本，再执行operate_eef_visual_pilot.py的本地预检和启动。run_eef_visual_broker.py只接受该独立授权、账本及冻结源码。若部署启动传输结果未知，先检查同一远端句柄，不重跑启动。
