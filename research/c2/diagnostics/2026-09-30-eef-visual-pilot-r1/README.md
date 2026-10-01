# 高清视觉试点 r1：第三次请求传输删失

所在阶段：首次Astra视觉试点。推进：5号两次有效响应，locate_point只读定位成功，check_path只读返回ik_not_found。阻塞：第三次POST抛出URLError，原代理仅保存异常类型，缺少底层reason，无法判断DNS/TLS/连接重置等具体原因，也无法确认供应商是否收到或计费。下一道门：修复错误证据保存后，单独提出重试授权；本批不自动重试。

## 结果边界

5号：api_censored；工具调用2、环境步0；55/58未启动。没有正常任务结局，不计为0/3，也没有新增严格Pick成功。前两请求已证明高清请求与轮内响应链可被接口接受，但不是完整闭环完成。

第一条note：看到桌上的红白罐，测量顶部表面；locate_point返回当前base_link表面点[0.85967,-0.13659,1.00296]m，深度0.805m。第二条check_path被IK拒绝，尚未执行运动。第三条没有可用响应，不能归因为模型推理错误。失败归因：基础设施传输删失，模型任务归因无法判定。

## 记录与媒体

[完整历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-b3aed777/delivery/history.html)路径另见下方；原始请求/有效响应/第三次预留在broker-records/，逐例结果在remote-results/results/plan-005-visual/。录像只有初始帧，没有环境运动；不制造动作视频或运动轨迹。初始帧保留在recording-frames/000000.png，delivery.json明确no_motion。不是忘录视频。

已注册交付目录：`docs/media/c2-pick-visual-2026-09-30-plan005-b3aed777/delivery`。后处理仅CPU，无额外仿真。

## 用量

GPU1 102.174793进程秒，API尝试3，其中2有效、1未知。可计算用量保守估计USD0.8397000，未知请求预留USD1.6134，供应商实账未知；三请求总预算预留USD3.3768。实验室项目收费0（用户确认、无发票）。剩余范围关闭，不转入新批。GPU1进程与监督进程均已退出；旧RunPod存储仍单列未知。

注意remote usage-ledger.json的api_requests=0是监督器未汇总逐例API的记录bug；实际API3以本地逐请求账本和case result为准，不篡改冻结远端原文件。需在下一版本修复汇总。

## 收尾CPU修复（未重跑）

代理新增credential-free传输诊断：异常/底层reason类型、errno/winerror/HTTP状态，不记录headers或异常文本中的潜在凭证。下一次异常可区分DNS/TLS/连接/超时；无法追溯本次已丢失reason。监督器补汇总case.api_requests。9项代理测试通过，监督脚本编译通过，见post-run-cpu-fixes.json；冻结r1部署与原始账本不改。后续必须重新冻结包，当前源码已不同于r1。
