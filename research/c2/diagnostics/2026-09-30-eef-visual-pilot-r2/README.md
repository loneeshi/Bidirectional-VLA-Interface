# 高清视觉试点r2：首请求连接被重置，删失

所在阶段：模型接口运行。推进：异常证据确认URLError的底层为ConnectionResetError，errno10054。阻塞：不能据此区分本机代理、中间网络或供应商断连；无法确认供应商是否接收或计费。下一步：按用户新要求，改低分辨率与有界图像历史，先CPU验证。

5号API尝试1、有效响应0、工具0、环境0步；55/58未启动。不是0/3任务失败；无新增严格Pick成功。Supervisor与GPU子进程均已退出，GPU1无compute进程。GPU1 50.188527秒，未知请求预留USD0.6378，供应商实账未知；剩余额度关闭，不自动重试。

只有初始帧，没有动作视频或运动轨迹；完整交付：`docs/media/c2-pick-visual-2026-09-30-plan005-902c6b1c/delivery/README.md`。requests/responses及初始sensor档案均保留。旧r1/r2结果不覆盖。
