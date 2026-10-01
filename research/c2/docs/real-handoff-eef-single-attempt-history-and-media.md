# 单任务历史与每例媒体交付（guidance-v3）

2026-09-30 UTC。用户授权代码修改；GPU/API 实验尚未启动。跨尝试记忆暂缓，不加入力反馈或进展不足事件中断，v13 运动参数不变。

## 每轮输入与输出

完整响应链继续使用样本独立的 previous_response_id，保留先前图像、指令和工具结果，不静默裁剪或降清晰度。每轮增加：

- 历史索引：turn、工具、起止控制步、是否到位；不是原始历史的替代。
- 上一次动作的参数及 task_state、工具回执；前后观察编号、TCP、本体里程计、剩余步数，以及在前次底盘坐标系表达的实测位移。
- 四张 2048 图仍有明确相机/模态标签。旧点和图像属于其来源观察，不自动刷新场景点。
- 模型输出 task_state 四项，各不超过 160 字符：belief、uncertainty、expected_change、next_check。明确是模型自述，不能作为测量真值。note 与 hindsight 保留。
- guidance-v3 要求先核对前次预期与本次测量，遮挡不能当作空闲空间；需要在中间位姿看图时拆成多次 move_to。每轮仍只执行一个工具。

工具开始执行前持久化指令，正常返回后补齐回执；异常中断保留已执行步数和未完成指令。读取模型请求的数据路径不访问评估端接触、力值、物体身份或真值位姿。

## 每例必交付

视觉 API0 接口 runner 与视觉 Astra runner 都在真实执行时录制：初始帧 + 每个环境步一帧。保存独立 PNG，进程被终止时已有帧仍在。无额外环境步。评估渲染相机不进入模型请求。

GPU 子进程退出后，批处理以 CPU 调用 scripts/finalize_eef_visual_attempt.py：

- delivery/history.html：全部请求、模型响应/自述、工具结果、四路图；包括无响应的删失请求。
- delivery/online-raw.mp4：本次在线录制。
- delivery/online-tcp-demo.mp4：复用用户 make_eef_demo_with_tcp_trajectory.py，运动与增长的末端轨迹逐帧同步。
- delivery/trajectory.html 与 trajectory.svg：复用用户 visualize_eef_trajectory.py，展示底盘/末端、工具时序、夹爪、累计力和结果，仅用于评估。
- delivery/README.md 与 delivery.json：准确文件名、哈希、结局与缺失情况。

失败、官方终止、删失同样交付。没有任何物理步时标注 no_motion；缺帧或编码失败明确标记交付不完整，停止后续批次，不自动物理重放补录像，不改实验原始结局。API0 视频标为 online interface check，不冒充 Astra 行为。CPU 生成媒体不计作仿真或模型调用。

## 本地收集与索引

远端 results 下整例目录（含 recording-frames、sensors、requests、responses、delivery）必须完整收回；不要只取 result.json。批回执中 delivery 指向每例 README。收回后运行：

```powershell
$env:PYTHONPATH='src;scripts;tests'
.venv/Scripts/python.exe scripts/finalize_eef_visual_attempt.py --attempt <收回的单例目录> --register-media docs/media
```

该命令把交付与传感器图片复制到 docs/media/c2-pick-visual-<录制UTC日期>-planNNN-<结果哈希>/，追加而不覆盖媒体 README 和 manifest；保留原始结果与哈希。重复目标目录拒绝覆盖。最终实验汇报必须给完整历史、同步视频、轨迹页面的准确链接，以及严格 Pick、删失和交付状态。

依赖：imageio 2.38.0、imageio-ffmpeg 0.6.0，以及项目已有 NumPy/Pillow。部署前须核实这些依赖，录制渲染成本仍受原GPU进程秒上限约束。

## 验证与授权边界

CPU 回执见 ../diagnostics/2026-09-30-eef-visual-history-cpu/。合成帧验证了编码、同步与索引，未验证真实GPU渲染。新增输入按 UTF-8 字节门限 14336 计预算；完整25轮及反思保守上下文为 1,014,624，小于 1,050,000。沿用已冻结计价公式的三例估算 USD 1000.8816（可提案取整1010），不是授权、不是实账，也不是新核验的价格。

旧 guidance-v2 部署包与提案哈希保留。此修改后必须重新冻结部署包、源码和提案，核实服务器及 finance，然后获得接口运行授权；不得拿旧包代表本版本或用旧额度启动。
