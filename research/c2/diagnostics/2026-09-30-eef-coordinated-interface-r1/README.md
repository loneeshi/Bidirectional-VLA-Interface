# Coordinated arm v1 API0接口：未过60步门

所在阶段：C2末端调度修订。推进：新关节路径/FK同源参考消除了5号IK等待；阻塞：第一条move_to在60步内仍未到位，后续目标及55/58未运行。下一门：末段跟踪/参考放行的CPU审查，API暂不启动。用户已经授权后续有界GPU/API，但接口门仍必须通过。

不是Pick试验，不计Astra成功率。本批使用原r3 Astra第一条指令及冻结快照做程序侧回放；没有SAC轨迹或场景过滤参与控制。第一条失败后只回休息位并停止，不伪装网络删失。supervisor统一censored字段不能取代result的interface_failed分类。

版本v1使用同一条IK8关节路径生成末端FK和躯干高度，原始笛卡尔直线路径的中间形状可能略有变化；目标和容差不变。v1平滑时间进度，v2线性进度并检查原关节速度及A1躯干动作限幅。底盘v13控制不改，v1/v2独立模块保留。CPU验证固定计划高度下的逐点IK7通过，不保证物理跟踪可用。

结果：参考点69，IK等待0步，60步位置残差0.016452916m、朝向残差0.079294280rad，reference_completed=false。回位19步且官方ee_rest/robot_rest/is_static通过。共83环境步（张开4+move60+rest19）。未放宽60步、1cm/0.05rad到位门；不是成功。

GPU1 56.637865进程秒，API0，实验室项目收费0（用户确认无发票）；全部本批PID已退出/GPU1无compute进程。未用额度关闭，不自动转入。旧未知API/RunPod存储继续单列。

- [online-command-vs-executed.mp4](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-b4fb98e1/delivery/online-command-vs-executed.mp4)
- [完整执行历史](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-b4fb98e1/delivery/history.html)
- [交互轨迹](../../../../docs/media/c2-pick-visual-2026-09-30-plan005-b4fb98e1/delivery/trajectory.html)

在线API0脚本视频，不是Astra新推理；命令线不是物理预测。全部媒体哈希已核对，源码/部署绑定见authorization.json；原始轨迹保留remote-results。
