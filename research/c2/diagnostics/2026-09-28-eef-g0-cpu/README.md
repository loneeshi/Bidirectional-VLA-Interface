# 2026-09-28 G0 CPU 准备

仅 CPU 数学/接口验证与只读 SSH。新 API 0、GPU 进程秒 0、仿真动作 0。
实验室收费 0 为用户确认（无发票）；历史 API 欠核账与 RunPod 存储不计作本轮费用。

- 11 开发集 SAC 首次成功快照：归档成员字节与冻结哈希全部匹配，运行时恢复待验收。
- 8/7 DOF 各 11 个起点 FK→IK 自洽检查通过；TCP FK 与归档官方 link
  评估端核对最大矩阵元素误差 8.42e-7。不是完整教师轨迹 G1。
- 60 个头/手相机外参 URDF FK 审计通过，最大矩阵元素误差 1.033e-6。
  世界位姿只在评估核对使用，没有进入工具或请求；不是 G2 表面精度通过。
- Python CPU 核心：src/bvi/eef_tools.py；pytest tests/test_eef_tools.py，13 通过、0 跳过。
- GPU1 只读状态：15 MiB、0%、无计算进程；约 636 GB 可用，工作目录可写，
  quota 无限制报告。sbatch/qsub 不在 PATH；实验室调度/预约规则未确认。
- 原 stationary_head=True 会清零头部动作；return_to_rest 需要保留头部通道。
- 本地三份方向文档单独提交 3e3f833；后续代码、测试、提案和诊断保留未提交供审阅，未发布。

机器证据：[样本与 IK](readiness.eval-only.json)、
[相机校准](camera-calibration-60.eval-only.json)。
测试结果、源码哈希及收尾资源：[准备回执](preparation-receipt.json)。
批准入口：[G0 提案](../../../../docs/design/real-handoff-eef-g0-authorization-proposal.md)。

## G0 运行后

上文为历史 CPU 准备口径；实际 G0 已运行并停在失败门。见[完整运行记录（已移出可信日志）](run-record.md)、[评分](g0-run02-score.eval-only.json)与[运行后 CPU dogbox 审计](dogbox-collected-teachers.eval-only.json)。新求解器未部署到 GPU。
