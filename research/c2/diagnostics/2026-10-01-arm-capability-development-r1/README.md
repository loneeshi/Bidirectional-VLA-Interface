# 修订二开发组：启动异常，停批待复核

当前为五阶段主线第2步。已按用户“批准”登记并启动；首例 arm-dev-000 / V 在初始化阶段退出，后14个进程未启动。阻塞是资产路径接口；下一验收门是修复路径解析并补CPU检查后，复核冻结与恢复安排。没有抓取能力结果。

## 执行器 CPU 验证

沿用 aac0487 冻结的协调v2与此前579 passed、3 skipped（948 deselected）的CPU回归。本次启动前100个源文件、1046个资产哈希通过；哈希和导入通过不代表环境初始化通过。执行器未改。

只读CPU检查确认：worker设置 MS_ASSET_DIR=/home/pshuai/bvi-research/assets/data，固定版ManiSkill自动追加data，最终解析为不存在的assets/data/data。检查点config可正常加载。原worker只保留FileNotFoundError类型，没有完整异常栈，故无法确定首个失败文件读取。此前CPU准入遗漏有效ASSET_DIR检查。详见 startup-cpu-diagnosis.json。

## 脚本参照

0/5例执行，严格成功率不可计算，4/5测试准入门尚未评估。SAC也未执行。不得报告0%或把缺失计为失败。

## Astra V

1次进程启动被基础设施截断，0个有效评估，物理动作0、API发送0；不归因为Astra失败。P已删除。200/600步结果、定位距离和脚本成功子集均不可计算。

## 资源与证据

GPU1进程耗时5.580406秒，剩余上限13494.419594秒；API已发送0/125，已知消费USD0，USD65上限未使用，供应商账单未核。剩余额度保持待复核，不自动重跑或转移。批次PID已退出，GPU1无计算进程，GPU0他人进程未触碰。历史Runpod存储账务未刷新，不能声称停止收费。

本次没有物理动作或采集帧，因此没有视频或轨迹可视化；原始media_complete=true仅表示零帧分支收尾成功，不能解释为已生成录像。完整原始归档为raw-run.tar.gz；raw/保留原始状态和结果，closeout.json提供明确解释。

只读诊断未实例化环境、未初始化CUDA、未发送API。未更改冻结执行器或重新运行。全部结果留在diagnostics，未写docs/log，未发布GitHub。
