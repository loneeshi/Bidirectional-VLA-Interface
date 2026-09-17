# S1 pilot 训练集归一化

仅使用20条训练父轨迹的889个pre-action帧，验证集使用0帧。使用服务器固定OpenPI RunningStats，state24/action13，统计量已下载并校验SHA。每条记录动作计数一次，不按动作chunk重复加权；这是明确的归一化选择，未声称与作者按chunk统计完全相同。没有额外delta转换，模型padding应在归一化后进行。

CPU命令：`envs/openpi/bin/python normalize_official_fetch.py --dataset runs/s1-official-pilot-2026-09-17-run01 --output NEW_OUTPUT`。

这些统计量只适用于本pilot。扩大训练父轨迹时重新计算并冻结，不复用旧V8统计。尚未接入新S1训练配置/在线执行器，训练0更新；下一步验证实际模型预处理和反归一化路径。不能据此宣布100更新吞吐门通过。

本次实验室CPU任务上限120秒并已退出；GPU/API/租机均未使用，实验室服务费未知。

## CPU runtime 接入

FetchPiSkill新增显式native24分支：检查state_source=env_native_agent，从环境_get_obs_agent读取qpos12/qvel12，与数据导出使用同一校验器。保持原始指令传递，不使用旧30维或相对底盘变换；旧15/30维分支保留。52项相关CPU测试通过，包括模拟环境与旧路径回归。尚无实际新S1检查点/仿真在线验证，归一化仍须在新模型配置中接入；不得将单元测试称为原生Pick成功。
