# S1 pilot 训练集归一化

仅使用20条训练父轨迹的889个pre-action帧，验证集使用0帧。使用服务器固定OpenPI RunningStats，state24/action13，统计量已下载并校验SHA。每条记录动作计数一次，不按动作chunk重复加权；这是明确的归一化选择，未声称与作者按chunk统计完全相同。没有额外delta转换，模型padding应在归一化后进行。

CPU命令：`envs/openpi/bin/python normalize_official_fetch.py --dataset runs/s1-official-pilot-2026-09-17-run01 --output NEW_OUTPUT`。

这些统计量只适用于本pilot。扩大训练父轨迹时重新计算并冻结，不复用旧V8统计。尚未接入新S1训练配置/在线执行器，训练0更新；下一步验证实际模型预处理和反归一化路径。不能据此宣布100更新吞吐门通过。

本次实验室CPU任务上限120秒并已退出；GPU/API/租机均未使用，实验室服务费未知。

## CPU runtime 接入

FetchPiSkill新增显式native24分支：检查state_source=env_native_agent，从环境_get_obs_agent读取qpos12/qvel12，与数据导出使用同一校验器。保持原始指令传递，不使用旧30维或相对底盘变换；旧15/30维分支保留。52项相关CPU测试通过，包括模拟环境与旧路径回归。尚无实际新S1检查点/仿真在线验证，归一化仍须在新模型配置中接入；不得将单元测试称为原生Pick成功。

## S1 模型数据配置接入

新增fetch_native_s1_config.py，强制显式初始化路径和训练集统计来源；接入native24元数据，禁用progress head、移除LIBERO专用进度repack字段，保留13维动作且不额外delta变换。服务器CPU检查通过repack→图像输入转换→分位数归一化→反归一化，零动作往返最大误差1.11e-16。首次测试遗漏严格反归一化要求的state，补齐测试输入后通过，未修改库行为。

这只是CPU数据变换检查：初始化使用未加载的显式占位符，未执行模型/tokenizer全路径，未加载权重或训练。下一步必须固定真实初始化权重并完成数据加载/模型变换验证，不能直接启动该占位配置。证据config-smoke.json。

## 真实数据完整预处理检查

在实验室CPU使用显式root加载真实LeRobot训练集（889帧），首样本按20Hz构建10动作chunk，经过repack、双相机转换、训练集归一化、模型resize/tokenizer/padding全部通过：state32、actions10×32、prompt200 token、图像224×224。原始数据仍是原生128RGB/state24，224/32仅为模型内部标准变换。证据real-sample-smoke.json。

尚未加载初始化权重、计算模型损失或更新参数。官方训练loader默认根据repo_id找缓存，不接受本次自定义root；正式入口需显式绑定该已验证数据目录，不能依赖默认下载路径。服务器目前已检查的checkpoints目录没有独立pi05_base/libero原始初始化，只见旧V8等；未把V8暗作新初始化。下一步定位/取得并固定原始权重，再实现有界S1启动。
