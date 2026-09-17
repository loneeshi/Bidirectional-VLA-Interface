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

## S1 初始化获取已启动

选定官方pi05_base作为新的Fetch目标域SFT初始化，不复用失败V8。公开来源gs://openpi-assets/checkpoints/pi05_base/params，列表20对象共12,441,721,931字节。实验室CPU下载PID1450841，外层1800秒，来源大小上限20GB；按GCS generation固定每个对象并计算本地SHA256。状态文件checkpoints/s1-pi05-base-2026-09-17/acquisition.json，尚未确认全量下载、参数加载或开始训练。

## 显式父轨迹划分 loader

新增native_dataset绑定显式train/validation root，两组分别889/230帧真实首样本经过完整预处理。S1显式progress_loss_weight=0、use_val_set=False，避免作者按episode hash再次切分；验证集独立loader，仍使用训练集统计量。该loader仅返回动作学习输入，不提供伪造学习进度。正式有界训练循环尚未实现/启动，不能直接用作者默认main替代这个显式loader。证据loader-smoke.json。

## 有界 S1 pilot 执行器

run_native_s1_pilot.py已实现并通过CPU数据预检：默认不训练，显式--train仅GPU1，最多100更新/内部3600秒，外部必须另加timeout。microbatch1/accumulation1，用于吞吐门，不宣称有效batch8。执行器调用作者train.main/train_step，进程内替换数据loader以绑定既有划分；S1关闭progress head/loss，作者四元组接口的progress占位零值不是学习标签。独立val数据只预检，pilot不按验证选点，后续长训仍需独立验证与选点。

当前只通过CPU路径，GPU初始化/反向/检查点保存尚未实测，未实际更新参数。下一步GPU前核查资源并登记100更新范围，外层限时，失败保留部分更新日志。runner-preflight.json记录实际配置。

### 吞吐计时修正（启动前）

CPU审查发现pilot原计时起点在样本预处理后，可能高估用于扩展数据规模的吞吐。已前移至样本读取前，包含读取、图像/tokenizer处理、设备传输、训练和日志；前10次热身不计稳定吞吐，另保留总墙钟。此前未运行GPU，故没有需要撤回的吞吐成绩。服务器脚本需在启动前同步此修正。
