# S1 → S2 衔接实测记录


## 等待 S1 期间的衔接实现（2026-09-17 15:03 EDT）

主线仍在 **S1 正式目标域 SFT**，不是 TAPT 已完成。最近确认 2510/6835 更新；1000/2000 步保存与验证成功，暂选1000步（留出动作loss 0.172855；2000步0.176180）。最终选点仍只用留出损失，原生10起点评估尚未运行。

已完成：
- 新 `serve_native_s1.py` / `eval_native_s1.py` / `run_native_s1_panel.py`，原生双相机128RGB、qpos12+qvel12、官方13维动作；固定训练模板指令、10种子、200动作和5cm EE-rest原生判据。默认CPU预检，不并发训练。服务器端CPU预检通过，实际GPU推理/渲染待训练释放后验证。
- 真实官方第一帧训推预处理比较：状态、两相机图像和指令token完全一致；这是预处理一致性，不是完整模型数值或任务成功验收。
- 新证据汇总按实际episode日志、检查点身份和文件哈希生成准入报告；不足3/10保留失败报告，基础设施失败不能计为完成评估。
- 独立 `train_native_s2.py` 接入新S1十起点证据门和native24错误交接门；实际加载参数再核对哈希。保留旧V8训练器与hold。当前入口仅允许20更新/1800秒验证，不自动放大训练窗口；尚未进行任何S2参数更新。
- 已在CPU实际生成四族cache：50父轨迹、101调用、71,042,184字节；train reach20/grasp20/move29/release9，validation reach5/grasp5/move9/release4。原生128RGB/state24、当前帧进度、真实N+1端点与endpoint action mask通过loader检查。12条Place未满足既定分段条件，未放宽阈值。
- 33项CPU测试通过；未新增模型API调用或GPU任务。

仍需完成（不得写成已验收）：
1. 错误交接native24实际观测/标注验证，禁止拿旧workspace/state30样本直接放行。当前 `wrong_handoff_validation_complete=false`。
2. S1训练结束后，串行执行新推理入口与十起点评估；首个在线回合同时核查动作、模型恢复和原生环境契约。
3. 新S1达到>=3/10且错误交接门通过后，运行S2真实20更新门；GPU梯度/路由正确性仍待实测。
4. 完整高层协议离线回归及C线任务UID绑定仍待做，不挤占当前训练GPU。

部署修复：远端脚本缺少PYTHONPATH及新bvi模块，CPU任务曾在import阶段失败；补齐后预检与cache完成，无模型调用/参数更新。新增builder排除原因记录是后续代码改进，已生成cache的实际builder哈希保存在summary。

资源：CPU缓存进程已退出；S1训练继续使用lab GPU1。新增租机USD0、API0；lab费用未知，历史停止Runpod存储持续计费且本轮未重核账单。本次没有生成视频。

## 训练结束后的评估启动命令

先确认 S1 `status=completed_one_epoch`、GPU1空闲，再执行；此命令尚未运行。默认去掉 `--execute` 仅CPU预检。运行上限一小时，每episode最多600秒；遇基础设施失败停止并保留目录，不覆盖失败。

```bash
cd /home/pshuai/bvi-research
timeout -k 20s 3660s env PYTHONPATH=/home/pshuai/bvi-research/src/Bidirectional-VLA-Interface/src TZ=America/New_York envs/openpi/bin/python run_native_s1_panel.py \
  --training-run /home/pshuai/bvi-research/runs/s1-medium-epoch-2026-09-17-run01 \
  --normalizer /home/pshuai/bvi-research/runs/s1-official-medium-norm-2026-09-17-run01 \
  --output /home/pshuai/bvi-research/runs/s1-native-panel-2026-09-17-run01 \
  --model-python /home/pshuai/bvi-research/envs/openpi/bin/python \
  --sim-python /home/pshuai/bvi-research/envs/acdit/bin/python --execute
```

启动前在finance登记该S1能力评估窗口；不新增API/租机。S2新入口会重读全部S1证据并校验真正加载的参数哈希，不能通过手填成功率放行。S2错误交接manifest尚未生成，当前不能启动训练。旧V8/AC持有文件不改。
