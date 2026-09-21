# S1-IA 输入、时间与动作消费合同审计（2026-09-19）

本轮以当时的两日交付计划（工作区文档，已作废）为执行依据，冻结 S1-IA `best/855`。截至本报告的静态审计，**没有发现有独立证据支持、足以进入路径 A 的输入或时点缺陷**。既有证据已经排除若干实现错误，但不证明原生操作能力成立。待补充预先冻结的 train 8 / development 8 样本语义检查后，如仍无明确缺陷，按新计划进入路径 B 的单共享 LoRA 小样本强拟合诊断；旧报告的暂停训练决定不覆盖新计划给出的有界诊断范围。

本报告初稿只读取本地源码和已有证据：没有 SSH、GPU、训练、仿真或付费调用。固定 OpenPI 本地 checkout 的 HEAD 已核对为 `f4eb160ba52b22c1e85fe432de59c24bbbac6187`，tracked working tree 无修改。MS-HAB/ManiSkill 本地来源是 `repo/AC-DiT-source/` 的源码快照；其静态行为与历史记录相符，但本次未据此宣称最新实验室安装逐文件相同。

## 合同表

| 项目 | 已核验 | 有证据的不一致或边界 | 尚未验证 / 本轮补充 |
| --- | --- | --- | --- |
| 相机与 RGB 类型 | H5 与在线适配器显式使用 `fetch_head`、`fetch_hand`；要求各自 `uint8 [128,128,3]`。固定上游把 head 接 `base_0_rgb`，hand 接 `left_wrist_0_rgb`，第三相机全零且在 PI05 中 mask=false。[1][2] | 未发现相机交换、RGB/BGR 交换、重复归一化的源码证据。静态相机名和训推一致都不足以单独证明采集时的物理相机正确。 | 固定 16 样本逐一导出原始两相机、模型实际 224 输入、相机标识，核对视角、通道、帧号及 resize。 |
| 图像缩放与强度 | 上游模型 transforms 使用 `ResizeImages(224,224)` → `resize_with_pad`；当前正方形 128 输入不会因长宽比产生裁切/letterbox。`Observation.from_dict` 对 uint8 执行一次 `x/255*2-1`。[3] | **既有 bit-exact 的边界是模型输入 transforms。** 实际训练 `train=True` 后有上游默认 augmentation：非 wrist 图像 95% crop、resize、±5°旋转；所有视图有 color jitter；推理 `train=False` 不启用。此为明确的预期训练行为，不是新发现缺陷。[4] | 16 样本报告应标明所导出的是确定性 224 输入还是带固定训练 RNG 的 augmentation 后输入，不把两者混称“训练实际像素”。 |
| state24 | `SequentialTaskEnv._get_obs_agent` 去掉完整 robot qpos/qvel 前三个 base 关节，保留各 12 维；适配器按 `[qpos12,qvel12]` 拼接，未使用旧 state30，也没有相对底盘平移。[5] | 元数据 `base_position_reference='world'` 不表示 state24 含世界坐标：base x/y/yaw 已被删掉。该旧兼容字段本身没有产生额外坐标变换的证据。 | 维名/单位见下表；将固定 16 样本的数值与源 H5、原始关节顺序一起保存。最新实验室环境源码和 manifest 身份由运行 preflight 复核。 |
| state 分位数与离散编码 | 归一化严格为 `2*(x-q01)/(q99-q01+1e-6)-1`，无显式 clip；tokenizer 在 256 个 bin 边界上 `digitize(...)-1`。[6] | 五个既有 reach 请求的 dim3 归一化值 1.1862–1.2127，seed2026 的 dim6 为 1.07054；这些上界外数值均编码成 **255**，与顶端区间相同。不能把轻微 q99 越界当根因；单纯上界 clip 对这些离散 state token 不改变结果。[7] | 16 样本逐维保存 raw、q01/q99、normalized、bin；下界 `<-1` 会编码成 `-1`，应与上界饱和区分。不得为了成绩随意裁剪 state。 |
| obs[t] → action[t] | IA reader 使用同一 `t` 的 qpos/qvel/head/hand 与 `actions[t:t+n]`；mask 必须是连续有效前缀，跨调用位置 mask，endpoint 不成为动作样本。真实 H5 6 边界样本已过检查。[8] | 未发现 off-by-one。已有专家回放对 `obs[0] --action[0]--> obs[1]` 的独立对齐证据：首步更贴近 source state1，而非 state0/2；无需重跑通用回放。[9] | 对固定 16 行保存父轨迹、调用、t、有效 action 源索引及 source hash；H5 单帧不能假定恢复完整接触/控制器状态。 |
| 指令与 tokenizer | IA reader 使用 window instruction；既有 preflight 的 reach/grasp/move token 不同。PaliGemma 输入是 `Task: {cleaned_z}, State: {24个bin};\nAction: `，先编码 state24，再将连续 state pad 到32；PI05 max token length=200。[6][10] | 普通 S1 原生 runner 固定 `Pick and stably hold the apple.`；独立 IA runner 分别固定 reach/grasp/move 窗口句。这是已知的**协议差异**，不是9/18某次运行用错 prompt 的证据。旧0/10不得与新 oracle 条件直接算提升。[11] | 保存16样本完整 tokenizer 字符串、未截断 tokens、最终 tokens/mask、截断数量。既有5份 reach 请求的有效 token 长度为106/106/107/107/105，均小于200；尚不能据此覆盖所有样本。 |
| chunk 与队列 | IA 原生调用 runner 每次预测10×13，只执行 `actions[0]`，clip至[-1,1]并置 head通道8:10为0。每步重新预测，无跨步/跨调用动作队列。通用 `FetchPiSkill.start()` 明确 `actions.clear()`。[12] | 通用 skill 默认 chunk_steps=3，与原生/独立调用 runner 执行1步不是同条件。未发现现有独立调用旧队列残留。 | 原基线与候选冻结同一执行方式；记录每次消费的预测索引、prompt切换和总动作预算。 |
| oracle 串接 | 可复用 `ia_call_predicates.py` 的句子与 completion predicates：reach≤8cm且未持物，grasp连续3观测持物，move持物且原生Pick成功。[13] | 本次所审查的 `eval_ia_s1_call.py` 是独立调用入口；未找到可直接视为已验收的 S1-IA 原生起点 reach→grasp→move 串接运行证据。15个SAC起点调用不能拼为连续链。 | 若新建串接入口，应在运行前冻结切换规则，切换清队列；总200动作及既有原生终止/累计力/恢复判据不变。严格命名“原生环境中的 oracle-assisted 调用串接诊断”，不得称GPT自主或历史固定整句门通过。 |

## state24 的逐维解释

顺序由完整 `JOINT_NAMES` 删除前3个base关节，再拼接同序速度得到。位置是**各关节自身坐标**，不是世界 TCP/目标位置；state 中不含 base x/y/yaw 或相应速度。[5]

| qpos索引 | qvel索引 | 关节 | 位置单位 | 速度单位 |
| ---: | ---: | --- | --- | --- |
| 0 | 12 | torso_lift_joint | m | m/s |
| 1 | 13 | head_pan_joint | rad | rad/s |
| 2 | 14 | shoulder_pan_joint | rad | rad/s |
| 3 | 15 | head_tilt_joint | rad | rad/s |
| 4 | 16 | shoulder_lift_joint | rad | rad/s |
| 5 | 17 | upperarm_roll_joint | rad | rad/s |
| 6 | 18 | elbow_flex_joint | rad | rad/s |
| 7 | 19 | forearm_roll_joint | rad | rad/s |
| 8 | 20 | wrist_flex_joint | rad | rad/s |
| 9 | 21 | wrist_roll_joint | rad | rad/s |
| 10 | 22 | r_gripper_finger_joint | m | m/s |
| 11 | 23 | l_gripper_finger_joint | m | m/s |

单位按 Fetch 原生关节位置/速度的 SI 物理语义列出，不是 action13 的归一化控制命令。动作7是夹爪绝对目标、动作11/12是底盘速度，不能把 action 索引直接套到 state 索引。当前缺少本轮最新实验室 URDF 身份复核；执行 preflight 应把运行时关节名和源文件哈希一起保存，不通过上述静态表声称重新测量过单位。

对 state 的归一化和离散化，设 `u_i=2*(x_i-q01_i)/(q99_i-q01_i+1e-6)-1`，边界 `b_k=-1+2k/256, k=0..255`，编码 `d_i=np.digitize(u_i,b)-1`。因此 `u_i<-1` 得 -1，`u_i>=0.9921875` 得255。上界外不产生256以上的数字；下界外没有被统一裁到0。已核验的上界越界 dim3 是 head tilt，而非 torso、base yaw或第三个动作通道。TokenizePrompt 在 PadStatesAndActions 之前，所以离散输入有24个值，而不是补零后的32个值。[6][10]

## 已有证据直接复用的范围

- 五个 reach exact-start 的训练 repack、LiberoInputs、quantile normalized、model input 全部 array_equal/hash相同；checkpoint normalizer与显式normalizer一致。这排除这些请求的两条确定性处理路径分叉，不证明原始相机视角充分或训练输入语义正确。[7]
- 跨进程原始 `1e-6` 复現队列仍是 invalid；独立 fresh-server 只确认5/5 same-key数值簇。不能用 exact RNG 把GPU输出要求成逐bit相同；后续配对记录rng并保留浮点容差及原始动作。[14]
- 已有 offline reconstruction 中，整体11个活动非head通道均优于zero reference，但yaw/torso有分族弱项。这支持新计划优先检验训练集可学习性；不支持已锁定相机/state bug或通过闭环门。[15]

## 缺陷裁决与下一最小实验

**候选缺陷：0个。** 固定整句与调用句的差异、模型内标准augmentation、少量state quantile越界均已明确记录，但目前没有独立证据证明它们是错误实现或9/18失败的原因。不要把“改变后动作更像SAC”单独当修复成立。

1. 完成本轮CPU固定16样本审计：train8/dev8，各4reach/2grasp/2move，按父轨迹隔离。绑定源码、H5、index、normalizer和checkpoint身份；输出两相机原始与224输入、state24逐维值、tokenizer完整输入/tokens/截断和action源索引。此项可以发现独立的可复核语义缺陷；没有缺陷则不再追加输入网格搜索。
2. 若16样本无明确缺陷，按路径B从最多2条成功训练父轨迹预先选64–128有效行，冻结原best/855、单共享LoRA、normalizer和输入合同；progress/family-bank关闭。至多100 optimizer updates或60分钟（含加载/编译/保存，先到停止）。原模型与诊断模型比较11个活动非head通道的首动作、有效chunk误差，另列reach/yaw/torso和回收阶段。
3. 训练快照短闭环只在确有可恢复完整快照时开展；不把训练轨迹成功算开发回归成功。小样本checkpoint不能直接充当泛化候选。未建立训练集改善证据即停止，不启动第二候选或新骨干。

本轮16样本实际结果尚未写入初稿；由执行任务追加本轮产物路径、样本名单、hash、截断统计、视觉核验结果与最终路径A/B决定。以上没有把计划中的检查写成已运行。

## 源码与证据定位

以下路径以 `D:\AI\embodied intelligence\repo\Bidirectional-VLA-Interface\` 为仓库根；`../` 指同级源码快照。行号用于定位本次所读版本，执行归档还须绑定文件hash。

1. `src/bvi/official_fetch_data.py:7–25,49–58`；`src/bvi/ia_fetch_data.py:48–54`；`scripts/eval_ia_s1_call.py:187–191`。物理相机声明：`../AC-DiT-source/third_party/ManiSkill/mani_skill/agents/robots/fetch/fetch.py:51–73`，head挂`head_camera_link`，hand挂`gripper_link`。
2. `../openpi-vlas-tools-audit/src/openpi/policies/libero_policy.py:20–26,52–68`。
3. `../openpi-vlas-tools-audit/src/openpi/training/config.py:130–140`；`../openpi-vlas-tools-audit/src/openpi/transforms.py:185–191`；`../openpi-vlas-tools-audit/src/openpi/models/model.py:115–120`。
4. `scripts/ia_s1_steps.py:32,59`；`../openpi-vlas-tools-audit/src/openpi/models/pi0.py:267,398`；`../openpi-vlas-tools-audit/src/openpi/models/model.py:168–187`。
5. `../AC-DiT-source/third_party/mshab/mshab/envs/sequential_task.py:1364–1368`；`src/bvi/fetch_pi_skill.py:9–12,85–94`；`src/bvi/official_fetch_data.py:13–17`；`docs/results/s1-official-pilot-2026-09-17-run01/README.md:8`。
6. `../openpi-vlas-tools-audit/src/openpi/transforms.py:141–145,250–270`；`../openpi-vlas-tools-audit/src/openpi/models/tokenizer.py:22–48`。
7. [已有确定性预处理审计](results/s1-preprocessing-parity-2026-09-18-run01/report.json)，各`requests[].state_quantile_ood`、`stages[-1].training_summary.tokenized_prompt_mask`、`provenance`、`transform_classes`；[报告解释](s1-first-action-attribution-2026-09-18.md)，第13–17行。
8. `src/bvi/ia_fetch_data.py:7–28,32–55`；`src/bvi/official_fetch_data.py:74–86`；[真实H5边界检查](results/s1-ia-index-2026-09-18-run01/h5-boundary-check.json)；[CPU实际token检查](results/s1-ia-index-2026-09-18-run01/runtime-preflight.json)。
9. [专家动作时间配对证据](s1-expert-replay-level2-2026-09-18.md)，第38行；[后续同状态首次分叉](s1-level3-first-divergence-2026-09-18.md)。
10. `../openpi-vlas-tools-audit/src/openpi/training/config.py:130–140`；`../openpi-vlas-tools-audit/src/openpi/models/pi0_config.py:39–40`；`scripts/serve_native_s1.py:121–123,181–189`。
11. `scripts/eval_native_s1.py:154–155`；`scripts/eval_ia_s1_call.py:156–162`；`src/bvi/ia_call_predicates.py:4–7`；`docs/s1-ia-sft-revision.md:25–27`。
12. `scripts/eval_ia_s1_call.py:193–205`；`scripts/eval_native_s1.py:193–198`；`src/bvi/fetch_pi_skill.py:16,68,119–124`。
13. `src/bvi/ia_call_predicates.py:10–21`；`src/bvi/fetch_segments.py:24–33`；`scripts/run_ia_s1_calls.py:39–40,95`；[独立调用协议](s1-ia-call-evaluation.md)，第5–9行。训练reach标注使用`not grasped[i-1]`，在线reach完成使用当前`not held`；尚无证据显示所选样本在这一边界发生不一致，不据此登记缺陷。
14. [same-key确认摘要](results/s1-same-key-confirmation-2026-09-18-run01/summary.json)；[首动作归因报告](s1-first-action-attribution-2026-09-18.md)，第19–33行。
15. [离线重建报告](s1-offline-action-reconstruction-2026-09-18.md)，第13–20、49–60行；其“训练暂停”是历史决定，本轮以新计划为准。


## 固定16样本实际核查（2026-09-19 UTC）

CPU导出完成，25.56秒、模型前向/优化器/仿真/API均0。train parents0/1与development20/21分离，每组4reach、2grasp、2move。16份raw RGB、确定性224模型输入、state、离散bin、tokenizer完整字符串和token/mask已保存；有效token为93–110/200，无截断。10/16有分位数越界，保留实际编码，未作裁剪。两张对照图均已逐行检查，未见转换引入的相机交换、通道交换或错误裁切；这不声称完成物理标定。尚无独立缺陷，因此选择路径B。

小样本名单在任何前向前冻结为成功train parent0/1的全部85个有效IA样本；原manifest与normalizer保持不变。H5单帧不构成可恢复接触/控制器快照，本轮不虚构训练快照短闭环。

证据：[input-audit-decision.json](results/two-day-delivery-2026-09-20/input-audit-decision.json)、[16样本与输入文件](results/two-day-delivery-2026-09-20/two-day-s1-input-audit-2026-09-19-run01/summary.json)、[train contact sheet](results/two-day-delivery-2026-09-20/two-day-s1-input-audit-2026-09-19-run01/train-contact-sheet.png)、[development contact sheet](results/two-day-delivery-2026-09-20/two-day-s1-input-audit-2026-09-19-run01/dev-contact-sheet.png)。


## 2026-09-19 原始身份核查更正：0/10的检查点归属

本轮重新读取实验室两批十种子运行的 `launch.json` 与 `server/metadata.json`：`s1-native-panel-2026-09-17-run01` 和 `s1-native-render-aligned-2026-09-18-run01` 均指向普通S1 `s1-medium-epoch-2026-09-17-run01/best/6000`，参数tree SHA256 `374dabe609e3bd117a1e42c373436aef9b4d9e5672a80a6aa0498b792853416d`，training_stage为 `S1_ordinary_target_domain_SFT_not_TAPT`。因此这两批固定整句0/10不能归给S1-IA best/855。

S1-IA调用批 `s1-ia-calls-2026-09-18-run01` 才是best/855，参数tree SHA256 `7854919f17de40ea8c62ee966327904a61503cfe2c0eb1715e3f30ae6e72892e`，training_stage为 `S1_IA_single_bank_no_progress`。已核实的直接证据为reach0/5、grasp/move调用诊断和离线重建；当前没有从所核实记录建立best/855独立固定整句0/10的身份链。此更正不把IA声明为成功，也不改变本轮冻结best855的路径B实验。保留历史文字与成绩，后续报告采用更正归属。

原始身份文件与SHA回执：repo `docs/results/two-day-delivery-2026-09-20/historical-identity-audit/receipt.json`。本次只读，不新增GPU前向、仿真、训练或API。


## Oracle串接入口的CPU准备

新增独立 `eval_native_s1_oracle.py` 与 `bvi.s1_oracle_protocol.PickOracle`，旧固定整句入口不改。协议命名 `ia_oracle_train_windows_v1`：每个chunk仅消费首动作；reach切换复现distance[t]≤0.08且not held[t-1]；grasp从near后的3个连续held观测切换；提示切换不重置RNG、不增加200总动作预算；move掉落仅记录训练支持范围偏离，不修改原生评分。6项CPU状态机测试通过，含与原分段函数的边界逐项比较。尚未实际执行原生评测，不能作为G1通过证据。
