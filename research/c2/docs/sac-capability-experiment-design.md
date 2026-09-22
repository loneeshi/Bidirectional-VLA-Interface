# SAC 初始状态恢复与反馈：C2 开发实验

更新：2026-09-22；协议 `c2-pose-feedback/3`。只新增 C2，不恢复 C1；C0 原路径不变。
状态：代码与CPU/控制smoke已通过；九次开发rollout已启动，首集已验证GPT调用和真实动作。尚无本轮最终结果，不代表恢复能力已经验证。

## 一、实验要证明什么

如果当前失败轨迹和 SAC 实测经验有用，那么在固定三个 MS-HAB TidyHouse 开发 episode 上，
相同工具与恢复预算下，加入这些反馈应比基础反馈完成更多最终正确放置对象；逐级去掉轨迹与案例检验信息贡献。
不假设 GPT 会选最优姿态：此前单帧判断未建立可靠预测能力，24个 spawn 样本也不是跨场景最优姿态库。

## 二、四根支柱

### 1. Baseline 与设置

| 设置 | GPT 输入 | 执行工具 | 用途 |
|---|---|---|---|
| H0（历史） | 上周输入 | PPO＋逐物体SAC；原生失败截停 | 16集、13/80对象、0/16成功；停止规则不同，不直接排名 |
| C0（保留） | 原C0输入 | Navigate500、Pick/Place200；每skill-target一次 | 历史参照，原prompt/schema与24请求门不变 |
| C2-basic | 当前两相机、任务、具名状态、终态历史、已试姿态 | PPO/SAC＋base preparation，操作最多额外3次 | 新协议恢复基线 |
| C2-trace | basic＋最近操作关键帧/结构化轨迹 | 与basic相同 | 当前失败上下文的增量价值 |
| C2-experience | trace＋最多两个相关历史SAC案例 | 与basic相同 | 实测经验的增量价值 |

**恢复 loop：Pick/Place未成功 → 编译反馈 → GPT给出新的明确位姿目标 → 当前几何与目标校验 →
真实重新站位 → 测量到达/持物 → GPT看新图像与误差 → 再调用SAC，或放弃/转向其他目标。**

首版只调base平面位置/朝向，arm/torso/head保持准备前测量值，gripper保持实际控制命令。
不执行自然语言、不直接控制关节、不替代SAC。GPT可提连续参数，不强制选择A/B/C。
spawn prior不再是上线必要条件；几何可行不等于SAC有优势。

### 2. 数据与隔离

从既有16个开发episode中按seed升序选前三个身份/日志完整的plan，冻结seed、UID、scene、初态SHA及选择理由。
已冻结seed 0/1/2（plan 557/90/147）；三者恰好来自同一scene，不能称为跨场景验证。
不可根据新C2结果换任务；每变体planned=3，总计九次rollout，不是24个新任务。
经验库只用剩余plan，排除所选三个plan的**全部attempts/帧**，按parent隔离而不是视频隔离。
成功失败均保留；先匹配skill/类别，再按可用起始几何排序，不用成功标签挑例。
无匹配/缺失状态明确unavailable，不补零；不能注入当前episode未来帧/结果。
已提取其余13个plan的66个历史调用；旧日志缺完整调用起始姿态，显式缺失。
这些调用是旧40步slice而非新的完整200步尝试，输入标注horizon；不得把slice timeout当作200步SAC失败。
三变体固定环境、checkpoint、初态、RNG、相机、controller与evaluator。
未验证simulator/RNG/控制缓存完整恢复前，不做“相同失败快照”分叉，主实验从冻结初态运行。
这三个反复开发的episode不是未见test，不能据此声称通用SR。

### 3. 评价指标

主要：最终满足独立放置/稳定谓词的唯一对象数0–5、完整成功/3。
同时：ever-completed、最终被破坏对象、subtask requested/executed/succeeded、准备拒绝率/到达率、
位置/yaw误差、持物变化、到达后SAC恢复率、全部实际SAC起始状态。
准备失败与SAC失败分开；基础设施失败/not_run/预算截断单列，planned分母不缩小。
报告全部动作、API请求/token/费用、墙钟、严格有效性、接触力峰值/累积违规和共同动作预算下的完成曲线。
预算终止的调用均值是截断观测；三集报告逐plan配对值，均值/离散程度仅作描述。

### 4. 变量控制

仅feedback variant变化；输出schema、控制器、恢复次数、模型、预算完全相同。
basic也保留必要终态和已试姿态，不能因消融而重复原姿态。
C0对比只能解释整个恢复机制收益，不能归因于经验模块；extra compute单列。

## 三、GPT输入输出与pipeline

### 1. Context compiler

公共：五对象目标/来源、当前图像、具名base/arm/body测量值及单位、工具集合、过去请求/终态、已试姿态/剩余额度。
本周使用**evaluator feedback**：native subtask success是特权反馈，不是learned progress。
continuous-progress旧分支保留但不混入本消融。

trace：采集最近操作调用的首末帧及grasp变化/最近目标距离事件帧，最多六帧；用真实sim step对齐图像/状态。
以带时间/事件/相机标签的contact sheet输入，保留两个当前相机；原始记录落盘，缺失不臆造。
experience：额外最多两个案例，每例最多四帧，提供parent UID、call ID、图像SHA、checkpoint SHA、
起始状态与真实历史结果。历史动作只是证据，不是当前待执行指令。
三组使用确定性编译，不为筛帧新增模型调用；不是复现GPT-Policy的VLM keyframe selector。

### 2. 结构化输出

普通工具调用 `recovery_goal=null`；reposition指定已失败且仍有余量的操作target，并给：

```json
{"frame":"base_at_request","x_m":0.15,"y_m":0.12,"yaw_rad":-0.25}
```

坐标相对请求时base：x向前、y向左、yaw逆时针，米/弧度。
执行前一次转为固定root-joint SE(2)目标，不随机器人移动重新解释。
平移≤0.5m，yaw变化≤π/2；与当前或已试SAC起点几何等价的目标拒绝。
解释文字只作日志，绝不能被模糊转换成动作。

### 3. 执行与返回

1. 读当前状态；校验schema/目标/新鲜度/预算，转换坐标并检查当前场景运动可行性。
2. 不支持/几何未验证返回rejected、零动作；不替换目标、不偷偷teleport。
3. 固定目标，低速闭环控制base，保持关节/夹爪；每步经env.step且计入总动作。
   C2独立跟踪器支持短距离倒退，不为后退先转身；线速度≤0.15m/s、角速度≤0.6rad/s。
4. 测量位置≤0.04m、yaw≤0.06rad、保持关节≤0.03rad，连续三检查才到达。
5. Place必须仍持有绑定物体。到达不是SAC成功；准备失败不得自动开始SAC。
6. 到达发放同target/同状态单次retry许可，其他动作使其失效。
7. 新图像和真实误差交给GPT，由GPT再调用200步以内SAC；结果回到compiler。

静态可通行不等于全身/动态碰撞证明。真实几何API/控制链未验收时fail closed；mock通过不能替代。
每操作目标最多3次准备和3次额外SAC；准备失败消耗准备机会，schema拒绝只消耗API。
普通Navigate仅首次调用，不追加重试；Pick/Place没有原地独立重试池。每轮恢复两次GPT决策：先站位，再根据到达证据决定SAC。未到达不得执行SAC，消耗本轮准备机会；GPT否决已到达姿态可用下一轮机会换姿态，不能无限检查。

## 四、GPT-Policy参考

固定版本 `4c43e57833d4314b066d386935e814d06940d027`：

- [motion/planner.py](https://github.com/cheng-haha/GPT-Policy/blob/4c43e57833d4314b066d386935e814d06940d027/src/gpt_policy/motion/planner.py)：明确目标、测量初态、坐标变换与执行报告。
- `input/video.py`、`input/action_sampling.py`：事件附近帧、真实时间对齐。
- `harness/video_selector.py`、`input/preparation.py`：有限上下文、来源/阶段标注。

只借鉴架构，不直接搬ARX/YAM机械臂IK到Fetch，不替换实验模型，不声称参考代码保证SAC提升。
本版平面base控制无需引入机械臂IK。

## 五、预算、停止与复现

Navigate500、Pick/Place200、reposition200。C0原24请求及7000动作/900秒保持不变。
C2每集7000动作、90次实际API请求（含网络/格式修复）、1800秒，单调用180秒、2048输出token。
20次首次调用＋10个操作目标×3轮×2次=80次逻辑调用，另10次请求余量；九集810次，另20次smoke，共830次、每请求USD0.005预留，总预留USD4.15，并非实际账单。
用户确认每轮两次并启动C2；上述包络取代未执行的1100次提案。原共享264请求/USD0.33保留，不重置历史ledger。

模拟器force/subtask fail记录而不截停整集；严格有效性一旦违规不得恢复true。
数值/基础设施异常及总预算仍停止；全对象最终完成可结束；不reset/teleport/清零累计力。
GPU1 only，无训练/RunPod；串行每次一集，最多两集小批，已完成跳过，失败保留attempt。
保存源码/dirty/config/prompt/manifest/初态SHA、随机种子、完整GPT输入输出/图像、视频与费用。

## 六、验收

1. CPU：C0隔离、坐标/schema/范围/重复、到达许可/计数、feedback分组、历史隔离/hash。
2. 无GPT真实Fetch smoke：手工目标验证平移/转向、关节/持物保持、误差与不可达拒绝。
3. 冻结三个plan及经验库；新增预算获批后GPT smoke，再九集对照。
4. 完整保存失败/not_run。接口可用≠姿态优化有效，到达≠SAC成功，九次rollout≠九个独立任务。

未过门保持blocked/not_run，不为“跑起来”删除运动、经验或费用校验。

2026-09-22验收证据：CPU 711 passed/6 skipped；空手前移/转向/侧移到达（202动作）；
最新持物后退/转向到达（99动作，含PPO导航/SAC抓取61动作），保持持物；一次向前越出静态网格的请求
零动作拒绝，原失败smoke保留。仅验证控制/接口，不是GPT优化有效或Place操作成功。
API桥接按每集实际provider attempts计数，网络重试与恢复attempt共享上限。

实现入口：`scripts/run_c2_feedback.py` 冻结/续跑小批 → `scripts/run_coordinator.py` 调用循环
→ `src/bvi/feedback/c2_context.py` 编译输入 → `src/bvi/coordinator.py` schema/GPT
→ `src/bvi/runtime.py` 准入/执行 → `src/bvi/pose_goal.py` 位姿转换/静态网格/控制
→ `src/bvi/pose_recovery.py` 到达检查与单次许可 → 原PPO/SAC。
资产生成用 `scripts/build_c2_experience.py`，无API控制验收用 `scripts/smoke_c2_pose.py`。
