# 阶段 1：冻结基线源码复现修复

五阶段主线仍在第 2 步。本次只修复本地 CPU 评分源码的复现门；没有新的 GPU 进程、模型 API 请求、PPO 或 SAC 动作。

## 问题与修复

Q5 评分器在运行前验证 `src/bvi/offline_spatial_baselines.py` 的冻结 SHA-256 `c787c302c23850afd6b5f40216080676f021fcd5eebccc4317ee01f2a6b56d9a`。后续 Q4 深度代理把一个新几何函数加进了同一文件，导致 Q5 重跑在源码门处停止；这不改变既有 Q5 归档，但破坏了复现入口。

现把 Q4 新函数移入独立的 [`offline_spatial_depth_geometry.py`](../../../src/bvi/offline_spatial_depth_geometry.py)，恢复冻结基线文件的原始字节，Q4 评分器从新文件导入。冻结清单、标签和原始归档均未修改。

## 验证

| 检查 | 结果 |
|---|---|
| 冻结基线源码 SHA-256 | `c787c302c23850afd6b5f40216080676f021fcd5eebccc4317ee01f2a6b56d9a` |
| Q5 位置 1 重跑与既有归档 JSON | 逐字节相同 |
| Q5 位置 1 输出 SHA-256 | `6b29a7a27133db0d019ca33d3206b542ca524b720d3a846a8a3c622bd9095ca2` |
| Q4 深度代理重跑 SHA-256 | `ccc02fe66912280b9fe6ff986a5152dea5a85f9ba5cd131a5fce092eabc873f8` |
| Q2 深度代理重跑 SHA-256 | `b27a3cc3139186af37b314de4bed38a1e59b0b34f58c8cc4538df68fc206dd32` |
| 对应 CPU 单元测试 | 12/12 通过 |

Q5 的单例重跑只证明冻结源码门与该例结果可复现；60 例的既有静态结果仍按原归档解释，不是动态 Pick 成功。模型开发集请求仍未发送。
