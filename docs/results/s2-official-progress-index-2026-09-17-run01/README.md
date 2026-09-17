# S2 当前帧监督索引（CPU 准备）

从已固定 Pick/Place manifest 生成 1,816 行调用局部监督，覆盖四族；101 个调用端点保留 progress=1，但 action_valid 全为 false，动作源索引为 null，不伪造零动作。索引读取当前 observation t，horizon=2；超出调用边界的未来标签使用 mask。

按源 H5 SHA256、父轨迹 ID、split、调用 index 与 observation index 唯一定位。完整父轨迹分组先于切分；重复/泄漏/缺失父轨迹拒绝。没有分段证据的尾部不生成完成标签。46 项相关测试通过。

这只是监督索引，尚未验证图像/状态训练缓存，未完成错误交接验证，未做任何 S2 参数更新。S1 归一化与在线接口仍待接入，100 更新吞吐门未启动。S2 必须在新 S1 ≥3/10 后才训练。

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe scripts/index_official_fetch_progress.py --manifests docs/results/s1-official-pilot-2026-09-17-run01/pick-manifest.json docs/results/s1-official-pilot-2026-09-17-run01/place-manifest.json --output NEW_OUTPUT
```

此次仅本地 CPU，无模型 API、GPU、服务器或租机操作。索引 SHA256 与来源见 summary.json。
