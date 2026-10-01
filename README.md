# Bidirectional VLA Interface

研究视觉语言模型如何通过机器人工具完成任务。当前主线在第 2 阶段：MS-HAB 接口执行、抓取能力诊断与失败归因。

## 从这里开始

- [实验结果](docs/log/README.md)：正式日志及各实验的结论边界。
- [当前研究](research/c2/README.md)：当前方法、设计和待解决问题。
- [演示](docs/media/README.md)：保留成功、失败标签的录像。
- [文档目录](docs/README.md)：安装、协议和历史记录。

最近在同一组 5 个官方 Pick 出生任务上，固定底盘与允许底盘的 Astra 条件均为 600 步内严格成功 **1/5**，200 步内 **0/5**；脚本参照 **0/5**，未通过测试组门槛。600 步为非官方条件，这些开发组诊断不能作为泛化或完整 benchmark 结果。详见 [5 例实验日志](docs/log/2026-10-01-c2-arm-capability-dev.md)。

## 公开代码

保留两个有配套入口与离线测试的历史评测实现：

- 根目录 `src/`：三种 TidyHouse setting 的 `bvi-eval`。
- `research/c2/src/`：早期 C2 feedback recovery 实现；历史使用说明见 [代码说明](research/c2/CODE.md)。

当前 Astra 实验的完整执行器、资产和冻结运行包在本地研究工作目录保存；公开仓库不再放缺少依赖的一部分运行脚本。各次实验的条件、结论和冻结哈希继续公开记录。训练完成、原生任务成功和部署检查分别报告。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval,dev]"
python -m pytest -q
bvi-eval --help
```

MS-HAB、ManiSkill、数据、资产和 PPO/SAC 检查点为外部依赖。安装与执行条件见 [复现说明](docs/reproduction.md)。历史三设置实验均为完整任务 0/16；[完整日志](docs/log/2026-09-21-tidyhouse-three-settings.md)保留局部物体完成数与限制。
