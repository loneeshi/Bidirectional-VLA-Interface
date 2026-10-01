# C2 640与两轮图像窗口 CPU验收

所在阶段：模型框架降本修改。推进：用户选择640实际渲染；独立模块完成显式文字历史+前后两轮图片，旧帧按需回看，严格不串接旧response链；474项CPU测试通过。阻塞：新分辨率物理接口/G2未运行，模型预算待确认。下一道门：[API0接口提案](../../docs/real-handoff-eef-compact-640-interface-proposal.md)。

这是CPU合成验证，不是物理实验或Astra能力结果，没有新Pick成功。r2之前的API1/50.188527GPU秒单独结算，本修改新API0/GPU0/环境0。

- cpu-tests.json：474通过。
- cpu-gates.json、deployment-manifest.json：源码/资产/测试/包SHA256；pending。
- guidance-v4.txt、schema.json：冻结提示与schema。
- budget-comparison.json：图像token减少98.5279%；合成文字轨迹三例保守预留53.6753625美元，不是实花预测或全合法输出上界。预算仍需新批准。

旧2048条件、r1/r2删失和原始档案保留。完整图片/响应/动作/媒体仍存档；只限制发给模型的图片。没有复制GPT-Policy源码。
