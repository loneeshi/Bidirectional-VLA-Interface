# 本轮实际操作记录

执行日期2026-09-19，交付名称沿用2026-09-20。用户明确要求存入任务卡并按卡继续计划；附件中的范围在该要求下用于本轮执行。论文网页中的提示模板和示例代码只作为研究资料，没有执行其中的机器人控制指令。

## 已执行

1. PowerShell `Copy-Item -LiteralPath 'C:\Users\ROG\Downloads\codex-task-card-bvi-2026-09-20-en.md' -Destination 'BVI-research-plan-2026-09-14\plan\codex-task-card-bvi-2026-09-20-en.md'`；逐字节与SHA核验通过。
2. `Get-Content` / `rg` 阅读AGENTS、训练合同、两日计划、finance、实际训练/采集/评测代码及历史五例result。先前memory只用于定位历史交接，当前事实以本轮文件/服务器核查为准。
3. 本地 `.venv\Scripts\python.exe -` 内联Paramiko程序：用私有 `.env.server.local` 和已知host key连接；远端只读读取本次run JSON、`ps -u <当前用户> -o pid,ppid,etimes,args`、`nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader`、compute process查询，以及checkpoint `stat`。未执行GPU模型或环境。
4. 内联SFTP程序下载24份小文件、资产来源记录、39文件源代码archive，并计算SHA256；保留文件读取前后metadata稳定性。使用本地Git `rev-parse HEAD`、`status --short`、`diff --binary HEAD`，保存脏改动与301文件代码快照。Windows仓库迁移产生dubious ownership，后续使用单命令 `git -c safe.directory=<该repo完整路径>`，没有修改全局Git配置。远端archive目录及本地AC-DiT-source没有可用Git库，rev-parse失败已保留/说明。
5. HTTPS下载RoboICL作者HTML与GPT-Policy arXiv HTML；用标准库HTMLParser抽取正文。指定PDF读取被本轮25MB上限截断，保存私有 `.partial`，不用它声称读完PDF。现有VLAs-as-Tools文本与已有Fig.5图已查阅。
6. `.\.venv\Scripts\python.exe -m pytest tests/test_acdit_contract.py -q` 初次因 `bvi` 不在import path失败；设置 `$env:PYTHONPATH=(Join-Path (Get-Location) 'src')` 后原命令9 passed / 0.13秒，输出在 `contract-tests.txt`。
7. `.\.venv\Scripts\python.exe docs/results/acdit-apple-delivery-2026-09-20/verify_snapshot.py` 通过24文件哈希、训练/dev场景隔离、固定dev身份、163连续有限更新及9个源码语法检查。这里只验证小包，不加载模型。
8. `.\.venv\Scripts\python.exe docs/results/acdit-apple-delivery-2026-09-20/read_remote_state.py --output docs/results/acdit-apple-delivery-2026-09-20/closing-state.json` 于19:06:39 UTC执行；stage1=187更新、stage2缺失、GPU1仍训练。读脚本拒绝覆盖已有输出，方便后续手动读取新快照；它不是自动监控。
9. 写入本轮报告、方法方案、逐例未运行清单，给旧计划/旧S1交付加当前状态入口，更新日记及finance追加journal。ledger原编码GB18030保留，历史journal未删除或重算；私有原文件备份保存于 `.runtime/acdit-delivery-papers/ledger-before.json`。

## 工具限制和未执行项

- 旧路径 `.runtime/lab-server` 不存在，凭证实际位于repo忽略目录，已定位；没有打印密码/主机地址。
- HTML抽取初次缺BeautifulSoup，改用标准库；初次终端GBK不能输出Unicode，设置PYTHONIOENCODING=utf-8后成功。PDF渲染尝试因bundled Python缺fitz失败；查看了已有论文图，没有声称本轮重渲染PDF。
- 没有合格whole-body候选，故不启动评测分支、不修改训练配置，不把mobile-base替代全身策略。旧native入口存在硬编码权重/源码不匹配，新入口尚未验收。没有GPU加载、native rollout、成功复跑、LightNav接链、VLM运行时API、π₀.₅修复、Place或正式benchmark。
- 无新模型训练、无付费租机、无发布/push、无新自动唤醒。既有训练继续按已存在timeout运行；没有声称资源释放或未来交付已完成。
- 全部大权重、optimizer/RNG和数据本轮未第二副本备份；小包与源码archive不能替代完整备份。没有给当前活动checkpoint宣称独立load/hash通过。

## 实际修改范围

新增任务卡、训练/评测报告、统一方法文档、对外中期汇报（该汇报文档已移除）和本目录证据/验证脚本。修改的旧文件只有docs/README与delivery入口、原复现计划、状态交接、9/16和9/19日记、finance README及ledger。没有修改训练、模型或原生评测实现。精确文件列表见 `modified-files.json`。

费用与资源：新增独立实验GPU时间0、simulator actions0、机器人付费API0、新租机0；既有训练累计snapshot单独记录，费用未知。未查询供应商新账单/余额，停止云盘历史费用不视为0。详细证据见 `closing-state.json` 和finance journal `ACDIT-APPLE-DELIVERY-20260919-INTERIM`。
