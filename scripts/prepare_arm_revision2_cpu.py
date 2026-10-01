"""Freeze the user-directed V-only revision without API or physical execution."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'research/c2/diagnostics/2026-10-01-arm-capability-revision2-cpu'
OLD=ROOT/'research/c2/diagnostics/2026-10-01-arm-capability-cpu'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):(D/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def select_dev(original):
    selected=copy.deepcopy(original[:5]);replacements=[]
    for incoming in original[5:]:
        counts=Counter(r['height_band'] for r in selected)
        if incoming['height_band'] not in counts:
            index=next(i for i,r in enumerate(selected) if counts[r['height_band']]>1)
            replacements.append({'out':selected[index]['case_id'],'in':incoming['case_id'],
                'reason':'first missing height in original order; replace first selected row whose height has duplicates'})
            selected[index]=copy.deepcopy(incoming)
    order={r['case_id']:i for i,r in enumerate(original)}
    selected.sort(key=lambda r:order[r['case_id']])
    assert {r['height_band'] for r in selected}=={r['height_band'] for r in original}
    for row in selected:row['condition_order']=['V','SAC','script']
    return selected,replacements


def history():
    responses={};sources={};processes=[];sends=0;logical=0
    for run in (1,2,3):
        folder=ROOT/f'research/c2/diagnostics/2026-09-30-eef-compact-640-pilot-r{run}'
        path=folder/'remote-results/usage-ledger.json';ledger=json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))]=sha(path);sends+=ledger['api_requests']
        finance_path=folder/'finance-receipt.json';finance=json.loads(finance_path.read_text(encoding='utf-8'))
        sources[str(finance_path.relative_to(ROOT))]=sha(finance_path)
        logical+=len({e.get('logical_request_id',e['request_id']) for e in finance['entries']})
        for row in ledger['records']:
            processes.append({'run':run,'position':row['position'],'elapsed_process_seconds':row['elapsed_process_seconds'],
                'returncode':row['returncode'],'censored':row.get('censored'),'timed_out':row['timed_out']})
        for path in sorted(folder.glob('broker-records/**/response-raw.json')):
            body=json.loads(path.read_text());usage=body.get('usage',{})
            if not isinstance(usage.get('input_tokens'),int):continue
            key=body['id']
            if key in responses:
                assert responses[key]['usage']==usage
                continue
            responses[key]={'usage':usage,'source':str(path.relative_to(ROOT)),'sha256':sha(path)}
    inputs=[r['usage']['input_tokens'] for r in responses.values()]
    outputs=[r['usage']['output_tokens'] for r in responses.values()]
    return {'processes':processes,'source_sha256':sources,'responses':responses,
        'tokens':{'logical_requests':logical,'recorded_sends':sends,'known_unique_usage_responses':len(inputs),
            'unknown_usage_sends':sends-len(inputs),'input_median':statistics.median(inputs),
            'input_max':max(inputs),'output_median':statistics.median(outputs),'output_max':max(outputs),
            'deduplication':'provider response id; duplicate archived copies are not extra requests'},
        'scope':'historical640 r1-r3; includes censored/failed processes, not600-step throughput validation'}


def main():
    D.mkdir(parents=True,exist_ok=True)
    for name in ('runtime-assets.cpu.json','horizon-reset-offset-check.json'):
        (D/name).write_bytes((OLD/name).read_bytes())
    prior=json.loads((OLD/'roster.eval-only.json').read_text());dev,replacements=select_dev(prior['dev'])
    roster=copy.deepcopy(prior);roster.update(dev=dev,status='revision2 CPU frozen; not authorized',
        selection='first five of original ten; deterministic height replacement only if missing',
        prior_roster_sha256=sha(OLD/'roster.eval-only.json'),replacements=replacements,
        original_dev_height_counts=dict(Counter(r['height_band'] for r in prior['dev'])),
        dev_height_counts=dict(Counter(r['height_band'] for r in dev)),
        test_roster_canonical_sha256=canonical_sha(prior['test']))
    roster['historical_scene_overlap']=roster.pop('scene_overlap')
    roster['scene_overlap']={'dev_scenes':sorted({r['scene'] for r in dev}),
        'dev_test_common_scenes':sorted({r['scene'] for r in dev}&{r['scene'] for r in prior['test']}),
        'note':'old census/dev2 overlaps retain original scene names in historical_scene_overlap; selected dev is a subset'}
    save('roster.eval-only.json',roster)
    h=history();save('historical-usage-and-process-time.json',h)
    from estimate_arm_capability_budget import estimate
    save('token-budget-estimate.json',estimate(h))
    from bvi.eef_arm_contract import prompt
    (D/'guidance-v4-arm-V.txt').write_text(prompt('V'),encoding='utf-8')
    frozen=json.loads((OLD/'executor-freeze.json').read_text())['source_sha256']
    old_sha=next(v for k,v in frozen.items() if k.replace('\\','/')=='src/bvi/eef_arm_executor.py')
    assert sha(ROOT/'src/bvi/eef_arm_executor.py')==old_sha
    save('executor-unchanged.json',{'version':'arm-coordinated-v2-600','sha256':old_sha,'modified':False,
        'prior_executor_freeze_sha256':sha(OLD/'executor-freeze.json'),
        'deferred_issue':'joint candidate desktop141/147/145 vs Cartesian87/95/96 unexpectedly longer; distant endpoint IK branch or non-shortest continuous-joint wrap are hypotheses only; no repair or new candidate run'})
    from freeze_arm_deployment_cpu import main as freeze
    freeze()
    deployment=json.loads((D/'deployment-freeze.json').read_text())
    if (D/'deployment-import-check.json').exists():
        imported=json.loads((D/'deployment-import-check.json').read_text())
        assert imported['archive_sha256']==deployment['archive_sha256'] and imported['imports_passed'] and not imported['cuda_initialized']
    auth=json.loads((OLD/'authorization.NOT-AUTHORIZED.json').read_text())
    auth.update(status='not_authorized',revision=2,authorization_evidence=None,expires_at_epoch=0,
        provider_requests_cap=125,gpu1_process_seconds_cap=13500,usd_cap='65',case_ids=[r['case_id'] for r in dev],
        conditions=['V','SAC','script'],source_sha256=deployment['source_sha256'],
        roster_sha256=sha(D/'roster.eval-only.json'),scheduling_clearance_confirmed=False,
        output_root='/home/pshuai/bvi-research/runs/arm-capability-development-revision2-20261001')
    save('authorization.NOT-AUTHORIZED.json',auth)
    tests=None
    if (D/'regression.xml').exists():
        suite=ET.parse(D/'regression.xml').getroot().find('testsuite');assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
        tests={'passed':int(suite.attrib['tests'])-int(suite.attrib['skipped']),'skipped':int(suite.attrib['skipped'])}
    save('cpu-tests.json',{'eef_regression':tests,'api_sends':0,'gpu_runs':0})
    times='\n'.join(f"| r{r['run']} | {r['position']:03d} | {r['elapsed_process_seconds']:.3f} | {r['returncode']} |" for r in h['processes'])
    names=', '.join(r['case_id'] for r in dev)
    common=f'''开发组：{names}。原前5例覆盖middle/high，无需替换；原开发10例没有low层，不挪用测试样本。测试30行逐字段保持不变，以清单canonical SHA验证。高度仍是初始碰撞网格最低world-z的支撑面估计。

清单SHA256 `{sha(D/'roster.eval-only.json')}`。

执行器协调v2源码SHA不变；每移动150/总600非官方条件、内部601/外层600补偿reset，五项判据与累计力不变。首次获批运行必须核验，不符合即停，不修改计数。SAC官方200；200/600是同次600步尝试前缀。

P入口、特权白名单、P提示、配对汇总已移除。locate_point仅在评估端记录返回点到目标碰撞形状三角形表面的最短无符号距离（米），当前base_link转目标局部坐标；内部点仍到表面计算距离，不能用AABB内部零距离替代。无有效返回点记NA。诊断独立保存，不写入工具结果、history或请求。

640 r1–r3历史40个逻辑请求，旧重试使实际发送为42次；37份唯一已知usage、5次发送usage未知（含补发）。输入中位数10169、最大14676，未知不当0，重复归档不重复统计。第25轮3万输入+4000输出按普通输入$10/百万及输出$50/百万为$0.50，125次$62.50。按全缓存写入$12.5/百万保守算则$71.875，可能提前触及$65；该额度不保证125次用满。保留逐次实际wire预留，已计/未知预留+新预留将超$65即发送前停批；不自动加钱，不改模型/提示/图像或截断历史。不以26.9万文字安全上限设批次额度。

历史Astra实测进程耗时（包括失败/删失，非600步耗时保证）：

| 历史批次 | case | 进程秒 | 返回码 |
|---|---:|---:|---:|
{times}

资源硬上限：125次API、USD65；每V1800秒、script600秒、SAC300秒，共13500秒（3小时45分）。每例V、SAC、script各一次，共15独立进程，同UID/index/seed核对初态。所有可能计费发送均计入125，每V最多25发送/25工具，输出4000token。未知响应停批核账、不重发，不继承旧额度。

脚本600步严格成功>=4/5才允许提出测试组提案；未达即停交用户决定，不修补。测试30例本次不运行。每次录像、同步TCP曲线及分析页、完整轨迹和媒体索引/哈希规则保持。开发结果只进diagnostics，测试正式日志需按预注册完成后用户确认。

关节空间候选141–147参考点比末端直线87–96更多，不合理。远端目标IK构型或连续关节未按最短方向绕均为待查假设；当前不修、不重跑候选。
'''
    (D/'README.md').write_text('# 修订二 CPU 回执：仅V、开发5例\n\n主线第2步：接口准备与失效归因。旧提案被否决；新提案未批准。当前API/GPU实验/物理动作均0，脚本成功率与Astra V均未运行。下一门：用户批准后真实初始化检查。\n\n'+common+f'\n相关CPU回归：{tests}。源码包/授权模板/原始usage及耗时的哈希均在本目录；未获运行授权，只本地提交，不发布。\n',encoding='utf-8')
    (D/'report-format.md').write_text('''# arm-report-revision2

三条件V/SAC/script分别给计划n、有效n、删失n、未运行n、严格成功k/n、Wilson95%；n=0写NA。固定分母不因删失缩小。200/600保存首次成功步数；SAC仅官方200，不外推600。按类别×高度与场景列k/n，明确场景内相关，Wilson仅未校正描述。报告脚本600成功子集的V；不含P与配对项。

漏斗保持：check_path接受、TCP到局部AABB<=0.05m、is_grasped、同一步抓住且目标升高>=0.05m、官方五项同时成立。原始标志/累积交集分列，严格成功直接取官方五项，SAC的check_path层NA。

每次locate_point单列case/turn/step/observation-id/valid/目标碰撞三角形表面距离m，缺点为NA；仅评估，不进反馈/history/请求，不单凭距离断言模型失败。逐例保留note/hindsight与实测行动，区分错误信念、执行偏离、环境挫败、无法判定。

脚本至少4/5且5例有效才可提出测试组提案；不足即停止，不修补。记录五项判据、累计力、抓空/重抓、工具调用、费用及媒体精确文件/日期/结果/哈希。执行器CPU、脚本、Astra V分开报告。
''',encoding='utf-8')
    proposal='# 官方出生站位抓取测评：修订二开发组授权提案\n\n**重新提交，未批准。旧USD1800/500次方案已撤回；以主设计修订二、已决定事项17为准。**\n\n'+common+'''\nAPI只走实验室服务器代理，密钥规则保持执行器计划第三节第1条：~/.config/bvi/openai.env，目录700/文件600，只运行时读，不复制或打印。获批后再确认调度许可、GPU1空闲、源码/资产哈希和真实初态。旧授权模板失效，新模板status=not_authorized。\n\n[CPU回执](../diagnostics/2026-10-01-arm-capability-revision2-cpu/README.md) · [冻结清单](../diagnostics/2026-10-01-arm-capability-revision2-cpu/roster.eval-only.json) · [历史实测凭据](../diagnostics/2026-10-01-arm-capability-revision2-cpu/historical-usage-and-process-time.json) · [源码哈希](../diagnostics/2026-10-01-arm-capability-revision2-cpu/deployment-freeze.json)\n\n批准后入口为冻结包内run_arm_capability_batch.py，参数authorization/roster/code/lab-root。缺媒体、接口错误、资源触界或未知响应即停批，不能自动重跑。当前不运行、不发布GitHub。\n'''
    (ROOT/'research/c2/docs/c2-eef-arm-development-authorization-proposal.md').write_text(proposal,encoding='utf-8')
    save('manifest.json',{'revision':2,'sha256':{p.name:sha(p) for p in D.iterdir() if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps({'dev':[r['case_id'] for r in dev],'replacements':replacements,'tokens':h['tokens']}))

if __name__=='__main__':main()
