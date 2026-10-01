"""Record the user's explicit new-scope launch approval before any provider send."""
from datetime import datetime,timezone
import json
from pathlib import Path

root=Path(__file__).resolve().parents[6]
finance=root/'BVI-research-plan-2026-09-14/finance'
main_path=finance/'ledger.json'
main=json.loads(main_path.read_text(encoding='gb18030'))
path=finance/'ledger-arm-mobile-five-20261001.json'
record=json.loads(path.read_text(encoding='utf-8'))
assert record['status']=='proposed_pending_budget_confirmation'
event=dict(id='arm-mobile-five-20261001-r1-approved',at=datetime.now(timezone.utc).isoformat(),
    status='authorized_ready_to_launch',authorization_evidence='User explicitly said 启动 after USD65/API125/GPU9000 proposal',
    currency='USD',usd_hard_cap='65',provider_requests_cap=125,gpu1_process_seconds_cap=9000,
    processes=5,condition='V-mobile',prior_budget_transferred=0,
    provider_sends_before_launch=0,provider_bill_usd=None)
assert not any(item.get('id')==event['id'] for item in main['journal'])
record.update(status='authorized_ready_to_launch',authorization=event)
record.setdefault('events',[]).append(event)
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
main['journal'].append(event)
main_path.write_text(json.dumps(main,ensure_ascii=False,indent=2)+'\n',encoding='gb18030')
with (finance/'README.md').open('a',encoding='utf-8') as stream:
    stream.write('\nArm mobile five r1 authorized by user launch instruction: new USD65/API125/GPU1 9000s scope, five V-mobile processes; no old balance transfer. Authorization recorded before sends; actual provider bill pending.\n')
Path(__file__).with_name('authorization-evidence.json').write_text(json.dumps(event,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(authorized=True,usd_cap='65',requests=125,gpu_seconds=9000)))
