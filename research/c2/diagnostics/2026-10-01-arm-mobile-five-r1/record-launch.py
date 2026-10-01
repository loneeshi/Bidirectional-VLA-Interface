"""Link live broker accounting to finance without treating reservations as bills."""
from datetime import datetime,timezone
from decimal import Decimal
import json
from pathlib import Path

diag=Path(__file__).resolve().parent
finance=diag.parents[5]/'BVI-research-plan-2026-09-14/finance'
path=finance/'ledger-arm-mobile-five-20261001.json'
record=json.loads(path.read_text(encoding='utf-8'))
launch=json.loads((diag/'launch.json').read_text())
ledger=json.loads((diag/'api-ledger-at-launch.json').read_text())
entries=ledger['entries']
known=sum((Decimal(e['estimated_usd']) for e in entries if e['status'] in ('valid','invalid_call')
           and e.get('estimated_usd') is not None),Decimal(0))
reserved=sum((Decimal(e['reserve_usd']) for e in entries if e['status'] not in ('valid','invalid_call')
             or e.get('estimated_usd') is None),Decimal(0))
event=dict(id='arm-mobile-five-20261001-r1-launched',at=datetime.now(timezone.utc).isoformat(),
    status='running',pid=launch['pid'],authorization_sha256=launch['authorization_sha256'],
    provider_sends_snapshot=len(entries),estimated_definitive_usd=str(known),
    outstanding_reserved_usd=str(reserved),provider_bill_usd=None,
    gpu1_process_state='first process active; elapsed time not yet closed',
    live_accounting='/home/pshuai/bvi-research/runs/arm-mobile-five-20261001-r1/api-ledger.json',
    gpu_accounting='/home/pshuai/bvi-research/runs/arm-mobile-five-20261001-r1/batch.json')
record.update(status='running',launch=launch,provider_sends=len(entries),
              estimated_consumption_usd=str(known),outstanding_reserved_usd=str(reserved),
              api_entries_snapshot=entries,live_accounting=event['live_accounting'],
              gpu1_process_state=event['gpu1_process_state'])
record.setdefault('events',[]).append(event)
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
main_path=finance/'ledger.json';main=json.loads(main_path.read_text(encoding='gb18030'))
assert not any(x.get('id')==event['id'] for x in main['journal'])
main['journal'].append(event)
main_path.write_text(json.dumps(main,ensure_ascii=False,indent=2)+'\n',encoding='gb18030')
(diag/'launch-accounting.json').write_text(json.dumps(event,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(event))
