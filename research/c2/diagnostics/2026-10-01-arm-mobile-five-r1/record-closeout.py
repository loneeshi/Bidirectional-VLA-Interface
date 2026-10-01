"""Reconcile all definitive usage and close unused authorization without a bill claim."""
from datetime import datetime,timezone
from decimal import Decimal
import json
from pathlib import Path

diag=Path(__file__).resolve().parent
finance=diag.parents[5]/'BVI-research-plan-2026-09-14/finance'
close=json.loads((diag/'closeout.json').read_text())
ledger=json.loads((diag/'api-ledger.json').read_text())
assert ledger['status']=='completed_scope_closed'
assert len(ledger['entries'])==close['provider_sends']==108
assert sum((Decimal(e['estimated_usd']) for e in ledger['entries']),Decimal(0))==Decimal(close['estimated_consumption_usd'])
path=finance/'ledger-arm-mobile-five-20261001.json'
record=json.loads(path.read_text(encoding='utf-8'))
event=dict(id='arm-mobile-five-20261001-r1-complete',at=datetime.now(timezone.utc).isoformat(),
    status='completed_scope_closed',**{k:close[k] for k in ('provider_sends','estimated_consumption_usd',
        'provider_bill_usd','outstanding_reservations','gpu1_process_seconds','gpu1_idle_verified',
        'unused_usd','unused_requests','unused_gpu_seconds','unused_scope_closed')})
record.update(status='completed_scope_closed',provider_sends=108,
    estimated_consumption_usd=close['estimated_consumption_usd'],outstanding_reserved_usd='0',
    gpu1_process_seconds=close['gpu1_process_seconds'],gpu1_process_state='exited; GPU1 idle verified',
    api_entries_snapshot=ledger['entries'],closeout=close)
assert not any(e.get('id')==event['id'] for e in record['events'])
record['events'].append(event)
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
main_path=finance/'ledger.json';main=json.loads(main_path.read_text(encoding='gb18030'))
assert not any(e.get('id')==event['id'] for e in main['journal'])
main['journal'].append(event)
main_path.write_text(json.dumps(main,ensure_ascii=False,indent=2)+'\n',encoding='gb18030')
with (finance/'README.md').open('a',encoding='utf-8') as stream:
    stream.write('\nArm mobile five r1 closed: five valid processes, strict600 1/5 (226 steps), strict200 0/5; API108 estimated USD22.1865, invoice pending, no unknown reservations; GPU1 1663.368258s and idle verified. Remaining USD42.8135/17 sends/7336.631742s scope closed; no test authorization. Historical Runpod storage unrefreshed and separate.\n')
print(json.dumps(event))
