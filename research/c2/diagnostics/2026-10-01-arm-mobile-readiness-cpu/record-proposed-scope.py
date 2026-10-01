"""Record proposal only; does not grant spending permission or launch anything."""
from datetime import datetime,timezone
import json
from pathlib import Path

finance=Path(__file__).resolve().parents[6]/'BVI-research-plan-2026-09-14/finance'
stamp=datetime.now(timezone.utc).isoformat()
record=dict(id='arm-mobile-five-20261001-r1',at=stamp,status='proposed_pending_budget_confirmation',
    user_instruction='fix code then start Astra; new cost ceiling awaiting confirmation',
    currency='USD',proposed_usd_hard_cap='65',provider_requests_cap=125,
    gpu1_process_seconds_cap=9000,processes=5,condition='V-mobile',prior_budget_transferred=0,
    provider_sends=0,estimated_consumption_usd='0',provider_bill_usd=None,
    gpu1_process_seconds=0,source_freeze_sha256='590063d9988a51814f311caa784d2d51565ba276883092a8ecad1404e07ebab2',
    evidence='repo/Bidirectional-VLA-Interface/research/c2/diagnostics/2026-10-01-arm-mobile-readiness-cpu/',
    historical_runpod_storage='not refreshed; outstanding reconciliation remains separate')
(finance/'ledger-arm-mobile-five-20261001.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
ledger_path=finance/'ledger.json'
ledger=json.loads(ledger_path.read_text(encoding='gb18030'))
assert isinstance(ledger['journal'],list)
assert not any(item.get('id')==record['id'] for item in ledger['journal'])
ledger['journal'].append(record)
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='gb18030')
with (finance/'README.md').open('a',encoding='utf-8') as stream:
    stream.write('\n\nArm mobile same-five preparation: 149 CPU regressions and server CPU hash/import gates passed; sends0/physical0. Proposed new scope USD65/API125/GPU1 9000s for five Astra processes, pending new ceiling confirmation; old balance0. See ledger-arm-mobile-five-20261001.json. No spending started.\n')
print(json.dumps(dict(proposal_recorded=True,authorized=False)))
