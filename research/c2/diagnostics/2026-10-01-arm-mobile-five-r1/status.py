"""Read-only compact status and accounting; credential material is never read."""
import json
from pathlib import Path
from decimal import Decimal
import subprocess

root=Path('/home/pshuai/bvi-research/runs/arm-mobile-five-20261001-r1')
batch=json.loads((root/'batch.json').read_text())
ledger=json.loads((root/'api-ledger.json').read_text())
entries=ledger['entries']
result=dict(batch_status=batch['status'],completed_jobs=sum(j['status']=='completed' for j in batch['jobs']),
    gpu1_closed_process_seconds=batch['gpu1_process_seconds'],
    last_job=batch['jobs'][-1] if batch['jobs'] else None,
    provider_sends=len(entries),api_status=ledger['status'],
    estimated_definitive_usd=str(sum((Decimal(e['estimated_usd']) for e in entries
        if e['status'] in ('valid','invalid_call') and e.get('estimated_usd') is not None),Decimal(0))),
    outstanding_reserved_usd=str(sum((Decimal(e['reserve_usd']) for e in entries
        if e['status'] not in ('valid','invalid_call') or e.get('estimated_usd') is None),Decimal(0))),
    results=[])
for path in sorted(root.glob('arm-dev-*/*/result.json')):
    r=json.loads(path.read_text())
    row={k:r.get(k) for k in ('case_id','condition','status','simulator_actions','api_requests',
        'strict_pick_success','first_strict_success_step','termination_category','error_type')}
    row['recording_frames']=r.get('recording',{}).get('frames')
    row['commands']=len(r.get('commands',[]))
    if r.get('commands'):
        row['last_tool']=(r['commands'][-1].get('call') or {}).get('tool')
        row['last_feedback']=r['commands'][-1].get('result')
    result['results'].append(row)
result['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
print(json.dumps(result))
