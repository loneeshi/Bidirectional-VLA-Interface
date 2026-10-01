"""Read final evidence, close unused scope and retain an exact raw archive."""
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile

root=Path('/home/pshuai/bvi-research/runs/arm-mobile-five-20261001-r1')
batch=json.loads((root/'batch.json').read_text());assert len(batch['jobs'])==5
assert all(j['status']=='completed' and j['media_complete'] for j in batch['jobs'])
assert batch['status']=='mobile_development_complete_no_test_authorization'
ledger=json.loads((root/'api-ledger.json').read_text());entries=ledger['entries']
assert len(entries)<=125 and all(e['status'] in ('valid','invalid_call') and e.get('estimated_usd') for e in entries)
cost=sum((Decimal(e['estimated_usd']) for e in entries),Decimal(0));assert cost<=65
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
assert 'GPU-b7ebba23-7824-7601-df32-be55628936c3' not in gpu
ledger.update(status='completed_scope_closed',closed_at=datetime.now(timezone.utc).isoformat(),unused_scope_closed=True)
(root/'api-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
cases=[]
for path in sorted(root.glob('arm-dev-*/*/result.json')):
    r=json.loads(path.read_text());start=r['initial_robot_state']['base_world_eval_only']
    def distance(frame):
        pose=frame['base_world_eval_only']
        return math.hypot(pose[0][3]-start[0][3],pose[1][3]-start[1][3])
    cases.append(dict(case_id=r['case_id'],condition=r['condition'],status=r['status'],
        success=r['strict_pick_success'],first_success=r['first_strict_success_step'],
        steps=r['simulator_actions'],termination=r['termination_category'],funnel=r['funnel'],
        max_base_displacement_m=max(map(distance,r['frames']),default=0),
        final_force=r['official_final_info'].get('robot_cumulative_force'),
        commands=r['commands'],localization_eval_only=r.get('localization_eval_only',[])))
(root/'case-evidence.json').write_text(json.dumps(cases,indent=2)+'\n')
receipt=dict(at=datetime.now(timezone.utc).isoformat(),status='completed_scope_closed',
    provider_sends=len(entries),estimated_consumption_usd=str(cost),provider_bill_usd=None,
    outstanding_reservations=0,gpu1_process_seconds=batch['gpu1_process_seconds'],
    gpu1_idle_verified=True,unused_usd=str(Decimal(65)-cost),unused_requests=125-len(entries),
    unused_gpu_seconds=9000-batch['gpu1_process_seconds'],unused_scope_closed=True,test_authorized=False,
    gpu_processes=gpu.splitlines(),historical_runpod_storage='Not refreshed; outstanding reconciliation remains separate')
(root/'closeout.json').write_text(json.dumps(receipt,indent=2)+'\n')
archive=Path('/tmp/bvi-mobile-five-r1-complete.tar.gz')
if not archive.exists():
    with tarfile.open(archive,'w:gz') as tar:tar.add(root,arcname=root.name)
h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
receipt.update(raw_archive_sha256=h.hexdigest(),raw_archive_bytes=archive.stat().st_size)
Path('/tmp/bvi-mobile-five-closeout.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
