import json
from pathlib import Path
from decimal import Decimal
root=Path('/home/pshuai/bvi-research/runs/arm-capability-development-revision2-20261001-r3')
b=json.loads((root/'batch.json').read_text())
a=json.loads((root/'api-ledger.json').read_text())
summary={'batch':b,'api_entries_total':len(a['entries']),
         'estimated_usd':str(sum((Decimal(e.get('estimated_usd') or e['reserve_usd']) for e in a['entries']),Decimal(0))),
         'api_status':a['status']}
summary['results']=[]
for p in sorted(root.glob('arm-dev-*/*/result.json')):
    r=json.loads(p.read_text())
    summary['results'].append({k:r.get(k) for k in ('case_id','condition','status','strict_pick_success','first_strict_success_step','simulator_actions','termination_category','error_type')})
last=b['jobs'][-1]
p=root/last['case_id']/last['condition']/'result.json'
if p.exists():
    r=json.loads(p.read_text())
    summary['last_result']={k:r.get(k) for k in ('status','error_type','simulator_actions','api_requests','strict_pick_success','termination_category')}
    summary['commands']=len(r.get('commands',[]))
    if r.get('commands'):
        summary['last_command']={k:r['commands'][-1].get(k) for k in ('turn','call','result')}
summary['batch']={k:b[k] for k in ('status','gpu1_process_seconds')}
summary['batch']['completed_jobs']=sum(j['status']=='completed' for j in b['jobs'])
summary['batch']['last_job']=b['jobs'][-1]
summary['results']=[{k:r[k] for k in ('case_id','condition','status','strict_pick_success','first_strict_success_step')} for r in summary['results']]
print(json.dumps(summary))
