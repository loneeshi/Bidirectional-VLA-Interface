"""One authorized serial lab batch; refusal is the default. No auto resume."""
import argparse
import copy
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from eef_mobile_server_broker import validate_scope,save
from run_arm_mobile_case import sha,GPU

CAPS={'V-mobile':1800}


def initial_api_entries(auth):
    entries=copy.deepcopy(auth.get('consumed_api_entries',[]))
    from bvi.eef_compact_api import committed_cost
    if len(entries)>=125 or committed_cost(entries)>Decimal(auth['usd_cap']):
        raise ValueError('prior API usage exhausts authorized scope')
    if any(e['status'] not in ('valid','invalid_call') or e.get('estimated_usd') is None for e in entries):
        raise ValueError('unreconciled prior API sends require review')
    for entry in entries:
        entry['original_request_id']=entry['request_id']
        entry['request_id']='prior:'+entry['request_id']
    return entries


def schedule(roster):
    return [(row['case_id'],condition) for row in roster['dev'] for condition in ('V-mobile',)]


def run(args):
    auth=json.loads(args.authorization.read_text());validate_scope(auth)
    if os.name!='posix' or not auth.get('scheduling_clearance_confirmed'):raise ValueError('lab scheduler clearance required')
    if sha(args.roster)!=auth['roster_sha256']:raise ValueError('roster hash mismatch')
    roster=json.loads(args.roster.read_text());jobs=schedule(roster)
    if len(jobs)!=5 or {case for case,_ in jobs}!=set(auth['case_ids']):raise ValueError('development scope')
    out=Path(auth['output_root']);out.mkdir(parents=True,exist_ok=False)
    from eef_server_transport import exclusive_lock
    with exclusive_lock(out.parent/'arm-capability.lock'):
        # Resource check is read-only, immediately before the first worker.
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if GPU in gpu:raise ValueError('approved GPU already in use')
        ledger=out/'api-ledger.json';save(ledger,{'authorization_sha256':sha(args.authorization),
            'status':'running','entries':initial_api_entries(auth)})
        prior_seconds=float(auth.get('consumed_gpu1_process_seconds',0))
        if not 0<=prior_seconds<auth['gpu1_process_seconds_cap']:raise ValueError('invalid prior GPU usage')
        state={'status':'running','jobs':[],'gpu1_process_seconds':prior_seconds,
               'prior_gpu1_process_seconds':prior_seconds,'api_ledger':'api-ledger.json'}
        save(out/'batch.json',state)
        for case,condition in jobs:
            if time.time()+CAPS[condition]>auth['expires_at_epoch']:raise ValueError('lease too short for next bounded process')
            if state['gpu1_process_seconds']+CAPS[condition]>auth['gpu1_process_seconds_cap']:raise ValueError('GPU total budget')
            folder=out/case/condition;folder.parent.mkdir(exist_ok=True)
            job={'case_id':case,'condition':condition,'status':'reserved','max_seconds':CAPS[condition]}
            state['jobs'].append(job);save(out/'batch.json',state)
            command=[sys.executable,str(args.code/'scripts/run_arm_mobile_case.py'),
                '--authorization',str(args.authorization),'--roster',str(args.roster),'--code',str(args.code),
                '--lab-root',str(args.lab_root),'--output',str(folder),'--ledger',str(ledger),
                '--case-id',case,'--condition',condition]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONPATH=str(args.code/'src'))
            start=time.monotonic()
            with (folder.parent/(condition+'-worker.log')).open('w') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:code=process.wait(timeout=CAPS[condition])
                except subprocess.TimeoutExpired:
                    process.kill();process.wait();code=-1
            elapsed=time.monotonic()-start;state['gpu1_process_seconds']+=elapsed
            job.update(status='completed' if code==0 else 'censored',exit_code=code,wall_seconds=elapsed)
            result_path=folder/'result.json'
            if result_path.exists():
                result=json.loads(result_path.read_text())
                if code!=0:
                    result.update(status='infrastructure_censored',termination_category='worker_process_limit_or_error');save(result_path,result)
                try:
                    from finalize_arm_capability_attempt import finalize
                    media=finalize(folder);job['media_complete']=media['complete']
                    index_path=out/'media-index.json'
                    index=json.loads(index_path.read_text()) if index_path.exists() else []
                    index.append({**media,'directory':str(folder.relative_to(out)/'media-source/delivery')})
                    save(index_path,index)
                except Exception as exc:job.update(media_complete=False,media_error_type=type(exc).__name__)
            else:job['media_complete']=False
            if code!=0 or not job['media_complete']:
                state['status']='stopped_requires_review';save(out/'batch.json',state);return 1
            save(out/'batch.json',state)
        from summarize_arm_mobile import summarize
        summary=summarize(out,roster);save(out/'summary.json',summary)
        state.update(status='mobile_development_complete_no_test_authorization',test_authorized=False);save(out/'batch.json',state)
    return 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('authorization','roster','code','lab-root'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
