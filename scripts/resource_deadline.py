"""Independent Pod-stop watchdog; must be live-tested before paid experiments.

Run detached on the cloud host, separately from the workload and SSH session.
Default backend uses the tested Pod-scoped runpodctl 1.14 stop command.
The optional legacy API backend requires RUNPOD_API_KEY. Neither logs keys.
This calls the provider stop API; killing a workload process is insufficient.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request


def watch_cli(pod_id,deadline,log,ready,max_lease_seconds=3600,
              clock=time.time,sleep=time.sleep,run=subprocess.run):
    """Pod-scoped CLI watchdog. A ready file proves arming, not authentication.

    Validate a real self-stop on a disposable Pod before relying on this mechanism.
    The 1.14.15 template CLI was live-tested; use MCP to verify the final state.
    """
    if not re.fullmatch(r'[a-z0-9]{8,32}',pod_id): raise ValueError('Invalid Pod ID')
    if not 0<max_lease_seconds<=5400 or deadline-clock()>max_lease_seconds:
        raise ValueError('Deadline exceeds the explicit bounded lease')
    expected=os.environ.get('RUNPOD_POD_ID')
    if expected and expected!=pod_id: raise ValueError('Pod identity differs from host environment')
    version=run(['runpodctl','--version'],capture_output=True,text=True,timeout=10)
    if version.returncode or not version.stdout.strip().startswith('runpodctl 1.14.'):
        raise ValueError('This backend requires the tested runpodctl 1.14 command contract')
    def emit(event,**fields):
        with log.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),
                'event':event,'pod_id':pod_id,**fields})+'\n')
    ready.write_text(json.dumps({'pod_id':pod_id,'deadline_unix':deadline,'pid':os.getpid(),
        'backend':'pod_scoped_cli','authentication_verified_by_this_process':False}))
    emit('armed',deadline_unix=deadline)
    while clock()<deadline: sleep(min(2,deadline-clock()))
    for attempt in range(1,21):
        try:
            result=run(['runpodctl','stop','pod',pod_id],capture_output=True,timeout=30)
            emit('stop_return',attempt=attempt,returncode=result.returncode)
            if result.returncode==0:
                emit('stop_submitted_final_state_requires_control_plane_audit')
                return
        except Exception as exc:
            emit('stop_attempt_error',attempt=attempt,error_type=type(exc).__name__)
        sleep(3)
    raise RuntimeError('Resource stop failed; control-plane intervention required')


def api(pod_id, key, stop=False):
    if not re.fullmatch(r'[a-z0-9]{8,32}', pod_id):
        raise ValueError('Invalid explicit Pod ID')
    url=f'https://rest.runpod.io/v1/pods/{pod_id}' + ('/stop' if stop else '')
    req=urllib.request.Request(url, headers={'Authorization':'Bearer '+key},
                              method='POST' if stop else 'GET')
    with urllib.request.urlopen(req, timeout=15) as response:
        body=response.read()
        return json.loads(body) if body else {}


def watch(pod_id, deadline, key, log, ready, clock=time.time, sleep=time.sleep, request=api):
    def emit(event, **fields):
        with log.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),
                'event':event,'pod_id':pod_id,**fields})+'\n')
    if deadline-clock() > 3600:
        raise ValueError('Each lease is limited to one hour')
    current=request(pod_id,key)
    if current.get('id') != pod_id:
        raise ValueError('Provider returned an unexpected Pod identity')
    ready.write_text(json.dumps({'pod_id':pod_id,'deadline_unix':deadline,
                                 'pid':os.getpid(),'authentication_verified':True}))
    emit('armed',deadline_unix=deadline)
    while clock() < deadline:
        sleep(min(2,deadline-clock()))
    # Retry transient provider/network failures; the workload must independently
    # reject actions after its deadline. Provider stop success still needs audit.
    for attempt in range(1,21):
        try:
            request(pod_id,key,stop=True)
            emit('stop_submitted',attempt=attempt)
            current=request(pod_id,key)
            status=current.get('desiredStatus',current.get('status'))
            emit('provider_state',status=status)
            if status in ('EXITED','STOPPED'):
                emit('stop_verified')
                return
        except Exception as exc:
            emit('stop_attempt_error',attempt=attempt,error_type=type(exc).__name__)
        sleep(3)
    raise RuntimeError('Provider stop remains unverified; requires control-plane intervention')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pod-id',required=True)
    p.add_argument('--deadline-utc',required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--ready',type=Path,required=True)
    p.add_argument('--backend',choices=('pod-scoped-cli','legacy-api'),default='pod-scoped-cli')
    p.add_argument('--max-lease-seconds',type=int,default=3600)
    args=p.parse_args()
    key=os.environ.get('RUNPOD_API_KEY','')
    if args.backend=='legacy-api' and not key: p.error('RUNPOD_API_KEY required for legacy-api backend')
    parsed=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00'))
    if parsed.tzinfo is None: p.error('Deadline must include UTC/timezone')
    args.log.parent.mkdir(parents=True,exist_ok=True)
    if args.backend=='pod-scoped-cli':
        watch_cli(args.pod_id,parsed.timestamp(),args.log,args.ready,args.max_lease_seconds)
    else:
        watch(args.pod_id,parsed.timestamp(),key,args.log,args.ready)


if __name__=='__main__':
    main()
