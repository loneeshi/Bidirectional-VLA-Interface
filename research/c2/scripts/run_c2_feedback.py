"""Three-plan C2 feedback panel, immutable inputs and explicit two-episode chunks.

No authorization is inferred from defaults. The local bridge separately enforces
the shared request/currency ledger including failed provider attempts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bvi.bridge import atomic_json
from bvi.feedback.c2_context import CaseBank, VARIANTS
from run_continuation_c012 import command as legacy_command
from run_sac_interface_baseline import OFFICIAL_MSHAB, runtime_environment, classify


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def command(row, variant, output, checkpoints, bridge, authorization, manifest, case_bank,
            reservation, episode_cost):
    argv=legacy_command(row,'C0',output,checkpoints,bridge,authorization,manifest,None,
                        row['initial_state_sha256'])
    changes={'--continuation-condition':'C2','--max-calls':'90','--max-wall-seconds':'1800',
             '--max-api-cost-usd':str(episode_cost),'--request-cost-ceiling-usd':str(reservation),
             '--max-input-bytes':'2000000','--feedback-profile':'raw_v0'}
    for flag,value in changes.items():argv[argv.index(flag)+1]=value
    argv+=['--c2-feedback',variant,'--evaluation-manifest',str(manifest)]
    if variant=='experience':argv+=['--c2-case-bank',str(case_bank)]
    return argv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','case-bank','smoke-evidence','checkpoint-root','bridge-dir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--authorization-id',required=True)
    p.add_argument('--request-reservation-usd',type=float,required=True)
    p.add_argument('--episode-cost-cap-usd',type=float,required=True)
    p.add_argument('--max-new-episodes',type=int,default=2,choices=(1,2))
    p.add_argument('--retry-infrastructure',action='store_true')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    plans=json.loads(a.manifest.read_text())['episodes']
    if len(plans)!=3 or len({r['plan_uid'] for r in plans})!=3:
        p.error('Expected three unique frozen plans')
    bank=CaseBank(a.case_bank,{r['plan_uid'] for r in plans})
    smoke=json.loads(a.smoke_evidence.read_text())
    if a.execute and (smoke.get('status')!='passed' or not smoke.get('held_place_verified')):
        p.error('Real base and held-Place controller smoke must pass before live C2')
    if a.request_reservation_usd<=0 or a.episode_cost_cap_usd < 90*a.request_reservation_usd:
        p.error('Cost reservation must cover 90 requests per episode')
    source=Path(__file__).resolve().parents[1]
    if a.execute and smoke.get('controller_sources') != {
            name:sha(source/'src/bvi'/name) for name in ('pose_goal.py','pose_recovery.py','lightnav_skill.py')}:
        p.error('Controller source differs from verified smoke; preserve old evidence and verify new version')
    files=[*sorted((source/'src').rglob('*.py')),*sorted((source/'scripts').glob('*.py'))]
    spec=dict(protocol='c2-pose-feedback/3',plans=plans,variants=list(VARIANTS),planned_per_variant=3,
        manifest_sha256=sha(a.manifest),bank_sha256=bank.sha256,smoke_sha256=sha(a.smoke_evidence),
        sources={str(f.relative_to(source)):sha(f) for f in files},authorization=a.authorization_id,
        request_reservation_usd=a.request_reservation_usd,episode_cost_cap_usd=a.episode_cost_cap_usd,
        max_calls=90,max_actions=7000,max_wall_seconds=1800,checkpoint_root=str(a.checkpoint_root.resolve()),
        bridge_dir=str(a.bridge_dir.resolve()))
    if not a.execute:
        print(json.dumps(spec,indent=2));return
    a.output.mkdir(parents=True,exist_ok=True)
    lock=a.output/'runner.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    path=a.output/'panel-status.json';state=None;child=None
    try:
        state=json.loads(path.read_text()) if path.exists() else dict(specification=spec,status='ready',
            episodes=[dict(**r,variant=v,status='not_run',attempts=[]) for r in plans for v in VARIANTS])
        if state['specification']!=spec:raise ValueError('Frozen resume inputs changed; create a new panel')
        if any(r['status']=='running' for r in state['episodes']):
            raise ValueError('Unresolved prior running attempt: inspect PID/summary before resume')
        count=0
        env=runtime_environment(OFFICIAL_MSHAB)
        env.update(PYTHONHASHSEED='0',BVI_DEBUG_ADAPTER_EXCEPTION='1')
        for row in state['episodes']:
            if row['status']!='not_run' and not (a.retry_infrastructure and row['status']=='infrastructure_failure'):continue
            if count>=a.max_new_episodes:break
            dest=a.output/row['variant']/f"seed-{row['seed']:03d}"/f"attempt-{len(row['attempts'])+1:03d}"
            if dest.exists():raise FileExistsError('Attempt already exists')
            dest.parent.mkdir(parents=True,exist_ok=True)
            argv=command(row,row['variant'],dest,a.checkpoint_root,a.bridge_dir,a.authorization_id,
                         a.manifest,a.case_bank,a.request_reservation_usd,a.episode_cost_cap_usd)
            attempt=dict(directory=str(dest),argv=argv,status='running',started_unix=time.time())
            row['attempts'].append(attempt);row['status']='running';state['status']='running';atomic_json(path,state)
            with dest.with_suffix('.log').open('w') as log:
                child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                attempt['pid']=child.pid;atomic_json(path,state)
                code=child.wait(timeout=1920);child=None
            raw=json.loads((dest/'summary.json').read_text()) if (dest/'summary.json').exists() else None
            status,result=classify(code,raw)
            row.update(status=status,result=result);attempt.update(status=status,result=result,finished_unix=time.time())
            atomic_json(path,state);count+=1
            if status=='infrastructure_failure':break
        state['status']='finished' if all(r['status']=='completed' for r in state['episodes']) else 'chunk_finished'
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if state is not None:
            state['summary']={v:dict(planned=3,**{s:sum(r['variant']==v and r['status']==s for r in state['episodes'])
                for s in ('completed','running','not_run','infrastructure_failure')}) for v in VARIANTS}
            atomic_json(path,state)
        lock.unlink()


if __name__=='__main__':main()
