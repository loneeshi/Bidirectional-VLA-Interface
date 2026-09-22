"""Run resumable C0/C1/C2 development episodes in two-plan chunks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bvi.bridge import atomic_json
from run_sac_interface_baseline import classify, runtime_environment, OFFICIAL_MSHAB
from run_ppo_sac_paired16 import select_plans


def command(row, condition, output, checkpoints, bridge, authorization,
            manifest, prior, initial_hash=None):
    argv=[sys.executable,str(Path(__file__).with_name('run_coordinator.py')),
        '--paired-ppo-episode','--goal-tools','--organizer',
        '--continuation-condition',condition,'--seed',str(row['seed']),
        '--expected-plan-uid',row['plan_uid'],'--checkpoint-root',str(checkpoints),
        '--policy-type','rl_per_obj','--navigation-policy','official',
        '--manipulation-policy','official','--navigation-camera','fetch_nav',
        '--workspace-camera','--max-env-steps','7000','--max-wall-seconds','900',
        '--skill-wall-seconds','180','--organizer-slice-steps','40',
        '--max-calls','40','--provider','openai','--model','gpt-5.6-luna',
        '--transport','bridge','--bridge-dir',str(bridge),'--bridge-timeout-seconds','120',
        '--authorization-id',authorization,'--max-api-cost-usd','0.05',
        '--request-cost-ceiling-usd','0.00125','--max-output-tokens','2048',
        '--max-input-bytes','512000','--feedback-profile','object_trajectory_v1',
        '--output',str(output)]
    if initial_hash:
        argv += ['--expected-initial-state-sha256',initial_hash]
    if condition=='C2':
        argv[argv.index('object_trajectory_v1')]='object_trajectory_prior_v1'
        argv += ['--spawn-prior',str(prior),'--evaluation-manifest',str(manifest)]
    return argv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--bridge-dir',type=Path,required=True)
    p.add_argument('--authorization-id',required=True)
    p.add_argument('--plans',type=int,default=2,choices=range(1,17))
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    selected=select_plans(json.loads(a.source_manifest.read_text()))[:a.plans]
    if not a.execute:
        print(json.dumps({'plans':selected,'conditions':['C0','C1','C2'],
                          'max_requests':len(selected)*3*40},indent=2)); return
    a.output.mkdir(parents=True,exist_ok=True)
    lock=a.output/'runner.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    path=a.output/'panel-status.json'
    spec={'schema':'gpt-sac-continuation/1','plans':selected,
          'conditions':['C0','C1','C2'],'max_calls':40,'max_steps':7000,
          'prior_sha256':__import__('hashlib').sha256(a.prior.read_bytes()).hexdigest()}
    state=json.loads(path.read_text()) if path.exists() else {
        'specification':spec,'status':'running',
        'episodes':[dict(**row,condition=c,status='not_run',attempts=[])
                    for row in selected for c in ('C0','C1','C2')]}
    if state['specification'] != spec:
        lock.unlink(); raise ValueError('Resume specification mismatch')
    env=runtime_environment(OFFICIAL_MSHAB)
    env.update(MS_ASSET_DIR='/home/pshuai/bvi-research/assets',PYTHONHASHSEED='0',
               BVI_DEBUG_ADAPTER_EXCEPTION='1')
    def save():
        state['summary']={c:{'planned':a.plans,
            'completed':sum(r['condition']==c and r['status']=='completed' for r in state['episodes']),
            'infrastructure_failed':sum(r['condition']==c and r['status']=='infrastructure_failure' for r in state['episodes']),
            'not_run':sum(r['condition']==c and r['status']=='not_run' for r in state['episodes'])}
            for c in ('C0','C1','C2')}
        atomic_json(path,state)
    save()
    try:
        for row in state['episodes']:
            if row['status']!='not_run': continue
            c0=next((r for r in state['episodes'] if r['seed']==row['seed'] and r['condition']=='C0'),None)
            initial_hash=None
            if row['condition']!='C0':
                if c0['status']!='completed':
                    state['status']='blocked_missing_c0_pair'; break
                initial_hash=json.loads((Path(c0['attempts'][-1]['directory'])/'initial-state.json').read_text())['state_sha256']
            dest=a.output/row['condition']/f"seed-{row['seed']:03d}"/'attempt-001'
            dest.mkdir(parents=True,exist_ok=False)
            argv=command(row,row['condition'],dest,a.checkpoint_root,a.bridge_dir,
                         a.authorization_id,a.source_manifest,a.prior,initial_hash)
            attempt={'directory':str(dest),'argv':argv,'status':'running','started_unix':time.time()}
            row['attempts'].append(attempt);row['status']='running';save()
            with dest.with_suffix('.log').open('w') as log:
                child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                attempt['pid']=child.pid;save()
                try: code=child.wait(timeout=990)
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM); child.wait(timeout=10); raise
            raw=json.loads((dest/'summary.json').read_text()) if (dest/'summary.json').exists() else None
            status,result=classify(code,raw)
            row.update(status=status,result=result)
            attempt.update(status=status,result=result,finished_unix=time.time());save()
            if status=='infrastructure_failure':
                state['status']='stopped_infrastructure';break
        else: state['status']='finished'
    except BaseException as exc:
        state.update(status='interrupted',error_type=type(exc).__name__);raise
    finally:
        save();lock.unlink()

if __name__=='__main__':main()
