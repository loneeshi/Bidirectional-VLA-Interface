"""Run C0/C2 development episodes; historical filename retained for compatibility."""
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
    if condition not in ('C0', 'C2'):
        raise ValueError('Active study supports C0/C2 only')
    # C0 has only twenty admissible one-shot skill/target executions. Four
    # additional API slots tolerate schema/incomplete responses without
    # granting a second physical attempt for any subtask.
    max_calls = 24 if condition == 'C0' else 40
    argv=[sys.executable,str(Path(__file__).with_name('run_coordinator.py')),
        '--paired-ppo-episode','--goal-tools','--organizer',
        '--continuation-condition',condition,'--seed',str(row['seed']),
        '--expected-plan-uid',row['plan_uid'],'--checkpoint-root',str(checkpoints),
        '--policy-type','rl_per_obj','--navigation-policy','official',
        '--manipulation-policy','official','--navigation-camera','fetch_nav',
        '--workspace-camera','--max-env-steps','7000','--max-wall-seconds','900',
        '--skill-wall-seconds','180','--organizer-slice-steps','500',
        '--max-calls',str(max_calls),'--provider','openai','--model','gpt-5.6-luna',
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
    p.add_argument('--prior',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--bridge-dir',type=Path,required=True)
    p.add_argument('--authorization-id',required=True)
    p.add_argument('--plans',type=int,default=2,choices=range(1,17))
    p.add_argument('--conditions',nargs='+',choices=('C0','C2'),
                   default=['C0','C2'])
    p.add_argument('--c0-panel',type=Path,
                   help='Existing C0 panel-status.json for a C2-only matched run')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if len(set(a.conditions)) != len(a.conditions):
        p.error('Duplicate conditions')
    if 'C2' in a.conditions and a.prior is None:
        p.error('C2 requires a pose prior')
    if 'C2' in a.conditions and 'C0' not in a.conditions and a.c0_panel is None:
        p.error('C2-only execution requires --c0-panel for initial-state binding')
    selected=select_plans(json.loads(a.source_manifest.read_text()))[:a.plans]
    if not a.execute:
        print(json.dumps({'plans':selected,'conditions':a.conditions,
                          'max_requests':len(selected)*sum(24 if c=='C0' else 40 for c in a.conditions),
                          'c2_budget_status':'legacy resource cap; recovery envelope not funded'},indent=2)); return
    if 'C2' in a.conditions:
        from bvi.feedback.spawn_prior import load_prior
        from bvi.pose_recovery import audit_pose_prior
        prior = load_prior(a.prior, [r['plan_uid'] for r in selected])
        audit = audit_pose_prior(prior['data'])
        if not audit['ready']:
            p.error('C2 pose recovery blocked: ' + ','.join(audit['reasons']))
    a.output.mkdir(parents=True,exist_ok=True)
    lock=a.output/'runner.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    path=a.output/'panel-status.json'
    spec={'schema':'gpt-sac-continuation/2','plans':selected,
          'conditions':a.conditions,'max_calls_by_condition':{
              c:(24 if c=='C0' else 40) for c in a.conditions},'max_steps':7000,
          'prior_sha256':(__import__('hashlib').sha256(a.prior.read_bytes()).hexdigest()
                          if 'C2' in a.conditions else None),
          'c0_panel_sha256':(__import__('hashlib').sha256(a.c0_panel.read_bytes()).hexdigest()
                             if a.c0_panel else None)}
    state=json.loads(path.read_text()) if path.exists() else {
        'specification':spec,'status':'running',
        'episodes':[dict(**row,condition=c,status='not_run',attempts=[])
                    for row in selected for c in a.conditions]}
    if state['specification'] != spec:
        lock.unlink(); raise ValueError('Resume specification mismatch')
    env=runtime_environment(OFFICIAL_MSHAB)
    env.update(PYTHONHASHSEED='0',
               BVI_DEBUG_ADAPTER_EXCEPTION='1')
    def save():
        state['summary']={c:{'planned':a.plans,
            'completed':sum(r['condition']==c and r['status']=='completed' for r in state['episodes']),
            'infrastructure_failed':sum(r['condition']==c and r['status']=='infrastructure_failure' for r in state['episodes']),
            'not_run':sum(r['condition']==c and r['status']=='not_run' for r in state['episodes'])}
            for c in a.conditions}
        atomic_json(path,state)
    save()
    try:
        for row in state['episodes']:
            if row['status']!='not_run': continue
            c0=next((r for r in state['episodes'] if r['seed']==row['seed'] and r['condition']=='C0'),None)
            if c0 is None and a.c0_panel is not None:
                c0 = next((r for r in json.loads(a.c0_panel.read_text())['episodes']
                           if r['seed']==row['seed'] and r['plan_uid']==row['plan_uid']
                           and r['condition']=='C0'), None)
            initial_hash=None
            if row['condition']!='C0':
                if c0 is None or c0['status']!='completed':
                    state['status']='blocked_missing_c0_pair'; break
                initial_hash=json.loads((Path(c0['attempts'][-1]['directory'])/'initial-state.json').read_text())['state_sha256']
            dest=a.output/row['condition']/f"seed-{row['seed']:03d}"/'attempt-001'
            # The coordinator atomically creates its own output directory.
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                raise FileExistsError('Attempt directory already exists; preserve it and use a new panel')
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
