"""Serial fixed PPO/SAC C1 runner. Default only validates/builds commands.

Requires real bound plan UIDs. Does not implement C2/C3, change scoring, or call
an API. Execution requires --execute and prechecked laboratory GPU availability.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from bvi.baseline_v2 import validate_bound_panel


def commands(config, python, checkpoint_root, output):
    rows=validate_bound_panel(config,'fixed','ppo')
    script=Path(__file__).with_name('run_coordinator.py')
    return [dict(seed=row['seed'],plan_uid=row['plan_uid'],argv=[
        str(python),str(script),'--dry-run','--seed',str(row['seed']),
        '--expected-plan-uid',row['plan_uid'],'--checkpoint-root',str(checkpoint_root),
        '--navigation-policy','official','--manipulation-policy','official',
        '--max-env-steps','7000','--max-calls','40','--max-wall-seconds','1200',
        '--output',str(Path(output)/f"seed-{row['seed']:03d}")]) for row in rows]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    jobs=commands(cfg,sys.executable,a.checkpoint_root.resolve(),a.output.resolve())
    if not a.execute:
        print(json.dumps(dict(status='commands_only_no_execution',jobs=jobs),indent=2));return
    if os.name!='posix':raise ValueError('Execution requires lab Linux process-group timeout handling')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(condition='C1_fixed_ppo_sac',planned_n=10,api_calls=0,episodes=[],
                status='running',not_run_seeds=list(range(10)))
    def save():
        (a.output/'panel-status.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3')
    started=time.monotonic()
    try:
        for job in jobs:
            if time.monotonic()-started>=12600:break
            row=dict(seed=job['seed'],plan_uid=job['plan_uid'])
            with (a.output/f"seed-{job['seed']:03d}.log").open('w') as log:
                child=subprocess.Popen(job['argv'],stdout=log,stderr=subprocess.STDOUT,
                                       env=env,start_new_session=True)
                try:
                    code=child.wait(timeout=min(1260,12600-(time.monotonic()-started)))
                    row.update(returncode=code,status='completed' if code==0 else 'infrastructure_failure')
                except BaseException as exc:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    row.update(status='infrastructure_failure',error=type(exc).__name__,returncode=child.returncode)
                    report['episodes'].append(row)
                    report['not_run_seeds'].remove(job['seed'])
                    save()
                    raise
            summary=a.output/f"seed-{job['seed']:03d}"/'summary.json'
            if summary.is_file():row['raw_summary']=json.loads(summary.read_text())
            else:row['status']='infrastructure_failure'
            report['episodes'].append(row);report['not_run_seeds'].remove(job['seed']);save()
            if row['status']=='infrastructure_failure':break
        report['status']='finished' if not report['not_run_seeds'] else 'stopped_partial'
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
